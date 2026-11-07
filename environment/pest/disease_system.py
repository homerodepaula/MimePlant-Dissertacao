"""Doencas das plantas (FarmGym).

FarmGym v3 (D-25): passou a existir INFECCAO PRIMARIA. Antes disso `espalhar_doenca` so propagava a partir de plantas ja
infectadas e nada infectava a primeira, entao nenhuma doenca ocorria em nenhum episodio.

FarmGym v4 (06/09/2026): cinco defeitos corrigidos, todos medidos.
  1. **A doenca nunca era apagada quando a celula esvaziava.** 47 celula-passos por episodio in-dist e 527 em humid
     mostravam doenca em celula SEM planta na observacao, e um terco dos plantios herdava a infeccao no proprio passo
     do plantio. Agora `sincronizar_plantas` e chamada pelo ambiente antes e depois da atualizacao das plantas.
  2. **A propagacao REESCREVIA o estagio do vizinho ja infectado para INICIAL.** Em surto denso a doenca se
     auto-limitava: um foco SEVERO (impacto 0,8) era rebaixado a INICIAL (0,2) pelo vizinho. Era bug, nao modelo.
  3. **`_condicoes_favoraveis` era um degrau binario sobre janelas retangulares.** Como os regimes tem desvio padrao de
     ~1,1 C e ~0,02 de umidade, qualquer janela ou contem a media (liga em ~100 % dos passos) ou nao contem (0 %). Por
     isso o regime seco era inerte POR CONSTRUCAO, e nao por agronomia. Agora a adequacao e um kernel suave em [0, 1]
     que modula a infeccao primaria, a propagacao e a progressao.
  4. **O defensivo curava 60 % de QUALQUER doenca, inclusive o mosaico viral** — que e justamente a que mais ocorre
     in-dist. Nao existe quimico curativo para virose de planta; era por isso que o pesticida derrubava o impacto de
     doenca em 97 %, uma bala de prata. Agora a eficacia e POR PATOGENO (`controle_quimico`), o que corrige a agronomia
     sem criar uma 16a acao: o professor passa a nao ter como curar parte da pressao.
  5. **A doenca era uma catraca**: das fases INICIAL e MODERADA so se avancava, e o clima desfavoravel apenas
     desacelerava. Agora ha recuperacao espontanea quando a adequacao e baixa.
Alem disso entra o **oidio**, o unico fungo que germina sem agua livre (quente e seco): e ele que da ao regime seco uma
doenca real e controlavel.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Iterable
from enum import Enum
import random

# Probabilidade por passo de uma planta SA ser infectada, quando a adequacao climatica da doenca e 1. Multiplicada pela
# adequacao (kernel suave) e pela suscetibilidade relativa da planta.
# v4 r4: 0,0015 -> 0,0008. Com o kernel continuo a doenca ocorria em 100 % dos episodios de TODAS as politicas e
# dominava o retorno (-98 por episodio na politica base contra -11 na que pulveriza), esmagando as demais decisoes.
P_INFECCAO_PRIMARIA = 0.0008

# Largura do kernel de adequacao, em unidades de meia-janela: 1,0 significa que a borda da janela otima ainda tem
# adequacao exp(-0,5) = 0,61, e que a adequacao cai suavemente fora dela em vez de zerar num degrau.
LARGURA_ADEQUACAO = 1.0


class EstagioDoenca(Enum):
    SAUDAVEL = 0
    INICIAL = 1
    MODERADO = 2
    SEVERO = 3
    RECUPERANDO = 4


class TipoDoenca:
    def __init__(self, nome: str, taxa_dispersao: float, severidade: float,
                 temp_otima: Tuple[float, float], umidade_otima: Tuple[float, float],
                 classe: str = 'fungo', controle_quimico: float = 0.6):
        self.nome = nome
        self.taxa_dispersao = taxa_dispersao
        self.severidade = severidade
        self.temp_otima = temp_otima
        self.umidade_otima = umidade_otima
        self.classe = classe
        # v4: fracao da doenca que o defensivo de amplo espectro leva a RECUPERANDO por aplicacao. Virose de planta
        # nao tem quimico curativo (0,00); fungo de folha responde bem (0,60-0,75); fungo de solo, pouco (0,30);
        # bacteriose, pouco (0,25).
        self.controle_quimico = controle_quimico


class SistemaDoencas:
    def __init__(self, tamanho_grade: Tuple[int, int] = (4, 4)):
        self.tamanho_grade = tamanho_grade
        self.doencas = {}
        self.ciclos_infeccao = {}
        self.doencas_plantas = {}

        self.tipos_doenca = {
            'infeccao_fungica': TipoDoenca(
                'infeccao_fungica', taxa_dispersao=0.15, severidade=0.7,
                temp_otima=(15, 25), umidade_otima=(0.7, 0.9),
                classe='fungo_folha', controle_quimico=0.60),
            'praga_bacteriana': TipoDoenca(
                'praga_bacteriana', taxa_dispersao=0.20, severidade=0.8,
                temp_otima=(20, 30), umidade_otima=(0.6, 0.8),
                classe='bacteria', controle_quimico=0.25),
            'mosaico_viral': TipoDoenca(
                'mosaico_viral', taxa_dispersao=0.10, severidade=0.5,
                temp_otima=(18, 28), umidade_otima=(0.4, 0.7),
                classe='virus', controle_quimico=0.00),
            'podridao_raiz': TipoDoenca(
                'podridao_raiz', taxa_dispersao=0.12, severidade=0.9,
                temp_otima=(10, 20), umidade_otima=(0.8, 1.0),
                classe='fungo_solo', controle_quimico=0.30),
            # v4: o unico patogeno que germina sem agua livre. Sem ele o regime seco fica sem doenca por construcao,
            # e la o pesticida nao teria nenhum efeito benefico possivel.
            'oidio': TipoDoenca(
                'oidio', taxa_dispersao=0.15, severidade=0.5,
                temp_otima=(20, 32), umidade_otima=(0.25, 0.60),
                classe='fungo_folha', controle_quimico=0.75),
        }

        self.taxas_progressao = {
            EstagioDoenca.INICIAL: 0.1,
            EstagioDoenca.MODERADO: 0.15,
            EstagioDoenca.SEVERO: 0.05,
            EstagioDoenca.RECUPERANDO: 0.2
        }

        self.taxa_recuperacao_base = 0.05
        self.impulso_recuperacao_tratamento = 0.15
        # v4: com adequacao baixa a infeccao pode REGREDIR, e nao so progredir mais devagar.
        self.taxa_regressao_max = 0.06

    # ── adequacao climatica ──────────────────────────────────────────────────────────────────────
    def adequacao(self, temp: float, umidade_ar: float, tipo_doenca: TipoDoenca) -> float:
        """v4: kernel suave em [0, 1], no lugar do degrau binario da v3. Vale 1 no centro da janela otima, ~0,61 na
        borda e decai continuamente fora dela."""
        def sino(valor: float, faixa: Tuple[float, float]) -> float:
            centro = 0.5 * (faixa[0] + faixa[1])
            meia = max(1e-6, 0.5 * (faixa[1] - faixa[0]))
            z = (valor - centro) / (meia * LARGURA_ADEQUACAO)
            return float(np.exp(-0.5 * z * z))
        return sino(temp, tipo_doenca.temp_otima) * sino(umidade_ar, tipo_doenca.umidade_otima)

    def _condicoes_favoraveis(self, temp: float, umidade_ar: float, tipo_doenca: TipoDoenca) -> bool:
        """Compatibilidade: 'favoravel' passa a ser adequacao acima da metade."""
        return self.adequacao(temp, umidade_ar, tipo_doenca) >= 0.5

    # ── ciclo de vida ────────────────────────────────────────────────────────────────────────────
    def sincronizar_plantas(self, posicoes_plantas: Iterable[Tuple[int, int]]):
        """v4: a doenca e DA PLANTA. Quando a celula e colhida ou a planta morre, a infeccao vai junto. Sem isto a
        observacao mostrava doenca em celula vazia e o replantio herdava a infeccao no mesmo passo."""
        vivas = set(posicoes_plantas)
        for pos in list(self.doencas_plantas.keys()):
            if pos not in vivas:
                del self.doencas_plantas[pos]

    def verificar_infeccao(self, x: int, y: int, nome_doenca: str) -> EstagioDoenca:
        pos = (x, y)
        if pos in self.doencas_plantas and nome_doenca in self.doencas_plantas[pos]:
            return self.doencas_plantas[pos][nome_doenca]
        return EstagioDoenca.SAUDAVEL

    def espalhar_doenca(self, dados_clima: Dict, posicoes_plantas: List[Tuple[int, int]]):
        temp = dados_clima.get('temperatura', 20)
        umidade_ar = dados_clima.get('umidade_ar', 0.5)
        plantas = set(posicoes_plantas)

        for nome_doenca, tipo_doenca in self.tipos_doenca.items():
            adeq = self.adequacao(temp, umidade_ar, tipo_doenca)
            if adeq <= 0.01:
                continue
            novas_infeccoes = []

            # infeccao primaria: planta sa, sorteio por passo, escalado pela adequacao
            for pos in posicoes_plantas:
                if nome_doenca in self.doencas_plantas.get(pos, {}):
                    continue
                if random.random() < P_INFECCAO_PRIMARIA * adeq * self._obter_susceptibilidade_planta(pos) / 0.3:
                    self.doencas_plantas.setdefault(pos, {})[nome_doenca] = EstagioDoenca.INICIAL
                    novas_infeccoes.append(pos)

            for pos in posicoes_plantas:
                estagio = self.doencas_plantas.get(pos, {}).get(nome_doenca)
                if estagio in (EstagioDoenca.MODERADO, EstagioDoenca.SEVERO):
                    taxa_dispersao = tipo_doenca.taxa_dispersao * adeq
                    if estagio == EstagioDoenca.SEVERO:
                        taxa_dispersao *= 1.5

                    for vizinho in self._obter_vizinhos(pos):
                        if vizinho not in plantas:
                            continue
                        # v4: NAO reescrever o vizinho que ja tem a doenca. Na v3 esta linha rebaixava um foco SEVERO
                        # a INICIAL e a epidemia se auto-limitava.
                        if nome_doenca in self.doencas_plantas.get(vizinho, {}):
                            continue
                        if random.random() < taxa_dispersao * self._obter_susceptibilidade_planta(vizinho):
                            self.doencas_plantas.setdefault(vizinho, {})[nome_doenca] = EstagioDoenca.INICIAL
                            novas_infeccoes.append(vizinho)

            if novas_infeccoes:
                self.ciclos_infeccao.setdefault(nome_doenca, []).append({
                    'passo': dados_clima.get('step', dados_clima.get('passo', 0)),
                    'novas_infeccoes': len(novas_infeccoes),
                    'condicoes': {'temp': temp, 'umidade_ar': umidade_ar, 'adequacao': adeq}
                })

    def progredir_doencas(self, dados_clima: Dict, tratamento_aplicado: bool = False):
        temp = dados_clima.get('temperatura', 20)
        umidade_ar = dados_clima.get('umidade_ar', 0.5)

        for pos, doencas in list(self.doencas_plantas.items()):
            for nome_doenca, estagio in list(doencas.items()):
                tipo_doenca = self.tipos_doenca[nome_doenca]
                adeq = self.adequacao(temp, umidade_ar, tipo_doenca)

                # v4: a adequacao modula a progressao continuamente (era x1,5 favoravel / x0,7 desfavoravel)
                taxa_progressao = self.taxas_progressao.get(estagio, 0) * (0.6 + 1.4 * adeq)

                taxa_recuperacao = self.taxa_recuperacao_base
                if tratamento_aplicado:
                    # v4: o impulso tambem e POR PATOGENO. Sem isto, `controle_quimico = 0` da virose nao adiantava
                    # nada: o defensivo nao a curava em `aplicar_tratamento`, mas acelerava a recuperacao dela aqui —
                    # o mesmo efeito nao declarado, por outra porta.
                    taxa_recuperacao += self.impulso_recuperacao_tratamento * tipo_doenca.controle_quimico

                if estagio == EstagioDoenca.RECUPERANDO:
                    if random.random() < taxa_recuperacao:
                        del doencas[nome_doenca]
                        if not doencas:
                            del self.doencas_plantas[pos]
                elif estagio == EstagioDoenca.SEVERO:
                    if random.random() < taxa_recuperacao:
                        doencas[nome_doenca] = EstagioDoenca.RECUPERANDO
                else:
                    # v4: com adequacao baixa a infeccao pode regredir, e nao so avancar mais devagar.
                    if random.random() < self.taxa_regressao_max * (1.0 - adeq):
                        doencas[nome_doenca] = EstagioDoenca.RECUPERANDO if estagio == EstagioDoenca.INICIAL \
                            else EstagioDoenca.INICIAL
                    elif random.random() < taxa_progressao:
                        doencas[nome_doenca] = self._obter_proximo_estagio(estagio)

    def aplicar_tratamento(self, x: int, y: int, tipo_tratamento: str = 'fungicida'):
        pos = (x, y)
        if pos not in self.doencas_plantas:
            return
        doencas = self.doencas_plantas[pos]

        for nome_doenca in list(doencas.keys()):
            tipo = self.tipos_doenca[nome_doenca]
            if tipo_tratamento == 'amplo_espectro':
                # v4: eficacia POR PATOGENO. Na v3 era 0,6 contra tudo, inclusive virose — o que fazia do pesticida
                # uma bala de prata (reduzia 97 % do impacto de doenca).
                if random.random() < tipo.controle_quimico:
                    doencas[nome_doenca] = EstagioDoenca.RECUPERANDO
            elif tipo_tratamento == 'fungicida' and tipo.classe.startswith('fungo'):
                doencas[nome_doenca] = EstagioDoenca.RECUPERANDO
            elif tipo_tratamento == 'antibiotico' and tipo.classe == 'bacteria':
                doencas[nome_doenca] = EstagioDoenca.RECUPERANDO

    # ── impacto ──────────────────────────────────────────────────────────────────────────────────
    _IMPACTO_ESTAGIO = {
        EstagioDoenca.INICIAL: 0.2,
        EstagioDoenca.MODERADO: 0.5,
        EstagioDoenca.SEVERO: 0.8,
        EstagioDoenca.RECUPERANDO: 0.1,
    }

    def calcular_impacto_doenca(self, x: int, y: int) -> float:
        pos = (x, y)
        if pos not in self.doencas_plantas:
            return 0.0
        impacto_total = sum(self.tipos_doenca[n].severidade * self._IMPACTO_ESTAGIO.get(e, 0)
                            for n, e in self.doencas_plantas[pos].items())
        return min(1.0, impacto_total)

    def calcular_impacto_controlavel(self, x: int, y: int) -> float:
        """v4: parte do impacto que o defensivo consegue tratar (pesa cada doenca por `controle_quimico`). E a
        estatistica que o professor usa para decidir pulverizar: pulverizar por causa de mosaico viral nao paga."""
        pos = (x, y)
        if pos not in self.doencas_plantas:
            return 0.0
        impacto = sum(self.tipos_doenca[n].severidade * self._IMPACTO_ESTAGIO.get(e, 0) * self.tipos_doenca[n].controle_quimico
                      for n, e in self.doencas_plantas[pos].items())
        return min(1.0, impacto)

    def obter_indicadores_doenca(self) -> List[float]:
        """v4: tres agregados de campo (a v3 devolvia 4 prevalencias por patogeno + a pressao; com 5 patogenos e um
        unico defensivo, a prevalencia por patogeno nao e acionavel e ficava constante em regimes inteiros).
          0 pressao media do campo | 1 pressao CONTROLAVEL media | 2 prevalencia (celulas infectadas / celulas)
        A diferenca entre 0 e 1 e exatamente a parte da doenca que o quimico nao resolve."""
        total = self.tamanho_grade[0] * self.tamanho_grade[1]
        pressao = sum(self.calcular_impacto_doenca(*pos) for pos in self.doencas_plantas)
        controlavel = sum(self.calcular_impacto_controlavel(*pos) for pos in self.doencas_plantas)
        infectadas = sum(1 for d in self.doencas_plantas.values() if d)
        return [min(1.0, pressao / total), min(1.0, controlavel / total), infectadas / total]

    def calcular_epidemiologia(self) -> Dict:
        total_posicoes = self.tamanho_grade[0] * self.tamanho_grade[1]

        contagem_estagios = {estagio: 0 for estagio in EstagioDoenca}
        for doencas in self.doencas_plantas.values():
            for estagio in doencas.values():
                contagem_estagios[estagio] += 1

        total_infectados = sum(contagem_estagios[e] for e in
                               (EstagioDoenca.INICIAL, EstagioDoenca.MODERADO, EstagioDoenca.SEVERO))
        prevalencia = total_infectados / total_posicoes if total_posicoes > 0 else 0

        pontuacao_severidade = (contagem_estagios[EstagioDoenca.MODERADO] * 1 +
                                contagem_estagios[EstagioDoenca.SEVERO] * 2) / total_posicoes if total_posicoes else 0

        return {
            'prevalencia': prevalencia,
            'pontuacao_severidade': pontuacao_severidade,
            'total_infectados': total_infectados,
            'distribuicao_estagios': {e.name: c for e, c in contagem_estagios.items()}
        }

    # ── auxiliares ───────────────────────────────────────────────────────────────────────────────
    def _obter_vizinhos(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx or dy) and 0 <= x + dx < self.tamanho_grade[0] and 0 <= y + dy < self.tamanho_grade[1]]

    def _obter_susceptibilidade_planta(self, pos: Tuple[int, int]) -> float:
        susceptibilidade = 0.3
        if pos in self.doencas_plantas:
            susceptibilidade += len(self.doencas_plantas[pos]) * 0.1
        return min(1.0, susceptibilidade)

    def _obter_proximo_estagio(self, estagio_atual: EstagioDoenca) -> EstagioDoenca:
        progressao = {
            EstagioDoenca.SAUDAVEL: EstagioDoenca.INICIAL,
            EstagioDoenca.INICIAL: EstagioDoenca.MODERADO,
            EstagioDoenca.MODERADO: EstagioDoenca.SEVERO,
            EstagioDoenca.SEVERO: EstagioDoenca.SEVERO,
            EstagioDoenca.RECUPERANDO: EstagioDoenca.SAUDAVEL
        }
        return progressao.get(estagio_atual, estagio_atual)

    def resetar(self):
        self.doencas = {}
        self.ciclos_infeccao = {}
        self.doencas_plantas = {}
