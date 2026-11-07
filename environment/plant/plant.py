import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional

# FarmGym v2 (S6, D-17): peso das ervas no fator de crescimento (era 0,5 fixo no código).
PESO_ERVAS_CRESCIMENTO = 0.5
# FarmGym v3 (06/09/2026, D-24): pragas, doencas, polinizadores e nutrientes passam a agir na planta.
# v4: 0,3 -> 0,12. O 0,3 nunca foi exercitado -- a populacao vivia congelada em ~4/celula (arrasto maximo de 1,3 %).
# Com a populacao viva ela satura em 100 e o arrasto vira permanente; e agora o fator tambem multiplica o RENDIMENTO,
# nao so a duracao, entao 0,3 sozinho tirava 30 % da colheita de quem nao pulveriza e esmagava as demais decisoes.
PESO_PRAGAS_CRESCIMENTO = 0.12     # fator x(1 - 0,12 . populacao/100): pragas retardam o ciclo e reduzem o fruto
# v4 r4: de volta a 0,5 (a v3 r8 tinha subido para 0,8 porque a doenca nao chegava a colheita; na v4 ela chega por
# dois caminhos, o fator e o rendimento, e 0,8 fazia a doenca sozinha valer -98 por episodio na politica base).
PESO_DOENCA_CRESCIMENTO = 0.5      # fator x(1 - 0,5 . impacto da doenca)
POP_MAX_PRAGAS = 100.0
FLORESCIMENTO_BASE, FLORESCIMENTO_POLINIZADORES = 0.7, 0.6   # fator x(0,7 + 0,6 . densidade) no florescimento (era 0,8 + 0,4)
# Lei do minimo (S9): o peso do fruto cresce a x(PISO + (1 - PISO) . min(1, min(N, P, K) / LIMIAR))
POLINIZACAO_PISO = 0.7                # peso do fruto x (0,7 + 0,3 . densidade media de polinizadores durante o florescimento)
LIMIAR_NUTRIENTE_FRUTO = 0.15
PISO_NUTRIENTE_FRUTO = 0.9
# FarmGym v4 (06/09/2026): o estresse passa a reduzir o RENDIMENTO, e nao so a duracao do ciclo. Ate a v3 o fruto
# acumulava `fator x 0,5` por passo e o estagio avancava quando o progresso (que soma a mesma coisa) atingia 25, entao
# a integral dava sempre ~25 x nutriente x polinizacao, INDEPENDENTE do estresse: clima, solo, ervas, doenca e pragas
# mudavam so o relogio. Agora, na saida da frutificacao, o peso e multiplicado pela media do fator de crescimento
# daquela janela, relativa a uma referencia agronomica FIXA (nunca re-derivada do professor, para nao virar circular).
FATOR_REFERENCIA_FRUTO = 1.2          # fator de crescimento medio de uma lavoura bem conduzida
FATOR_FRUTO_MIN, FATOR_FRUTO_MAX = 0.25, 1.5   # teto acima de 1: cuidado acima da referencia rende mais

class EstagioPlanta(Enum):
    SEMENTE = "semente"
    GERMINACAO = "germinacao"
    CRESCIMENTO = "crescimento"
    FLORESCIMENTO = "florescimento"
    FRUTIFICACAO = "frutificacao"
    COLHEITA = "colheita"
    MORTA = "morta"

class Planta:
    def __init__(self, tamanho_grade: Tuple[int, int] = (10, 10)):
        self.tamanho_grade = tamanho_grade
        self.resetar()

    def resetar(self):
        self.plantas = {}
        self.passo_tempo = 0

    def adicionar_planta(self, x: int, y: int, tipo_planta: str = "padrao"):
        if 0 <= x < self.tamanho_grade[0] and 0 <= y < self.tamanho_grade[1]:
            self.plantas[(x, y)] = EstadoPlanta(tipo_planta)

    def atualizar(self, clima: Dict, solo: Dict, polinizadores: Dict, ervas_daninhas: Dict,
                  pragas: Dict = None, doencas: Dict = None):
        """`pragas`: {pos: populacao}; `doencas`: {pos: impacto em [0, 1]} (v3; opcionais por compatibilidade)."""
        self.passo_tempo += 1
        plantas_para_remover = []
        pragas = pragas or {}
        doencas = doencas or {}

        for pos, planta in self.plantas.items():
            if planta.estagio == EstagioPlanta.MORTA:
                continue

            fator_crescimento = self._calcular_fator_crescimento(clima, solo, polinizadores, ervas_daninhas, pos,
                                                                 pragas.get(pos, 0), doencas.get(pos, 0.0))
            nutrientes = solo.get(pos, {}).get('nutrientes', {})
            minimo = min(nutrientes.get('N', 0.0), nutrientes.get('P', 0.0), nutrientes.get('K', 0.0)) if nutrientes else 0.0
            fator_nutriente = PISO_NUTRIENTE_FRUTO + (1.0 - PISO_NUTRIENTE_FRUTO) * min(1.0, minimo / LIMIAR_NUTRIENTE_FRUTO)
            val_pol = polinizadores.get(pos, 0)
            densidade_polinizadores = val_pol.get('densidade', 0) if isinstance(val_pol, dict) else val_pol
            planta.atualizar(fator_crescimento, fator_nutriente, float(densidade_polinizadores))

            if planta.estagio == EstagioPlanta.MORTA:
                plantas_para_remover.append(pos)

        for pos in plantas_para_remover:
            del self.plantas[pos]

    def _calcular_fator_crescimento(self, clima: Dict, solo: Dict, polinizadores: Dict, ervas_daninhas: Dict, pos: Tuple[int, int],
                                    populacao_pragas: float = 0.0, impacto_doenca: float = 0.0) -> float:
        fator = 1.0

        temp = clima.get('temperatura', 20)
        umidade_ar = clima.get('umidade_ar', 0.5)
        chuva = clima.get('chuva', 0)

        if temp < 10 or temp > 35:
            fator *= 0.7
        if umidade_ar < 0.3 or umidade_ar > 0.9:
            fator *= 0.8
        if chuva < 2.0:
            fator *= 0.9

        qualidade_solo = solo.get(pos, {}).get('qualidade', 0.5)
        fator *= (1.0 + qualidade_solo)

        if self.plantas[pos].estagio == EstagioPlanta.FLORESCIMENTO:
            val_pol = polinizadores.get(pos, 0)
            densidade_polinizadores = val_pol.get('densidade', 0) if isinstance(val_pol, dict) else val_pol
            fator *= (FLORESCIMENTO_BASE + FLORESCIMENTO_POLINIZADORES * densidade_polinizadores)

        densidade_ervas = ervas_daninhas.get(pos, {}).get('densidade', 0)
        fator *= (1.0 - PESO_ERVAS_CRESCIMENTO * densidade_ervas)

        # v3: pragas e doencas retardam o ciclo (o pesticida age nas duas; as aves comem pragas)
        fator *= (1.0 - PESO_PRAGAS_CRESCIMENTO * min(1.0, float(populacao_pragas) / POP_MAX_PRAGAS))
        fator *= (1.0 - PESO_DOENCA_CRESCIMENTO * min(1.0, float(impacto_doenca)))

        return max(0.0, fator)

    def obter_observacao(self) -> Dict:
        obs = {}
        for pos, planta in self.plantas.items():
            obs[pos] = {
                'estagio': planta.estagio.value,
                'saude': planta.saude,
                'progresso_crescimento': planta.progresso_crescimento,
                'peso_fruto': planta.peso_fruto
            }
        return obs

    def pode_colher(self, pos: Tuple[int, int]) -> bool:
        if pos in self.plantas:
            return self.plantas[pos].estagio == EstagioPlanta.COLHEITA
        return False

    def colher(self, pos: Tuple[int, int]) -> float:
        if self.pode_colher(pos):
            quantidade_rendimento = self.plantas[pos].peso_fruto
            del self.plantas[pos]
            return quantidade_rendimento
        return 0.0

class EstadoPlanta:
    def __init__(self, tipo_planta: str = "padrao"):
        self.tipo_planta = tipo_planta
        self.estagio = EstagioPlanta.SEMENTE
        self.saude = 1.0
        self.progresso_crescimento = 0.0
        self.peso_fruto = 0.0
        self.idade = 0
        # Rastreio da qualidade de cuidado acumulada nas fases ativas.
        # Usado para calcular o multiplicador de rendimento na colheita:
        # yield_efetivo = peso_fruto × max(0.3, 0.3 + fator_medio)
        # Baseline fator≈0.70 (clima típico) → mult=1.0 (sem mudança)
        # Bot com solo fértil e sem ervas fator≈1.0+ → mult=1.3+ (+30%)
        # Negligência fator≈0.4 → mult=0.7 (-30%)
        self._fator_soma = 0.0
        self._fator_n    = 0
        # v3: polinizacao -- media da densidade de polinizadores vista durante o florescimento; congelada ao entrar em
        # frutificacao como fator do peso do fruto (POLINIZACAO_PISO + (1 - POLINIZACAO_PISO) . media)
        self._polin_soma = 0.0
        self._polin_n = 0
        self.fator_polinizacao = 1.0
        # v4: media do fator de crescimento durante a FRUTIFICACAO, congelada na saida do estagio como fator do peso
        self._frut_soma = 0.0
        self._frut_n = 0
        self.fator_enchimento = 1.0

        self.limiares_estagio = {
            EstagioPlanta.SEMENTE: 5,
            EstagioPlanta.GERMINACAO: 10,
            EstagioPlanta.CRESCIMENTO: 20,
            EstagioPlanta.FLORESCIMENTO: 15,
            EstagioPlanta.FRUTIFICACAO: 25,
            EstagioPlanta.COLHEITA: 10
        }

    @property
    def fator_medio(self) -> float:
        """Fator de crescimento médio durante as fases ativas (CRESCIMENTO→COLHEITA)."""
        return self._fator_soma / self._fator_n if self._fator_n > 0 else 1.0

    @property
    def multiplicador_rendimento(self) -> float:
        # v4: NAO e mais usado na colheita (env.py). Fica como KPI de diagnostico do cuidado medio.
        """KPI de diagnostico (v4): ate a v3 multiplicava a colheita, o que descontava o estresse duas vezes e
        expunha um exploit (a qualidade do solo era lida no passo da colheita)."""
        """Multiplicador de rendimento na colheita baseado na qualidade do cuidado.

        Fórmula: max(0.3, 0.3 + fator_medio)
          fator_medio ≈ 0.70 (baseline climático) → mult = 1.00  (sem diferença)
          fator_medio ≈ 1.00 (bot fertilizando)   → mult = 1.30  (+30%)
          fator_medio ≈ 1.40 (condições ótimas)   → mult = 1.70  (+70%)
          fator_medio ≈ 0.40 (negligência)        → mult = 0.70  (-30%)
        """
        return max(0.3, 0.3 + self.fator_medio)

    def atualizar(self, fator_crescimento: float, fator_nutriente: float = 1.0, densidade_polinizadores: float = None):
        self.idade += 1
        self.progresso_crescimento += fator_crescimento * 0.5

        if self.estagio == EstagioPlanta.FLORESCIMENTO and densidade_polinizadores is not None:
            self._polin_soma += float(densidade_polinizadores)
            self._polin_n += 1

        limiar_atual = self.limiares_estagio.get(self.estagio, float('inf'))
        if self.progresso_crescimento >= limiar_atual:
            estagio_anterior = self.estagio
            self._avancar_estagio()
            if estagio_anterior == EstagioPlanta.FLORESCIMENTO and self._polin_n > 0:
                self.fator_polinizacao = POLINIZACAO_PISO + (1.0 - POLINIZACAO_PISO) * (self._polin_soma / self._polin_n)
            if estagio_anterior == EstagioPlanta.FRUTIFICACAO and self._frut_n > 0:
                # v4: o rendimento passa a depender do cuidado MEDIO durante o enchimento do fruto
                media = self._frut_soma / self._frut_n
                self.fator_enchimento = float(min(FATOR_FRUTO_MAX, max(FATOR_FRUTO_MIN, media / FATOR_REFERENCIA_FRUTO)))
                self.peso_fruto *= self.fator_enchimento

        self._atualizar_saude(fator_crescimento)

        # Acumular fator nas fases ativas para calcular multiplicador_rendimento
        _FASES_ATIVAS = (EstagioPlanta.CRESCIMENTO, EstagioPlanta.FLORESCIMENTO,
                         EstagioPlanta.FRUTIFICACAO, EstagioPlanta.COLHEITA)
        if self.estagio in _FASES_ATIVAS:
            self._fator_soma += fator_crescimento
            self._fator_n    += 1

        if self.estagio == EstagioPlanta.FRUTIFICACAO:
            # lei do minimo (N, P, K) e polinizacao (fixada na saida do florescimento); a media do fator desta janela
            # multiplica o total na saida do estagio (v4)
            self.peso_fruto += fator_crescimento * 0.5 * fator_nutriente * self.fator_polinizacao
            self._frut_soma += float(fator_crescimento)
            self._frut_n += 1

    def _avancar_estagio(self):
        ordem_estagios = [
            EstagioPlanta.SEMENTE,
            EstagioPlanta.GERMINACAO,
            EstagioPlanta.CRESCIMENTO,
            EstagioPlanta.FLORESCIMENTO,
            EstagioPlanta.FRUTIFICACAO,
            EstagioPlanta.COLHEITA
        ]

        indice_atual = ordem_estagios.index(self.estagio)
        if indice_atual < len(ordem_estagios) - 1:
            self.estagio = ordem_estagios[indice_atual + 1]
            self.progresso_crescimento = 0.0
        elif self.estagio == EstagioPlanta.COLHEITA:
            if self.progresso_crescimento >= self.limiares_estagio[self.estagio]:
                self.estagio = EstagioPlanta.MORTA

    def _atualizar_saude(self, fator_crescimento: float):
        if fator_crescimento < 0.3:
            self.saude -= 0.05
        elif fator_crescimento > 0.8:
            self.saude = min(1.0, self.saude + 0.03)
        elif fator_crescimento > 0.5:
            self.saude = min(1.0, self.saude + 0.01)

        self.saude = max(0.0, self.saude)

        if self.saude <= 0:
            self.estagio = EstagioPlanta.MORTA
