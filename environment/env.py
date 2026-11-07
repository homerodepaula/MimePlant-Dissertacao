"""FarmGym — ambiente de manejo agrícola (reimplementação própria do Farm-gym sobre Gymnasium).

FarmGym v2 (defesa; D-17 em 0_contexto/DECISOES.md, 05/09/2026) — o que mudou em relação à qualificação (v1):
  S1   a colheita é paga UMA vez, no passo em que ocorre (o terminal não soma mais `total_colheita × 5`);
  S2a  sem bônus de aves residentes (+0,2/passo era pago a qualquer política, inclusive "só esperar");
  S2b  sem bônus de estágio × saúde por passo (pagava ~700/episódio por plantar e esperar; shaping, se
       necessário, fica no wrapper do agente como potencial — Ng et al. 1999);
  S4   penalidade terminal de químicos LINEAR (2,0 por aplicação por célula), sem o teto de −10, e contando
       TODAS as aplicações do episódio (antes só as ainda ativas no dia 365 eram contadas);
  S5   `info['recompensa_comp']` traz a decomposição da recompensa em todo passo (custo_acao, colheita, clima,
       ervas, doenca; no último passo também frutos_pendentes, solo_final, polinizadores_final, quimicos_final);
  RNG  `reset(seed)` semeia também o módulo `random` (doenças, migração e eventos extremos o usam) — sem isso
       a "avaliação pareada" não era pareada nesses subsistemas;
  OBS  a observação respeita de fato o espaço Box[0, 1]: o bloco de migração (5 valores) era contagem crua
       (chegava a 10+) e a população de aves por célula podia passar de 1; agora são normalizados e o vetor
       inteiro é recortado em [0, 1] (só a temperatura abaixo de 0 °C, rara, perde informação);
  CLIMA `Clima.resetar()` volta a estação para 'primavera' (antes a estação vazava de um episódio para o
       seguinte; com episódios de 365 passos ela coincidia por acaso, com episódios truncados não).
Observação (148 dimensões, mesma ordem), ações (15), máscara e KPIs são os mesmos da qualificação.

FarmGym v3 (defesa; D-24 em 0_contexto/DECISOES.md, 06/09/2026) -- os subsistemas que nao tinham efeito passam a agir:
  DOENCA  infeccao primaria sob clima favoravel (`disease_system.P_INFECCAO_PRIMARIA`); a doenca retarda o ciclo da planta
          (`plant.PESO_DOENCA_CRESCIMENTO`); o pesticida trata (60 % das doencas da celula -> recuperacao) e acelera a
          recuperacao enquanto esta ativo; mascara: pesticida valido sempre que ha plantas;
  PRAGAS  a populacao retarda o ciclo (`plant.PESO_PRAGAS_CRESCIMENTO`); o pesticida abate 80 % na aplicacao (como o
          herbicida faz com as ervas); as aves comem pragas (ja existia);
  AVES    `Passaros` (o modelo que o espantalho afasta) passa a ser a populacao observada por celula e come uma fracao do
          peso do fruto por passo nas plantas em frutificacao/colheita (`TAXA_PREDACAO_AVES`) -- o espantalho protege;
  EVENTOS eventos extremos somam `obter_impacto_extremo(categoria)` ao custo da acao executada durante o evento
          (componente `eventos` da decomposicao); limiares de calor/frio 35 C / 5 C;
  SOLO    nutrientes decaem 0,004/passo (era 0,01) e o peso do fruto segue a lei do minimo de N, P, K (`plant.py`);
  POLIN.  clima e cerca viva modulam o crescimento dos polinizadores, nao o estoque (nao colapsam mais a zero);
  INSTAL. espantalho e cerca coexistem na mesma celula (`facilities.py`).
"""
import random
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, List, Any, Optional
import sys
import os

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
for subdir in ['plant', 'birds', 'pollinators', 'soil', 'weather', 'weeds', 'pest', 'cides-fertilizers', 'facilities']:
    sys.path.insert(0, os.path.join(diretorio_atual, subdir))

from plant import Planta, EstagioPlanta
from birds import Passaros
from pollinators import Polinizadores
from soil import Solo
from weather import Clima
from weeds import ErvasDaninhas
from pest import Pragas
from cides_fertilizers import DefensivosFertilizantes, TipoNutriente, TipoDefensivo
from facilities import Instalacoes, TipoInstalacao, TipoEspantalho

from weather import EventosClimaticosExtremos
from pest import SistemaDoencas

# ── Constantes da recompensa (FarmGym v2) ────────────────────────────────────────────────────────
PENALIDADE_TEMP_ALTA = 1.0            # por passo, com plantas, se temperatura > 35 °C
PENALIDADE_TEMP_BAIXA = 0.8           # por passo, com plantas, se temperatura < 5 °C
PESO_ERVAS = 0.5                      # × densidade média de ervas nas células plantadas, por passo
PESO_DOENCA = 0.25                    # × impacto total de doença, por passo (teto abaixo); v3: 0,5 -> 0,25 (o efeito principal
TETO_DOENCA = 1.0                     # da doença passa pela colheita, via plant.PESO_DOENCA_CRESCIMENTO); teto 2,0 -> 1,0
PESO_FRUTOS_PENDENTES = 2.0           # terminal: × peso dos frutos não colhidos
PESO_SOLO_FINAL = 10.0                # terminal: × qualidade média do solo
PESO_POLINIZADORES_FINAL = 5.0        # terminal: × densidade média de polinizadores
PENALIDADE_QUIMICO_POR_CELULA = 2.0   # terminal: × aplicações de defensivo por célula (linear, sem teto)
ACOES_QUIMICAS = ('herbicida', 'pesticida')
# S7b (FarmGym v2): o herbicida abate imediatamente esta fração da densidade de ervas em cada célula (o efeito
# residual de 14 dias continua). Antes só havia o residual (~8 %/dia), menor que o crescimento das ervas, e o
# herbicida nunca compensava o custo e o dano ao solo em nenhuma configuração medida.
EFICACIA_HERBICIDA = 0.7
# FarmGym v3 (06/09/2026, D-24)
EFICACIA_PESTICIDA = 0.8              # fracao da populacao de pragas abatida na aplicacao (o residual continua em pest_core)
TAXA_PREDACAO_AVES = 0.02             # fracao do peso do fruto comida por passo quando a populacao de aves da celula esta no maximo
POP_MAX_AVES = 50.0                   # `Passaros.populacao_maxima`
POP_MAX_PRAGAS_OBS = 100.0            # `Pragas.populacao_maxima`, usada para normalizar a observacao
# FarmGym v4: a praga tambem DESTROI o produto, nao so atrasa o ciclo. Ate a v3 o unico canal era o fator de
# crescimento, que muda so a DURACAO e e indistinguivel de ervas, clima e solo -- o agente nao conseguia atribuir a
# perda a praga, e o pesticida so pagaria por throughput.
TAXA_DANO_PRAGAS = 0.003              # fracao do peso do fruto destruida por passo com a praga da celula no teto

# ── FarmGym v4: layout da observacao (uma fonte so; protocolo.py, rede_lstm.py e wrapper.py espelham daqui) ──────
# 0-3 clima | 4 passo | 5-9 eventos extremos | 10 pressao de doenca | 11 pressao CONTROLAVEL | 12 prevalencia
# 13 pressao migratoria de aves | 14 praga maxima nas celulas PLANTADAS | 15 fracao de celulas com cerca
# 16.. celulas x 10: [tem_planta, saude, estagio, umidade_solo, qualidade_solo, ervas, doenca, aves, pragas, espantalho]
OBS_BASE = 5
OBS_AMBIENTAL = 11
INICIO_GRADE = OBS_BASE + OBS_AMBIENTAL   # 16
TAM_CELULA = 10
N_CELULAS = 16
OBS_DIM = INICIO_GRADE + TAM_CELULA * N_CELULAS   # 176
FORCA_MAX_ESPANTALHO = 2.0            # espantalho avancado; usada so para normalizar a observacao
CATEGORIA_EVENTO = {'plantar': 'plantar', 'colher': 'colher', 'regar': 'irrigar',
                    'fertilizar_N': 'fertilizar', 'fertilizar_P': 'fertilizar', 'fertilizar_K': 'fertilizar', 'fertilizar_C': 'fertilizar',
                    'herbicida': 'controle_pragas', 'pesticida': 'controle_pragas'}


class AmbienteFazendaGym(gym.Env):
    def __init__(self, tamanho_grade: Tuple[int, int] = (4, 4), passos_maximos: int = 365):
        self.tamanho_grade = tamanho_grade
        self.passos_maximos = passos_maximos
        self.passo_atual = 0

        self.planta = Planta(tamanho_grade)
        self.passaros = Passaros(tamanho_grade)
        self.polinizadores = Polinizadores(tamanho_grade)
        self.solo = Solo(tamanho_grade)
        self.clima = Clima(tamanho_grade)
        self.ervas_daninhas = ErvasDaninhas(tamanho_grade)
        self.pragas = Pragas(tamanho_grade)
        self.defensivos_fertilizantes = DefensivosFertilizantes(tamanho_grade)
        self.instalacoes = Instalacoes(tamanho_grade)

        self.eventos_extremos = EventosClimaticosExtremos()
        self.sistema_doencas = SistemaDoencas(tamanho_grade)

        self._configurar_espaco_acoes()
        self._configurar_espaco_observacao()

        self.recompensa_episodio = 0
        self.historico_colheita = []
        self.total_aplicacoes_quimicas = 0
        self.fruto_perdido_aves = 0.0
        self.fruto_perdido_pragas = 0.0

    def _configurar_espaco_acoes(self):
        self.nomes_acoes = [
            'plantar', 'colher', 'regar', 'fertilizar_N', 'fertilizar_P',
            'fertilizar_K', 'fertilizar_C', 'herbicida', 'pesticida',
            'colocar_espantalho_basico', 'colocar_espantalho_avancado', 'remover_espantalho',
            'colocar_cerca', 'esperar'
        ]
        # v4: 'observar' saiu. Era um no-op puro (a observacao ja e integral) que custava 0,1 e ficava sempre valida,
        # contra a regra declarada em mascara_acoes.py ("uma acao e invalida quando nao tem efeito nenhum no estado").
        # Medido: uma politica que so observa fazia -32,5 por episodio. N_ACOES 15 -> 14.

        n_acoes = len(self.nomes_acoes)
        self.action_space = gym.spaces.Discrete(n_acoes)

    def _configurar_espaco_observacao(self):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(OBS_DIM,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)   # FarmGym v2: doenças, migração e eventos extremos usam o módulo random

        self.planta.resetar()
        self.passaros.resetar()
        self.polinizadores.resetar()
        self.solo.resetar()
        self.clima.resetar()
        self.ervas_daninhas.resetar()
        self.pragas.resetar()
        self.defensivos_fertilizantes.resetar()
        self.instalacoes.resetar()

        self.eventos_extremos.resetar()
        self.sistema_doencas.resetar()

        self.passo_atual = 0
        self.recompensa_episodio = 0
        self.historico_colheita = []
        self.total_aplicacoes_quimicas = 0
        self.fruto_perdido_aves = 0.0

        return self._obter_observacao(), {}

    def step(self, action):
        nome_acao = self.nomes_acoes[action]

        terminado = False
        truncado = False
        info = {}

        # S5: decomposição da recompensa. `_executar_acao` devolve a colheita (>= 0) para 'colher'
        # e o custo (<= 0) para as demais ações.
        recompensa_acao = self._executar_acao(nome_acao, action)
        comp = {'custo_acao': 0.0, 'colheita': 0.0, 'eventos': 0.0}
        if nome_acao == 'colher':
            comp['colheita'] = float(recompensa_acao)
        else:
            comp['custo_acao'] = float(recompensa_acao)
        if nome_acao in ACOES_QUIMICAS:
            self.total_aplicacoes_quimicas += self.tamanho_grade[0] * self.tamanho_grade[1]
        # v3: agir durante um evento extremo custa mais (impactos negativos x severidade, definidos em extreme_events.py)
        categoria = CATEGORIA_EVENTO.get(nome_acao)
        if categoria is not None:
            comp['eventos'] = float(self.eventos_extremos.obter_impacto_extremo(categoria))

        self._atualizar_ambiente()

        comp.update(self._componentes_recompensa_passo())

        self.passo_atual += 1

        # Coleta de KPIs Agrícolas para o artigo e análise de HPO
        info['kpis'] = {
            'produtividade_acumulada': float(sum(self.historico_colheita)),
            'fruto_perdido_aves': float(self.fruto_perdido_aves),
            'fruto_perdido_pragas': float(self.fruto_perdido_pragas),           # v4
            'populacao_pragas_media': float(np.mean(list(self.pragas.populacao_pragas.values()))),   # v4
            'populacao_pragas_max': float(self.pragas.obter_populacao_maxima_no_campo()),            # v4
            'populacao_aves_media': float(self.passaros.obter_populacao_total() / max(1, N_CELULAS)),  # v4
            'pressao_migratoria': float(self.passaros.pressao_migratoria),                           # v4
            'impacto_doenca_medio': float(np.mean([self.sistema_doencas.calcular_impacto_doenca(x, y)
                                                   for x in range(self.tamanho_grade[0]) for y in range(self.tamanho_grade[1])])),
            'populacao_pragas_total': int(self.pragas.obter_populacao_total()),
            'qualidade_solo_media': float(np.mean([c.qualidade for c in self.solo.estado_solo.values()])),
            'umidade_solo_media': float(np.mean([c.umidade for c in self.solo.estado_solo.values()])),
            'densidade_polinizadores': float(self.polinizadores.obter_densidade_media()),
            'uso_quimicos_total': int(self.total_aplicacoes_quimicas),
            'num_plantas_ativas': int(len(self.planta.plantas)),
            'progresso_episodio': float(self.passo_atual / self.passos_maximos)
        }

        if self.passo_atual >= self.passos_maximos:
            truncado = True
            comp.update(self._componentes_recompensa_final())

        recompensa = float(sum(comp.values()))
        self.recompensa_episodio += recompensa
        info['recompensa_comp'] = comp

        return self._obter_observacao(), recompensa, terminado, truncado, info

    def _executar_acao(self, nome_acao: str, id_acao: int) -> float:
        recompensa = 0

        if nome_acao == 'plantar':
            espacos_vazios = []
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    if (x, y) not in self.planta.plantas:
                        espacos_vazios.append((x, y))
            if espacos_vazios:
                local = espacos_vazios[np.random.randint(len(espacos_vazios))]
                self.planta.adicionar_planta(local[0], local[1])
                recompensa -= 1.0

        elif nome_acao == 'colher':
            for pos in list(self.planta.plantas.keys()):
                planta_obj = self.planta.plantas[pos]
                if planta_obj.estagio == EstagioPlanta.COLHEITA:
                    # v4: a colheita e o proprio peso do fruto. O multiplicador de rendimento saiu por tres motivos
                    # medidos: (i) descontava o estresse uma segunda vez, agora que o peso ja carrega a media do fator
                    # da janela de enchimento (plant.fator_enchimento); (ii) contava a qualidade do solo duas vezes,
                    # porque ela ja entra no fator de crescimento; (iii) `mult_solo = 0,5 + qualidade` lia a umidade do
                    # solo NO PASSO DA COLHEITA, entao regar no passo anterior valia ~+2,9 de recompensa por 0,5 de
                    # custo -- um exploit de ~50 pontos por episodio. Sem ele, a fruta colhida e a fruta pendente
                    # (termo terminal) passam a valer a mesma coisa por unidade de peso.
                    quantidade_colhida = planta_obj.peso_fruto
                    self.historico_colheita.append(quantidade_colhida)
                    self.planta.colher(pos)
                    recompensa += quantidade_colhida * 5.0

        elif nome_acao == 'regar':
            for pos in self.planta.plantas:
                celula_solo = self.solo.obter_solo_em(pos[0], pos[1])
                celula_solo.umidade = min(1.0, celula_solo.umidade + 0.2)
            recompensa -= 0.5

        elif nome_acao == 'fertilizar_N':
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    self.defensivos_fertilizantes.aplicar_fertilizante(
                        x, y, TipoNutriente.NITROGENIO, 2.0)
            recompensa -= 2.0

        elif nome_acao == 'fertilizar_P':
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    self.defensivos_fertilizantes.aplicar_fertilizante(
                        x, y, TipoNutriente.FOSFORO, 2.0)
            recompensa -= 2.0

        elif nome_acao == 'fertilizar_K':
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    self.defensivos_fertilizantes.aplicar_fertilizante(
                        x, y, TipoNutriente.POTASSIO, 2.0)
            recompensa -= 2.0

        elif nome_acao == 'fertilizar_C':
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    self.defensivos_fertilizantes.aplicar_fertilizante(
                        x, y, TipoNutriente.CARBONO, 2.0)
            recompensa -= 2.0

        elif nome_acao == 'herbicida':
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    self.defensivos_fertilizantes.aplicar_defensivo(
                        x, y, TipoDefensivo.HERBICIDA, 0.8, 2.0)
                    self.ervas_daninhas.remover_ervas(x, y, EFICACIA_HERBICIDA)   # S7b: abate imediato
            recompensa -= 1.5

        elif nome_acao == 'pesticida':
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    self.defensivos_fertilizantes.aplicar_defensivo(
                        x, y, TipoDefensivo.PESTICIDA, 0.8, 2.0)
                    self.pragas.aplicar_tratamento(x, y, EFICACIA_PESTICIDA)          # v3: abate imediato das pragas
                    self.sistema_doencas.aplicar_tratamento(x, y, 'amplo_espectro')  # v3: trata a doenca da celula
            recompensa -= 1.5

        elif nome_acao == 'colocar_espantalho_basico':
            posicoes_com_planta = list(self.planta.plantas.keys())
            if posicoes_com_planta:
                pos = posicoes_com_planta[np.random.randint(len(posicoes_com_planta))]
                self.instalacoes.colocar_espantalho(pos[0], pos[1], TipoEspantalho.BASICO)
                recompensa -= 1.0

        elif nome_acao == 'colocar_espantalho_avancado':
            posicoes_com_planta = list(self.planta.plantas.keys())
            if posicoes_com_planta:
                pos = posicoes_com_planta[np.random.randint(len(posicoes_com_planta))]
                self.instalacoes.colocar_espantalho(pos[0], pos[1], TipoEspantalho.AVANCADO)
                recompensa -= 3.0

        elif nome_acao == 'remover_espantalho':
            posicoes_espantalhos = self.instalacoes.obter_todas_posicoes_por_tipo(
                TipoInstalacao.ESPANTALHO)
            if posicoes_espantalhos:
                pos = posicoes_espantalhos[np.random.randint(len(posicoes_espantalhos))]
                self.instalacoes.remover_espantalho(pos[0], pos[1])

        elif nome_acao == 'colocar_cerca':
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    self.instalacoes.colocar_cerca_viva(x, y)
            recompensa -= 5.0


        elif nome_acao == 'esperar':
            pass

        return recompensa

    def _atualizar_ambiente(self):
        obs_solo = self.solo.obter_observacao()
        obs_instalacoes = self.instalacoes.obter_observacao()
        # v3 (06/09/2026): o solo, as ervas e as pragas leem `fertilizante_N/P/K/C`, `efeito_herbicida`, `efeito_pesticida` e
        # `quantidade_defensivo`, que so `obter_contribuicoes_solo()` fornece. Ate a v2 recebiam `obter_observacao()` (contagem
        # de aplicacoes), entao o fertilizante nunca chegou ao solo, o herbicida nunca teve residual nem dano ao solo e o
        # pesticida nunca agiu nas pragas; so os polinizadores recebiam o dicionario certo (criticas.md, secao 13).
        contrib_defensivos = self.defensivos_fertilizantes.obter_contribuicoes_solo()

        self.clima.atualizar()
        obs_clima = self.clima.obter_observacao()
        obs_clima['step'] = self.passo_atual

        self.eventos_extremos.verificar_condicoes_extremas(obs_clima)

        self.solo.atualizar(obs_clima, contrib_defensivos)
        self.defensivos_fertilizantes.atualizar()
        self.instalacoes.atualizar()

        # v4: as aves passam a depender da estacao, da onda migratoria e do alimento da celula. O dicionario de
        # alimento ja existia e era consumido SO pela `MigracaoSazonal`, que era inerte.
        self.passaros.atualizar(obs_instalacoes, obs_clima, self._calcular_disponibilidade_alimento())
        self.polinizadores.atualizar(obs_clima, obs_instalacoes, contrib_defensivos)
        self.ervas_daninhas.atualizar(obs_clima, obs_solo, contrib_defensivos)
        self.pragas.atualizar(obs_clima, self.planta.obter_observacao(), contrib_defensivos,
                          self.passaros.obter_observacao())

        posicoes_plantas = list(self.planta.plantas.keys())
        # v4: a doenca e da planta. Sem esta sincronizacao a infeccao sobrevivia a colheita e a morte da planta, a
        # observacao mostrava doenca em celula vazia e um terco dos plantios nascia infectado.
        self.sistema_doencas.sincronizar_plantas(posicoes_plantas)
        self.sistema_doencas.espalhar_doenca(obs_clima, posicoes_plantas)
        pesticida_ativo = any(c.get('efeito_pesticida', 0.0) > 0.01 for c in contrib_defensivos.values())
        self.sistema_doencas.progredir_doencas(obs_clima, tratamento_aplicado=pesticida_ativo)   # v3

        doencas_por_pos = {pos: self.sistema_doencas.calcular_impacto_doenca(pos[0], pos[1]) for pos in posicoes_plantas}
        self.planta.atualizar(obs_clima, obs_solo, self.polinizadores.obter_observacao(),
                          self.ervas_daninhas.obter_observacao(), self.pragas.obter_observacao(), doencas_por_pos)
        # v4: `planta.atualizar` remove as plantas mortas; sem sincronizar de novo, a doenca delas ficaria orfa
        self.sistema_doencas.sincronizar_plantas(self.planta.plantas.keys())
        self._aves_comem_frutos()
        self._pragas_danificam_frutos()

    def _aves_comem_frutos(self):
        """v3: nas plantas em frutificacao/colheita, as aves da celula comem `TAXA_PREDACAO_AVES x pop/POP_MAX` do peso do
        fruto por passo. O espantalho reduz a populacao (`Passaros._aplicar_efeito_espantalho`), logo a perda."""
        for pos, planta_obj in self.planta.plantas.items():
            if planta_obj.estagio in (EstagioPlanta.FRUTIFICACAO, EstagioPlanta.COLHEITA) and planta_obj.peso_fruto > 0:
                pop = self.passaros.obter_populacao_em(pos[0], pos[1])
                perda = planta_obj.peso_fruto * TAXA_PREDACAO_AVES * min(1.0, pop / POP_MAX_AVES)
                planta_obj.peso_fruto -= perda
                self.fruto_perdido_aves += perda

    def _pragas_danificam_frutos(self):
        """v4: nas plantas em frutificacao/colheita, a praga da celula destroi `TAXA_DANO_PRAGAS x pop/pop_max` do peso
        do fruto por passo -- o mesmo canal das aves, sobre o mesmo estoque. E o que torna o pesticida uma decisao de
        TIMING (pulverizar as vesperas do enchimento vale mais) em vez de um imposto de calendario."""
        for pos, planta_obj in self.planta.plantas.items():
            if planta_obj.estagio in (EstagioPlanta.FRUTIFICACAO, EstagioPlanta.COLHEITA) and planta_obj.peso_fruto > 0:
                pop = self.pragas.obter_populacao_em(pos[0], pos[1])
                perda = planta_obj.peso_fruto * TAXA_DANO_PRAGAS * min(1.0, pop / POP_MAX_PRAGAS_OBS)
                planta_obj.peso_fruto -= perda
                self.fruto_perdido_pragas += perda

    def _calcular_disponibilidade_alimento(self) -> Dict:
        disponibilidade_alimento = {}

        for x in range(self.tamanho_grade[0]):
            for y in range(self.tamanho_grade[1]):
                pos = (x, y)
                disponibilidade = 0.0

                if pos in self.planta.plantas:
                    planta_obj = self.planta.plantas[pos]
                    if planta_obj.estagio in [EstagioPlanta.FLORESCIMENTO, EstagioPlanta.FRUTIFICACAO, EstagioPlanta.COLHEITA]:
                        disponibilidade += 0.8
                    elif planta_obj.estagio in [EstagioPlanta.CRESCIMENTO, EstagioPlanta.FLORESCIMENTO]:
                        disponibilidade += 0.4

                densidade_ervas = self.ervas_daninhas.obter_densidade_em(x, y)
                disponibilidade += densidade_ervas * 0.2

                densidade_polinizadores = self.polinizadores.obter_densidade_em(x, y)
                disponibilidade += densidade_polinizadores * 0.1

                disponibilidade_alimento[pos] = min(1.0, disponibilidade)

        return disponibilidade_alimento

    def _componentes_recompensa_passo(self) -> Dict[str, float]:
        """FarmGym v2: só penalidades ligadas às plantas que o agente cultiva (clima extremo, ervas, doença).
        Sem bônus de estágio (S2b) nem de aves (S2a): nada é pago por existir."""
        comp = {'clima': 0.0, 'ervas': 0.0, 'doenca': 0.0}
        plantas = self.planta.plantas
        num_plantas = len(plantas)

        if num_plantas > 0:
            temp = self.clima.obter_observacao().get('temperatura', 20)
            if temp > 35:
                comp['clima'] = -PENALIDADE_TEMP_ALTA
            elif temp < 5:
                comp['clima'] = -PENALIDADE_TEMP_BAIXA

            densidade_media_ervas = sum(
                self.ervas_daninhas.obter_densidade_em(pos[0], pos[1]) for pos in plantas) / num_plantas
            comp['ervas'] = -PESO_ERVAS * densidade_media_ervas

        impacto_doenca_total = sum(
            self.sistema_doencas.calcular_impacto_doenca(pos[0], pos[1]) for pos in plantas)
        comp['doenca'] = -min(TETO_DOENCA, impacto_doenca_total * PESO_DOENCA)
        return comp

    def _calcular_recompensa_passo(self) -> float:
        return float(sum(self._componentes_recompensa_passo().values()))

    def _componentes_recompensa_final(self) -> Dict[str, float]:
        """Terminal (passo 365), FarmGym v2: frutos pendentes, solo, polinizadores e químicos (linear, sem teto).
        A colheita NÃO é somada de novo (S1)."""
        comp = {}
        comp['frutos_pendentes'] = float(sum(
            planta_obj.peso_fruto * PESO_FRUTOS_PENDENTES
            for planta_obj in self.planta.plantas.values()
            if planta_obj.estagio in (EstagioPlanta.FRUTIFICACAO, EstagioPlanta.COLHEITA)))

        qualidade_media_solo = np.mean([celula.qualidade for celula in self.solo.estado_solo.values()])
        comp['solo_final'] = float(qualidade_media_solo * PESO_SOLO_FINAL)

        comp['polinizadores_final'] = float(self.polinizadores.obter_densidade_media() * PESO_POLINIZADORES_FINAL)

        num_celulas = self.tamanho_grade[0] * self.tamanho_grade[1]
        quimicos_por_celula = self.total_aplicacoes_quimicas / max(1, num_celulas)
        comp['quimicos_final'] = float(-PENALIDADE_QUIMICO_POR_CELULA * quimicos_por_celula)
        return comp

    def _calcular_recompensa_final(self) -> float:
        return float(sum(self._componentes_recompensa_final().values()))

    def _obter_observacao(self) -> np.ndarray:
        obs = []

        obs_clima = self.clima.obter_observacao()
        # Correcao: o subsistema Clima devolve as chaves em PORTUGUES
        # ('umidade_ar','chuva','vento'). Antes liam-se 'humidity'/'rain'/'wind'
        # (ingles), que nao existem no dict, fazendo .get() cair sempre no default
        # => 3 das 5 features do bloco global ficavam CONSTANTES na observacao.
        obs.extend([
            obs_clima.get('temperatura', 20) / 40,
            obs_clima.get('umidade_ar', 0.5),
            obs_clima.get('chuva', 0) / 50.0,
            obs_clima.get('vento', 0.5)
        ])

        obs.append(self.passo_atual / self.passos_maximos)

        impactos_extremos = self.eventos_extremos.obter_impactos_atuais()
        obs.extend(impactos_extremos)

        # v4: 3 agregados de doenca (pressao, pressao CONTROLAVEL, prevalencia). A v3 punha 4 prevalencias por
        # patogeno + a pressao; com um unico defensivo, a prevalencia por patogeno nao e acionavel e ficava constante
        # em regimes inteiros. A diferenca entre pressao e pressao controlavel e o que o quimico NAO resolve.
        obs.extend(self.sistema_doencas.obter_indicadores_doenca())

        # v4: pressao migratoria de aves (0 ou 1). Substitui as 5 dimensoes do modulo `MigracaoSazonal`, removido: ele
        # lia a chave 'passo' enquanto o ambiente escreve 'step', entao 4 das 5 eram constantes em 96 % dos passos.
        # Ocupa slot proprio porque e ESTOCASTICA -- a estacao e dedutivel do passo, a onda nao.
        obs.append(float(self.passaros.pressao_migratoria))

        # v4: praga maxima entre as celulas PLANTADAS. E a estatistica que decide a acao (o pesticida e do campo
        # inteiro); sobre as celulas plantadas, e nao sobre todas, para nao descasar da regra do professor logo apos
        # uma colheita, quando a celula esvazia com a populacao ainda alta.
        plantadas = list(self.planta.plantas.keys())
        praga_max = max((self.pragas.obter_populacao_em(x, y) for x, y in plantadas), default=0.0)
        obs.append(min(1.0, praga_max / POP_MAX_PRAGAS_OBS))

        # v4: fracao de celulas com cerca viva. `colocar_cerca` e do campo inteiro, entao uma dimensao global basta --
        # e ate a v3 a cerca valia +10 % de retorno e era completamente invisivel ao agente.
        obs_inst = self.instalacoes.obter_observacao()
        obs.append(sum(1.0 for c in obs_inst.values() if c.get('tem_cerca')) / float(N_CELULAS))

        for x in range(self.tamanho_grade[0]):
            for y in range(self.tamanho_grade[1]):
                pos = (x, y)

                if pos in self.planta.plantas:
                    planta_obj = self.planta.plantas[pos]
                    obs.extend([
                        1.0,
                        planta_obj.saude,
                        list(EstagioPlanta).index(planta_obj.estagio) / len(EstagioPlanta)
                    ])
                else:
                    obs.extend([0.0, 0.0, 0.0])

                celula_solo = self.solo.obter_solo_em(x, y)
                obs.extend([
                    celula_solo.umidade,
                    celula_solo.qualidade
                ])

                densidade_ervas = self.ervas_daninhas.obter_densidade_em(x, y)
                obs.append(densidade_ervas)

                impacto_doenca = self.sistema_doencas.calcular_impacto_doenca(x, y)
                obs.append(impacto_doenca)

                # v3: a populacao observada e a que o espantalho afasta e que come fruto (`Passaros`, max. 50);
                # antes era `MigracaoSazonal`, um segundo modelo de aves sem efeito em nada
                pop_passaros = self.passaros.obter_populacao_em(x, y)
                obs.append(min(1.0, pop_passaros / POP_MAX_AVES))

                # v4: praga da celula. Ate a v3 a praga nao aparecia em nenhuma das 148 dimensoes -- o agente era
                # punido por algo que nao via. Por celula (e nao so agregada) para a rede poder parear praga com
                # ESTAGIO na mesma celula: o dano ao fruto so existe enquanto a planta enche.
                obs.append(min(1.0, self.pragas.obter_populacao_em(x, y) / POP_MAX_PRAGAS_OBS))

                # v4: forca do espantalho da celula. Mesma razao: mudava a dinamica e era invisivel.
                obs.append(min(1.0, (obs_inst.get(pos, {}).get('forca_espantalho', 0.0) or 0.0) / FORCA_MAX_ESPANTALHO))

        # FarmGym v2 (OBS): garantia do contrato Box[0, 1] do espaço de observação.
        return np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)

    def render(self, mode='human'):
        if mode == 'human':
            grade = [['  ' for _ in range(self.tamanho_grade[1])] for _ in range(self.tamanho_grade[0])]

            for (x, y), planta_obj in self.planta.plantas.items():
                if 0 <= x < self.tamanho_grade[0] and 0 <= y < self.tamanho_grade[1]:
                    grade[x][y] = f'P{planta_obj.estagio.value}'

            for linha in grade:
                print(' '.join(linha))
            print(f"Passo: {self.passo_atual}/{self.passos_maximos}")

    def close(self):
        pass
