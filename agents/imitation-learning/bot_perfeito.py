"""
Bot Ajustável — o oráculo do MimePlant. Duas políticas hierárquicas de regras agronômicas, ambas com acesso ao estado
interno do ambiente (FarmGym v4) e nenhuma delas ótima:

  `BotIntensivo` (v5, D-33, 18/09/2026)  manejo INTENSIVO: solo fértil, ervas baixas, colheita em lote, espantalhos e
                                         cerca viva proativos. É o **professor** dos experimentos desde a D-33.
  `BotPerfeito`  (v4, nome histórico)    manejo REATIVO: age quando um limiar de dano é cruzado, espera em 64 % dos
                                         passos. Foi o professor da rodada 2 (06–10/09/2026) e fica no código para
                                         reproduzi-la e para comparar qualidade de professor (`protocolo.novo_professor_v4()`).

Com p = 0 (sem pausas) o bot é o **professor** (D-27): gera as demonstrações, rotula o DAgger e é a âncora 1 da
recompensa normalizada — agentes de RL passam de 1. Com p > 0 (pausas de inação semeadas por episódio, L passos cada)
vira uma variante degradada, mantida como possibilidade para estudos de qualidade do professor; nenhum experimento da
defesa a usa.

Papéis: (1) gerar as demonstrações (`gerar_demos.py`); (2) rotular o DAgger (`treinar_warmstart_il.py`);
(3) referências de avaliação (`referencias.py`). Constantes do protocolo em `protocolo.py` (nunca redigitar aqui):
construa por `protocolo.novo_professor()` ou `protocolo.novo_oraculo_variante()`.

Recompensa (FarmGym v4): colheita = peso do fruto × 5, paga uma vez, no passo. O peso já carrega o cuidado: acumula
`fator × 0,5` por passo de frutificação e, na saída do estágio, é multiplicado pela média do fator de crescimento da
janela sobre 1,2 (teto 1,5). Tudo o que sobe o fator (solo, ervas, pragas, doença, polinizadores) encurta o ciclo E
aumenta o fruto — é por aí que o manejo intensivo ganha do reativo.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, '../../environment'))
for _sub in ['plant', 'birds', 'pollinators', 'soil', 'weather',
             'weeds', 'pest', 'cides-fertilizers', 'facilities']:
    sys.path.insert(0, os.path.join(_DIR, f'../../environment/{_sub}'))

from plant import EstagioPlanta
from soil import Solo                          # eficiência do fertilizante por nutriente (fonte única)
from cides_fertilizers import TipoDefensivo

# ─── Índices de ação (FarmGym v4: derivados do protocolo, nunca redigitados) ──────────────────
# Redigitar os índices aqui foi o que quase quebrou a remoção de 'observar': `ESPERAR = 14` continuaria apontando para
# fora do espaço de ações. Agora qualquer mudança em `protocolo.NOMES_ACOES` propaga sozinha.
import protocolo as _P   # protocolo importa bot_perfeito só dentro de novo_professor(), então não há ciclo

NOMES_ACOES = list(_P.NOMES_ACOES)
PLANTAR              = NOMES_ACOES.index('plantar')
COLHER               = NOMES_ACOES.index('colher')
REGAR                = NOMES_ACOES.index('regar')
FERTILIZAR_N         = NOMES_ACOES.index('fertilizar_N')
FERTILIZAR_P         = NOMES_ACOES.index('fertilizar_P')
FERTILIZAR_K         = NOMES_ACOES.index('fertilizar_K')
FERTILIZAR_C         = NOMES_ACOES.index('fertilizar_C')
HERBICIDA            = NOMES_ACOES.index('herbicida')
PESTICIDA            = NOMES_ACOES.index('pesticida')
ESP_BASICO           = NOMES_ACOES.index('colocar_espantalho_basico')
ESP_AVANCADO         = NOMES_ACOES.index('colocar_espantalho_avancado')
REMOVER_ESP          = NOMES_ACOES.index('remover_espantalho')
COLOCAR_CERCA        = NOMES_ACOES.index('colocar_cerca')
ESPERAR              = NOMES_ACOES.index('esperar')

# ─── Parâmetros de controle ───────────────────────────────────────────────────
# Watering: soil moisture drops ~0.03-0.05/step; water adds 0.2 → needed every ~6 steps
LIMIAR_UMIDADE          = 0.55   # regar se umidade média das células plantadas < este valor (v2: era 0,40; medido)
LIMIAR_CHUVA_ATIVA      = 3.0    # mm de chuva no step atual para considerar "chovendo agora"
LIMIAR_UMIDADE_AR_ALTA  = 0.75   # umidade_ar acima disto sugere chuva iminente
# FarmGym v2 (oráculo v2, D-17): com o limiar 0,15 o oráculo aplicava herbicida em 27–69 % dos passos (a densidade
# de equilíbrio das ervas fica acima disso) e se sabotava pelo solo e pelos custos. Agora: limiar mais alto e uma
# carência mínima entre aplicações (meia-vida do herbicida no solo = 14 dias).
LIMIAR_ERVAS            = 0.30   # herbicida se densidade máxima de ervas > este valor
CARENCIA_HERBICIDA      = 10     # passos mínimos entre duas aplicações de herbicida (v2: medido)
# FarmGym v4: o limiar passa a ser lido sobre o impacto CONTROLÁVEL (o que o defensivo consegue tratar). Na v3 o
# defensivo curava 60 % de qualquer doença, inclusive virose — e é a virose que mais ocorre in-dist. Pulverizar por
# causa de mosaico viral (controle químico 0) é custo puro.
LIMIAR_DOENCA           = 0.10   # pesticida se impacto CONTROLÁVEL máximo de doença > este valor
# FarmGym v3 (oraculo v3, D-24): pesticida tambem por pragas, com carencia; espantalho reativo a aves com fruto no campo.
CARENCIA_PESTICIDA      = 7      # passos minimos entre pesticidas (meia-vida do pesticida = 7 dias)
# FarmGym v4: com a população de pragas viva (a da v3 era constante), 30 está acima do nível de dano econômico —
# a praga satura em 100 por volta do passo 30 e o dano ao fruto é proporcional à população.
LIMIAR_PRAGAS           = 15     # pesticida se a populacao de pragas de alguma celula plantada > este valor (max. 100)
LIMIAR_AVES             = 15     # espantalho se a media de aves nas celulas em frutificacao/colheita > este valor (max. 50)
MAX_ESPANTALHOS         = 4      # no maximo um espantalho por quatro celulas (o espantalho repele as 4 vizinhas)
# FarmGym v4: a cerca viva sustenta os polinizadores e, por eles, o peso do fruto (+10 % medidos na v3). Na v3 o
# professor não a usava, de propósito, para deixar margem ao RL; com o ambiente corrigido a margem passa a vir do
# TEMPO das decisões (antecipar a onda de praga e a onda migratória, que são estocásticas e não dedutíveis do passo),
# e não de uma decisão que a heurística ignora. Um professor que ignora um recurso barato não é "a melhor heurística".
PASSO_CERCA_VIVA        = 20     # instala a cerca uma vez, cedo, depois de o campo estar plantado
CARENCIA_ESPANTALHO     = 5      # passos minimos entre dois espantalhos
# FarmGym v2 (oráculo v2): o fertilizante libera ao longo de ~20 dias; re-checar o nutriente no dia seguinte
# empilhava aplicações (na v1 isso era mascarado pelo herbicida diário, que tinha prioridade). Agora: limiar
# mais baixo e carência por nutriente.
LIMIAR_NUTRIENTE        = 0.15   # fertilizar se o nutriente mais escasso (elegível) < este valor
CARENCIA_FERTILIZANTE   = 25     # passos minimos entre duas aplicacoes do MESMO nutriente (v3: o fertilizante libera em ~20 dias; era 120)
# Nutrientes: N decai mais rápido com uso (-0.01/step), eficiência alta (0.15/unit aplicado)
# P tem eficiência baixíssima (0.05/unit), K médio (0.12), C médio-baixo (0.08)
PRIORIDADE_NUTRIENTES   = ['N', 'P', 'K', 'C']  # ordem de prioridade quando deficientes
# FarmGym v4: o ciclo do v3/v4 leva ~140-160 passos, não 120-135 (número herdado do v2). Com 125 o professor parava
# de replantar no passo 240 e desperdiçava o fim do episódio: medido, ele fazia ~32 plantios contra ~53 de um agente
# de RL. O valor passa a ser o ciclo medido, e o professor replanta enquanto houver tempo de fechar um ciclo.
PASSOS_MINIMOS_REPLANTE = 150


class BotPerfeito:
    """
    Oracle com acesso direto ao estado interno do AmbienteFazendaGym.

    Política hierárquica determinística (em ordem de prioridade):
      1. COLHER    — planta em COLHEITA perde valor se não colhida (morre em ~13 steps)
      2. PLANTAR   — preencher grade imediatamente; replantio se há tempo (≥125 steps)
      3. REGAR     — umidade média das parcelas < 0.55 (v2)
      4. HERBICIDA — densidade máxima de ervas > 0.30, no máximo a cada 10 passos (v2)
      5. PESTICIDA — doenca > 0.10 OU pragas > 30 em alguma celula plantada, no maximo a cada 7 passos (v3)
      6. FERTILIZAR — nutriente mais escasso < 0.15, no maximo a cada 25 passos por nutriente (v3)
      7. ESPANTALHO — reativo: aves > 15 (media) nas celulas com fruto e menos de 4 espantalhos (v3)
      8. ESPERAR   — quando tudo está em ordem
    """

    def __init__(self, imperfeicao: float = 0.0, lazy_steps: int = 5,
                  seed: Optional[int] = None):
        """
        Args:
            imperfeicao: probabilidade p ∈ [0,1] por step de o bot entrar em
                modo PREGUIÇOSO (rajada de ESPERARs consecutivos). Cria um
                oráculo "quase ótimo" com lacunas de inação suficientemente
                longas para deixar plantas morrerem, soil secar, etc.
                p=0.0 ⇒ comportamento ótimo (default).
            lazy_steps: nº de steps consecutivos em ESPERAR quando entra em
                modo preguiçoso (default 5).
            seed: seed do RNG interno. Se None, usa np.random global.
        """
        self.imperfeicao = float(imperfeicao)
        self.lazy_steps  = int(lazy_steps)
        self._seed_base  = None if seed is None else int(seed)
        self._rng = np.random.RandomState(seed) if seed is not None else None
        self.resetar_episodio()

    def _rand(self) -> float:
        if self._rng is not None:
            return float(self._rng.random())
        return float(np.random.random())

    def resetar_episodio(self, seed_episodio: Optional[int] = None):
        """Zera o estado do episódio.

        S13 (FarmGym v2, D-17): com `seed_episodio`, as pausas de imperfeição deste episódio ficam
        reproduzíveis — RandomState(seed_base + seed_episodio), onde seed_base é a semente dada no
        construtor (ORACULO_SEED = 12345 no protocolo). Sem `seed_episodio`, o RNG interno continua
        de onde estava (comportamento da qualificação).
        """
        self._ultimo_espantalho = -CARENCIA_ESPANTALHO
        self._ultimo_herbicida = -CARENCIA_HERBICIDA
        self._ultimo_pesticida = -CARENCIA_PESTICIDA
        self._ultimo_fertilizante = {k: -CARENCIA_FERTILIZANTE for k in PRIORIDADE_NUTRIENTES}
        self._lazy_remaining = 0
        if seed_episodio is not None:
            base = 0 if self._seed_base is None else self._seed_base
            self._rng = np.random.RandomState(base + int(seed_episodio))

    # ── Interface principal ───────────────────────────────────────────────────

    def selecionar_acao(self, env_raw) -> int:
        """Retorna o índice da ação ótima dado acesso direto ao ambiente.

        Se self.imperfeicao > 0, periodicamente entra em modo "preguiçoso":
        rajada de ESPERARs por self.lazy_steps steps consecutivos, durante
        os quais plantas em COLHEITA podem morrer, soil seca, ervas crescem
        — i.e., criando lacunas de inação suficientemente longas para impactar.
        """
        if self._lazy_remaining > 0:
            self._lazy_remaining -= 1
            return ESPERAR
        acao = self._selecionar_acao_otima(env_raw)
        if self.imperfeicao > 0.0 and self._rand() < self.imperfeicao:
            self._lazy_remaining = max(0, self.lazy_steps - 1)
            return ESPERAR
        # v2: o estado interno (último herbicida/espantalho) só muda quando a ação é de fato executada
        # (antes, uma pausa que caía em cima de um espantalho contava como se ele tivesse sido colocado).
        self._registrar_acao(acao, env_raw.passo_atual)
        return acao

    _FERTILIZANTE_POR_ACAO = {FERTILIZAR_N: 'N', FERTILIZAR_P: 'P', FERTILIZAR_K: 'K', FERTILIZAR_C: 'C'}

    def _registrar_acao(self, acao: int, passo: int):
        if acao == HERBICIDA:
            self._ultimo_herbicida = passo
        elif acao == PESTICIDA:
            self._ultimo_pesticida = passo
        elif acao == ESP_AVANCADO:
            self._ultimo_espantalho = passo
        elif acao in self._FERTILIZANTE_POR_ACAO:
            self._ultimo_fertilizante[self._FERTILIZANTE_POR_ACAO[acao]] = passo

    def _selecionar_acao_otima(self, env_raw) -> int:
        """Política hierárquica determinística pura (sem imperfeição)."""
        plantas        = env_raw.planta.plantas
        passo          = env_raw.passo_atual
        passos_max     = env_raw.passos_maximos
        linhas, colunas = env_raw.tamanho_grade
        n_celulas       = linhas * colunas
        n_plantas       = len(plantas)
        espacos_vazios  = n_celulas - n_plantas
        passos_restantes = passos_max - passo

        # ── 1. COLHER ─────────────────────────────────────────────────────────
        # Máxima prioridade: planta em COLHEITA começa a morrer em ~13 steps
        for planta_obj in plantas.values():
            if planta_obj.estagio == EstagioPlanta.COLHEITA:
                return COLHER

        # ── 2. PLANTAR ────────────────────────────────────────────────────────
        # Plantar apenas se há espaço E tempo suficiente para outro ciclo completo
        if espacos_vazios > 0 and passos_restantes >= PASSOS_MINIMOS_REPLANTE:
            return PLANTAR

        # ── 3. REGAR ─────────────────────────────────────────────────────────
        # Umidade baixa → growth factor cai → ciclos mais lentos.
        # Domain knowledge: NÃO regar se está chovendo agora OU se umidade do ar
        # está alta (chuva iminente) — a chuva faz o papel da rega naturalmente,
        # liberando o step para colher/plantar/fertilizar.
        if n_plantas > 0 and self._umidade_plantas(env_raw) < LIMIAR_UMIDADE:
            if not self._chovendo_ou_iminente(env_raw):
                return REGAR

        # ── 4. HERBICIDA ──────────────────────────────────────────────────────
        # Ervas: reduzem growth factor em (1 - 0.5 × densidade)
        # Ex: densidade 0.3 → penalidade de 15% no fator de crescimento
        if (self._ervas_max(env_raw) > LIMIAR_ERVAS
                and passo - self._ultimo_herbicida >= CARENCIA_HERBICIDA):
            return HERBICIDA

        # ── 5. PESTICIDA ──────────────────────────────────────────────────────
        # v4: a doenca conta pelo impacto CONTROLAVEL (o defensivo nao cura virose) e a praga pelo nivel de dano
        # economico. Na v3 o ramo de praga era codigo morto (a populacao nao crescia) e o de doenca disparava por
        # virose, que o quimico nao trata.
        if ((self._doenca_controlavel_max(env_raw) > LIMIAR_DOENCA or self._pragas_max(env_raw) > LIMIAR_PRAGAS)
                and passo - self._ultimo_pesticida >= CARENCIA_PESTICIDA):
            return PESTICIDA

        # ── 6. FERTILIZAR ─────────────────────────────────────────────────────
        # Manter nutrientes altos → soil quality alta → growth factor alto
        # N decai em -0.01/step; fertilizar_N aplica 2 units × 0.15 efic. = +0.30 N
        acao_fert = self._acao_fertilizar(env_raw, passo)
        if acao_fert is not None:
            return acao_fert

        # ── 7. ESPANTALHO AVANÇADO ────────────────────────────────────────────
        # v3: reativo -- aves comem fruto; instalar quando ha fruto no campo e aves acima do limiar
        if (self._aves_em_frutificacao(env_raw) > LIMIAR_AVES
                and env_raw.instalacoes.n_espantalhos() < MAX_ESPANTALHOS
                and passo - self._ultimo_espantalho >= CARENCIA_ESPANTALHO):
            return ESP_AVANCADO

        # ── 7b. CERCA VIVA ────────────────────────────────────────────────────
        # v4: uma vez, cedo, depois de o campo estar plantado. Sustenta os polinizadores e, por eles, o peso do fruto.
        if (passo >= PASSO_CERCA_VIVA and env_raw.instalacoes.n_cercas() < len(env_raw.planta.plantas)
                and len(env_raw.planta.plantas) > 0):
            return COLOCAR_CERCA

        # ── 8. ESPERAR ────────────────────────────────────────────────────────
        return ESPERAR

    # ── Helpers de estado ─────────────────────────────────────────────────────

    def _chovendo_ou_iminente(self, env_raw) -> bool:
        """Retorna True se está chovendo agora (chuva > LIMIAR_CHUVA_ATIVA) ou
        se a umidade do ar está alta (LIMIAR_UMIDADE_AR_ALTA), sugerindo
        chuva iminente. Em ambos os casos, a chuva natural dispensará a rega."""
        clima = getattr(env_raw, 'clima', None)
        if clima is None:
            return False
        atual = getattr(clima, 'clima_atual', None) or {}
        if atual.get('chuva', 0.0) > LIMIAR_CHUVA_ATIVA:
            return True
        if atual.get('umidade_ar', 0.0) > LIMIAR_UMIDADE_AR_ALTA:
            return True
        return False

    def _umidade_plantas(self, env_raw) -> float:
        """Umidade média das células com plantas (as que precisam de água)."""
        plantas = env_raw.planta.plantas
        if not plantas:
            return 1.0
        total = sum(
            env_raw.solo.obter_solo_em(pos[0], pos[1]).umidade
            for pos in plantas
        )
        return total / len(plantas)

    def _ervas_max(self, env_raw) -> float:
        """Densidade máxima de ervas daninhas nas células com plantas."""
        plantas = env_raw.planta.plantas
        if not plantas:
            return 0.0
        return max(
            env_raw.ervas_daninhas.obter_densidade_em(pos[0], pos[1])
            for pos in plantas
        )

    def _pragas_max(self, env_raw) -> float:
        """v3: maior populacao de pragas nas celulas com plantas."""
        plantas = env_raw.planta.plantas
        if not plantas:
            return 0.0
        return max(env_raw.pragas.obter_populacao_em(pos[0], pos[1]) for pos in plantas)

    def _aves_em_frutificacao(self, env_raw) -> float:
        """v3: media de aves nas celulas com planta em frutificacao/colheita (0 se nao ha fruto no campo)."""
        celulas = [pos for pos, p in env_raw.planta.plantas.items()
                   if p.estagio in (EstagioPlanta.FRUTIFICACAO, EstagioPlanta.COLHEITA)]
        if not celulas:
            return 0.0
        return sum(env_raw.passaros.obter_populacao_em(pos[0], pos[1]) for pos in celulas) / len(celulas)

    def _doenca_controlavel_max(self, env_raw) -> float:
        """v4: maior impacto de doenca que o defensivo CONSEGUE tratar (cada doenca pesada por `controle_quimico`).
        Pulverizar por causa de mosaico viral, que nao tem quimico curativo, e custo puro."""
        plantas = env_raw.planta.plantas
        if not plantas:
            return 0.0
        return max(env_raw.sistema_doencas.calcular_impacto_controlavel(pos[0], pos[1]) for pos in plantas)

    def _doenca_max(self, env_raw) -> float:
        """Impacto máximo de doença/praga nas células com plantas."""
        plantas = env_raw.planta.plantas
        if not plantas:
            return 0.0
        return max(
            env_raw.sistema_doencas.calcular_impacto_doenca(pos[0], pos[1])
            for pos in plantas
        )

    def _acao_fertilizar(self, env_raw, passo: int = 0) -> Optional[int]:
        """
        Fertiliza o nutriente mais deficiente abaixo do limiar, entre os que já cumpriram a carência
        (v2). Prioridade em caso de empate: N → P → K → C.
        """
        linhas, colunas = env_raw.tamanho_grade
        n = linhas * colunas
        medias = {'N': 0.0, 'P': 0.0, 'K': 0.0, 'C': 0.0}

        for x in range(linhas):
            for y in range(colunas):
                solo = env_raw.solo.obter_solo_em(x, y)
                for k in medias:
                    medias[k] += solo.nutrientes.get(k, 0.0)
        for k in medias:
            medias[k] /= n

        mapa = {'N': FERTILIZAR_N, 'P': FERTILIZAR_P, 'K': FERTILIZAR_K, 'C': FERTILIZAR_C}
        elegiveis = [k for k in PRIORIDADE_NUTRIENTES
                     if passo - self._ultimo_fertilizante[k] >= CARENCIA_FERTILIZANTE]
        if not elegiveis:
            return None
        deficiente = min(elegiveis, key=lambda k: medias[k])
        if medias[deficiente] < LIMIAR_NUTRIENTE:
            return mapa[deficiente]
        return None


# ─── Professor v5: manejo intensivo (D-33, 18/09/2026) ───────────────────────────────────────────

class BotIntensivo(BotPerfeito):
    """Professor v5 — manejo INTENSIVO. É o professor dos experimentos a partir da D-33 (`protocolo.novo_professor()`).

    Por que existe. O `BotPerfeito` (v4) é reativo: só age quando um limiar de dano é cruzado e espera em 64 % dos passos.
    No FarmGym v4 isso desperdiça o mecanismo que mais pesa no retorno: o fator de crescimento é proporcional a
    (1 + qualidade do solo) × (1 − 0,5 × ervas), e esse fator define tanto a duração do ciclo quanto o enchimento do
    fruto (`plant.py`). O v4 aduba só até o mínimo da lei do mínimo (0,15) e deixa a qualidade do solo em ≈ 0,30; faz
    ≈ 2 ciclos por célula. Na rodada 2 o PPO treinado do zero chegou a 1,48 × o professor, ou seja, o professor estava
    abaixo do que o reforço encontra sozinho e ancorava o aluno na espera (IL + RL padrão: 0,87).

    O que muda em relação ao v4 (cada item medido em sementes de calibragem próprias, 20000–20049, disjuntas de treino,
    validação, teste e demonstrações — `versao final/experimentos/professor_v5/`):
      solo      aduba para manter cada nutriente perto de `ALVO_NUTRIENTE` (nível + o que ainda vai ser liberado),
                em vez de só evitar a deficiência; é o que leva a qualidade do solo de ≈ 0,30 a ≈ 0,80;
      ervas     herbicida pela densidade MÉDIA com carência curta: aplicações próximas empilham o efeito residual
                (meia-vida de 14 dias) e seguram as ervas em ≈ 0,15, contra ≈ 0,22 no v4 com solo pobre;
      colheita  em LOTE: `colher` colhe todas as plantas prontas de uma vez, então esperar o lote libera passos para
                plantar e cuidar; uma planta pronta nunca passa de `PROGRESSO_MAX_COLHEITA` (morre em 10);
      aves      espantalhos PROATIVOS (até `N_ESPANTALHOS`), antes de haver fruto, em vez de reagir à população;
      cerca     viva assim que houver passo livre: com herbicida frequente, é ela que sustenta os polinizadores
                (sem a cerca o retorno in-dist cai ≈ 1.100 pontos);
      replantio enquanto restarem `PASSOS_MINIMOS_REPLANTE` passos (o ciclo caiu para ≈ 95 passos).
    Medido e DESCARTADO: cuidado de emergência à frente de colher/plantar (−200 a −380) e alternar plantio e colheita.

    Sem memória própria: a carência dos defensivos é lida do ESTADO do ambiente (idade da aplicação mais recente), e
    não do que o bot lembra ter feito. No DAgger quem executa a ação pode ser o aluno; um rotulador que confia na
    própria memória rotula errado a partir da primeira divergência (era o caso das carências do v4).

    Continua sendo uma heurística, não um ótimo: decide localmente, não antecipa clima nem onda de praga ou de aves,
    não pondera o custo de oportunidade do passo (que aperta no regime seco) e não ajusta nada ao regime.
    """
    VERSAO = 'v5'
    LOTE_COLHEITA            = 3      # colhe quando há >= 3 plantas prontas...
    PROGRESSO_MAX_COLHEITA   = 4.0    # ...ou quando alguma pronta já andou 4 dos 10 de progresso em que morre
    PASSOS_MINIMOS_REPLANTE  = 100    # com menos que isso a planta não chega a encher fruto (no inverno o ciclo alonga)
    LIMIAR_UMIDADE           = 0.58   # rega abaixo disso: +0,2 leva a 0,78, ainda sob o 0,8 em que os micróbios param
    LIMIAR_ERVAS_MEDIA       = 0.11   # herbicida se a densidade MÉDIA nas células plantadas passa disso
    CARENCIA_HERBICIDA       = 4
    LIMIAR_PRAGAS            = 25.0   # o pesticida custa ≈ 25 % dos polinizadores por aplicação: limiar acima do v4 (15)
    LIMIAR_DOENCA            = LIMIAR_DOENCA
    CARENCIA_PESTICIDA       = CARENCIA_PESTICIDA
    ALVO_NUTRIENTE           = 0.90   # nível médio + liberação pendente, por nutriente
    N_ESPANTALHOS            = 8      # avançados; a célula é sorteada pelo ambiente entre as plantadas

    def _selecionar_acao_otima(self, env_raw) -> int:
        plantas = env_raw.planta.plantas
        linhas, colunas = env_raw.tamanho_grade
        n_celulas = linhas * colunas
        passos_restantes = env_raw.passos_maximos - env_raw.passo_atual
        prontas = [p for p in plantas.values() if p.estagio == EstagioPlanta.COLHEITA]

        # ── 1. COLHER em lote ─────────────────────────────────────────────────
        if prontas and (len(prontas) >= self.LOTE_COLHEITA
                        or max(p.progresso_crescimento for p in prontas) >= self.PROGRESSO_MAX_COLHEITA
                        or passos_restantes <= 2):
            return COLHER

        # ── 2. PLANTAR ────────────────────────────────────────────────────────
        if len(plantas) < n_celulas and passos_restantes >= self.PASSOS_MINIMOS_REPLANTE:
            return PLANTAR

        if plantas:
            # ── 3. REGAR ──────────────────────────────────────────────────────
            if self._umidade_plantas(env_raw) < self.LIMIAR_UMIDADE:
                return REGAR
            # ── 4. HERBICIDA ──────────────────────────────────────────────────
            # `_ervas_max > 0,1` é a condição da máscara: sem ela a ação seria inválida
            if (self._ervas_media(env_raw) > self.LIMIAR_ERVAS_MEDIA and self._ervas_max(env_raw) > 0.1
                    and self._passos_desde_defensivo(env_raw, TipoDefensivo.HERBICIDA) >= self.CARENCIA_HERBICIDA):
                return HERBICIDA
            # ── 5. PESTICIDA ──────────────────────────────────────────────────
            if ((self._pragas_max(env_raw) > self.LIMIAR_PRAGAS
                 or self._doenca_controlavel_max(env_raw) > self.LIMIAR_DOENCA)
                    and self._passos_desde_defensivo(env_raw, TipoDefensivo.PESTICIDA) >= self.CARENCIA_PESTICIDA):
                return PESTICIDA

        # ── 6. FERTILIZAR ─────────────────────────────────────────────────────
        acao_fert = self._acao_fertilizar_alvo(env_raw)
        if acao_fert is not None:
            return acao_fert

        # ── 7. ESPANTALHOS proativos ──────────────────────────────────────────
        if plantas and env_raw.instalacoes.n_espantalhos() < self.N_ESPANTALHOS:
            return ESP_AVANCADO

        # ── 8. CERCA VIVA ─────────────────────────────────────────────────────
        if env_raw.instalacoes.n_cercas() < n_celulas:
            return COLOCAR_CERCA

        # ── 9. nada a fazer: colhe o que houver, senão espera ─────────────────
        return COLHER if prontas else ESPERAR

    def _ervas_media(self, env_raw) -> float:
        plantas = env_raw.planta.plantas
        if not plantas:
            return 0.0
        return sum(env_raw.ervas_daninhas.obter_densidade_em(pos[0], pos[1]) for pos in plantas) / len(plantas)

    def _passos_desde_defensivo(self, env_raw, tipo: TipoDefensivo) -> int:
        """Passos desde a última aplicação de `tipo`, lidos do ambiente. Herbicida e pesticida são ações do campo
        inteiro, então a lista de aplicações de uma célula qualquer basta."""
        aplicacoes = next(iter(env_raw.defensivos_fertilizantes.aplicacoes.values()), [])
        idades = [a.idade for a in aplicacoes if a.tipo == 'defensivo' and a.tipo_defensivo == tipo]
        return min(idades) if idades else 10 ** 6

    _ACAO_POR_NUTRIENTE = {'N': FERTILIZAR_N, 'P': FERTILIZAR_P, 'K': FERTILIZAR_K, 'C': FERTILIZAR_C}

    def _acao_fertilizar_alvo(self, env_raw) -> Optional[int]:
        """Aduba o nutriente com o maior déficit em relação a `ALVO_NUTRIENTE`, contando o que já foi aplicado e ainda
        vai ser liberado (o fertilizante libera 0,1 por passo, durante 20 passos)."""
        n = len(env_raw.solo.estado_solo)
        previsto = {k: 0.0 for k in self._ACAO_POR_NUTRIENTE}
        for solo in env_raw.solo.estado_solo.values():
            for k in previsto:
                previsto[k] += solo.nutrientes.get(k, 0.0) / n
        for aplicacoes in env_raw.defensivos_fertilizantes.aplicacoes.values():
            for a in aplicacoes:
                if a.tipo == 'fertilizante' and a.tipo_nutriente is not None:
                    k = a.tipo_nutriente.value
                    previsto[k] += a.quantidade * Solo._EFICIENCIA_FERTILIZANTE[k] / n
        k = min(PRIORIDADE_NUTRIENTES, key=lambda q: previsto[q])
        return self._ACAO_POR_NUTRIENTE[k] if previsto[k] < self.ALVO_NUTRIENTE else None


# ─── Treinador / Executor ─────────────────────────────────────────────────────

class ExecutorBot:
    """
    Executa o BotPerfeito, coleta estatísticas e salva trajetórias de especialista
    para uso posterior em Imitation Learning.
    """

    def __init__(self, env, dir_resultados: str, imperfeicao: float = 0.0,
                  lazy_steps: int = 5, seed: Optional[int] = None):
        self.env = env
        self.bot = BotPerfeito(imperfeicao=imperfeicao, lazy_steps=lazy_steps, seed=seed)
        self.dir = os.path.join(dir_resultados, f"bot_perfeito_{int(time.time())}")
        os.makedirs(self.dir, exist_ok=True)

        self.recompensas   = []
        self.colheitas     = []
        self.duracoes      = []
        self.acoes_total   = defaultdict(int)

    def _env_raw(self):
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env

    def executar(self, num_episodios: int = 100, passos_maximos: int = 365,
                 salvar_trajetorias: bool = True) -> str:

        print(f"\n{'='*60}")
        print(f"  BOT PERFEITO — {num_episodios} episódios × {passos_maximos} passos")
        print(f"  Dir: {self.dir}")
        print(f"{'='*60}\n")

        todas_trajetorias = []
        t0 = time.time()

        for ep in range(num_episodios):
            obs, info = self.env.reset()
            self.bot.resetar_episodio()
            env_raw = self._env_raw()

            rec_ep  = 0.0
            colh_ep = 0
            traj_ep = [] if salvar_trajetorias else None

            for _ in range(passos_maximos):
                acao    = self.bot.selecionar_acao(env_raw)
                obs_ant = obs.copy()
                mask    = info.get('mascara_acoes', np.ones(15, dtype=np.float32))

                obs, r, done, trunc, info = self.env.step(acao)
                rec_ep += r
                self.acoes_total[NOMES_ACOES[acao]] += 1

                if acao == COLHER:
                    colh_ep += 1

                if salvar_trajetorias:
                    traj_ep.append({
                        'obs':      obs_ant.tolist(),
                        'acao':     int(acao),
                        'recompensa': float(r),
                        'prox_obs': obs.tolist(),
                        'feito':    bool(done or trunc),
                        'mascara':  mask.tolist(),
                    })

                if done or trunc:
                    break

            self.recompensas.append(rec_ep)
            self.colheitas.append(colh_ep)
            self.duracoes.append(_ + 1)

            if salvar_trajetorias and traj_ep:
                todas_trajetorias.append(traj_ep)

            if (ep + 1) % 10 == 0:
                ult = self.recompensas[-10:]
                print(f"  Ep {ep+1:4d}/{num_episodios} | "
                      f"Média(10): {np.mean(ult):9.2f} | "
                      f"Última: {rec_ep:9.2f} | "
                      f"Colheitas: {colh_ep:3d}")

        tempo = time.time() - t0

        if salvar_trajetorias and todas_trajetorias:
            n_trans = sum(len(t) for t in todas_trajetorias)
            cam = os.path.join(self.dir, "trajetorias_especialista.json")
            with open(cam, 'w') as f:
                json.dump(todas_trajetorias, f)
            print(f"\n  Trajetórias: {cam}")
            print(f"  {len(todas_trajetorias)} episódios, {n_trans} transições")

        self._salvar_stats(tempo)
        self._gerar_graficos()
        self._resumo()

        return self.dir

    def _salvar_stats(self, tempo: float):
        stats = {
            'recompensas':       self.recompensas,
            'colheitas':         self.colheitas,
            'duracoes':          self.duracoes,
            'acoes_total':       dict(self.acoes_total),
            'tempo_segundos':    tempo,
            'media_recompensa':  float(np.mean(self.recompensas)),
            'melhor_recompensa': float(np.max(self.recompensas)),
            'pior_recompensa':   float(np.min(self.recompensas)),
            'std_recompensa':    float(np.std(self.recompensas)),
            'media_colheitas':   float(np.mean(self.colheitas)),
        }
        with open(os.path.join(self.dir, 'estatisticas.json'), 'w') as f:
            json.dump(stats, f, indent=2)

    def _gerar_graficos(self):
        try:
            fig, eixos = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle('Bot Perfeito — Oracle Ótimo', fontsize=14, fontweight='bold')

            recs = self.recompensas
            colh = self.colheitas

            # Recompensa por episódio
            eixos[0].plot(recs, alpha=0.35, linewidth=0.8, color='steelblue')
            mm = [np.mean(recs[max(0, i-10):i+1]) for i in range(len(recs))]
            eixos[0].plot(mm, color='red', linewidth=2, label=f'MM-10')
            eixos[0].axhline(np.mean(recs), color='green', linestyle='--',
                             linewidth=1.5, label=f'Média: {np.mean(recs):.0f}')
            eixos[0].set_title('Recompensa por Episódio')
            eixos[0].set_xlabel('Episódio')
            eixos[0].set_ylabel('Recompensa')
            eixos[0].legend()
            eixos[0].grid(True, alpha=0.3)

            # Colheitas por episódio
            eixos[1].plot(colh, color='darkorange', alpha=0.7, linewidth=1.2)
            eixos[1].axhline(np.mean(colh), color='red', linestyle='--',
                             linewidth=1.5, label=f'Média: {np.mean(colh):.1f}')
            eixos[1].set_title('Colheitas por Episódio')
            eixos[1].set_xlabel('Episódio')
            eixos[1].set_ylabel('Número de colheitas')
            eixos[1].legend()
            eixos[1].grid(True, alpha=0.3)

            # Distribuição de ações
            acoes_ord = sorted(self.acoes_total.items(), key=lambda x: x[1], reverse=True)
            if acoes_ord:
                nomes, vals = zip(*acoes_ord)
                cores = ['#2196F3' if v > np.mean(vals) else '#90CAF9' for v in vals]
                eixos[2].barh(nomes, vals, color=cores)
                eixos[2].set_title('Ações Executadas (total)')
                eixos[2].set_xlabel('Frequência')
                eixos[2].grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig(os.path.join(self.dir, 'desempenho.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  [aviso] Erro nos gráficos: {e}")

    def _resumo(self):
        recs = self.recompensas
        colh = self.colheitas
        print(f"\n{'='*60}")
        print(f"  BOT PERFEITO — RESULTADO FINAL")
        print(f"{'='*60}")
        print(f"  Episódios executados:   {len(recs)}")
        print(f"  Recompensa média:       {np.mean(recs):>10.2f}")
        print(f"  Melhor episódio:        {np.max(recs):>10.2f}")
        print(f"  Pior episódio:          {np.min(recs):>10.2f}")
        print(f"  Desvio padrão:          {np.std(recs):>10.2f}")
        print(f"  Colheitas por episódio: {np.mean(colh):>10.2f}")
        print(f"{'='*60}")
        print(f"  Top ações:")
        for nome, cnt in sorted(self.acoes_total.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {nome:25s}: {cnt:6d}")
        print(f"{'='*60}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from env import AmbienteFazendaGym
    from mascara_acoes import InvolucroMascaraAcoes

    # --- argumentos CLI (protocolo v2.0) ---
    import argparse
    import random
    parser = argparse.ArgumentParser(description='Geração de demonstrações do BotPerfeito')
    parser.add_argument('--n-eps', type=int, default=200,
                        help='Nº de episódios de demonstração (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed para reprodutibilidade (default: 42)')
    parser.add_argument('--passos-maximos', type=int, default=365,
                        help='Passos por episódio (default: 365)')
    parser.add_argument('--imperfeicao', type=float, default=0.0,
                        help='Prob por step de entrar em modo preguicoso (default: 0 = o professor da D-27; 0.14 = variante)')
    parser.add_argument('--lazy-steps', type=int, default=8,
                        help='Steps consecutivos em ESPERAR ao entrar em modo preguicoso (default: 8)')
    args = parser.parse_args()

    # Fixar todas as fontes de aleatoriedade
    np.random.seed(args.seed)
    random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    env = InvolucroMascaraAcoes(AmbienteFazendaGym(passos_maximos=args.passos_maximos))
    try:
        env.reset(seed=args.seed)
    except TypeError:
        env.reset()

    dir_res = os.path.join(_DIR, '../../results/bot_perfeito')
    os.makedirs(dir_res, exist_ok=True)

    print(f"[BotPerfeito] seed={args.seed}  n_eps={args.n_eps}  horizonte={args.passos_maximos}")
    print(f"[BotPerfeito] imperfeicao={args.imperfeicao}  lazy_steps={args.lazy_steps}")
    executor = ExecutorBot(env, dir_res, imperfeicao=args.imperfeicao,
                            lazy_steps=args.lazy_steps, seed=args.seed)
    executor.executar(num_episodios=args.n_eps, passos_maximos=args.passos_maximos,
                       salvar_trajetorias=True)
