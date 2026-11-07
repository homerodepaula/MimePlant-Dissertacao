"""avaliacao.py — a ÚNICA função de avaliação da defesa (D-06, D-10, D-18).

Todo número reportado (oráculo, aleatório, IL, RL, HPO) sai de `avaliar(...)`:
  - episódios identificados pela SEMENTE (reset(seed=s)); os mesmos para todos os agentes (common random numbers);
  - regime aplicado depois do reset (in_dist / dry / humid), como na qualificação;
  - políticas de rede decodificadas por AMOSTRAGEM da distribuição mascarada (τ = 1; D-26, 06/09/2026), com um gerador
    próprio por episódio semeado pela semente do episódio — o fluxo de RNG do ambiente não muda com a política (pareamento
    por semente preservado) e a avaliação é determinística dada a semente; `PoliticaRede(agente, tau=0)` dá o argmax
    (análise secundária: até 06/09 o argmax era a regra, D-05). Hidden zerado no início de cada episódio;
  - pausas do oráculo semeadas por episódio (S13);
  - tudo dentro de `protocolo.rng_isolado()`: a avaliação não altera o fluxo de RNG do treino.

Interface de política (três adaptadores prontos abaixo):
    politica.iniciar_episodio(seed_episodio, env_bruto)   # zera hidden / semeia pausas / etc.
    politica.agir(obs, mascara) -> int
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

import protocolo as P

P.preparar_sys_path()


# ── Adaptadores de política ──────────────────────────────────────────────────────────────────────
SEMENTE_DECODIFICACAO = 7_919_000   # deslocamento da semente do gerador de amostragem (D-26); + semente do episódio


class PoliticaRede:
    """Agente com `.rede.inicializar_hidden(1, dispositivo)`, `.dispositivo` e `.selecionar_acao(obs, mask, hidden, ...)`
    (AgenteBC, AgenteDAgger, AgentePolitica, agentes de RL).

    Decodificação (D-26, 06/09/2026): amostragem da distribuição mascarada com temperatura `tau` (padrão 1,0 = a política
    tal como treinada; o PPO é uma distribuição, e seu argmax é outra política — no bake-off da rodada 2 o argmax entrava em
    ciclos colocar/remover espantalho e fazia 3.347 in-dist onde a mesma rede amostrada fazia 6.111). `tau = 0` dá o argmax
    (análise secundária). A amostragem usa `np.random.default_rng(SEMENTE_DECODIFICACAO + semente_do_episódio)`: não toca
    `np.random`/`random` (usados pelo ambiente) nem o RNG do torch, então a decodificação não perturba o fluxo do ambiente e
    duas avaliações da mesma rede são idênticas. Atenção ao alcance do pareamento: mesma semente significa mesmo estado
    inicial e mesmo fluxo de sorteios, mas o ambiente consome `np.random` dentro de algumas ações (plantar, espantalho),
    então a partir da primeira ação em que duas políticas discordam a realização climática delas diverge. O pareamento é
    por episódio, não passo a passo (auditoria de 06/09)."""

    def __init__(self, agente, tau: float = 1.0):
        self.agente = agente
        self.tau = float(tau)
        self._hidden = None
        self._rng = None

    def iniciar_episodio(self, seed_episodio: int, env_bruto) -> None:
        self._hidden = self.agente.rede.inicializar_hidden(1, self.agente.dispositivo)
        self._rng = np.random.default_rng(SEMENTE_DECODIFICACAO + int(seed_episodio))

    def agir(self, obs: np.ndarray, mascara: np.ndarray) -> int:
        acao, self._hidden = self.agente.selecionar_acao(obs, mascara, self._hidden, temperatura=self.tau, rng=self._rng)
        return int(acao)


class PoliticaOraculo:
    """O oráculo (`BotPerfeito`) com acesso ao estado interno. O professor dos experimentos é o de p = 0 (D-27); a
    variante com pausas continua construível por `protocolo.novo_oraculo_variante()`, com as pausas semeadas por episódio."""

    def __init__(self, bot):
        self.bot = bot
        self._env_bruto = None

    def iniciar_episodio(self, seed_episodio: int, env_bruto) -> None:
        self._env_bruto = env_bruto
        self.bot.resetar_episodio(seed_episodio=seed_episodio)

    def agir(self, obs: np.ndarray, mascara: np.ndarray) -> int:
        return int(self.bot.selecionar_acao(self._env_bruto))


class PoliticaAleatoria:
    """Ação uniforme entre as válidas; RNG próprio semeado por episódio (reprodutível e pareado)."""

    def __init__(self, seed_base: int = 0):
        self.seed_base = int(seed_base)
        self._rng = np.random.RandomState(self.seed_base)

    def iniciar_episodio(self, seed_episodio: int, env_bruto) -> None:
        self._rng = np.random.RandomState(self.seed_base + int(seed_episodio))

    def agir(self, obs: np.ndarray, mascara: np.ndarray) -> int:
        validas = np.where(np.asarray(mascara) > 0.5)[0]
        return int(self._rng.choice(validas)) if len(validas) else P.ACAO_ESPERAR


class PoliticaFixa:
    """Política dada por uma função `f(env_bruto, obs, mascara, t) -> acao` (usada nas análises de sensibilidade)."""

    def __init__(self, funcao):
        self.funcao = funcao
        self._env_bruto = None
        self._t = 0

    def iniciar_episodio(self, seed_episodio: int, env_bruto) -> None:
        self._env_bruto = env_bruto
        self._t = 0

    def agir(self, obs: np.ndarray, mascara: np.ndarray) -> int:
        acao = int(self.funcao(self._env_bruto, obs, mascara, self._t))
        self._t += 1
        return acao


# ── A função de avaliação ────────────────────────────────────────────────────────────────────────
def avaliar(politica, env, sementes: Sequence[int], regime: Optional[str] = 'in_dist') -> Dict:
    """Roda um episódio por semente e devolve um dicionário serializável.

    Retorno:
      regime, sementes, recompensas (lista por episódio), duracoes, contagem_acoes (total), acoes_por_episodio
      (matriz n_eps × 15), kpis (listas por episódio: produtividade, qualidade_solo, uso_quimicos, biodiversidade),
      comp (soma por episódio de cada componente de info['recompensa_comp'], se o ambiente a fornecer).
    """
    bruto = P.env_bruto(env)
    res = {
        'regime': regime or 'in_dist', 'sementes': [int(s) for s in sementes],
        'recompensas': [], 'duracoes': [], 'acoes_por_episodio': [],
        'kpis': {'produtividade': [], 'qualidade_solo': [], 'uso_quimicos': [], 'biodiversidade': []},
        'kpis_brutos': [],      # v3: o dicionario info['kpis'] inteiro no ultimo passo de cada episodio
        'comp': {},
    }
    with P.rng_isolado():
        for s in sementes:
            obs, info = env.reset(seed=int(s))
            P.aplicar_regime(env, regime)
            obs = bruto._obter_observacao()          # a observação do dia 0 muda com o regime
            info['mascara_acoes'] = env._gerar_mascara(obs) if hasattr(env, '_gerar_mascara') else info.get('mascara_acoes')
            politica.iniciar_episodio(int(s), bruto)
            contagem = np.zeros(P.N_ACOES, dtype=np.int64)
            comp_ep: Dict[str, float] = {}
            total, dur, fim = 0.0, 0, False
            while not fim:
                mascara = info.get('mascara_acoes', np.ones(P.N_ACOES, dtype=np.float32))
                acao = politica.agir(obs, mascara)
                obs, r, terminado, truncado, info = env.step(acao)
                total += float(r); dur += 1; contagem[acao] += 1
                for k, v in info.get('recompensa_comp', {}).items():
                    comp_ep[k] = comp_ep.get(k, 0.0) + float(v)
                fim = terminado or truncado
            kp = info.get('kpis', {})
            res['kpis']['produtividade'].append(float(kp.get('produtividade_acumulada', np.nan)))
            res['kpis']['qualidade_solo'].append(float(kp.get('qualidade_solo_media', np.nan)))
            res['kpis']['uso_quimicos'].append(float(kp.get('uso_quimicos_total', np.nan)))
            res['kpis']['biodiversidade'].append(float(kp.get('densidade_polinizadores', np.nan)))
            res['kpis_brutos'].append({k: float(v) for k, v in kp.items()})
            res['recompensas'].append(total); res['duracoes'].append(dur)
            res['acoes_por_episodio'].append(contagem.tolist())
            for k, v in comp_ep.items():
                res['comp'].setdefault(k, []).append(v)
        P.restaurar_regime(env)
    contagem_total = np.sum(np.array(res['acoes_por_episodio']), axis=0) if res['acoes_por_episodio'] else np.zeros(P.N_ACOES)
    res['contagem_acoes'] = {P.NOMES_ACOES[i]: int(contagem_total[i]) for i in range(P.N_ACOES)}
    return res


def avaliar_regimes(politica, env, sementes: Sequence[int], regimes: Sequence[str] = tuple(P.NOMES_REGIMES)) -> Dict[str, Dict]:
    return {reg: avaliar(politica, env, sementes, reg) for reg in regimes}


# ── Estatísticas de resumo ───────────────────────────────────────────────────────────────────────
def iqm(valores: Sequence[float]) -> float:
    """Média interquartil (Agarwal et al. 2021): média dos valores entre os percentis 25 e 75."""
    v = np.sort(np.asarray(valores, dtype=float))
    n = len(v)
    if n == 0:
        return float('nan')
    ini, fim = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(v[ini:fim].mean()) if fim > ini else float(v.mean())


def resumo(res: Dict) -> Dict[str, float]:
    """`std` e `erro_padrao` com ddof = 1 (desvio amostral), consistentes entre si (06/09/2026)."""
    r = np.asarray(res['recompensas'], dtype=float)
    return {'n': int(len(r)), 'media': float(r.mean()), 'std': float(r.std(ddof=1)) if len(r) > 1 else 0.0, 'iqm': iqm(r),
            'min': float(r.min()), 'max': float(r.max()),
            'erro_padrao': float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else float('nan')}


# `observar` não altera estado nenhum (a observação já é integral) e custa 0,1: é dominada por `esperar`. Fica válida
# porque mascará-la mudaria o espaço de ações e obrigaria a refazer tudo desde as demonstrações, mas entra na
# ociosidade — senão a métrica subestima a inação, como acontecia com `remover_espantalho` antes da D-23.
ACOES_OCIOSAS = ('esperar', 'remover_espantalho', 'observar')


def acoes_pct(res: Dict) -> Dict[str, float]:
    """Percentual de cada ação sobre todos os passos avaliados (a partir de `contagem_acoes`)."""
    total = float(sum(res['contagem_acoes'].values())) or 1.0
    return {nome: 100.0 * n / total for nome, n in res['contagem_acoes'].items()}


def ociosidade_pct(res: Dict) -> float:
    """Percentual de passos em que o agente não interveio: `esperar`, `remover_espantalho` e `observar`. As duas
    últimas são inação disfarçada: `remover_espantalho` era um segundo "esperar" gratuito até a correção da máscara
    (D-23) e `observar` é um no-op que custa 0,1 (auditoria de 06/09). `esperar_pct` sozinho subestimava a inação em
    até 79 pontos (criticas.md, §2.3)."""
    total = float(sum(res['contagem_acoes'].values())) or 1.0
    return 100.0 * sum(res['contagem_acoes'].get(a, 0) for a in ACOES_OCIOSAS) / total


def normalizar(recompensa: float, r_aleatorio: float, r_perfeito: float) -> float:
    """Retorno normalizado por regime (D-18): 0 = agente aleatório, 1 = o **professor** (o oráculo sem pausas, D-27).
    O 1 é o nível de uma heurística, não um ótimo: valores acima de 1 são esperados e significam que o agente decidiu
    melhor do que a regra que o ensinou."""
    return (recompensa - r_aleatorio) / max(1e-9, (r_perfeito - r_aleatorio))
