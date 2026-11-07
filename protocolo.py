"""protocolo.py — a única fonte das constantes e das utilidades do protocolo experimental da defesa (FarmGym v3).

Regras (D-10, D-17, D-18 em ../../0_contexto/DECISOES.md):
  - Sementes têm papel. Quem escolhe alguma coisa (checkpoint, rodada, trial, candidato) só olha VALIDAÇÃO.
      treino              {42, 123, 2024, 7, 1337}   pesos da rede, embaralhamento, coleta DAgger/rollouts
      validação-rollback  500..519                    early stopping / rollback / seleção de checkpoint
      validação-objetivo  520..549                    objetivo da HPO e do bake-off
      teste               999..1048                   só para reportar (uma vez por agente)
      demos               10000..10199                episódios das demonstrações do oráculo
  - O PROFESSOR dos experimentos é o oráculo SEM pausas (`PROFESSOR_IMPERFEICAO = 0`, D-27), na política de manejo
    INTENSIVO (`PROFESSOR_VERSAO = 'v5'`, D-33). O professor da rodada 2 (manejo reativo, v4) continua construível por
    `novo_professor_v4()`, só para reproduzi-la e comparar. A variante com pausas existe
    (`ORACULO_IMPERFEICAO_VARIANTE`) e é reproduzível — RandomState(ORACULO_SEED + semente_do_episodio) —, mas nenhum
    experimento da defesa a usa. Construa o professor por `novo_professor()`, nunca à mão.
  - Toda avaliação roda dentro de `rng_isolado()`: o `reset(seed)` do ambiente semeia os RNG globais, e sem o
    isolamento uma avaliação no meio do treino re-semearia o fluxo de treino de todos os trials.
  - Nenhum outro arquivo redigita 0, 0.14, 8, 12345, 365, 500, 999 etc.: importa daqui.
"""
import os
import sys
import json
import time
import random
import hashlib
import platform
import contextlib
import subprocess
from typing import Dict, List, Optional

import numpy as np

# ── Caminhos ─────────────────────────────────────────────────────────────────────────────────────
RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR_AMBIENTE = os.path.join(RAIZ, 'environment')
DIR_IL = os.path.join(RAIZ, 'agents', 'imitation-learning')
DIR_RL = os.path.join(RAIZ, 'agents', 'rl')
DIR_RESULTS = os.path.join(RAIZ, 'results')
SUBPASTAS_AMBIENTE = ['plant', 'birds', 'pollinators', 'soil', 'weather', 'weeds', 'pest',
                      'cides-fertilizers', 'facilities']


def preparar_sys_path() -> None:
    """Coloca ambiente, agentes e subpastas no sys.path (idempotente). Chamar antes de importar env/bot/agentes."""
    caminhos = [RAIZ, DIR_AMBIENTE, DIR_IL, DIR_RL] + \
               [os.path.join(DIR_AMBIENTE, s) for s in SUBPASTAS_AMBIENTE]
    for p in caminhos:
        if p not in sys.path:
            sys.path.insert(0, p)


# ── Ambiente ─────────────────────────────────────────────────────────────────────────────────────
VERSAO_AMBIENTE = 'FarmGym v4'   # D-24 (06/09/2026): subsistemas ligados; ver environment/env.py
PASSOS_MAXIMOS = 365
TAMANHO_GRADE = (4, 4)
# FarmGym v4: espelho do layout definido em environment/env.py (unica fonte). Um teste trava a igualdade.
OBS_DIM = 176          # 5 base + 11 ambiental + 16 celulas x 10
INICIO_GRADE = 16      # primeiro indice do bloco por celula
TAM_CELULA = 10
N_CELULAS = 16
N_ACOES = 14           # v4: 'observar' saiu (no-op com custo)
NOMES_ACOES = ['plantar', 'colher', 'regar', 'fertilizar_N', 'fertilizar_P', 'fertilizar_K', 'fertilizar_C',
               'herbicida', 'pesticida', 'colocar_espantalho_basico', 'colocar_espantalho_avancado',
               'remover_espantalho', 'colocar_cerca', 'esperar']
ACAO_PLANTAR, ACAO_COLHER, ACAO_REGAR = 0, 1, 2
ACAO_HERBICIDA, ACAO_PESTICIDA = 7, 8
ACAO_ESPERAR = 13   # v4: era 14, antes de 'observar' sair

# ── Oráculo (Bot Ajustável) ──────────────────────────────────────────────────────────────────────
# D-27 (06/09/2026): o professor dos experimentos é o oráculo sem pausas. A margem acima dele é estreita porém real e
# medida no próprio ambiente: a melhor política reativa completa fica 4,6 % abaixo, e a regra não usa cerca viva, não
# ajusta a irrigação à temperatura e não explora o multiplicador de cuidado contínuo. Não é preciso degradar o professor
# de propósito para haver espaço de aprendizado. A variante com pausas continua disponível para estudos de qualidade.
PROFESSOR_IMPERFEICAO = 0.0          # p do PROFESSOR: demonstrações, DAgger, âncora do KL e comparações
# D-33 (18/09/2026): a POLÍTICA do professor passa a ser o manejo intensivo (`bot_perfeito.BotIntensivo`). O professor
# reativo da rodada 2 (`BotPerfeito`, 'v4') ficava abaixo do que o PPO aprende do zero na configuração padrão (1,48 × o
# professor) e ancorava o aluno na espera. Medido em validação (500–549): ver experimentos/professor_v5/.
PROFESSOR_VERSAO = 'v5'              # 'v5' = BotIntensivo (D-33) | 'v4' = BotPerfeito (rodada 2, 06–10/09/2026)
ORACULO_IMPERFEICAO_VARIANTE = 0.14  # p da variante degradada (referência reportada; nenhum experimento da defesa a usa)
ORACULO_LAZY_STEPS = 8               # L: duração da pausa, quando há pausa
ORACULO_SEED = 12345                 # base das pausas: RandomState(ORACULO_SEED + semente_do_episodio)

# ── Sementes por papel ───────────────────────────────────────────────────────────────────────────
SEMENTES_TREINO: List[int] = [42, 123, 2024, 7, 1337]
SEED_VALIDACAO_ROLLBACK, N_VALIDACAO_ROLLBACK = 500, 20
SEED_VALIDACAO_OBJETIVO, N_VALIDACAO_OBJETIVO = 520, 30
SEED_TESTE, N_TESTE = 999, 50
SEED_DEMOS, N_DEMOS = 10000, 200


def sementes_validacao_rollback() -> List[int]:
    return list(range(SEED_VALIDACAO_ROLLBACK, SEED_VALIDACAO_ROLLBACK + N_VALIDACAO_ROLLBACK))


def sementes_validacao_objetivo() -> List[int]:
    return list(range(SEED_VALIDACAO_OBJETIVO, SEED_VALIDACAO_OBJETIVO + N_VALIDACAO_OBJETIVO))


def sementes_validacao() -> List[int]:
    """Validação completa (rollback + objetivo): usada nas referências (oráculo, aleatório, IL)."""
    return sementes_validacao_rollback() + sementes_validacao_objetivo()


def sementes_teste() -> List[int]:
    return list(range(SEED_TESTE, SEED_TESTE + N_TESTE))


def sementes_demos() -> List[int]:
    return list(range(SEED_DEMOS, SEED_DEMOS + N_DEMOS))


def seed_pausas_oraculo(seed_episodio: int) -> int:
    return ORACULO_SEED + int(seed_episodio)


def classe_do_professor(versao: Optional[str] = None):
    """A classe da política do professor: 'v5' = manejo intensivo (D-33), 'v4' = manejo reativo (rodada 2)."""
    preparar_sys_path()
    from bot_perfeito import BotPerfeito, BotIntensivo
    classes = {'v4': BotPerfeito, 'v5': BotIntensivo}
    versao = PROFESSOR_VERSAO if versao is None else versao
    if versao not in classes:
        raise ValueError(f'versão de professor desconhecida: {versao!r} (use {sorted(classes)})')
    return classes[versao]


def novo_bot(imperfeicao: float = PROFESSOR_IMPERFEICAO, lazy_steps: int = ORACULO_LAZY_STEPS, versao: Optional[str] = None):
    """Um bot na política do professor (ou na `versao` pedida), com a imperfeição dada e a semente de pausas do protocolo."""
    return classe_do_professor(versao)(imperfeicao=imperfeicao, lazy_steps=lazy_steps, seed=ORACULO_SEED)


def novo_professor():
    """O professor dos experimentos (D-27 + D-33): manejo intensivo, sem pausas. Único ponto de construção —
    `gerar_demos.py`, `treinar_warmstart_il.py` e as referências passam por aqui, então trocar o professor é trocar
    uma constante (e regenerar as demonstrações: o hash delas é conferido no warm-start)."""
    return novo_bot(PROFESSOR_IMPERFEICAO)


def novo_professor_v4():
    """O professor da rodada 2 (manejo reativo, sem pausas). Só para reproduzi-la e para comparar qualidade de professor."""
    return novo_bot(PROFESSOR_IMPERFEICAO, versao='v4')


def novo_oraculo_variante():
    """A variante degradada por pausas (p = 0,14, L = 8) da política do professor. Só referência reportada e estudos
    de qualidade do professor."""
    return novo_bot(ORACULO_IMPERFEICAO_VARIANTE)


PAPEIS_SEMENTES = {
    'treino': SEMENTES_TREINO,
    'validacao_rollback': sementes_validacao_rollback(),
    'validacao_objetivo': sementes_validacao_objetivo(),
    'teste': sementes_teste(),
    'demos': sementes_demos(),
}

# ── Regimes climáticos (cap. 4: in-dist, dry, humid) ─────────────────────────────────────────────
CLIMA_NORMAL = dict(temperatura_base=20.0, umidade_ar_base=0.5, probabilidade_chuva_base=0.3)
REGIMES: Dict[str, Optional[dict]] = {
    'in_dist': None,
    'dry': dict(temperatura_base=32.0, umidade_ar_base=0.25, probabilidade_chuva_base=0.05),
    'humid': dict(temperatura_base=20.0, umidade_ar_base=0.85, probabilidade_chuva_base=0.70),
}
NOMES_REGIMES = ['in_dist', 'dry', 'humid']


def env_bruto(env):
    """Desembrulha wrappers do Gymnasium até o AmbienteFazendaGym."""
    while hasattr(env, 'env'):
        env = env.env
    return env


def aplicar_regime(env, regime: Optional[str]) -> None:
    """Aplica o regime OOD depois de um reset (mesmo procedimento da qualificação: muda as bases do gerador de
    clima, liga `modo_ood` e re-sorteia o dia 0). `None`/'in_dist' só restaura os parâmetros normais, SEM re-sortear:
    assim o episódio in-dist de semente s é exatamente o mesmo em `avaliacao.avaliar` e nos rollouts de treino
    (`EnvLote`) — o dia 0 é o que `reset(seed)` já gerou."""
    clima = env_bruto(env).clima
    if regime in (None, 'in_dist'):
        for k, v in CLIMA_NORMAL.items():
            setattr(clima, k, v)
        clima.modo_ood = None
        return
    if regime not in REGIMES:
        raise ValueError(f'regime desconhecido: {regime!r} (use {NOMES_REGIMES})')
    for k, v in REGIMES[regime].items():
        setattr(clima, k, v)
    clima.modo_ood = regime
    clima.resetar()


def restaurar_regime(env) -> None:
    aplicar_regime(env, None)


# ── Demonstrações ────────────────────────────────────────────────────────────────────────────────
DIR_DEMOS = os.path.join(DIR_RESULTS, 'demos_v5')
DEMOS_PADRAO = os.path.join(DIR_DEMOS, 'trajetorias_especialista.json')
HASH_DEMOS_PADRAO: Optional[str] = '9e92a38adce50419d68c310a959899e8ee3de5d3b9da444792f4ca1b2ddbaf11'   # demos do
# PROFESSOR v5 (manejo intensivo, D-33) no FarmGym v4 (18/09/2026 16:34, git 6a19c71): 200 episodios, 73.000 transicoes,
# retorno 5.469,8 +- 236,9, 23,1 acoes de colheita (em lote), espera 36,4 %, observacao de 176 colunas. As demos do
# professor v4 da rodada 2 (665902b6..., retorno 3.036,1 +- 100,2, espera 64,2 %) continuam em results/demos_v2 e NAO
# carregam aqui, de proposito: o warm-start aborta se o hash do arquivo nao for este.


# ── Rodada de experimentos ───────────────────────────────────────────────────────────────────────
# D-33 (18/09/2026): a rodada 3 é a cadeia refeita com o professor v5. Cada rodada grava numa pasta PRÓPRIA e todo script
# toma daqui o seu caminho padrão — até a rodada 2 os padrões (`results/referencias_v2`, `results/il_warmstart_v2`,
# `results/bakeoff`, ...) estavam redigitados em oito scripts, e bastava esquecer um `--referencias` para normalizar um
# agente da rodada nova pela âncora da antiga, em silêncio. A rodada 2 (professor v4) fica onde estava, intocada.
RODADA = 'rodada3'
DIR_RODADA = os.path.join(DIR_RESULTS, RODADA)
DIR_REFERENCIAS = os.path.join(DIR_RODADA, 'referencias')
REFERENCIAS_PADRAO = os.path.join(DIR_REFERENCIAS, 'referencias.json')
DIR_WARMSTART = os.path.join(DIR_RODADA, 'il_warmstart')          # clone do professor (BC + DAgger), 5 sementes
DIR_RL_PADRAO = os.path.join(DIR_RODADA, 'rl_padrao')             # RL na configuração padrão, SEM HPO (ppo_il e o controle ppo_zero)
DIR_REAVALIACAO = os.path.join(DIR_RODADA, 'hpo_reavaliacao')     # top-k da HPO e os padrões, 5 sementes, validação-objetivo e teste
DIR_ABLACOES = os.path.join(DIR_RODADA, 'hpo_ablacoes')
DIR_POS_HPO = os.path.join(DIR_RODADA, 'pos_hpo')
DIR_CONTROLE_ZERO = os.path.join(DIR_RODADA, 'controle_zero')     # D-32: HPO sobre o RL treinado do zero (opcional)
DIR_REAVALIACAO_ZERO = os.path.join(DIR_RODADA, 'hpo_reavaliacao_zero')
STUDIES_HPO = ('tpe_r3', 'rs_r3')                                 # bancos em ../experimentos/hpo/<study>.db
# DEHB (D-16/D-34): estudo de COMPARAÇÃO. Fica FORA de STUDIES_HPO de propósito: quem está em STUDIES_HPO concorre a MimePlant
# (`reavaliar_topk.escolher_melhor_candidato`) e vira base das ablações; a reavaliação do DEHB vai para pasta própria.
STUDY_DEHB = 'dehb_r3'
DIR_REAVALIACAO_DEHB = os.path.join(DIR_RODADA, 'hpo_reavaliacao_dehb')
SUFIXO_STUDIES_ZERO = 'r3'                                        # tpe_zero_r3, rs_zero_r3
# Passos de ambiente de UMA execução de RL (1 FRE). Fonte única: bake-off, HPO, reavaliação, ablações e controle leem
# daqui, porque a reavaliação só reaproveita uma execução padrão se o orçamento for idêntico. Na rodada 2 eram 250.000
# (57 atualizações do PPO com V = 12) e TODAS as curvas de validação ainda subiam no último bloco (12 de 15 execuções com
# o melhor bloco = o último; `results/hpo_reavaliacao/*/seed_*/metricas.json`). O valor da rodada 3 é decisão do Homero
# (a discutir antes de lançar); o encadeador o passa explicitamente a cada etapa.
ORCAMENTO_RL_PASSOS = 1_000_000


# ── Utilidades de reprodutibilidade e registro ───────────────────────────────────────────────────
def hash_arquivo(caminho: str) -> str:
    """SHA-256 do arquivo inteiro (hex)."""
    h = hashlib.sha256()
    with open(caminho, 'rb') as f:
        for bloco in iter(lambda: f.read(1 << 22), b''):
            h.update(bloco)
    return h.hexdigest()


def hash_git() -> str:
    try:
        return subprocess.check_output(['git', '-C', RAIZ, 'rev-parse', '--short', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'sem-git'


def git_sujo() -> bool:
    try:
        saida = subprocess.check_output(['git', '-C', RAIZ, 'status', '--porcelain'],
                                        text=True, stderr=subprocess.DEVNULL)
        return bool(saida.strip())
    except Exception:
        return False


def versoes() -> Dict[str, str]:
    out = {'python': platform.python_version()}
    for mod in ('torch', 'numpy', 'gymnasium', 'optuna', 'scipy'):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = 'ausente'
    return out


def dispositivo() -> str:
    try:
        import torch
        return 'cuda:' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def fixar_sementes(seed: int) -> None:
    """Fixa todas as fontes de aleatoriedade do treino (random, numpy, torch, cuda; cudnn determinístico)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@contextlib.contextmanager
def rng_isolado():
    """Salva e restaura os estados de `np.random`, `random` e (se houver) `torch`/`cuda`.

    Usar em toda avaliação: o `reset(seed)` do ambiente semeia os RNG globais; sem isto, uma avaliação em
    sementes fixas no meio do treino tornaria o fluxo de treino idêntico para todos os trials (D-18)."""
    estado_np = np.random.get_state()
    estado_py = random.getstate()
    estado_torch = estado_cuda = None
    torch = None
    try:
        import torch  # noqa: F811
        estado_torch = torch.get_rng_state()
        if torch.cuda.is_available():
            estado_cuda = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    try:
        yield
    finally:
        np.random.set_state(estado_np)
        random.setstate(estado_py)
        if torch is not None and estado_torch is not None:
            torch.set_rng_state(estado_torch)
            if estado_cuda is not None:
                torch.cuda.set_rng_state_all(estado_cuda)


def snapshot_protocolo() -> dict:
    return {
        'versao_ambiente': VERSAO_AMBIENTE, 'passos_maximos': PASSOS_MAXIMOS, 'tamanho_grade': list(TAMANHO_GRADE),
        'obs_dim': OBS_DIM, 'n_acoes': N_ACOES,
        'oraculo': {'professor_versao': PROFESSOR_VERSAO, 'professor_imperfeicao': PROFESSOR_IMPERFEICAO,
                    'variante_imperfeicao': ORACULO_IMPERFEICAO_VARIANTE, 'lazy_steps': ORACULO_LAZY_STEPS, 'seed': ORACULO_SEED},
        'sementes': {k: (v if len(v) <= 5 else [v[0], '...', v[-1], f'n={len(v)}']) for k, v in PAPEIS_SEMENTES.items()},
        'regimes': REGIMES,
        'demos': {'caminho': DEMOS_PADRAO, 'hash': HASH_DEMOS_PADRAO},
        'rodada': {'nome': RODADA, 'dir': DIR_RODADA, 'orcamento_rl_passos': ORCAMENTO_RL_PASSOS, 'studies_hpo': list(STUDIES_HPO)},
    }


def registrar_config(dir_out: str, **campos) -> str:
    """Grava `dir_out/config.json` com git hash, versões, dispositivo, data, snapshot do protocolo e `campos`."""
    os.makedirs(dir_out, exist_ok=True)
    cfg = {
        'data': time.strftime('%Y-%m-%d %H:%M:%S'),
        'maquina': platform.node(),
        'hash_git': hash_git(),
        'git_sujo': git_sujo(),
        'versoes': versoes(),
        'dispositivo': dispositivo(),
        'protocolo': snapshot_protocolo(),
    }
    cfg.update(campos)
    caminho = os.path.join(dir_out, 'config.json')
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1, default=str)
    return caminho


def escrever_flag(dir_out: str, nome: str, mensagem: str = '') -> str:
    """Grava `_DONE.flag` / `_FAILED.flag` (o lançador e o validador leem isto)."""
    os.makedirs(dir_out, exist_ok=True)
    caminho = os.path.join(dir_out, nome)
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {mensagem}\n")
    return caminho


def limpar_flags(dir_out: str) -> None:
    """Remove _DONE.flag/_FAILED.flag antigos ao iniciar um run (flags são artefatos do run corrente, não do diretório)."""
    for nome in ('_DONE.flag', '_FAILED.flag'):
        caminho = os.path.join(dir_out, nome)
        if os.path.exists(caminho):
            os.remove(caminho)


def escrever_status(dir_out: str, etapa: str, detalhe: str = '') -> None:
    """Acrescenta uma linha com timestamp em `dir_out/_STATUS.txt`."""
    os.makedirs(dir_out, exist_ok=True)
    with open(os.path.join(dir_out, '_STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {etapa} | {detalhe}\n")
