"""hp.py — espaços de hiperparâmetros da fase de RL (DESENHO_FASE3_BAKEOFF_RL.md §3) e amostragem (random search / Optuna).

Cada entrada: nome -> ('log', lo, hi) | ('float', lo, hi) | ('int', lo, hi) | ('cat', [opções]).
"""
from typing import Dict

import numpy as np

ESPACO_PPO_IL: Dict = {
    'lr': ('log', 1e-5, 3e-4), 'clip': ('float', 0.1, 0.3), 'c_ent': ('log', 1e-4, 3e-2),
    'gamma': ('cat', [0.98, 0.99, 0.995]), 'lambda_gae': ('float', 0.9, 0.98), 'n_epocas': ('int', 2, 6),
    'minilote_seqs': ('cat', [10, 20, 30]), 'beta_kl0': ('log', 1e-3, 1.0), 'beta_kl_decay': ('float', 0.9, 0.999),
    'escala': ('cat', [50.0, 100.0, 200.0]), 'k_aquec': ('int', 0, 20), 'temperatura_inicial': ('log', 1.0, 10.0),
}
# D-30 (07/09/2026) — a inicialização SAIU da busca e o método é sempre "partir da imitação do professor e soltar a
# mão". Motivo (Homero, literal): "o Agent deve aprender com ele e com o tempo largar a mão do professor e seguir
# sozinho, e daí sim entra HPO nos hiperparâmetros". Treinar do zero é o CONTROLE, e o lugar dele é o braço `ppo_zero`
# do bake-off, não o espaço de busca do método: com 'inicializacao' aqui, metade do orçamento da HPO ia otimizar um
# agente que nunca vê o professor (medido na rodada abortada: 19 de 39 trials). Supera a D-21, que tratava a
# inicialização como hiperparâmetro. Os quatro condicionais deixam de ser condicionais — existem em todo trial.
ESPACO_MIMEPLANT: Dict = dict(ESPACO_PPO_IL)
INICIALIZACAO_DO_METODO = 'professor'
CONDICIONAIS_PROFESSOR = ('temperatura_inicial', 'beta_kl0', 'beta_kl_decay', 'k_aquec')
ESPACO_ELITE: Dict = {
    'lr': ('log', 1e-5, 3e-4), 'tam_elite': ('int', 20, 80), 'n_eps_coleta': ('cat', [4, 8, 12, 16]),
    'updates_por_iter': ('int', 4, 32), 'fracao_demo0': ('float', 0.1, 0.7), 'decay_demo': ('float', 0.9, 0.995),
    'lambda_kl0': ('log', 1e-3, 0.3), 'decay_kl': ('float', 0.9, 0.995), 'tau0': ('float', 0.3, 1.0),
    'epsilon0': ('float', 0.0, 0.15), 'p_antipausa0': ('float', 0.0, 0.5), 'pause_pen': ('cat', [0.0, 5.0]),
}
ESPACOS = {'mimeplant': ESPACO_MIMEPLANT, 'ppo_il': ESPACO_PPO_IL, 'ppo_zero': {k: v for k, v in ESPACO_PPO_IL.items() if k not in ('beta_kl0', 'beta_kl_decay', 'k_aquec', 'temperatura_inicial')},
           'elite': ESPACO_ELITE}


def amostrar(espaco: Dict, rng: np.random.Generator) -> Dict:
    """Uma configuração aleatória (random search)."""
    hp = {}
    for nome, spec in espaco.items():
        tipo = spec[0]
        if tipo == 'log':
            hp[nome] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        elif tipo == 'float':
            hp[nome] = float(rng.uniform(spec[1], spec[2]))
        elif tipo == 'int':
            hp[nome] = int(rng.integers(spec[1], spec[2] + 1))
        elif tipo == 'cat':
            hp[nome] = spec[1][int(rng.integers(0, len(spec[1])))]
    return hp


def sugerir(trial, espaco: Dict) -> Dict:
    """Uma configuração proposta pelo Optuna (TPE)."""
    hp = {}
    for nome, spec in espaco.items():
        tipo = spec[0]
        if tipo == 'log':
            hp[nome] = trial.suggest_float(nome, spec[1], spec[2], log=True)
        elif tipo == 'float':
            hp[nome] = trial.suggest_float(nome, spec[1], spec[2])
        elif tipo == 'int':
            hp[nome] = trial.suggest_int(nome, spec[1], spec[2])
        elif tipo == 'cat':
            hp[nome] = trial.suggest_categorical(nome, spec[1])
    return hp
