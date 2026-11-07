"""Configuração comum dos testes: sys.path do projeto e fixtures do ambiente."""
import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

import protocolo as P  # noqa: E402

P.preparar_sys_path()


@pytest.fixture
def env():
    """Ambiente oficial: AmbienteFazendaGym(365) + InvolucroMascaraAcoes."""
    from env import AmbienteFazendaGym
    from mascara_acoes import InvolucroMascaraAcoes
    return InvolucroMascaraAcoes(AmbienteFazendaGym(passos_maximos=P.PASSOS_MAXIMOS))


@pytest.fixture
def bot_subotimo():
    """A variante degradada por pausas (D-27: existe, mas nenhum experimento a usa)."""
    return P.novo_oraculo_variante()


@pytest.fixture
def bot_perfeito():
    """O PROFESSOR dos experimentos (D-27): oráculo sem pausas."""
    return P.novo_professor()


def rodar_episodio(env, seed, escolher_acao, n_passos=None):
    """Roda um episódio com `escolher_acao(obs, mascara, env_bruto, t)`; devolve (obs_seq, recompensas, infos)."""
    obs, info = env.reset(seed=seed)
    bruto = P.env_bruto(env)
    obs_seq, recs, infos = [obs.copy()], [], []
    fim, t = False, 0
    while not fim and (n_passos is None or t < n_passos):
        acao = escolher_acao(obs, info['mascara_acoes'], bruto, t)
        obs, r, terminado, truncado, info = env.step(acao)
        obs_seq.append(obs.copy()); recs.append(float(r)); infos.append(info)
        fim = terminado or truncado
        t += 1
    return np.array(obs_seq), np.array(recs), infos


def acao_aleatoria_valida(rng):
    def f(obs, mascara, bruto, t):
        validas = np.where(mascara > 0.5)[0]
        return int(rng.choice(validas))
    return f
