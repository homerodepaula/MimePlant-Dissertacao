"""Testes do ambiente (FarmGym): forma da observação, truncamento, máscara e determinismo por semente."""
import numpy as np

import protocolo as P
from conftest import rodar_episodio, acao_aleatoria_valida


def test_observacao_forma_e_intervalo(env):
    obs, info = env.reset(seed=11)
    assert obs.shape == (P.OBS_DIM,) and obs.dtype == np.float32
    assert 'mascara_acoes' in info and info['mascara_acoes'].shape == (P.N_ACOES,)
    obs_seq, recs, infos = rodar_episodio(env, 11, acao_aleatoria_valida(np.random.RandomState(0)))
    assert obs_seq.shape == (P.PASSOS_MAXIMOS + 1, P.OBS_DIM)
    assert np.all(np.isfinite(obs_seq))
    assert obs_seq.min() >= 0.0 and obs_seq.max() <= 1.0 + 1e-6


def test_episodio_trunca_em_365_passos(env):
    obs, info = env.reset(seed=5)
    for t in range(P.PASSOS_MAXIMOS):
        obs, r, terminado, truncado, info = env.step(P.ACAO_ESPERAR)
        assert not terminado
        assert truncado == (t == P.PASSOS_MAXIMOS - 1)


def test_mascara_basica(env):
    obs, info = env.reset(seed=2)
    m = info['mascara_acoes']
    assert m[P.ACAO_PLANTAR] == 1.0      # grade vazia: plantar é válido
    assert m[P.ACAO_COLHER] == 0.0       # nada para colher
    assert m[13] == 1.0 and m[P.ACAO_ESPERAR] == 1.0
    for _ in range(16):                   # preenche a grade
        obs, r, _, _, info = env.step(P.ACAO_PLANTAR)
    assert info['mascara_acoes'][P.ACAO_PLANTAR] == 0.0


def test_mesma_semente_mesma_trajetoria_em_instancias_diferentes():
    """Duas instâncias novas, mesma semente, mesmas ações ⇒ mesmas observações e recompensas.
    Cobre também os subsistemas que usam o módulo `random` (migração, doenças, eventos extremos)."""
    from env import AmbienteFazendaGym
    from mascara_acoes import InvolucroMascaraAcoes
    import random
    saidas = []
    for k in range(2):
        random.seed(1000 + k)            # estado global diferente antes de cada instância
        np.random.seed(2000 + k)
        env = InvolucroMascaraAcoes(AmbienteFazendaGym(passos_maximos=P.PASSOS_MAXIMOS))
        obs_seq, recs, _ = rodar_episodio(env, 77, acao_aleatoria_valida(np.random.RandomState(9)), n_passos=120)
        saidas.append((obs_seq, recs))
    np.testing.assert_array_equal(saidas[0][0], saidas[1][0])
    np.testing.assert_array_equal(saidas[0][1], saidas[1][1])


def test_recompensa_decomposta_soma_o_total(env):
    """S5 (FarmGym v2): info['recompensa_comp'] existe em todo passo e soma a recompensa do passo."""
    obs, info = env.reset(seed=21)
    rng = np.random.RandomState(4)
    for t in range(P.PASSOS_MAXIMOS):
        validas = np.where(info['mascara_acoes'] > 0.5)[0]
        obs, r, _, truncado, info = env.step(int(rng.choice(validas)))
        comp = info['recompensa_comp']
        assert abs(sum(comp.values()) - r) < 1e-9, (t, r, comp)
        if not truncado:
            assert 'solo_final' not in comp
    assert 'solo_final' in info['recompensa_comp']            # terminal só no último passo
    assert 'colheita_final' not in info['recompensa_comp']    # S1: colheita não é paga de novo no terminal


def test_sem_recompensa_por_existir(env):
    """S2 (FarmGym v2): quem só espera não ganha bônus por passo; o terminal fica pequeno (solo, polinizadores)."""
    obs_seq, recs, infos = rodar_episodio(env, 31, lambda o, m, b, t: P.ACAO_ESPERAR)
    assert np.all(recs[:-1] == 0.0)
    assert 0.0 <= recs[-1] <= 20.0


def test_penalidade_de_quimicos_linear_sem_teto(env):
    """S4 (FarmGym v2): o terminal desconta 2,0 por aplicação por célula, sem teto de −10."""
    obs, info = env.reset(seed=8)
    n_aplicacoes = 12
    for t in range(P.PASSOS_MAXIMOS):
        acao = P.ACAO_PESTICIDA if t < n_aplicacoes else P.ACAO_ESPERAR
        obs, r, _, truncado, info = env.step(acao)
    comp = info['recompensa_comp']
    assert abs(comp['quimicos_final'] - (-2.0 * n_aplicacoes)) < 1e-9
