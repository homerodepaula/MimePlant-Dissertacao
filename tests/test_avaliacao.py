"""Testes da função única de avaliação: determinismo, isolamento de RNG, regimes e políticas de referência."""
import random

import numpy as np

import protocolo as P
import avaliacao as A


def _agente_cpu():
    import torch
    from bc import AgenteBC
    torch.manual_seed(0)
    return AgenteBC(dim_lstm=64, dim_encoder=32, dispositivo='cpu')


def test_politica_rede_amostra_com_gerador_proprio_sem_tocar_os_rngs_globais():
    """D-26: a amostragem usa `default_rng(SEMENTE_DECODIFICACAO + semente)`; np.random, random e torch ficam intactos."""
    import random, torch
    ag = _agente_cpu()
    pol = A.PoliticaRede(ag)                       # tau = 1 por padrao
    assert pol.tau == 1.0
    pol.iniciar_episodio(500, None)
    rs = np.random.RandomState(3)
    obs = [rs.rand(P.OBS_DIM).astype(np.float32) for _ in range(30)]
    mask = np.ones(P.N_ACOES, dtype=np.float32)
    est_np, est_py, est_torch = np.random.get_state(), random.getstate(), torch.get_rng_state().clone()
    acoes = [pol.agir(o, mask) for o in obs]
    assert np.array_equal(est_np[1], np.random.get_state()[1]) and est_py == random.getstate() and torch.equal(est_torch, torch.get_rng_state())
    assert len(set(acoes)) > 1                      # rede nao treinada e quase uniforme: a amostragem varia
    pol.iniciar_episodio(500, None)
    assert [pol.agir(o, mask) for o in obs] == acoes  # deterministica dada a semente do episodio
    pol.iniciar_episodio(501, None)
    assert [pol.agir(o, mask) for o in obs] != acoes  # outra semente, outro fluxo


def test_politica_rede_tau_zero_e_o_argmax(env):
    """`tau = 0` reproduz a regra antiga (argmax) e nao depende da semente de amostragem."""
    ag = _agente_cpu()
    r0 = A.avaliar(A.PoliticaRede(ag, tau=0.0), env, [500, 501])
    r1 = A.avaliar(A.PoliticaRede(ag, tau=0.0), env, [500, 501])
    assert r0['recompensas'] == r1['recompensas'] and r0['acoes_por_episodio'] == r1['acoes_por_episodio']
    ra = A.avaliar(A.PoliticaRede(ag, tau=1.0), env, [500, 501])
    assert ra['acoes_por_episodio'] != r0['acoes_por_episodio']   # amostrada e argmax sao politicas diferentes


def test_amostragem_nao_perturba_o_fluxo_aleatorio_do_ambiente(env):
    """Se a amostragem consumisse np.random/random (o fluxo do ambiente), repetir as MESMAS acoes sem amostrar daria
    outro episodio (outro clima, outras recompensas)."""
    ag = _agente_cpu()
    bruto = P.env_bruto(env)
    pol = A.PoliticaRede(ag, tau=1.0)
    with P.rng_isolado():
        obs, info = env.reset(seed=502); pol.iniciar_episodio(502, bruto)
        acoes, recs = [], []
        for t in range(80):
            a = pol.agir(obs, info['mascara_acoes']); acoes.append(a)
            obs, r, te, tr, info = env.step(a); recs.append(float(r))
    with P.rng_isolado():
        env.reset(seed=502)
        recs2 = [float(env.step(a)[1]) for a in acoes]
    assert recs == recs2


def test_avaliacao_do_oraculo_e_deterministica(env, bot_subotimo):
    pol = A.PoliticaOraculo(bot_subotimo)
    sementes = [500, 501, 502]
    r1 = A.avaliar(pol, env, sementes)
    for _ in range(50):          # avança os RNG globais entre as duas chamadas
        random.random(); np.random.rand()
    r2 = A.avaliar(pol, env, sementes)
    assert r1['recompensas'] == r2['recompensas']
    assert r1['acoes_por_episodio'] == r2['acoes_por_episodio']
    assert all(d == P.PASSOS_MAXIMOS for d in r1['duracoes'])


def test_avaliacao_nao_perturba_o_rng_global(env, bot_subotimo):
    np.random.seed(11); random.seed(11)
    esperado = (np.random.RandomState(11).rand(), random.Random(11).random())
    A.avaliar(A.PoliticaOraculo(bot_subotimo), env, [500])
    assert (np.random.rand(), random.random()) == esperado


def test_politica_aleatoria_e_reproduzivel_e_pareada(env):
    r1 = A.avaliar(A.PoliticaAleatoria(0), env, [999, 1000])
    r2 = A.avaliar(A.PoliticaAleatoria(0), env, [999, 1000])
    assert r1['recompensas'] == r2['recompensas']
    assert sum(r1['contagem_acoes'].values()) == 2 * P.PASSOS_MAXIMOS


def test_regimes_mudam_o_resultado_e_restauram_o_clima(env, bot_perfeito):
    pol = A.PoliticaOraculo(bot_perfeito)
    res = A.avaliar_regimes(pol, env, [999, 1000, 1001])
    assert set(res) == {'in_dist', 'dry', 'humid'}
    assert np.mean(res['dry']['recompensas']) < np.mean(res['in_dist']['recompensas'])
    assert P.env_bruto(env).clima.modo_ood is None


def test_resumo_e_iqm():
    assert A.iqm([1, 2, 3, 4, 5, 6, 7, 8]) == np.mean([3, 4, 5, 6])
    r = A.resumo({'recompensas': [1.0, 2.0, 3.0]})
    assert r['n'] == 3 and r['media'] == 2.0 and abs(r['erro_padrao'] - 1 / np.sqrt(3)) < 1e-12
    assert A.normalizar(5.0, 0.0, 10.0) == 0.5


def test_politica_fixa_recebe_o_tempo(env):
    vistos = []
    def f(bruto, obs, mascara, t):
        vistos.append(t); return P.ACAO_ESPERAR
    A.avaliar(A.PoliticaFixa(f), env, [500])
    assert vistos == list(range(P.PASSOS_MAXIMOS))
