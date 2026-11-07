"""Testes da fase de RL (CPU, rápidos): coleta em lote, transformador de recompensa e shaping por potencial."""
import numpy as np

import protocolo as P
import avaliacao as A
from envs_lote import EnvLote, fabrica_env_padrao
from wrapper import TransformadorRecompensa, potencial


def politica_obs(obs, mascara, t):
    """Política determinística só de observação: colher > plantar (até t<250) > regar a cada 4 > esperar."""
    if mascara[P.ACAO_COLHER] > 0.5:
        return P.ACAO_COLHER
    if mascara[P.ACAO_PLANTAR] > 0.5 and t < 250:
        return P.ACAO_PLANTAR
    if mascara[P.ACAO_REGAR] > 0.5 and t % 4 == 0:
        return P.ACAO_REGAR
    return P.ACAO_ESPERAR


def test_env_lote_reproduz_avaliacao_um_a_um():
    lote = EnvLote(2, fabrica_env_padrao, seed_base=500)
    t = np.zeros(2, dtype=int)
    obs, masc = lote.obs.copy(), lote.mascaras.copy()
    while len(lote.episodios_concluidos) < 4:
        acoes = [politica_obs(obs[v], masc[v], t[v]) for v in range(2)]
        obs, masc, rec, fins, _ = lote.step(acoes)
        t = np.where(fins, 0, t + 1)
    lote_por_semente = {e['semente']: e['retorno'] for e in lote.coletar_concluidos()}
    assert set(lote_por_semente) == {500, 501, 502, 503}
    env = fabrica_env_padrao()
    ref = A.avaliar(A.PoliticaFixa(lambda bruto, o, m, tt: politica_obs(o, m, tt)), env, [500, 501, 502, 503])
    for s, r in zip(ref['sementes'], ref['recompensas']):
        assert abs(lote_por_semente[s] - r) < 1e-9, (s, lote_por_semente[s], r)
    assert all(d == P.PASSOS_MAXIMOS for d in ref['duracoes'])


def test_env_lote_registra_decomposicao_e_acoes():
    lote = EnvLote(1, fabrica_env_padrao, seed_base=520)
    for _ in range(P.PASSOS_MAXIMOS):
        lote.step([P.ACAO_ESPERAR])
    ep = lote.coletar_concluidos()
    assert len(ep) == 1 and ep[0]['passos'] == P.PASSOS_MAXIMOS and ep[0]['acoes'][P.ACAO_ESPERAR] == P.PASSOS_MAXIMOS
    assert abs(sum(ep[0]['comp'].values()) - ep[0]['retorno']) < 1e-6
    assert lote.sementes_atuais[0] == 521        # já reiniciou com a semente seguinte


def test_transformador_escala_e_pbrs_telescopa():
    tr = TransformadorRecompensa(escala=100.0, gamma=1.0, pbrs=True)
    lote = EnvLote(1, fabrica_env_padrao, seed_base=500)
    obs = lote.obs.copy()
    soma_f, soma_bruta, t = 0.0, 0.0, 0
    fim = np.array([False])
    while not fim[0]:
        a = [politica_obs(obs[0], lote.mascaras[0], t)]
        obs_dep, _, rec, fim, _ = lote.step(a)
        r_treino = tr(rec, obs, obs_dep, fim)
        soma_f += float(r_treino[0]) - float(rec[0]) / 100.0
        soma_bruta += float(rec[0])
        obs = obs_dep
        t += 1
    assert abs(soma_f - (-potencial(np.zeros(P.OBS_DIM)))) < 1e-4     # γ = 1: Σ F = −Φ(s0) = 0
    sem = TransformadorRecompensa(escala=50.0)
    assert np.allclose(sem(np.array([10.0, -5.0]), None, None, np.array([False, False])), [0.2, -0.1])


def test_potencial_conta_estagio_e_saude():
    obs = np.zeros(P.OBS_DIM, dtype=np.float32)
    i = P.INICIO_GRADE
    obs[i:i + 8] = [1.0, 0.5, 5 / 7, 0, 0, 0, 0, 0]   # uma planta em COLHEITA com saúde 0,5 (índices do protocolo)
    assert abs(potencial(obs) - 0.30 * 0.5) < 1e-6
    assert potencial(np.zeros(P.OBS_DIM)) == 0.0
