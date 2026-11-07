"""Testes do agente de imitação (rede LSTM): máscara respeitada, persistência e avaliação pelo adaptador."""
import numpy as np
import torch

import protocolo as P
import avaliacao as A


def _agente_cpu():
    from bc import AgenteBC
    torch.manual_seed(0)
    return AgenteBC(dim_lstm=64, dim_encoder=32, dispositivo='cpu')


def test_logits_mascarados_zeram_acoes_invalidas():
    ag = _agente_cpu()
    obs = torch.zeros(1, 1, P.OBS_DIM)
    mask = torch.ones(1, 1, P.N_ACOES); mask[0, 0, [0, 1, 7]] = 0.0
    logits, _ = ag.rede(obs, mask)
    probs = torch.softmax(logits, -1)[0, 0]
    assert probs[[0, 1, 7]].max().item() == 0.0
    assert abs(probs.sum().item() - 1.0) < 1e-6


def test_selecionar_acao_respeita_mascara_com_exploracao():
    ag = _agente_cpu()
    obs = np.zeros(P.OBS_DIM, dtype=np.float32)
    mask = np.ones(P.N_ACOES, dtype=np.float32); mask[[0, 1, 2, 7, 8]] = 0.0
    hidden = ag.inicializar_hidden()
    np.random.seed(0); torch.manual_seed(0)
    for _ in range(200):
        a, hidden = ag.selecionar_acao(obs, mask, hidden, temperatura=1.0, epsilon=0.5)
        assert mask[a] == 1.0


def test_salvar_e_carregar_preserva_saida(tmp_path):
    ag = _agente_cpu()
    ag.obs_media = np.random.RandomState(1).rand(P.OBS_DIM).astype(np.float32)
    caminho = str(tmp_path / 'm.pt')
    ag.salvar(caminho)
    from bc import AgenteBC
    ag2 = AgenteBC(dim_lstm=64, dim_encoder=32, dispositivo='cpu')
    ag2.carregar(caminho)
    obs = np.random.RandomState(2).rand(P.OBS_DIM).astype(np.float32)
    mask = np.ones(P.N_ACOES, dtype=np.float32)
    a1, _ = ag.selecionar_acao(obs, mask, ag.inicializar_hidden())
    a2, _ = ag2.selecionar_acao(obs, mask, ag2.inicializar_hidden())
    assert a1 == a2 and np.allclose(ag.obs_media, ag2.obs_media)


def test_politica_rede_na_avaliacao_oficial(env):
    ag = _agente_cpu()
    res = A.avaliar(A.PoliticaRede(ag), env, [500, 501])
    assert len(res['recompensas']) == 2 and all(d == P.PASSOS_MAXIMOS for d in res['duracoes'])
    res2 = A.avaliar(A.PoliticaRede(ag), env, [500, 501])
    assert res['recompensas'] == res2['recompensas']
