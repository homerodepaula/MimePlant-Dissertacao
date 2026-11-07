"""Smoke da auto-imitação de elite (braço B), CPU, rede pequena. Marcado `lento` (~1 min)."""
import json

import numpy as np
import pytest
import torch

import protocolo as P
import avaliacao as A
from envs_lote import fabrica_env_padrao


def _ckpt_e_demos(tmp_path):
    from bc import AgenteBC
    torch.manual_seed(0)
    ag = AgenteBC(dim_lstm=32, dim_encoder=16, dispositivo='cpu')
    ag.obs_media = np.full(P.OBS_DIM, 0.3, np.float32); ag.obs_std = np.full(P.OBS_DIM, 0.2, np.float32)
    ckpt = str(tmp_path / 'il.pt'); ag.salvar(ckpt)
    # demos mínimas: 2 episódios de "só esperar" no formato oficial
    env = fabrica_env_padrao()
    eps = []
    for s in (10000, 10001):
        obs, info = env.reset(seed=s); traj = []
        for _ in range(P.PASSOS_MAXIMOS):
            m = info['mascara_acoes']; o = obs.copy()
            obs, r, _, _, info = env.step(P.ACAO_ESPERAR)
            traj.append({'obs': o.tolist(), 'acao': P.ACAO_ESPERAR, 'mascara': m.tolist()})
        eps.append(traj)
    demos = str(tmp_path / 'demos.json')
    with open(demos, 'w', encoding='utf-8') as f:
        json.dump(eps, f)
    return ckpt, demos


@pytest.mark.lento
def test_elite_roda_desconecta_e_chama_callback(tmp_path):
    from elite import treinar_rl
    ckpt, demos = _ckpt_e_demos(tmp_path)
    hp = {'n_eps_coleta': 2, 'tam_elite': 3, 'updates_por_iter': 2, 'blocos': 2, 'dim_lstm': 32, 'dim_enc': 16}
    chamadas = []
    res = treinar_rl(ckpt, hp, seed=42, orcamento_passos=3 * 2 * P.PASSOS_MAXIMOS,
                     callback=lambda passos, ag: chamadas.append(passos) or False, dispositivo='cpu', log=lambda *a: None, demos=demos)
    assert res['passos'] >= 3 * 2 * P.PASSOS_MAXIMOS and chamadas
    assert res['nivel_il_rollouts'] is not None and len(res['curva']) >= 2
    assert all('retorno_elite' in c and c['kl'] is not None for c in res['curva'])
    env = fabrica_env_padrao()
    r = A.avaliar(A.PoliticaRede(res['agente']), env, [500])
    assert r['duracoes'][0] == P.PASSOS_MAXIMOS


def test_elite_exige_warm_start():
    from elite import treinar_rl
    with pytest.raises(ValueError):
        treinar_rl(None, None, seed=1, orcamento_passos=10)
