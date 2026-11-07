"""Smoke do PPO-IL e do PPO do zero (CPU, rede pequena, 2 ambientes, 1 aquecimento + 1 atualização). Marcado `lento` (~1 min)."""
import numpy as np
import pytest
import torch

import protocolo as P
import avaliacao as A
from envs_lote import fabrica_env_padrao


def _ckpt(tmp_path):
    from bc import AgenteBC
    torch.manual_seed(0)
    ag = AgenteBC(dim_lstm=32, dim_encoder=16, dispositivo='cpu')
    ag.obs_media = np.full(P.OBS_DIM, 0.3, np.float32); ag.obs_std = np.full(P.OBS_DIM, 0.2, np.float32)
    caminho = str(tmp_path / 'il.pt'); ag.salvar(caminho)
    return caminho


HP_MINI = {'V': 2, 'len_seq': 73, 'k_aquec': 1, 'n_epocas': 1, 'minilote_seqs': 4, 'dim_lstm': 32, 'dim_enc': 16, 'blocos': 2}


@pytest.mark.lento
def test_ppo_il_roda_e_chama_callback(tmp_path):
    from ppo import treinar_rl
    caminho = _ckpt(tmp_path)
    chamadas = []
    res = treinar_rl(caminho, HP_MINI, seed=42, orcamento_passos=2 * 2 * P.PASSOS_MAXIMOS,
                     callback=lambda passos, ag: chamadas.append(passos) or False, dispositivo='cpu', log=lambda *a: None)
    assert res['passos'] >= 2 * 2 * P.PASSOS_MAXIMOS and not res['podado']
    fases = [c['fase'] for c in res['curva']]
    assert fases[0] == 'aquecimento' and 'ppo' in fases
    assert chamadas and chamadas[-1] <= res['passos']
    ppo_est = [c for c in res['curva'] if c['fase'] == 'ppo'][0]
    assert ppo_est['kl'] is not None and ppo_est['kl'] >= 0 and 0 <= ppo_est['clipfrac'] <= 1
    assert all(e['passos'] == P.PASSOS_MAXIMOS for e in res['episodios_treino'])
    env = fabrica_env_padrao()
    r = A.avaliar(A.PoliticaRede(res['agente']), env, [500])
    assert len(r['recompensas']) == 1 and r['duracoes'][0] == P.PASSOS_MAXIMOS


@pytest.mark.lento
def test_ppo_do_zero_roda_sem_checkpoint_e_poda():
    from ppo import treinar_rl
    res = treinar_rl(None, HP_MINI, seed=7, orcamento_passos=4 * 2 * P.PASSOS_MAXIMOS,
                     callback=lambda passos, ag: True, dispositivo='cpu', log=lambda *a: None)
    assert res['podado'] and res['passos'] < 4 * 2 * P.PASSOS_MAXIMOS
    assert all(c['fase'] == 'ppo' for c in res['curva']) and res['curva'][0]['kl'] is None
    assert res['hp']['k_aquec'] == 0 or res['curva'][0]['fase'] == 'ppo'


@pytest.mark.lento
def test_inicializacao_zero_ignora_o_checkpoint(tmp_path):
    from ppo import treinar_rl
    caminho = _ckpt(tmp_path)
    res = treinar_rl(caminho, {**HP_MINI, 'inicializacao': 'zero', 'blocos': 1}, seed=3, orcamento_passos=2 * P.PASSOS_MAXIMOS,
                     dispositivo='cpu', log=lambda *a: None)
    assert all(c['fase'] == 'ppo' for c in res['curva']) and res['curva'][0]['kl'] is None   # sem aquecimento nem KL: partiu do zero


def test_espaco_mimeplant_e_so_do_agente():
    """D-30: o espaço de busca é o do AGENTE. `inicializacao` saiu — treinar do zero é o braço de controle do
    bake-off, não uma configuração do método — e os quatro hiperparâmetros da soltura da mão existem sempre."""
    from hp import ESPACOS, CONDICIONAIS_PROFESSOR, INICIALIZACAO_DO_METODO, amostrar
    esp = ESPACOS['mimeplant']
    assert 'inicializacao' not in esp, 'inicializacao voltou ao espaço de busca (ver D-30)'
    assert all(k in esp for k in CONDICIONAIS_PROFESSOR)
    assert INICIALIZACAO_DO_METODO == 'professor'
    hp = amostrar(esp, np.random.default_rng(0))
    assert 1.0 <= hp['temperatura_inicial'] <= 10.0 and 0.9 <= hp['beta_kl_decay'] <= 0.999
