"""Testes do ator-crítico da fase de RL: carga do IL, valor zerado no início, KL nula entre cópias, AgenteRL avaliável."""
import os

import numpy as np
import torch

import protocolo as P
import avaliacao as A
from politica import AtorCritico, AgenteRL, referencia_congelada, kl_referencia

CKPT_SMOKE = os.path.join(P.RAIZ, 'results', '_smoke_warmstart', 'seed_42', 'modelo_final.pt')


def _ckpt_pequeno(tmp_path):
    """Checkpoint de IL pequeno (rede 64) no formato oficial, para os testes não dependerem de arquivos externos."""
    from bc import AgenteBC
    torch.manual_seed(0)
    ag = AgenteBC(dim_lstm=64, dim_encoder=32, dispositivo='cpu')
    ag.obs_media = np.full(P.OBS_DIM, 0.3, np.float32); ag.obs_std = np.full(P.OBS_DIM, 0.2, np.float32)
    caminho = str(tmp_path / 'il.pt')
    ag.salvar(caminho)
    return caminho, ag


def test_carrega_il_e_valor_comeca_em_zero(tmp_path):
    caminho, ag = _ckpt_pequeno(tmp_path)
    # a rede salva por AgenteBC(dim_lstm=64, dim_encoder=32) tem config padrão no ckpt; construir com as mesmas dimensões
    m = AtorCritico(dim_enc=32, dim_lstm=64); m.eval(); ag.rede.eval()
    media, std = m.carregar_il(caminho, 'cpu')
    assert np.allclose(media, 0.3) and np.allclose(std, 0.2)
    obs = torch.rand(2, 5, P.OBS_DIM); mask = torch.ones(2, 5, P.N_ACOES)
    v, _ = m.valor(obs)
    assert v.shape == (2, 5) and torch.allclose(v, torch.zeros_like(v))
    lg_a, _ = m.logits(obs, mask); lg_il, _ = ag.rede(obs, mask)
    assert torch.allclose(lg_a, lg_il)


def test_kl_referencia_zero_entre_copias_e_positiva_apos_mudanca(tmp_path):
    caminho, _ = _ckpt_pequeno(tmp_path)
    m = AtorCritico(dim_enc=32, dim_lstm=64); m.carregar_il(caminho, 'cpu'); m.eval()
    ref = referencia_congelada(m.ator)
    obs = torch.rand(3, 7, P.OBS_DIM); mask = torch.ones(3, 7, P.N_ACOES); mask[..., [0, 7]] = 0
    lg, _ = m.logits(obs, mask); lg_ref, _ = ref(obs, mask)
    kl = kl_referencia(lg_ref, lg)
    assert kl.shape == (3, 7) and torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)
    with torch.no_grad():
        for p in m.ator.head.parameters():
            p.add_(0.5 * torch.randn_like(p))
    lg2, _ = m.logits(obs, mask)
    assert kl_referencia(lg_ref, lg2).mean().item() > 0
    assert not any(p.requires_grad for p in ref.parameters())


def test_agente_rl_avalia_e_salva_no_formato_do_il(tmp_path):
    caminho, ag = _ckpt_pequeno(tmp_path)
    m = AtorCritico(dim_enc=32, dim_lstm=64); media, std = m.carregar_il(caminho, 'cpu')
    arl = AgenteRL(m.ator, media, std, 'cpu')
    env = A.__dict__  # noqa: F841 (só para garantir que o módulo carregou)
    from envs_lote import fabrica_env_padrao
    e = fabrica_env_padrao()
    r1 = A.avaliar(A.PoliticaRede(arl), e, [500])
    r2 = A.avaliar(A.PoliticaRede(ag), e, [500])
    assert r1['recompensas'] == r2['recompensas']            # mesma política ⇒ mesmo episódio
    saida = str(tmp_path / 'rl.pt'); arl.salvar(saida)
    arl2 = AgenteRL.carregar_de(saida, 'cpu')
    assert A.avaliar(A.PoliticaRede(arl2), e, [500])['recompensas'] == r1['recompensas']


def test_temperatura_inicial_suaviza_sem_mudar_o_argmax(tmp_path):
    caminho, _ = _ckpt_pequeno(tmp_path)
    m1 = AtorCritico(dim_enc=32, dim_lstm=64); m1.carregar_il(caminho, 'cpu'); m1.eval()
    m2 = AtorCritico(dim_enc=32, dim_lstm=64); m2.carregar_il(caminho, 'cpu', temperatura=4.0); m2.eval()
    obs = torch.rand(3, 6, P.OBS_DIM); mask = torch.ones(3, 6, P.N_ACOES); mask[..., [1, 7]] = 0
    lg1, _ = m1.logits(obs, mask); lg2, _ = m2.logits(obs, mask)
    assert torch.equal(lg1.argmax(-1), lg2.argmax(-1))
    e1 = torch.distributions.Categorical(logits=lg1).entropy().mean(); e2 = torch.distributions.Categorical(logits=lg2).entropy().mean()
    assert e2 > e1
    validos = mask[..., :] > 0.5
    assert torch.allclose(lg2[validos] * 4.0, lg1[validos], atol=1e-4)
