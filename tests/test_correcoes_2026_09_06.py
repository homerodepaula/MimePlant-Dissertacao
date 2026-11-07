"""Testes das correções de 06/09/2026 (criticas.md): máscara de remover_espantalho/cerca, janela do IL cobrindo o fim
do episódio, âncora KL antes da temperatura, ablação "sem KL" de fato sem KL, referências por conjunto de sementes,
`std` consistente e ociosidade medida como esperar + remover_espantalho."""
import numpy as np
import sys as _sys


def test_normalizacao_da_observacao_tem_piso_e_recorte():
    """D-28: dimensão com desvio zero nas demonstrações não pode virar fator 1e8 fora da distribuição."""
    P.preparar_sys_path()
    from rede_lstm import estatisticas_obs, normalizar_obs, PISO_STD_OBS, CLIP_OBS_NORM
    import torch
    dados = np.random.RandomState(0).rand(500, P.OBS_DIM).astype(np.float32)
    dados[:, 5] = 0.0                                    # dimensão que só dispara fora da distribuição
    media, std = estatisticas_obs(dados)
    assert std.min() >= PISO_STD_OBS and std[5] == PISO_STD_OBS
    fora = dados[0].copy(); fora[5] = 0.62               # valor medido no regime seco
    z = normalizar_obs(fora, media, std)
    assert abs(z[5]) <= CLIP_OBS_NORM and np.abs(z).max() <= CLIP_OBS_NORM
    # o mesmo vale para um checkpoint antigo, gravado com desvio zero
    std_antigo = dados.std(axis=0); std_antigo[5] = 0.0
    assert np.abs(normalizar_obs(fora, media, std_antigo)).max() <= CLIP_OBS_NORM
    # e a versão torch dá o mesmo resultado
    zt = normalizar_obs(torch.as_tensor(fora), torch.as_tensor(media), torch.as_tensor(std))
    assert np.allclose(zt.numpy(), z, atol=1e-6)


def test_observar_conta_como_ociosidade():
    """`observar` é um no-op com custo; a métrica de ociosidade tem de incluí-lo (auditoria de 06/09)."""
    assert 'observar' in A.ACOES_OCIOSAS and 'esperar' in A.ACOES_OCIOSAS
import pytest
import torch

import protocolo as P
import avaliacao as A

REMOVER, CERCA, ESP_BASICO = 11, 12, 9


# ── máscara (criticas.md §2.3) ───────────────────────────────────────────────────────────────────
def test_remover_espantalho_so_e_valida_com_espantalho(env):
    obs, info = env.reset(seed=4)
    assert info['mascara_acoes'][REMOVER] == 0.0          # campo vazio: nada a remover
    obs, r, _, _, info = env.step(P.ACAO_PLANTAR)
    assert info['mascara_acoes'][REMOVER] == 0.0          # com planta mas sem espantalho
    assert info['mascara_acoes'][ESP_BASICO] == 1.0
    obs, r, _, _, info = env.step(ESP_BASICO)
    assert info['mascara_acoes'][REMOVER] == 1.0          # agora há o que remover
    obs, r, _, _, info = env.step(REMOVER)
    assert info['mascara_acoes'][REMOVER] == 0.0          # removido: volta a ser inválida


def test_cerca_e_invalida_quando_todas_as_celulas_ja_tem_cerca(env):
    obs, info = env.reset(seed=4)
    assert info['mascara_acoes'][CERCA] == 1.0
    obs, r, _, _, info = env.step(CERCA)
    assert info['mascara_acoes'][CERCA] == 0.0
    # a máscara é exposta também pelo caminho usado em avaliacao.avaliar (env._gerar_mascara)
    assert env._gerar_mascara(obs)[CERCA] == 0.0


def test_politica_aleatoria_nao_usa_remover_espantalho_sem_espantalho(env):
    res = A.avaliar(A.PoliticaAleatoria(0), env, [500, 501])
    # a aleatória só escolhe entre válidas: remover_espantalho exige espantalho, que ela só coloca às vezes
    assert res['contagem_acoes']['remover_espantalho'] <= res['contagem_acoes']['colocar_espantalho_basico'] + res['contagem_acoes']['colocar_espantalho_avancado']


# ── janela do IL (criticas.md §4.1) ──────────────────────────────────────────────────────────────
def test_janelas_cobrem_o_fim_do_episodio():
    from rede_lstm import ConjuntoDadosBC
    n, L = P.PASSOS_MAXIMOS, 32
    ep = [{'obs': np.full(P.OBS_DIM, t / n, np.float32), 'acao': t % P.N_ACOES, 'mascara': np.ones(P.N_ACOES, np.float32)} for t in range(n)]
    ds = ConjuntoDadosBC([ep], np.zeros(P.OBS_DIM, np.float32), np.ones(P.OBS_DIM, np.float32), len_seq=L)
    assert len(ds) == n // L + 1                          # 11 janelas alinhadas ao início + 1 alinhada ao fim
    ultimas = ds.sequencias[-1][2]
    assert ultimas[-1] == (n - 1) % P.N_ACOES             # a última transição do episódio está no conjunto
    assert len(ds.sequencias[-1][0]) == L
    cobertos = set()
    for ini in list(range(0, n - L + 1, L)) + [n - L]:
        cobertos.update(range(ini, ini + L))
    assert cobertos == set(range(n))


# ── âncora KL e temperatura (criticas.md §5.2) ───────────────────────────────────────────────────
def _ckpt(tmp_path, dim_lstm=32, dim_enc=16):
    from bc import AgenteBC
    torch.manual_seed(0)
    ag = AgenteBC(dim_lstm=dim_lstm, dim_encoder=dim_enc, dispositivo='cpu')
    ag.obs_media = np.full(P.OBS_DIM, 0.3, np.float32); ag.obs_std = np.full(P.OBS_DIM, 0.2, np.float32)
    caminho = str(tmp_path / 'il.pt'); ag.salvar(caminho)
    return caminho


def test_referencia_kl_e_o_clone_sem_temperatura(tmp_path):
    from politica import AtorCritico, referencia_congelada, kl_referencia
    caminho = _ckpt(tmp_path)
    m = AtorCritico(dim_enc=16, dim_lstm=32); m.carregar_il(caminho, 'cpu', temperatura=1.0); m.eval()
    obs = torch.rand(2, 5, P.OBS_DIM); mask = torch.ones(2, 5, P.N_ACOES)
    lg_clone, _ = m.logits(obs, mask)
    ref = referencia_congelada(m.ator)                  # a ordem usada em ppo.treinar_rl: congela antes da temperatura
    m.aplicar_temperatura(4.0)
    lg_ref, _ = ref(obs, mask); lg_ator, _ = m.logits(obs, mask)
    assert torch.allclose(lg_ref, lg_clone)            # a referência é o clone tal como treinado
    assert torch.allclose(lg_ator * 4.0, lg_clone, atol=1e-4)
    assert kl_referencia(lg_ref, lg_ator).mean().item() > 0   # e por isso o KL ao professor é positivo com τ > 1


@pytest.mark.lento
def test_beta_kl0_zero_desliga_o_kl_em_todas_as_atualizacoes(tmp_path):
    from ppo import treinar_rl
    caminho = _ckpt(tmp_path)
    hp = {'V': 2, 'len_seq': 73, 'k_aquec': 0, 'n_epocas': 1, 'minilote_seqs': 4, 'dim_lstm': 32, 'dim_enc': 16, 'blocos': 1,
          'beta_kl0': 0.0, 'beta_kl_min': 1e-3}
    res = treinar_rl(caminho, hp, seed=1, orcamento_passos=3 * 2 * P.PASSOS_MAXIMOS, dispositivo='cpu', log=lambda *a: None)
    ppo = [c for c in res['curva'] if c['fase'] == 'ppo']
    assert len(ppo) >= 3
    assert all(c['kl'] is None and c['beta_kl'] == 0.0 for c in ppo)   # antes: 1e-3 a partir da 2ª atualização


# ── referências por conjunto (criticas.md §7.3) ──────────────────────────────────────────────────
def test_referencias_tem_um_conjunto_por_papel_de_semente():
    import referencias as R
    assert set(R.CONJUNTOS) == {'validacao_rollback', 'validacao_objetivo', 'teste'}
    assert R.CONJUNTOS['validacao_rollback']() == P.sementes_validacao_rollback()
    assert R.CONJUNTOS['validacao_objetivo']() == P.sementes_validacao_objetivo()
    assert R.CONJUNTOS['teste']() == P.sementes_teste()


def test_normalizado_recusa_referencias_antigas():
    from bakeoff import normalizado
    antigo = {'aleatorio': {'validacao': {'in_dist': {'media': 0.0}}}, 'oraculo_perfeito': {'validacao': {'in_dist': {'media': 1.0}}}}
    with pytest.raises(KeyError):
        normalizado(0.5, antigo, 'validacao_objetivo', 'in_dist')
    novo = {'aleatorio': {'validacao_objetivo': {'in_dist': {'media': 0.0}}}, 'oraculo_perfeito': {'validacao_objetivo': {'in_dist': {'media': 2.0}}}}
    assert normalizado(1.0, novo, 'validacao_objetivo', 'in_dist') == 0.5


# ── estatística e ociosidade (criticas.md §2.3, §7.2) ────────────────────────────────────────────
def test_resumo_std_e_erro_padrao_consistentes():
    r = A.resumo({'recompensas': [1.0, 2.0, 4.0, 7.0]})
    assert abs(r['std'] - np.std([1.0, 2.0, 4.0, 7.0], ddof=1)) < 1e-12
    assert abs(r['erro_padrao'] - r['std'] / 2.0) < 1e-12


def test_ociosidade_soma_esperar_e_remover_espantalho():
    contagem = {n: 0 for n in P.NOMES_ACOES}
    contagem.update(esperar=30, remover_espantalho=20, plantar=50)
    res = {'contagem_acoes': contagem}
    assert abs(A.ociosidade_pct(res) - 50.0) < 1e-9
    pct = A.acoes_pct(res)
    assert abs(pct['plantar'] - 50.0) < 1e-9 and abs(sum(pct.values()) - 100.0) < 1e-9


def test_iqm_e_a_media_aparada_a_25_por_cento():
    from scipy import stats
    for n in (5, 8, 30, 50):
        v = np.random.RandomState(n).rand(n) * 100
        assert abs(A.iqm(v) - stats.trim_mean(v, 0.25)) < 1e-9
