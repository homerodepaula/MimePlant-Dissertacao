"""Testes do protocolo: papéis de semente disjuntos, isolamento de RNG, registro de config, regimes."""
import hashlib
import json
import os
import random

import numpy as np

import protocolo as P


def test_papeis_de_semente_sao_disjuntos():
    papeis = P.PAPEIS_SEMENTES
    nomes = list(papeis)
    for i, a in enumerate(nomes):
        for b in nomes[i + 1:]:
            assert not (set(papeis[a]) & set(papeis[b])), f'{a} e {b} compartilham sementes'
    assert len(papeis['validacao_rollback']) == 20 and papeis['validacao_rollback'][0] == 500
    assert len(papeis['validacao_objetivo']) == 30 and papeis['validacao_objetivo'][-1] == 549
    assert len(papeis['teste']) == 50 and papeis['teste'][0] == 999
    assert len(papeis['demos']) == 200 and papeis['demos'][0] == 10000


def test_seed_pausas_oraculo():
    assert P.seed_pausas_oraculo(0) == P.ORACULO_SEED
    assert P.seed_pausas_oraculo(7) == P.ORACULO_SEED + 7


def test_rng_isolado_restaura_numpy_e_random():
    np.random.seed(1); random.seed(1)
    np.random.rand(); random.random()
    esperado_np, esperado_py = np.random.rand(), random.random()
    np.random.seed(1); random.seed(1)
    np.random.rand(); random.random()
    with P.rng_isolado():
        np.random.seed(999); random.seed(999)
        np.random.rand(10); [random.random() for _ in range(10)]
    assert np.random.rand() == esperado_np
    assert random.random() == esperado_py


def test_rng_isolado_restaura_mesmo_com_excecao():
    np.random.seed(3)
    esperado = np.random.RandomState(3).rand()
    try:
        with P.rng_isolado():
            np.random.seed(12345)
            raise RuntimeError('proposital')
    except RuntimeError:
        pass
    assert np.random.rand() == esperado


def test_fixar_sementes_reproduz():
    P.fixar_sementes(42); a = (np.random.rand(), random.random())
    P.fixar_sementes(42); b = (np.random.rand(), random.random())
    assert a == b


def test_hash_arquivo(tmp_path):
    arq = tmp_path / 'x.bin'
    dados = b'mimeplant' * 1000
    arq.write_bytes(dados)
    assert P.hash_arquivo(str(arq)) == hashlib.sha256(dados).hexdigest()


def test_registrar_config_e_flags(tmp_path):
    d = str(tmp_path / 'saida')
    caminho = P.registrar_config(d, experimento='teste', sementes_treino=[42])
    cfg = json.load(open(caminho, encoding='utf-8'))
    for chave in ('data', 'hash_git', 'versoes', 'dispositivo', 'protocolo', 'experimento', 'sementes_treino'):
        assert chave in cfg
    assert cfg['protocolo']['oraculo'] == {'professor_versao': 'v5', 'professor_imperfeicao': 0.0, 'variante_imperfeicao': 0.14,
                                           'lazy_steps': 8, 'seed': 12345}
    P.escrever_flag(d, '_DONE.flag', 'ok'); P.escrever_status(d, 'etapa', 'detalhe')
    assert os.path.exists(os.path.join(d, '_DONE.flag')) and os.path.exists(os.path.join(d, '_STATUS.txt'))


def test_aplicar_e_restaurar_regime(env):
    env.reset(seed=1)
    clima = P.env_bruto(env).clima
    P.aplicar_regime(env, 'dry')
    assert clima.modo_ood == 'dry' and clima.temperatura_base == 32.0 and clima.probabilidade_chuva_base == 0.05
    P.aplicar_regime(env, 'humid')
    assert clima.modo_ood == 'humid' and clima.umidade_ar_base == 0.85
    P.restaurar_regime(env)
    assert clima.modo_ood is None and clima.temperatura_base == 20.0 and clima.probabilidade_chuva_base == 0.3
    P.aplicar_regime(env, None)
    assert clima.modo_ood is None
