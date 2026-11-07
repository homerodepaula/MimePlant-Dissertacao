"""Testes do braço de controle do zero (D-32, 11/09/2026).

O que precisa ser verdade para o experimento responder à pergunta que ele existe para responder:
  1. o estudo do controle sorteia só os hiperparâmetros que AGEM sem política de partida;
  2. todo trial dele sai com `inicializacao = 'zero'`, e o estudo do método continua saindo com 'professor';
  3. a reavaliação reconhece a inicialização do estudo — senão retreinaria o candidato do controle A PARTIR DA
     IMITAÇÃO, medindo o oposto do que o estudo mediu;
  4. as condições que têm de ser idênticas entre método e controle continuam idênticas.
"""
import os
import sys

import optuna
import pytest

import protocolo as P

sys.path.insert(0, os.path.join(P.RAIZ, 'hpo'))
sys.path.insert(0, os.path.join(P.RAIZ, 'agents', 'rl'))
import hpo_rl  # noqa: E402
import reavaliar_topk  # noqa: E402
from hp import ESPACO_MIMEPLANT, CONDICIONAIS_PROFESSOR, ESPACOS  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _sugerir(inicializacao, completo=False):
    """Roda `sugerir` num trial real do Optuna e devolve (hp, nomes sorteados pelo amostrador)."""
    est = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    caixa = {}

    def obj(t):
        caixa['hp'] = hpo_rl.sugerir(t, inicializacao=inicializacao, espaco_completo=completo)
        return 0.0

    est.optimize(obj, n_trials=1)
    return caixa['hp'], set(est.trials[0].params)


def test_controle_sorteia_so_os_hiperparametros_que_agem():
    """Sem política de partida, `ppo.treinar_rl` anula temperatura, KL e aquecimento: sorteá-los gastaria
    orçamento em dimensões inertes."""
    hp, sorteados = _sugerir('zero')
    assert sorteados == set(ESPACOS['ppo_zero'])
    assert len(sorteados) == 8
    assert not (sorteados & set(CONDICIONAIS_PROFESSOR))


def test_metodo_continua_com_os_doze():
    hp, sorteados = _sugerir('professor')
    assert sorteados == set(ESPACO_MIMEPLANT)
    assert len(sorteados) == 12
    assert hp['inicializacao'] == 'professor'


def test_espaco_completo_iguala_a_dimensao():
    """A alternativa declarada: 12 dimensões também no controle, para quem preferir igualar a dimensão da busca."""
    _, sorteados = _sugerir('zero', completo=True)
    assert sorteados == set(ESPACO_MIMEPLANT)


def test_todo_trial_do_controle_sai_com_zero():
    hp, _ = _sugerir('zero')
    assert hp['inicializacao'] == 'zero'


def test_o_padrao_continua_sendo_o_metodo():
    """Quem chamar `sugerir` sem dizer nada tem de receber o método, não o controle."""
    est = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    caixa = {}
    est.optimize(lambda t: (caixa.__setitem__('hp', hpo_rl.sugerir(t)), 0.0)[1], n_trials=1)
    assert caixa['hp']['inicializacao'] == 'professor'


def test_reavaliacao_reconhece_a_inicializacao_do_estudo(tmp_path, monkeypatch):
    """O defeito que este teste impede: sem ler o `user_attr`, o candidato do controle chegaria ao
    `rodar_candidato` sem a chave, cairia no padrão 'professor' e seria retreinado a partir da imitação."""
    monkeypatch.setattr(reavaliar_topk, 'DIR_HPO', str(tmp_path))
    for nome, ini in (('zz_zero', 'zero'), ('zz_prof', 'professor')):
        st = optuna.create_study(study_name=nome, direction='maximize',
                                 storage=f"sqlite:///{str(tmp_path / (nome + '.db')).replace(os.sep, '/')}")
        st.set_user_attr('inicializacao', ini)
        st.optimize(lambda t: t.suggest_float('lr', 1e-5, 3e-4, log=True) * 0 + 1.0, n_trials=2)
    assert all(hp['inicializacao'] == 'zero' for _, hp, _ in reavaliar_topk.candidatos_do_study('zz_zero', 2))
    assert all(hp['inicializacao'] == 'professor' for _, hp, _ in reavaliar_topk.candidatos_do_study('zz_prof', 2))


def test_estudo_sem_o_atributo_cai_no_professor(tmp_path, monkeypatch):
    """Estudos anteriores à D-30 não gravavam a inicialização; para eles o padrão 'professor' é o correto."""
    monkeypatch.setattr(reavaliar_topk, 'DIR_HPO', str(tmp_path))
    st = optuna.create_study(study_name='zz_velho', direction='maximize',
                             storage=f"sqlite:///{str(tmp_path / 'zz_velho.db').replace(os.sep, '/')}")
    st.optimize(lambda t: t.suggest_float('lr', 1e-5, 3e-4, log=True) * 0 + 1.0, n_trials=1)
    assert reavaliar_topk.candidatos_do_study('zz_velho', 1)[0][1]['inicializacao'] == 'professor'


def test_condicoes_identicas_entre_metodo_e_controle():
    """As condições que o experimento promete manter idênticas não podem depender da inicialização: elas são
    argumentos com o MESMO valor padrão para os dois estudos."""
    import argparse
    ap = argparse.ArgumentParser()
    fonte = __import__('inspect').getsource(hpo_rl.main)
    # o orçamento por execução vem de UMA constante do protocolo, nos dois estudos e na cadeia do controle (D-33)
    import controle_zero
    assert "'--orcamento', type=int, default=P.ORCAMENTO_RL_PASSOS" in fonte
    assert "'--orcamento', type=int, default=P.ORCAMENTO_RL_PASSOS" in __import__('inspect').getsource(controle_zero.main)
    for arg, valor in (("'--n-fre'", '64.0'),
                       ("'--seeds-trial'", '[42]'), ("'--n-startup'", '20')):
        assert arg in fonte and valor in fonte, f'{arg} deixou de ter o padrão {valor}'
    # o podador e o objetivo não são parametrizados por inicialização
    assert 'min_resource=max(1, N // 3)' in fonte
    assert "int(passos) < N and trial.should_prune()" in fonte


def test_controle_zero_nao_e_candidato_do_mimeplant():
    """`escolher_melhor_candidato` escolhe o MimePlant entre os candidatos do estudo do MÉTODO. O controle mora em
    outro resumo (results/hpo_reavaliacao_zero), então nunca pode ser escolhido como MimePlant por engano."""
    import controle_zero
    fonte = __import__('inspect').getsource(controle_zero.comparar)
    assert 'P.DIR_REAVALIACAO' in fonte and 'saida_reav' in fonte
    assert 'escolher_melhor_candidato(res_met' in fonte and 'escolher_melhor_candidato(res_zero' in fonte
