"""Testes do driver de HPO: o espaço do MÉTODO (D-30) e a cobertura das chaves.

A D-30 tirou `inicializacao` do espaço de busca: o método sempre parte da imitação do professor e solta a mão pelo
decaimento da âncora KL; treinar do zero é o braço de CONTROLE do bake-off. Estes testes travam as duas coisas que
isso implica e que a auditoria de 07/09 mostrou serem frágeis: (a) nenhum trial pode sair com `inicializacao` que
não seja 'professor', e (b) o espaço não pode voltar a ser condicional, porque `optuna.importance` usa a INTERSEÇÃO
dos espaços dos trials e descartava em silêncio os quatro condicionais — entre eles `temperatura_inicial`, que é o
hiperparâmetro que decide o resultado."""
import os
import sys

import optuna

import protocolo as P
from hp import ESPACO_MIMEPLANT, CONDICIONAIS_PROFESSOR

sys.path.insert(0, os.path.join(P.RAIZ, 'hpo'))
from hpo_rl import sugerir  # noqa: E402


def _fixo():
    params = {nome: (spec[1][0] if spec[0] == 'cat' else spec[1]) for nome, spec in ESPACO_MIMEPLANT.items()}
    return optuna.trial.FixedTrial(params)


def test_o_metodo_parte_sempre_do_professor():
    """D-30: `inicializacao` não é sorteada e todo trial parte da imitação."""
    hp = sugerir(_fixo())
    assert hp['inicializacao'] == 'professor'
    assert 'inicializacao' not in ESPACO_MIMEPLANT, 'inicializacao voltou ao espaço de busca (ver D-30)'
    assert all(k in hp for k in CONDICIONAIS_PROFESSOR)
    assert set(ESPACO_MIMEPLANT) | {'inicializacao'} == set(hp)


def test_o_espaco_nao_e_condicional_e_a_importancia_ve_tudo():
    """Trava o defeito de 07/09: com espaço condicional, a interseção usada por `optuna.importance` perdia
    `temperatura_inicial`, `beta_kl0`, `beta_kl_decay` e `k_aquec`. Falha se alguém reintroduzir a condicionalidade."""
    from optuna.search_space import intersection_search_space
    est = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    est.optimize(lambda t: (sugerir(t), 0.0)[1], n_trials=5)
    inter = set(intersection_search_space(est.get_trials()))
    assert set(ESPACO_MIMEPLANT) == inter, f'espaço condicional de novo: faltam {set(ESPACO_MIMEPLANT) - inter}'
    assert all(k in inter for k in CONDICIONAIS_PROFESSOR)
    assert {h.params['inicializacao'] for h in est.trials if 'inicializacao' in h.params} == set()


def test_smoke_reduz_o_orcamento_por_trial():
    hp = sugerir(_fixo(), smoke=True)
    assert hp['V'] == 2 and hp['blocos'] == 2 and hp['k_aquec'] <= 1


def _partida_ate_as_referencias(tmp_path, monkeypatch, argumentos):
    """Roda o main() de verdade só até a leitura das referências (que não existem): o que interessa é o que ele fez com as
    flags do estudo ANTES disso. Nada treina."""
    import pytest
    import hpo_rl
    monkeypatch.setattr(hpo_rl, 'DIR_HPO', str(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['hpo_rl.py', '--study', 'x', '--referencias', str(tmp_path / 'nao_existe.json'), *argumentos])
    with pytest.raises(FileNotFoundError):
        hpo_rl.main()


def test_retomar_apaga_a_failed_flag_de_uma_sessao_anterior_e_preserva_a_done(tmp_path, monkeypatch):
    """Com --retomar a _FAILED.flag velha sumia nunca: o validador reportava FALHA para sempre num estudo que se recuperou."""
    (tmp_path / 'x').mkdir()
    (tmp_path / 'x' / '_FAILED.flag').write_text('falha antiga', encoding='utf-8')
    (tmp_path / 'x' / '_DONE.flag').write_text('pronto', encoding='utf-8')
    _partida_ate_as_referencias(tmp_path, monkeypatch, ['--retomar'])
    assert not (tmp_path / 'x' / '_FAILED.flag').exists() and (tmp_path / 'x' / '_DONE.flag').exists()


def test_sem_retomar_as_duas_flags_saem_como_antes(tmp_path, monkeypatch):
    (tmp_path / 'x').mkdir()
    (tmp_path / 'x' / '_FAILED.flag').write_text('falha antiga', encoding='utf-8')
    (tmp_path / 'x' / '_DONE.flag').write_text('pronto', encoding='utf-8')
    _partida_ate_as_referencias(tmp_path, monkeypatch, [])
    assert not (tmp_path / 'x' / '_FAILED.flag').exists() and not (tmp_path / 'x' / '_DONE.flag').exists()
