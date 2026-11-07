"""Testes do pós-HPO: a espera por trials em voo (07/09/2026).

Com o study em vários trabalhadores, a `_DONE.flag` é gravada pelo primeiro processo que vê o orçamento estourado,
enquanto os outros ainda têm um trial RUNNING cada. O pós-HPO não pode ler o banco nesse instante: o top-k sairia sem
os últimos trials. `trials_em_voo` é a guarda; estes testes travam o que ela precisa fazer sem importar o optuna nem
escrever no banco.
"""
import os
import sqlite3
import sys
from datetime import datetime, timedelta

import protocolo as P

sys.path.insert(0, os.path.join(P.RAIZ, 'hpo'))
import pos_hpo  # noqa: E402


def _banco(tmp_path, linhas):
    """Cria um SQLite com o schema mínimo da tabela `trials` do Optuna."""
    p = tmp_path / 'x.db'
    con = sqlite3.connect(str(p))
    con.execute('create table trials (trial_id integer primary key, state text, datetime_start text, datetime_complete text)')
    for i, (estado, idade) in enumerate(linhas, start=1):
        ini = (datetime.now() - idade).isoformat(sep=' ')
        con.execute('insert into trials values (?, ?, ?, ?)', (i, estado, ini, None if estado == 'RUNNING' else ini))
    con.commit()
    con.close()
    return p


def test_conta_so_running_recente(tmp_path, monkeypatch):
    _banco(tmp_path, [('RUNNING', timedelta(minutes=10)), ('COMPLETE', timedelta(hours=1)), ('PRUNED', timedelta(hours=2))])
    monkeypatch.setattr(pos_hpo, 'DIR_HPO', str(tmp_path))
    assert pos_hpo.trials_em_voo('x') == 1


def test_zumbi_nao_segura_a_espera(tmp_path, monkeypatch):
    """Um RUNNING mais velho que a tolerância é processo morto que o Optuna não fechou: não se espera por ele."""
    _banco(tmp_path, [('RUNNING', timedelta(minutes=10)), ('RUNNING', timedelta(hours=5))])
    monkeypatch.setattr(pos_hpo, 'DIR_HPO', str(tmp_path))
    assert pos_hpo.trials_em_voo('x') == 1
    assert pos_hpo.trials_em_voo('x', tolerancia_h=6.0) == 2


def test_banco_ausente_nao_bloqueia(tmp_path, monkeypatch):
    monkeypatch.setattr(pos_hpo, 'DIR_HPO', str(tmp_path))
    assert pos_hpo.trials_em_voo('nao_existe') == 0


def test_abre_somente_leitura(tmp_path, monkeypatch):
    """A guarda não pode escrever no banco que os trabalhadores estão usando."""
    p = _banco(tmp_path, [('RUNNING', timedelta(minutes=1))])
    monkeypatch.setattr(pos_hpo, 'DIR_HPO', str(tmp_path))
    antes = p.stat().st_mtime_ns
    pos_hpo.trials_em_voo('x')
    assert p.stat().st_mtime_ns == antes


def test_tolerancia_ao_zumbi_acompanha_o_orcamento_e_nunca_cai_abaixo_de_tres_horas():
    assert pos_hpo.tolerancia_zumbi_h(250_000) == 3.0 and pos_hpo.tolerancia_zumbi_h(100_000) == 3.0
    assert pos_hpo.tolerancia_zumbi_h(500_000) == 6.0 and pos_hpo.tolerancia_zumbi_h(1_000_000) == 12.0


class _Esperou(Exception):
    pass


def _rodar_pos_hpo(tmp_path, monkeypatch, orcamento):
    """O main() de verdade com um study 'x' pronto (_DONE.flag) e UM trial RUNNING há 5 h. Devolve 'esperou' se ele ficou
    esperando o trial, ou a lista de etapas que lançou (subprocess de mentira: nada treina)."""
    _banco(tmp_path, [('RUNNING', timedelta(hours=5)), ('COMPLETE', timedelta(hours=6))])
    (tmp_path / 'x').mkdir()
    (tmp_path / 'x' / '_DONE.flag').write_text('ok', encoding='utf-8')
    monkeypatch.setattr(pos_hpo, 'DIR_HPO', str(tmp_path))
    monkeypatch.setattr(P, 'DIR_POS_HPO', str(tmp_path / 'pos_hpo'))
    lancadas = []
    monkeypatch.setattr(pos_hpo.subprocess, 'run', lambda cmd, **kw: lancadas.append(os.path.basename(cmd[3])) or type('R', (), {'returncode': 0})())
    monkeypatch.setattr(pos_hpo.time, 'sleep', lambda s: (_ for _ in ()).throw(_Esperou()))
    monkeypatch.setattr(sys, 'argv', ['pos_hpo.py', '--studies', 'x', '--orcamento', str(orcamento), '--sem-ablacoes'])
    try:
        pos_hpo.main()
    except _Esperou:
        return 'esperou'
    return lancadas


def test_com_um_milhao_de_passos_um_trial_de_cinco_horas_esta_vivo_e_a_reavaliacao_espera(tmp_path, monkeypatch):
    """Com a tolerância fixa de 3 h, o trial de 5 h (saudável com 1 milhão de passos) seria dado por morto e o top-k sairia sem ele."""
    assert _rodar_pos_hpo(tmp_path, monkeypatch, 1_000_000) == 'esperou'


def test_com_duzentos_e_cinquenta_mil_passos_o_mesmo_trial_e_zumbi_e_nao_segura_a_reavaliacao(tmp_path, monkeypatch):
    assert _rodar_pos_hpo(tmp_path, monkeypatch, 250_000) == ['reavaliar_topk.py', 'analise_hpo.py']


# ── escolha do MimePlant: definição única para a análise final e para as ablações (10/09/2026) ──

from reavaliar_topk import escolher_melhor_candidato  # noqa: E402


def _cand(media, n=5):
    return {'n_sementes': n, 'validacao_objetivo': {'normalizada_media_3_regimes': {'media': media}},
            'teste': {'normalizada_media_3_regimes': {'media': 0.0}}}


def test_escolhe_pela_media_de_validacao_e_ignora_os_padroes():
    """O padrão pode ter a maior validação e mesmo assim não é candidato: é termo de comparação."""
    res = {'tpe_v1_trial34': _cand(1.655), 'tpe_v1_trial70': _cand(1.661),
           'rs_v1_trial28': _cand(1.568), 'padrao_ppo_zero': _cand(9.9)}
    assert escolher_melhor_candidato(res) == 'tpe_v1_trial70'


def test_nunca_usa_o_teste_para_escolher():
    """Um candidato pior em validação e melhor em teste não pode ser escolhido."""
    res = {'a_trial1': _cand(1.60), 'b_trial2': _cand(1.50)}
    res['b_trial2']['teste']['normalizada_media_3_regimes']['media'] = 99.0
    assert escolher_melhor_candidato(res) == 'a_trial1'


def test_recusa_reavaliacao_incompleta():
    import pytest
    res = {'tpe_v1_trial70': _cand(1.661), 'rs_v1_trial53': _cand(1.70, n=1)}
    with pytest.raises(ValueError):
        escolher_melhor_candidato(res, n_sementes=5)


def test_analise_e_ablacoes_usam_a_mesma_funcao():
    """Trava a propriedade que motivou a mudança: se alguém voltar a escrever a escolha à mão em um dos dois
    arquivos, a configuração ablada e a da comparação final podem divergir de novo."""
    import inspect
    import analise_hpo
    import ablacoes
    assert 'escolher_melhor_candidato' in inspect.getsource(analise_hpo)
    assert 'escolher_melhor_candidato' in inspect.getsource(ablacoes.escolher_base)
    assert 'max(conc' not in inspect.getsource(ablacoes), 'ablacoes voltou a escolher pelo valor de ajuste'
