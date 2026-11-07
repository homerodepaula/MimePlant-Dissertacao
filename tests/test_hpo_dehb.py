"""Testes do coordenador do DEHB (`hpo/hpo_dehb.py`) com um avaliador FALSO (`_avaliador_falso_dehb.py`): nenhum usa GPU.
Todos os caminhos têm espaço e acento, como a pasta real do projeto."""
import os
import sys
import json
import time
import threading
import subprocess

import numpy as np
import pytest

import protocolo as P

sys.path.insert(0, os.path.join(P.RAIZ, 'hpo'))
import hpo_dehb as H        # noqa: E402
import dehb_nucleo as NUC   # noqa: E402

FALSO = os.path.join(P.RAIZ, 'tests', '_avaliador_falso_dehb.py')
N = 900_000                 # fidelidades 300.000 e 900.000: cobre o aquecimento do crítico (91.980 passos)


def comando_falso(dir_trabalho):
    return [sys.executable, '-X', 'utf8', FALSO, dir_trabalho]


@pytest.fixture
def area(tmp_path, monkeypatch):
    """DIR_HPO temporário + precondições mínimas (checkpoint de partida e referências), em caminho com acento e espaço."""
    raiz = tmp_path / 'Área de Trabalho' / 'estudo dehb'
    hpo = raiz / 'hpo'
    hpo.mkdir(parents=True)
    (raiz / 'il' / 'seed_42').mkdir(parents=True)
    (raiz / 'il' / 'seed_42' / 'modelo_final.pt').write_bytes(b'x')
    refs = {k: {c: {r: {'media': m} for r in P.NOMES_REGIMES} for c in ('validacao_rollback', 'validacao_objetivo', 'teste')}
            for k, m in (('oraculo_perfeito', 5000.0), ('aleatorio', 600.0))}
    (raiz / 'referencias.json').write_text(json.dumps(refs), encoding='utf-8')
    monkeypatch.setattr(H, 'DIR_HPO', str(hpo))
    monkeypatch.delenv('DEHB_FALSO', raising=False)
    return {'hpo': str(hpo), 'il': str(raiz / 'il'), 'refs': str(raiz / 'referencias.json'), 'raiz': raiz}


def argumentos(area, study='s', n_fre=6.0, k=3, extra=()):
    return ['--study', study, '--n-fre', str(n_fre), '--orcamento', str(N), '--paralelo', str(k), '--warmstart', area['il'],
            '--referencias', area['refs'], '--intervalo-s', '0.05', '--retomar', *extra]


def rodar(area, monkeypatch, controle=None, **kw):
    monkeypatch.setenv('DEHB_FALSO', json.dumps(controle or {}))
    return H.main(argumentos(area, **kw), comando_avaliador=comando_falso)


def jornal(area, study='s'):
    return [json.loads(ln) for ln in open(os.path.join(area['hpo'], study, 'jornal.jsonl'), encoding='utf-8')]


def trials(area, study='s'):
    import optuna
    st = optuna.load_study(study_name=study, storage=f"sqlite:///{os.path.join(area['hpo'], study + '.db').replace(os.sep, '/')}")
    ts = st.get_trials(deepcopy=True)
    H.liberar_banco(st)                     # no Windows um .db com conexão aberta não pode ser renomeado nem apagado
    return ts


# ── o estudo inteiro e os consumidores ───────────────────────────────────────────────────────────
def test_estudo_completo_orcamento_espelho_e_consumidores_sem_mudanca(area, monkeypatch):
    import optuna
    assert rodar(area, monkeypatch) == 0
    d = os.path.join(area['hpo'], 's')
    assert os.path.exists(os.path.join(d, '_DONE.flag')) and not os.path.exists(os.path.join(d, '_FAILED.flag'))
    ev = jornal(area)
    assert [e['i'] for e in ev] == list(range(len(ev))) and ev[0]['ev'] == 'cabecalho'
    pedidos, informados = [e for e in ev if e['ev'] == 'pedir'], [e for e in ev if e['ev'] == 'informar']
    assert len(pedidos) == len(informados) and sorted(e['id'] for e in informados) == [e['trabalho']['id'] for e in pedidos]
    fre = sum(e['fre'] for e in informados)
    prog = json.load(open(os.path.join(d, 'PROGRESSO.json'), encoding='utf-8'))
    assert 6.0 <= fre <= 6.0 + 2 * 1.0 and prog['fre_consumidos'] == pytest.approx(fre, abs=1e-3)   # excesso de no máximo K-1 em voo
    assert prog['em_voo'] == [] and prog['promocoes'] >= 1 and sum(prog['pais_por_origem'].values()) > 0
    # espelho: COMPLETE só na fidelidade máxima, com value = objetivo; PRUNED nas menores; número do trial = ordem do jornal
    ts = trials(area)
    assert len(ts) == len(informados)
    for t, e in zip(ts, informados):
        assert t.user_attrs['evento_i'] == e['i'] and t.user_attrs['id_trabalho'] == e['id'] and t.user_attrs['fre'] == pytest.approx(e['fre'])
        assert list(t.params) == list(H.ESPACO_MIMEPLANT) and t.params == e['hp']
        if e['fidelidade_maxima']:
            assert t.state == optuna.trial.TrialState.COMPLETE and t.value == pytest.approx(e['objetivo'])
        else:
            assert t.state == optuna.trial.TrialState.PRUNED and t.value is None and len(t.intermediate_values) == 1
    assert [t.datetime_complete for t in ts] == sorted(t.datetime_complete for t in ts)
    assert prog['melhor']['valor'] == pytest.approx(max(t.value for t in ts if t.value is not None))
    indice = json.load(open(os.path.join(d, 'INDICE.json'), encoding='utf-8'))
    assert all(v['pasta'] == f"trabalho_{v['id_trabalho']}" and os.path.isdir(os.path.join(d, v['pasta'])) for v in indice.values())
    # consumidores, SEM mudança: reavaliar_topk, analise_hpo e pos_hpo leem o estudo DEHB pelo espelho
    import reavaliar_topk, analise_hpo, pos_hpo
    for mod in (reavaliar_topk, analise_hpo, pos_hpo):
        monkeypatch.setattr(mod, 'DIR_HPO', area['hpo'])
    top = reavaliar_topk.candidatos_do_study('s', 3)
    cheios = sorted((e for e in informados if e['fidelidade_maxima']), key=lambda e: -e['objetivo'])
    assert [v for _, _, v in top] == pytest.approx([e['objetivo'] for e in cheios[:3]])
    assert all(hp['inicializacao'] == 'professor' and len(hp) == 13 for _, hp, _ in top)
    xs, ys = analise_hpo.curva_melhor_ate_agora(analise_hpo.carregar('s'))
    assert xs[-1] == pytest.approx(fre) and ys[-1] == pytest.approx(cheios[0]['objetivo'])
    assert pos_hpo.trials_em_voo('s') == 0
    assert analise_hpo.importancia(analise_hpo.carregar('s'))['n_concluidos'] == len(cheios)


def test_retomada_continua_a_mesma_busca_de_uma_execucao_sem_interrupcao(area, monkeypatch):
    """K = 1 torna a ordem determinística: parar em 2 FRE e retomar até 6 tem de dar a MESMA sequência de um estudo direto a 6."""
    assert rodar(area, monkeypatch, study='direto', k=1) == 0
    assert rodar(area, monkeypatch, study='partes', k=1, n_fre=2.0) == 0
    tamanho = len(jornal(area, 'partes'))
    assert rodar(area, monkeypatch, study='partes', k=1, n_fre=2.0) == 0             # relançar um estudo concluído não toca o jornal
    assert len(jornal(area, 'partes')) == tamanho
    assert rodar(area, monkeypatch, study='partes', k=1, n_fre=6.0) == 0             # orçamento maior: continua
    def seq(study):
        ev = jornal(area, study)
        return ([(e['trabalho']['id'], e['trabalho']['vetor'], e['trabalho']['origem']) for e in ev if e['ev'] == 'pedir'],
                [(e['id'], e['aptidao']) for e in ev if e['ev'] == 'informar'])
    # o nome do study entra na semente: para comparar, os dois precisam da MESMA semente
    assert seq('direto') != seq('partes')
    assert rodar(area, monkeypatch, study='direto2', k=1, extra=('--seed', '5')) == 0
    assert rodar(area, monkeypatch, study='partes2', k=1, n_fre=2.0, extra=('--seed', '5')) == 0
    assert rodar(area, monkeypatch, study='partes2', k=1, n_fre=6.0, extra=('--seed', '5')) == 0
    assert seq('direto2') == seq('partes2')
    assert not any(e['ev'] == 'divergencia' for e in jornal(area, 'partes2'))
    assert [t.value for t in trials(area, 'direto2')] == [t.value for t in trials(area, 'partes2')]


def test_queda_com_trabalhos_em_voo_e_retomada_sem_duplicar_nem_perder(area, monkeypatch):
    """Mata o coordenador (processo) com avaliações em voo. Os filhos morrem junto (Job Object); a retomada reexecuta os
    pendentes, o jornal segue contíguo, cada trabalho é informado uma vez só e o espelho fecha com o jornal."""
    contador = str(area['raiz'] / 'lancamentos.txt')
    env = dict(os.environ, DEHB_FALSO=json.dumps({'dorme_s': 1.5, 'contador': contador}), PYTHONUTF8='1')
    codigo = ("import sys, os; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s'); import hpo_dehb as H; H.DIR_HPO = r'%s'; "
              "sys.exit(H.main(%r, comando_avaliador=lambda d: [sys.executable, '-X', 'utf8', r'%s', d]))"
              % (os.path.join(P.RAIZ, 'hpo'), P.RAIZ, area['hpo'], argumentos(area, n_fre=4.0), FALSO))
    proc = subprocess.Popen([sys.executable, '-X', 'utf8', '-c', codigo], env=env, cwd=P.RAIZ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    arq = os.path.join(area['hpo'], 's', 'jornal.jsonl')
    for _ in range(400):
        if os.path.exists(arq) and sum(1 for ln in open(arq, encoding='utf-8') if '"pedir"' in ln) >= 3:
            break
        time.sleep(0.05)
    proc.kill(); proc.wait(timeout=30)
    em_voo = {e['trabalho']['id'] for e in jornal(area) if e['ev'] == 'pedir'} - {e['id'] for e in jornal(area) if e['ev'] == 'informar'}
    assert em_voo                                                                    # a queda pegou trabalhos em voo
    assert rodar(area, monkeypatch, controle={'contador': contador}, n_fre=4.0) == 0
    ev = jornal(area)
    assert [e['i'] for e in ev] == list(range(len(ev))) and sum(1 for e in ev if e['ev'] == 'sessao') == 2
    informados = [e['id'] for e in ev if e['ev'] == 'informar']
    assert len(informados) == len(set(informados)) == sum(1 for e in ev if e['ev'] == 'pedir')
    assert len(trials(area)) == len(informados) and not any(e['ev'] == 'divergencia' for e in ev)
    assert os.path.exists(os.path.join(area['hpo'], 's', '_DONE.flag'))


# ── jornal ───────────────────────────────────────────────────────────────────────────────────────
def test_jornal_tolera_cauda_cortada_e_recusa_corrupcao_no_meio(tmp_path):
    arq = str(tmp_path / 'jornal.jsonl')
    j = H.Jornal(arq)
    for k in range(4):
        j.acrescentar({'ev': 'x', 'k': k})
    inteiro = open(arq, 'rb').read()
    for estrago in (inteiro[:-7], inteiro + b'\x00' * 64, inteiro[:-1], inteiro + b'{"ev": "x", "i": 4'):
        open(arq, 'wb').write(estrago)
        j = H.Jornal(arq)
        assert [e['i'] for e in j.eventos] == list(range(len(j.eventos))) and len(j.eventos) in (3, 4)
        j.acrescentar({'ev': 'y'})
        linhas = open(arq, 'rb').read().split(b'\n')
        assert linhas[-1] == b'' and all(json.loads(ln) for ln in linhas[:-1])       # JSONL válido linha a linha depois do acréscimo
        for f in os.listdir(tmp_path):
            if '.corrompido.' in f:
                os.remove(os.path.join(tmp_path, f))
    linhas = inteiro.split(b'\n')
    open(arq, 'wb').write(b'\n'.join([linhas[0], b'{lixo', linhas[2], linhas[3], b'']))
    with pytest.raises(RuntimeError, match='linha 2'):
        H.Jornal(arq)
    open(arq, 'wb').write(b'\n'.join([linhas[0], linhas[2], linhas[3], b'']))          # salto no índice
    with pytest.raises(RuntimeError, match='linha 2'):
        H.Jornal(arq)
    with pytest.raises(ValueError):
        H.Jornal(str(tmp_path / 'novo.jsonl')).acrescentar({'ev': 'x', 'v': float('nan')})   # nada não finito chega ao jornal


def test_partida_recusa_estudo_existente_sem_retomar_e_cabecalho_incompativel(area, monkeypatch):
    assert rodar(area, monkeypatch, n_fre=1.0) == 0
    monkeypatch.setenv('DEHB_FALSO', '{}')
    sem_retomar = [a for a in argumentos(area, n_fre=1.0) if a != '--retomar']
    with pytest.raises(SystemExit, match='--retomar'):
        H.main(sem_retomar, comando_avaliador=comando_falso)
    with pytest.raises(SystemExit, match='outro estudo'):
        H.main(argumentos(area, n_fre=1.0, extra=('--niveis', '3')), comando_avaliador=comando_falso)
    os.makedirs(os.path.join(area['hpo'], 'orfao', 'trabalho_0'))
    with pytest.raises(SystemExit, match='restos'):
        H.main(argumentos(area, study='orfao'), comando_avaliador=comando_falso)
    with pytest.raises(SystemExit, match='warm-start ausente'):
        H.main(argumentos(area, study='z', extra=('--seeds-trial', '7')), comando_avaliador=comando_falso)
    with pytest.raises(SystemExit, match='aquecimento'):
        H.main(['--study', 'curto', '--orcamento', '250000', '--warmstart', area['il'], '--referencias', area['refs']], comando_avaliador=comando_falso)


# ── falhas ───────────────────────────────────────────────────────────────────────────────────────
def test_falha_da_configuracao_vira_fail_cobrado_e_nao_trava_o_bracket(area, monkeypatch):
    import optuna
    assert rodar(area, monkeypatch, controle={'falha_config_ids': [0], 'aptidao_nan_ids': [1]}, n_fre=4.0) == 0
    ev = jornal(area)
    f0 = next(e for e in ev if e['ev'] == 'informar' and e['id'] == 0)
    assert f0['falhou'] and f0['aptidao'] is None and f0['motivo'] == 'excecao_no_treino' and f0['fre'] == pytest.approx(150_000 / N)
    f1 = [e for e in ev if e.get('id') == 1 and e['ev'] in ('informar', 'tentativa_falhou')]
    assert f1[-1]['ev'] == 'informar' and f1[-1]['falhou']                           # resultado não finito nunca vira aptidão
    assert not any('NaN' in ln or 'Infinity' in ln for ln in open(os.path.join(area['hpo'], 's', 'jornal.jsonl'), encoding='utf-8'))
    ts = trials(area)
    falhos = [t for t in ts if t.state == optuna.trial.TrialState.FAIL]
    assert falhos and all(t.value is None and 'fre_parcial' in t.user_attrs for t in falhos)
    assert any(e['ev'] == 'pedir' and e['trabalho']['degrau'] == 1 for e in ev)      # o bracket 0 fechou o degrau e seguiu


def test_falha_transitoria_e_repetida_cobrada_e_esgotada_vira_falha(area, monkeypatch):
    contador = str(area['raiz'] / 'lancamentos.txt')
    assert rodar(area, monkeypatch, controle={'transitoria_uma_vez_ids': [1], 'transitoria_sempre_ids': [2], 'sem_resultado_ids': [4],
                                              'contador': contador}, n_fre=5.0) == 0
    ev = jornal(area)
    t1 = [e for e in ev if e.get('id') == 1 and e['ev'] in ('tentativa_falhou', 'informar')]
    assert [e['ev'] for e in t1] == ['tentativa_falhou', 'informar'] and not t1[1]['falhou'] and t1[0]['fre'] == pytest.approx(75_000 / N)
    t2 = [e for e in ev if e.get('id') == 2 and e['ev'] in ('tentativa_falhou', 'informar')]
    assert [e['ev'] for e in t2] == ['tentativa_falhou', 'tentativa_falhou', 'informar'] and t2[2]['falhou'] and t2[2]['motivo'] == 'transitoria_esgotada'
    t4 = [e for e in ev if e.get('id') == 4 and e['ev'] == 'tentativa_falhou']
    assert t4 and t4[0]['motivo'] == 'sem_resultado'                                  # saída 0 sem resultado é falha, não sucesso
    lancamentos = [int(x) for x in open(contador, encoding='utf-8').read().split()]
    assert lancamentos.count(1) == 2 and lancamentos.count(2) == 2
    gasto = sum(e['fre'] for e in ev if e['ev'] in ('informar', 'tentativa_falhou'))
    assert json.load(open(os.path.join(area['hpo'], 's', 'PROGRESSO.json'), encoding='utf-8'))['fre_consumidos'] == pytest.approx(gasto, abs=1e-3)
    assert len(trials(area)) == sum(1 for e in ev if e['ev'] in ('informar', 'tentativa_falhou'))


def test_disjuntor_para_o_estudo_e_a_retomada_nao_queima_gpu(area, monkeypatch):
    contador = str(area['raiz'] / 'lancamentos.txt')
    controle = {'falha_config_ids': list(range(50)), 'contador': contador}
    assert rodar(area, monkeypatch, controle=controle) == 1
    d = os.path.join(area['hpo'], 's')
    assert os.path.exists(os.path.join(d, '_FAILED.flag')) and not os.path.exists(os.path.join(d, '_DONE.flag'))
    n = len(open(contador, encoding='utf-8').read().split())
    assert 4 <= n <= 4 + 3
    assert rodar(area, monkeypatch, controle=controle) == 1                          # o validador relança: aborta de novo SEM lançar nada
    assert len(open(contador, encoding='utf-8').read().split()) == n
    assert rodar(area, monkeypatch, controle={'contador': contador}, n_fre=3.0, extra=('--ignorar-disjuntor',)) == 0


def test_resultado_em_disco_so_vale_com_a_impressao_digital_do_pedido(area, monkeypatch):
    """Monta um estudo parado com um pedido em voo. Um resultado íntegro DESTE pedido é aproveitado sem retreinar; um resultado
    de outra encarnação (impressão diferente) ou cortado é renomeado e o trabalho é reexecutado."""
    contador = str(area['raiz'] / 'lancamentos.txt')
    for caso in ('valido', 'alheio', 'cortado'):
        monkeypatch.setenv('DEHB_FALSO', json.dumps({'contador': contador}))
        args = H.argparse.Namespace(study=caso, orcamento=N, niveis=2, eta=3, seed=None, inicializacao='professor', seeds_trial=[42], smoke=False,
                                    retomar=True, n_fre=1.0, paralelo=1)
        c = H.Coordenador(args, comando_avaliador=comando_falso, log=lambda m: None)
        c.abrir()
        t = c.dehb.pedir()
        ev = c.jornal.acrescentar({'ev': 'pedir', 'trabalho': t.registro(), 'gerador': c.dehb.estado_do_gerador(), 'data': H.agora_iso()})
        d = os.path.join(area['hpo'], caso, 'trabalho_0')
        os.makedirs(d)
        certo = H.impressao_digital(c.cab['id_execucao'], ev['trabalho'], [42], N, c.hash_git)
        res = {'impressao': certo if caso == 'valido' else 'outra', 'id': 0, 'aptidao': 0.123, 'objetivo': None, 'indicador': -0.123,
               'passos': 300_000, 'fre': 300_000 / N, 'inicio': H.agora_iso(), 'fim': H.agora_iso(), 'tempo_s': 1.0, 'curva': []}
        open(os.path.join(d, 'resultado.json'), 'w', encoding='utf-8').write(json.dumps(res)[:-20] if caso == 'cortado' else json.dumps(res))
        def n_lancados():
            return len(open(contador, encoding='utf-8').read().split()) if os.path.exists(contador) else 0
        antes = n_lancados()
        assert H.main(argumentos(area, study=caso, n_fre=0.3, k=1), comando_avaliador=comando_falso) == 0
        lancados = n_lancados() - antes
        inf = next(e for e in jornal(area, caso) if e['ev'] == 'informar' and e['id'] == 0)
        if caso == 'valido':
            assert lancados == 0 and inf['aptidao'] == 0.123
        else:
            assert lancados == 1 and inf['aptidao'] != 0.123 and any('.rejeitado.' in f for f in os.listdir(d))


def test_replay_divergente_adota_o_estado_em_vez_de_abortar(area, monkeypatch):
    assert rodar(area, monkeypatch, n_fre=2.0, k=1) == 0
    arq = os.path.join(area['hpo'], 's', 'jornal.jsonl')
    ev = jornal(area)
    alvo = next(e for e in ev if e['ev'] == 'pedir' and e['trabalho']['id'] == 1)
    alvo['trabalho']['vetor'][0] = 0.123456                                          # como se o núcleo de outra versão tivesse pedido outra coisa
    open(arq, 'w', encoding='utf-8').write(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in ev))
    assert rodar(area, monkeypatch, n_fre=4.0, k=1) == 0
    ev2 = jornal(area)
    assert any(e['ev'] == 'divergencia' and e['no_evento'] == alvo['i'] for e in ev2)
    assert [e['i'] for e in ev2] == list(range(len(ev2))) and sum(e['fre'] for e in ev2 if e['ev'] == 'informar') >= 4.0
    assert rodar(area, monkeypatch, n_fre=5.0, k=1) == 0                             # depois de adotar, as retomadas seguintes já entram adotando


# ── exclusão mútua, espelho e higiene ────────────────────────────────────────────────────────────
def test_segundo_coordenador_espera_a_trava_e_sai_limpo_quando_o_estudo_termina(area, monkeypatch):
    d = os.path.join(area['hpo'], 's')
    os.makedirs(d)
    trava = H.Trava(os.path.join(d, '_COORDENADOR.lock'))
    assert trava.tentar() and not H.Trava(os.path.join(d, '_COORDENADOR.lock')).tentar()
    monkeypatch.setenv('DEHB_FALSO', '{}')
    saida = {}
    fio = threading.Thread(target=lambda: saida.update(rc=H.main(argumentos(area), comando_avaliador=comando_falso)))
    fio.start()
    time.sleep(0.6)
    assert fio.is_alive() and not os.path.exists(os.path.join(d, 'jornal.jsonl'))    # não escreveu nada nem lançou filhos
    P.escrever_flag(d, '_DONE.flag', 'o outro coordenador terminou')
    fio.join(timeout=20)
    trava.soltar()
    assert saida.get('rc') == 0 and not os.path.exists(os.path.join(d, 'jornal.jsonl'))


def test_espelho_e_projecao_do_jornal_pode_ser_refeito_e_nunca_derruba_o_estudo(area, monkeypatch):
    assert rodar(area, monkeypatch, controle={'falha_config_ids': [3]}, n_fre=4.0) == 0
    foto = [(t.number, t.state.name, t.value, t.params, t.user_attrs['evento_i'], sorted(t.intermediate_values.items())) for t in trials(area)]
    os.remove(os.path.join(area['hpo'], 's.db'))
    assert H.main(['--study', 's', '--reconstruir-espelho']) == 0
    assert [(t.number, t.state.name, t.value, t.params, t.user_attrs['evento_i'], sorted(t.intermediate_values.items())) for t in trials(area)] == foto
    # banco que não é projeção deste jornal: o estudo segue, e a _DONE.flag NÃO sai (o pos_hpo leria um banco errado)
    assert rodar(area, monkeypatch, study='b', n_fre=1.0) == 0
    import optuna
    st = optuna.load_study(study_name='b', storage=f"sqlite:///{os.path.join(area['hpo'], 'b.db').replace(os.sep, '/')}")
    st.add_trial(optuna.trial.create_trial(value=9.9, params={}, distributions={}))      # um trial que NÃO veio do jornal
    H.liberar_banco(st)
    assert rodar(area, monkeypatch, study='b', n_fre=2.0) == 1
    assert not os.path.exists(os.path.join(area['hpo'], 'b', '_DONE.flag'))
    assert sum(e['fre'] for e in jornal(area, 'b') if e['ev'] == 'informar') >= 2.0   # o jornal, que é a verdade, andou


def test_escrita_atomica_insiste_quando_o_arquivo_esta_preso(tmp_path, monkeypatch):
    real, falhas = os.replace, {'n': 0}

    def preso(a, b):
        falhas['n'] += 1
        if falhas['n'] <= 2:
            raise PermissionError('WinError 32')
        return real(a, b)
    monkeypatch.setattr(H.os, 'replace', preso)
    monkeypatch.setattr(H.time, 'sleep', lambda s: None)
    H.escrever_json_atomico(str(tmp_path / 'x.json'), {'a': 'ção'})
    assert json.load(open(tmp_path / 'x.json', encoding='utf-8')) == {'a': 'ção'} and falhas['n'] == 3


def test_coordenador_nao_carrega_torch_e_o_filho_roda_em_utf8(area):
    codigo = ("import sys, os, json; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s'); import hpo_dehb as H; H.DIR_HPO = r'%s'; "
              "rc = H.main(%r, comando_avaliador=lambda d: [sys.executable, '-X', 'utf8', r'%s', d]); "
              "assert rc == 0, rc; assert 'torch' not in sys.modules, 'o coordenador importou torch'"
              % (os.path.join(P.RAIZ, 'hpo'), P.RAIZ, area['hpo'], argumentos(area, n_fre=1.0, k=1), FALSO))
    r = subprocess.run([sys.executable, '-X', 'utf8', '-c', codigo], cwd=P.RAIZ, capture_output=True, text=True, encoding='utf-8',
                       env=dict(os.environ, DEHB_FALSO='{}', PYTHONUTF8='1'))
    assert r.returncode == 0, r.stderr[-1500:]
    log = open(os.path.join(area['hpo'], 's', 'trabalho_0', 'avaliacao.log'), encoding='utf-8').read()
    assert 'β validação' in log


def test_constantes_e_espaco_espelham_o_ppo_e_o_hpo_rl():
    import inspect
    import ppo
    import hpo_rl
    assert H.BLOCOS_POR_EXECUCAO == ppo.HP_PADRAO['blocos'] and H.EPISODIOS_POR_COLETA == ppo.HP_PADRAO['V']
    for ini in ('professor', 'zero'):
        assert H.espaco_de(ini) is hpo_rl.espaco_de(ini)
    assert set(H.HP_SMOKE) <= set(ppo.HP_PADRAO)
    assert "'--orcamento', type=int, default=P.ORCAMENTO_RL_PASSOS" in inspect.getsource(H.main)
    assert P.STUDY_DEHB not in P.STUDIES_HPO                                         # comparação (D-16): não concorre a MimePlant
    assert os.path.dirname(P.DIR_REAVALIACAO_DEHB) == P.DIR_RODADA and P.DIR_REAVALIACAO_DEHB != P.DIR_REAVALIACAO
    # blocos de tamanho constante: N // 3 com 3 blocos tem as MESMAS fronteiras do trial completo com 9
    for n in (1_000_000, 500_000):
        fid = NUC.escada_de_passos(n, 2)
        assert [max(1, round(f / (n // 9))) for f in fid] == [3, 9] and fid[0] // 3 == n // 9
