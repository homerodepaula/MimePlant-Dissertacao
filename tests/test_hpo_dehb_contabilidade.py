"""Testes de CONTABILIDADE do coordenador do DEHB (`hpo/hpo_dehb.py`): FRE real, impressão digital, cabeçalho, disjuntor, trabalho.json,
INDICE.json, espelho no Optuna, jornal, atributos do estudo, --reconstruir-espelho, manter_a_maquina_acordada e partida. Nasceram de
uma revisão por MUTAÇÃO: o código estava certo, mas trocas plausíveis nessas áreas passavam pelos testes de `test_hpo_dehb.py`.

Quase todos exercitam o objeto Coordenador À MÃO (sem processo nenhum). Quando é preciso um estudo inteiro, o avaliador FALSO
(`_avaliador_falso_dehb.py`) roda DENTRO do processo de teste, no lugar do `subprocess.Popen` (fixture `em_processo`): o estudo sai
em ~1 s, com K = 1 a ordem é determinística, e nenhum assert depende de relógio. Caminhos com espaço e acento, como a pasta real."""
import os
import sys
import json
import time
import types
import datetime
import subprocess

import pytest

import protocolo as P

sys.path.insert(0, os.path.join(P.RAIZ, 'hpo'))
sys.path.insert(0, os.path.join(P.RAIZ, 'tests'))
import hpo_dehb as H                        # noqa: E402
import dehb_nucleo as NUC                   # noqa: E402
import _avaliador_falso_dehb as AVALIADOR   # noqa: E402

FALSO = os.path.join(P.RAIZ, 'tests', '_avaliador_falso_dehb.py')
N = 900_000                 # fidelidades 300.000 e 900.000: cobre o aquecimento do crítico (91.980 passos)
CHAVES_DO_CABECALHO = ('semente', 'espaco', 'N', 'fidelidades', 'eta', 'inicializacao', 'seeds_trial', 'smoke')


# ── apoio ────────────────────────────────────────────────────────────────────────────────────────
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


class ProcessoFalso:
    """No lugar do Popen: roda o avaliador FALSO na hora do lançamento, dentro do processo de teste; `poll()` já devolve a saída."""

    def __init__(self, comando, **kw):
        self.pid, self.morto = 4321, False
        self.returncode = AVALIADOR.main(comando[-1])

    def poll(self):
        return self.returncode

    def kill(self):
        self.morto = True

    def wait(self, timeout=None):
        return self.returncode


class ProcessoMudo(ProcessoFalso):
    """Um filho que nunca termina e não faz nada: para olhar o que o coordenador escreveu ANTES de lançar, e para o _abortar."""

    def __init__(self, comando=None, quebra_no_kill=False, **kw):
        self.pid, self.morto, self.returncode, self.quebra_no_kill = 1234, False, None, quebra_no_kill

    def kill(self):
        if self.quebra_no_kill:
            raise OSError('processo já morreu')
        self.morto = True


def trocar_popen(monkeypatch, classe):
    # só o nome `subprocess` DENTRO do hpo_dehb: o módulo de verdade segue intacto para o resto do processo (hash_git, optuna...)
    monkeypatch.setattr(H, 'subprocess', types.SimpleNamespace(Popen=classe, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT,
                                                               TimeoutExpired=subprocess.TimeoutExpired))


class ModuloCom:
    """Um módulo com alguns nomes trocados, para pôr no lugar do NOME dele dentro do hpo_dehb (`H.time`, `H.os`). Trocar o atributo
    no módulo de verdade (H.time.sleep, H.os.name...) valeria para o processo inteiro — Optuna, SQLAlchemy e pytest incluídos."""

    def __init__(self, modulo, **trocas):
        self.__dict__.update(trocas, _modulo=modulo)

    def __getattr__(self, nome):
        return getattr(self._modulo, nome)


@pytest.fixture
def sem_esperas(monkeypatch):
    """As esperas do código (entre ciclos, entre tentativas de E/S) são cortesia com o disco: aqui não esperam. Com TETO, porque sem espera
    um laço que não sai do lugar (a trava do estudo que nunca volta, por exemplo) giraria para sempre e penduraria o validador horário
    sem mensagem nenhuma: depois de 5.000 esperas num teste só — o maior usa ~150 — o teste FALHA."""
    esperas = []

    def sleep(segundos):
        esperas.append(segundos)
        if len(esperas) > 5000:
            raise AssertionError(f'{len(esperas)} esperas num teste só: o código está preso num laço de espera')
    monkeypatch.setattr(H, 'time', ModuloCom(time, sleep=sleep))


@pytest.fixture
def em_processo(monkeypatch, sem_esperas):
    trocar_popen(monkeypatch, ProcessoFalso)                  # o filho já terminou quando o laço volta: não há o que esperar entre os ciclos


def argumentos(area, study='s', n_fre=1.0, k=1, extra=()):
    return ['--study', study, '--n-fre', str(n_fre), '--orcamento', str(N), '--paralelo', str(k), '--warmstart', area['il'],
            '--referencias', area['refs'], '--intervalo-s', '0.05', '--retomar', *extra]


def rodar(area, monkeypatch, controle=None, **kw):
    monkeypatch.setenv('DEHB_FALSO', json.dumps(controle or {}))
    return H.main(argumentos(area, **kw), comando_avaliador=lambda d: ['falso', d])


def args_de(area, **kw):
    base = dict(study='s', orcamento=N, niveis=2, eta=3, seed=None, inicializacao='professor', seeds_trial=[42], smoke=False, retomar=True,
                n_fre=1.0, paralelo=1, warmstart=area['il'], referencias=area['refs'], dispositivo=None, max_tentativas=2,
                max_falhas_seguidas=4, ignorar_disjuntor=False, limite_sem_batimento_h=3.0, intervalo_s=0.05)
    return H.argparse.Namespace(**dict(base, **kw))


class EspelhoMudo:
    """Espelho que só anota as chamadas: para os testes em que o Optuna não é o assunto (abrir o SQLite custa ~0,5 s)."""

    def __init__(self):
        self.chamadas = []

    def reconciliar(self, eventos):
        self.chamadas.append(len(eventos))
        return True

    def atributos_do_estudo(self, atributos, sessao):
        pass

    def indice(self):
        return {}

    def fechar(self):
        pass


def coordenador(area, abrir=True, **kw):
    mensagens = []
    c = H.Coordenador(args_de(area, **kw), comando_avaliador=lambda d: ['falso', d], log=mensagens.append)
    c.mensagens, c.espelho = mensagens, EspelhoMudo()
    if abrir:
        assert c.abrir()
    return c


def pedir(c):
    t = c.dehb.pedir()
    return c.jornal.acrescentar({'ev': 'pedir', 'trabalho': t.registro(), 'gerador': c.dehb.estado_do_gerador(), 'data': H.agora_iso()})['trabalho']


def resultado_de(c, registro, passos, valor=0.5, **kw):
    topo = registro['nivel'] == len(c.fidelidades) - 1
    return dict({'impressao': 'x', 'id': registro['id'], 'aptidao': -valor, 'objetivo': valor if topo else None, 'indicador': valor,
                 'passos': passos, 'fre': passos / (c.N * len(c.args.seeds_trial)), 'inicio': '2026-09-18T10:00:00.000',
                 'fim': '2026-09-18T10:30:00.000', 'tempo_s': 1800.0, 'curva': [{'passos': passos, 'proxy_norm': valor}], 'hash_git': 'falso'}, **kw)


def jornal(area, study='s'):
    return [json.loads(ln) for ln in open(os.path.join(area['hpo'], study, 'jornal.jsonl'), encoding='utf-8')]


def caminho_do_banco(area, study='s'):
    return os.path.join(area['hpo'], study + '.db')


def estudo_do_optuna(area, study='s'):
    import optuna
    return optuna.load_study(study_name=study, storage=f"sqlite:///{caminho_do_banco(area, study).replace(os.sep, '/')}")


def trials(area, study='s'):
    st = estudo_do_optuna(area, study)
    ts = st.get_trials(deepcopy=True)
    H.liberar_banco(st)                     # no Windows um .db com conexão aberta não pode ser renomeado nem apagado
    return ts


def atributos(area, study='s'):
    st = estudo_do_optuna(area, study)
    attrs = dict(st.user_attrs)
    H.liberar_banco(st)
    return attrs


def foto_da_pasta(raiz):
    """nome -> bytes de TODO arquivo abaixo de `raiz`: para provar que uma operação não escreveu nada."""
    foto = {}
    for pasta, _, arquivos in os.walk(raiz):
        for nome in arquivos:
            with open(os.path.join(pasta, nome), 'rb') as f:
                foto[os.path.relpath(os.path.join(pasta, nome), raiz)] = f.read()
    return foto


def terminais(eventos):
    return [e for e in eventos if e['ev'] in ('informar', 'tentativa_falhou')]


# ── a) orçamento pelo FRE REAL ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('fracao', [0.5, 1.25])
def test_orcamento_conta_os_passos_que_o_avaliador_relatou_e_nao_o_nominal_da_fidelidade(area, monkeypatch, em_processo, fracao):
    """O treino para na última coleta inteira (menos que o nominal) ou passa dele: jornal, PROGRESSO.json e o trial do Optuna
    levam passos/N RELATADOS. Com o FRE nominal, o estudo de fração 0,5 pararia na metade do orçamento de verdade."""
    assert rodar(area, monkeypatch, controle={'fracao_dos_passos': fracao}, n_fre=1.5) == 0
    informados = [e for e in jornal(area) if e['ev'] == 'informar']
    assert informados and all(not e['falhou'] for e in informados)
    for e in informados:
        assert e['passos'] == int(e['fidelidade'] * fracao) and e['fre'] == pytest.approx(e['passos'] / N)
        assert e['fre'] != pytest.approx(e['fidelidade'] / N)                         # o nominal NÃO é o que se cobra
    gasto = sum(e['fre'] for e in informados)
    assert gasto >= 1.5 and gasto - informados[-1]['fre'] < 1.5                        # parou quando o REAL cruzou o alvo (K = 1: sem excesso)
    prog = json.load(open(os.path.join(area['hpo'], 's', 'PROGRESSO.json'), encoding='utf-8'))
    assert prog['fre_consumidos'] == round(gasto, 3) and prog['fre_alvo'] == 1.5
    ts = trials(area)
    assert len(ts) == len(informados)                                                 # senão o zip abaixo encolheria calado
    for t, e in zip(ts, informados):
        assert t.user_attrs['fre'] == pytest.approx(e['fre']) and t.user_attrs['passos'] == e['passos'] and t.user_attrs['fidelidade'] == e['fidelidade']


def test_fre_real_tambem_chega_pelo_subprocesso_de_verdade(area, monkeypatch):
    """O mesmo, uma vez só, pela cadeia inteira (Popen, trabalho.json, resultado.json com impressão): trava a chave nova do avaliador falso."""
    monkeypatch.setenv('DEHB_FALSO', json.dumps({'fracao_dos_passos': 0.5}))
    assert H.main(argumentos(area, n_fre=0.3, k=2), comando_avaliador=lambda d: [sys.executable, '-X', 'utf8', FALSO, d]) == 0
    informados = [e for e in jornal(area) if e['ev'] == 'informar']
    assert len(informados) >= 2 and all(e['passos'] == e['fidelidade'] // 2 and e['fre'] == pytest.approx(e['passos'] / N) for e in informados)


def test_informar_grava_o_fre_do_resultado_e_a_falha_divide_os_passos_por_todas_as_sementes(area):
    """Com duas sementes de treino, `passos` é a soma das duas execuções e 1 FRE = N passos POR semente: o sucesso leva o `fre` que o
    avaliador calculou (não passos/N), e a falha divide o parcial por N x nº de sementes."""
    c = coordenador(area, seeds_trial=[42, 43])
    r0, r1, r2 = pedir(c), pedir(c), pedir(c)
    c._informar(r0, resultado=resultado_de(c, r0, passos=2 * 250_000))
    c._informar(r1, falha={'passos': 120_000, 'motivo': 'excecao_no_treino', 'detalhe': 'x' * 5000, 'tempo_s': 3.0})
    c._informar(r2, motivo='transitoria_esgotada')
    e0, e1, e2 = [e for e in c.jornal.eventos if e['ev'] == 'informar']
    assert e0['passos'] == 500_000 and e0['fre'] == pytest.approx(500_000 / (2 * N)) and not e0['falhou'] and e0['aptidao'] == -0.5
    assert e1['falhou'] and e1['aptidao'] is None and e1['passos'] == 120_000 and e1['fre'] == pytest.approx(120_000 / (2 * N))
    assert e1['motivo'] == 'excecao_no_treino' and len(e1['detalhe']) == 600          # o traceback inteiro fica no falha.json, não no jornal
    assert e2['falhou'] and e2['motivo'] == 'transitoria_esgotada' and e2['passos'] == 0 and e2['fre'] == 0.0
    assert c.fre_consumidos() == pytest.approx(e0['fre'] + e1['fre'])


def test_fre_consumidos_soma_informar_e_tentativa_falhada_e_nao_o_parcial_perdido_em_queda(area):
    c = coordenador(area)
    c.jornal.acrescentar({'ev': 'informar', 'id': 0, 'fre': 0.25, 'falhou': False})
    c.jornal.acrescentar({'ev': 'tentativa_falhou', 'id': 1, 'fre': 0.125, 'tentativa': 1})
    c.jornal.acrescentar({'ev': 'informar', 'id': 1, 'fre': 0.0625, 'falhou': True})
    c.jornal.acrescentar({'ev': 'relancado_apos_queda', 'id': 2, 'fre_nao_cobrado': 0.5, 'fre': 0.5})   # rastro, não cobrança
    c.jornal.acrescentar({'ev': 'pedir', 'trabalho': {'id': 3, 'origem': 'evolucao'}, 'fre': 9.0})
    assert c.fre_consumidos() == 0.25 + 0.125 + 0.0625


def test_informar_segue_a_ordem_jornal_nucleo_espelho_e_nada_acontece_se_o_jornal_nao_gravou(area, monkeypatch):
    c = coordenador(area)
    r0 = pedir(c)
    with monkeypatch.context() as m:
        m.setattr(c.jornal, 'acrescentar', lambda ev: (_ for _ in ()).throw(OSError('WinError 32')))
        with pytest.raises(OSError):
            c._informar(r0, resultado=resultado_de(c, r0, passos=300_000))
    # write-ahead: sem a linha no jornal, o núcleo não viu o resultado e o espelho não foi tocado — o próximo ciclo repete tudo
    assert 0 in c.dehb.pendentes and not c.dehb.historico and c.espelho.chamadas == []
    c._informar(r0, resultado=resultado_de(c, r0, passos=300_000, curva=[{'passos': 1, 'proxy_norm': 0.1}, {'passos': 3}]))
    ev = c.jornal.eventos[-1]
    assert 0 not in c.dehb.pendentes and c.dehb.historico[0].aptidao == -0.5 and c.espelho.chamadas == [len(c.jornal.eventos)]
    assert ev['hp'] == c.espaco.para_hp(r0['vetor']) and ev['hash_git'] == 'falso' and ev['inicio'] == '2026-09-18T10:00:00.000'
    assert ev['curva'] == [{'passos': 1, 'proxy_norm': 0.1}]                          # ponto sem medida finita não vai para o jornal (nem para o Optuna)
    assert os.path.exists(os.path.join(c.dir, 'PROGRESSO.json'))


def test_ponto_da_curva_com_proxy_norm_nulo_e_descartado_em_vez_de_derrubar_o_coordenador(area):
    """TypeError não é OSError: sai do laço, vira _FAILED.flag, e a retomada reencontra o MESMO resultado.json em disco e cai de novo."""
    c = coordenador(area)
    r0 = pedir(c)
    c._informar(r0, resultado=resultado_de(c, r0, passos=300_000, curva=[{'passos': 1, 'proxy_norm': 0.1}, {'passos': 2, 'proxy_norm': None}]))
    assert c.jornal.eventos[-1]['curva'] == [{'passos': 1, 'proxy_norm': 0.1}]


def batimento_em_disco(c, id_trabalho, conteudo):
    os.makedirs(c._dir_trabalho(id_trabalho), exist_ok=True)
    with open(os.path.join(c._dir_trabalho(id_trabalho), '_BATIMENTO.json'), 'w', encoding='utf-8') as f:
        f.write(conteudo if isinstance(conteudo, str) else json.dumps(conteudo))


def test_tentativa_que_morreu_sem_falha_json_cobra_os_passos_do_ultimo_batimento_divididos_pelas_sementes(area):
    """Processo morto pelo SO (OOM, kill do vigia) não grava falha.json: o parcial cobrado é o que o último batimento viu — é o
    'orçamento pelos passos REAIS, inclusive o parcial de avaliação que morreu'. Com duas sementes, 1 FRE = N passos POR semente."""
    c = coordenador(area, seeds_trial=[42, 43], max_tentativas=3)
    r0 = pedir(c)
    batimento_em_disco(c, 0, {'passos': 120_000, 'pid': 1})
    c._tentativa_falhou(r0, motivo='processo_morreu', rc=-9)
    e = c.jornal.eventos[-1]
    assert (e['ev'], e['id'], e['passos'], e['codigo_saida'], e['tentativa'], e['motivo']) == ('tentativa_falhou', 0, 120_000, -9, 1, 'processo_morreu')
    assert e['fre'] == pytest.approx(120_000 / (2 * N)) and c.fre_consumidos() == pytest.approx(120_000 / (2 * N))
    assert (e['bracket'], e['degrau'], e['nivel'], e['fidelidade'], e['hp']) == (r0['bracket'], r0['degrau'], 0, N // 3, c.espaco.para_hp(r0['vetor']))
    assert c.tentativas == {0: 1} and c._fila == [r0] and c.espelho.chamadas == [len(c.jornal.eventos)]   # volta para a FRENTE da fila, e o espelho já viu o FAIL
    c._tentativa_falhou(r0, motivo='gpu_ou_sistema', falha={'passos': 30_000, 'detalhe': 'z' * 900}, rc=4)   # com falha.json vale o que o avaliador contou
    e = c.jornal.eventos[-1]
    assert (e['passos'], e['tentativa'], e['codigo_saida'], len(e['detalhe'])) == (30_000, 2, 4, 600) and e['fre'] == pytest.approx(30_000 / (2 * N))


def test_batimento_cortado_pela_queda_cobra_zero_em_vez_de_derrubar_o_coordenador(area):
    c = coordenador(area)
    r0 = pedir(c)
    batimento_em_disco(c, 0, '{"passos": 12')                                         # arquivo pela metade (disco cheio, queda): ilegível não é motivo para cair
    assert c._passos_do_batimento(0) == 0 and c._passos_do_batimento(999) == 0        # ilegível ou ausente: nada a cobrar
    c._tentativa_falhou(r0, motivo='processo_morreu', rc=-9)                          # não levanta
    assert (c.jornal.eventos[-1]['ev'], c.jornal.eventos[-1]['passos'], c.jornal.eventos[-1]['fre']) == ('tentativa_falhou', 0, 0.0)


def test_tentativas_ja_gastas_valem_na_retomada_e_a_ultima_esgota_o_trabalho(area):
    """Sem restaurar `tentativas` no replay, cada reinício da máquina devolveria --max-tentativas inteiras ao mesmo trabalho."""
    c = coordenador(area)
    r0 = pedir(c)
    c._tentativa_falhou(r0, motivo='processo_morreu', rc=-9)
    assert c.tentativas == {0: 1} and c._fila == [r0]
    c2 = coordenador(area)
    assert c2.tentativas == {0: 1}
    c2._tentativa_falhou(r0, motivo='processo_morreu', rc=-9)                         # a 2ª de 2: esgota, mesmo com a 1ª noutra sessão
    do_trabalho = [e for e in c2.jornal.eventos if e.get('id') == 0 and e['ev'] in ('tentativa_falhou', 'informar')]
    assert [(e['ev'], e.get('tentativa')) for e in do_trabalho] == [('tentativa_falhou', 1), ('tentativa_falhou', 2), ('informar', None)]
    assert do_trabalho[-1]['falhou'] and do_trabalho[-1]['motivo'] == 'transitoria_esgotada' and c2._fila == [] and 0 not in c2.dehb.pendentes
    assert coordenador(area).tentativas == {0: 2}                                     # o replay fica com a MAIOR tentativa gravada


def test_trabalho_em_voo_na_queda_do_coordenador_volta_para_a_fila_e_o_parcial_fica_registrado_sem_ser_cobrado(area):
    """Quem ESCREVE o `relancado_apos_queda`: os passos do último batimento viram rastro (passos_perdidos, fre_nao_cobrado por semente),
    o trabalho vai para a fila e o orçamento não é tocado."""
    c = coordenador(area, seeds_trial=[42, 43])
    r0, r1 = pedir(c), pedir(c)
    batimento_em_disco(c, 1, {'passos': 60_000, 'pid': 1})
    c._informar(r0, resultado=resultado_de(c, r0, passos=2 * 250_000))                # o 0 terminou antes da queda: não é pendente
    c2 = coordenador(area, seeds_trial=[42, 43])
    c2._retomar_pendentes()
    quedas = [e for e in c2.jornal.eventos if e['ev'] == 'relancado_apos_queda']
    assert [(e['id'], e['passos_perdidos']) for e in quedas] == [(1, 60_000)] and quedas[0]['fre_nao_cobrado'] == pytest.approx(60_000 / (2 * N))
    assert c2._fila == [r1] and not c2.em_voo and not c2.adotados
    assert c2.fre_consumidos() == pytest.approx(500_000 / (2 * N))                    # só o que o trabalho 0 gastou
    assert c2.progresso()['fre_perdido_em_quedas_nao_cobrado'] == round(60_000 / (2 * N), 3)


def test_evento_do_jornal_leva_a_posicao_do_pedido_na_escada(area):
    c = coordenador(area)
    r0 = dict(pedir(c), bracket=2, degrau=1, origem='promocao')                       # bracket != degrau: no primeiro pedido (0, 0) uma troca é invisível
    c._informar(r0, resultado=resultado_de(c, r0, passos=300_000))
    ev = c.jornal.eventos[-1]
    assert (ev['bracket'], ev['degrau'], ev['origem'], ev['nivel'], ev['fidelidade'], ev['fidelidade_maxima']) == (2, 1, 'promocao', 0, N // 3, False)


def test_status_preso_nao_derruba_o_informar_depois_de_o_jornal_e_o_nucleo_terem_andado(area, monkeypatch):
    """O _STATUS.txt é a ÚLTIMA coisa do _informar: se o erro dele subisse, o ciclo repetiria o _tratar_termino de um trabalho já informado."""
    c = coordenador(area)
    r0 = pedir(c)
    monkeypatch.setattr(P, 'escrever_status', lambda *a, **kw: (_ for _ in ()).throw(PermissionError('WinError 32')))
    c._informar(r0, resultado=resultado_de(c, r0, passos=300_000))                    # não levanta
    assert [e['ev'] for e in c.jornal.eventos].count('informar') == 1 and 0 not in c.dehb.pendentes and len(c.dehb.historico) == 1


# ── b) impressão digital ─────────────────────────────────────────────────────────────────────────
REGISTRO = {'id': 7, 'vetor': [0.25, 0.5, 0.125], 'fidelidade': 300000, 'nivel': 0, 'id_pai': 1, 'bracket': 0, 'degrau': 0,
            'origem': 'evolucao', 'pais': {'pool': 3}}


@pytest.mark.parametrize('campo, troca', [
    ('id_execucao', dict(id_execucao='abc124')), ('id', dict(registro=dict(REGISTRO, id=8))),
    ('vetor', dict(registro=dict(REGISTRO, vetor=[0.25, 0.5, 0.12500000000000003]))),    # um bit do vetor já é outro pedido
    ('fidelidade', dict(registro=dict(REGISTRO, fidelidade=900000))), ('seeds', dict(semente_treino=[42, 43])),
    ('seeds (outra)', dict(semente_treino=[7])), ('N', dict(passos_max=900_001)), ('hash_git', dict(hash_git='deadbef'))])
def test_impressao_digital_muda_com_cada_campo_que_identifica_o_pedido(campo, troca):
    base = dict(id_execucao='abc123', registro=REGISTRO, semente_treino=[42], passos_max=900_000, hash_git='deadbee')
    assert H.impressao_digital(**dict(base, **troca)) != H.impressao_digital(**base), campo


def test_impressao_digital_e_estavel_e_e_um_contrato_em_disco():
    base = dict(id_execucao='abc123', registro=REGISTRO, semente_treino=[42], passos_max=900_000, hash_git='deadbee')
    certo = H.impressao_digital(**base)
    # O valor literal é um ALARME de mudança deliberada do formato em disco, não uma regra: nenhum comportamento depende DESTE número, mas
    # mexer na fórmula (até tirar o sort_keys) faz a retomada recusar todo resultado.json gravado pela versão anterior — horas de GPU.
    # O hash_git também entra na impressão, então com git um commit já invalida os resultados em voo; nesta pasta, que não é repositório
    # (hash_git == 'sem-git'), só a fórmula os invalida. Quem mudar a fórmula de propósito atualiza o literal e retoma com a fila vazia.
    assert certo == 'd25e8d6eb45c07d5ca2a8ef6ab631f75d8bf549cd732e350031d87eaf62ac9bd'
    ao_contrario = dict(reversed(list(REGISTRO.items())))                             # a ordem das chaves (JSON relido, dict remontado) não importa
    assert list(ao_contrario) != list(REGISTRO) and H.impressao_digital(**dict(base, registro=ao_contrario)) == certo
    assert H.impressao_digital(**dict(base, semente_treino=(42,), passos_max=900_000.0)) == certo    # tupla do argparse, N relido como float
    # o que NÃO identifica o pedido (posição na escada, contabilidade dos pais) não entra: o replay ADOTADO pode remontá-los
    assert H.impressao_digital(**dict(base, registro=dict(REGISTRO, origem='promocao', pais={}, bracket=3))) == certo


# ── c) cabeçalho: --retomar só no MESMO estudo ───────────────────────────────────────────────────
@pytest.mark.parametrize('chave', CHAVES_DO_CABECALHO)
def test_retomar_com_qualquer_chave_do_cabecalho_diferente_e_recusado_citando_a_chave(area, chave):
    """O cabeçalho gravado é adulterado numa chave só (assim eta, N e fidelidades ficam isolados uns dos outros) e a retomada com os
    argumentos originais tem de recusar, dizer QUAL chave divergiu e não tocar no jornal."""
    coordenador(area)
    arq = os.path.join(area['hpo'], 's', 'jornal.jsonl')
    ev = jornal(area)
    outro = {'semente': ev[0]['semente'] + 1, 'espaco': list(reversed(ev[0]['espaco'])), 'N': N + 9, 'fidelidades': [N // 9, N], 'eta': 2,
             'inicializacao': 'zero', 'seeds_trial': [42, 43], 'smoke': True}[chave]
    ev[0][chave] = outro
    open(arq, 'wb').write(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in ev).encode('utf-8'))
    antes = open(arq, 'rb').read()
    with pytest.raises(SystemExit, match=f'--retomar com {chave} diferente.*outro estudo'):
        coordenador(area)
    assert open(arq, 'rb').read() == antes                                           # recusou ANTES de registrar a sessão


def test_retomar_aceita_o_mesmo_estudo_e_recusa_cada_argumento_trocado(area, monkeypatch):
    coordenador(area)
    arq = os.path.join(area['hpo'], 's', 'jornal.jsonl')
    ev = jornal(area)
    assert [e['ev'] for e in ev] == ['cabecalho', 'sessao'] and set(CHAVES_DO_CABECALHO) < set(ev[0]) and ev[0]['semente'] == coordenador(area, abrir=False).semente
    assert ev[0]['N'] == N and ev[0]['fidelidades'] == [N // 3, N] and ev[0]['espaco'] == [[k, [v[0]] + [list(x) if isinstance(x, list) else x for x in v[1:]]] for k, v in H.ESPACO_MIMEPLANT.items()]
    coordenador(area, seeds_trial=(42,), n_fre=7.0, paralelo=2)                       # tupla x lista, orçamento e K: continua sendo o mesmo estudo
    assert [e['ev'] for e in jornal(area)] == ['cabecalho', 'sessao', 'sessao']
    antes = open(arq, 'rb').read()
    for chave, troca in (('semente', dict(seed=ev[0]['semente'] + 1)), ('N', dict(orcamento=2 * N)), ('fidelidades', dict(niveis=3)),
                         ('fidelidades', dict(eta=2)), ('seeds_trial', dict(seeds_trial=[42, 43])), ('seeds_trial', dict(seeds_trial=[43])),
                         ('smoke', dict(smoke=True)), ('espaco', dict(inicializacao='zero'))):
        with pytest.raises(SystemExit, match=f'--retomar com {chave} diferente'):
            coordenador(area, **troca)
    monkeypatch.setattr(H, 'espaco_de', lambda inicializacao: H.ESPACO_MIMEPLANT)      # mesmo espaço: sobra só a inicialização (o clone x do zero)
    with pytest.raises(SystemExit, match='--retomar com inicializacao diferente'):
        coordenador(area, inicializacao='zero')
    assert open(arq, 'rb').read() == antes


def test_semente_do_estudo_vem_do_nome_e_o_argumento_seed_prevalece(area):
    a, b = coordenador(area, study='um', abrir=False), coordenador(area, study='outro', abrir=False)
    assert a.semente == H.zlib.crc32('um'.encode('utf-8')) % 100_000 and a.semente != b.semente
    assert coordenador(area, study='um', seed=0, abrir=False).semente == 0              # 0 é semente válida, não "ausente"


# ── d) disjuntor ─────────────────────────────────────────────────────────────────────────────────
def test_falhas_finais_seguidas_conta_so_a_cauda_de_falhas_e_o_sucesso_zera(area):
    c = coordenador(area)
    assert c.falhas_finais_seguidas() == 0

    def informa(falhou, ev='informar'):
        c.jornal.acrescentar({'ev': ev, 'id': len(c.jornal.eventos), 'falhou': falhou, 'fre': 0.1})
    for falhou in (True, True, True, False, True):
        informa(falhou)
    assert c.falhas_finais_seguidas() == 1                                            # 4 falhas no total, mas só UMA depois do último sucesso
    informa(True)
    c.jornal.acrescentar({'ev': 'tentativa_falhou', 'id': 99, 'tentativa': 1, 'fre': 0.1})   # repetição transitória não é falha FINAL: nem conta nem zera
    c.jornal.acrescentar({'ev': 'pedir', 'trabalho': {'id': 100, 'origem': 'evolucao'}})
    assert c.falhas_finais_seguidas() == 2
    informa(True)
    assert c.falhas_finais_seguidas() == 3
    informa(False)
    assert c.falhas_finais_seguidas() == 0


def test_falhas_espalhadas_nao_disparam_o_disjuntor(area, monkeypatch, em_processo):
    """Duas falhas, um sucesso, duas falhas...: mais falhas no TOTAL que o limite, nunca `max` SEGUIDAS — configurações ruins não são defeito sistêmico."""
    assert rodar(area, monkeypatch, controle={'falha_config_ids': [i for i in range(60) if i % 3 != 2]}, n_fre=2.0, extra=('--max-falhas-seguidas', '3')) == 0
    falhou = [e['falhou'] for e in jornal(area) if e['ev'] == 'informar']
    assert sum(falhou) >= 4 and falhou[:6] == [True, True, False, True, True, False]
    assert os.path.exists(os.path.join(area['hpo'], 's', '_DONE.flag')) and not os.path.exists(os.path.join(area['hpo'], 's', '_FAILED.flag'))


def test_disjuntor_dispara_na_falha_final_de_numero_max_e_ignorar_disjuntor_passa(area, monkeypatch, em_processo):
    contador = str(area['raiz'] / 'lancamentos.txt')
    controle = {'falha_config_ids': [3, 4], 'contador': contador}
    assert rodar(area, monkeypatch, controle=controle, n_fre=3.0, extra=('--max-falhas-seguidas', '2')) == 1
    d = os.path.join(area['hpo'], 's')
    informados = [e for e in jornal(area) if e['ev'] == 'informar']
    assert [(e['id'], e['falhou']) for e in informados] == [(0, False), (1, False), (2, False), (3, True), (4, True)]   # parou NA segunda, não na terceira
    assert 'seguidas' in open(os.path.join(d, '_FAILED.flag'), encoding='utf-8').read() and not os.path.exists(os.path.join(d, '_DONE.flag'))
    n = len(open(contador, encoding='utf-8').read().split())
    assert rodar(area, monkeypatch, controle=controle, n_fre=3.0, extra=('--max-falhas-seguidas', '2')) == 1              # relançado: aborta de novo, sem lançar
    assert len(open(contador, encoding='utf-8').read().split()) == n and os.path.exists(os.path.join(d, '_FAILED.flag'))
    assert rodar(area, monkeypatch, controle=controle, n_fre=3.0, extra=('--max-falhas-seguidas', '3')) == 0              # abaixo do limite: segue
    assert os.path.exists(os.path.join(d, '_DONE.flag')) and not os.path.exists(os.path.join(d, '_FAILED.flag'))


def test_ignorar_disjuntor_deixa_o_estudo_seguir_mesmo_com_a_cauda_de_falhas(area, monkeypatch, em_processo):
    controle = {'falha_config_ids': [0, 1]}
    assert rodar(area, monkeypatch, controle=controle, extra=('--max-falhas-seguidas', '2')) == 1
    assert rodar(area, monkeypatch, controle=controle, extra=('--max-falhas-seguidas', '2', '--ignorar-disjuntor')) == 0
    informados = [e for e in jornal(area) if e['ev'] == 'informar']
    assert [e['falhou'] for e in informados[:3]] == [True, True, False] and sum(e['fre'] for e in informados) >= 1.0
    assert not os.path.exists(os.path.join(area['hpo'], 's', '_FAILED.flag'))
    # e o LAÇO também respeita a opção: falhas seguidas DURANTE a sessão não param quem pediu para ignorar (acima, o sucesso do trabalho 2 já tinha zerado a contagem)
    assert rodar(area, monkeypatch, controle=controle, study='desde o começo', extra=('--max-falhas-seguidas', '2', '--ignorar-disjuntor')) == 0
    assert [e['falhou'] for e in jornal(area, 'desde o começo') if e['ev'] == 'informar'][:3] == [True, True, False]


# ── e) trabalho.json e a escada ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('niveis', [2, 3])
def test_trabalho_json_leva_blocos_topo_hp_decodificado_impressao_e_caminhos_absolutos(area, monkeypatch, niveis):
    trocar_popen(monkeypatch, ProcessoMudo)
    monkeypatch.chdir(area['raiz'])
    c = coordenador(area, niveis=niveis, warmstart='il', referencias='referencias.json', dispositivo='cuda:0', paralelo=3)
    primeiro = pedir(c)
    registros = [dict(primeiro, id=k, nivel=k, fidelidade=c.fidelidades[k]) for k in range(niveis)]
    for registro in registros:
        c._lancar(registro)
    blocos_esperados = {2: [3, 9], 3: [1, 3, 9]}[niveis]
    for k, registro in enumerate(registros):
        d = os.path.join(c.dir, f'trabalho_{k}')
        tr = json.load(open(os.path.join(d, 'trabalho.json'), encoding='utf-8'))
        assert tr['blocos'] == blocos_esperados[k] and tr['fidelidade'] == c.fidelidades[k] and isinstance(tr['fidelidade'], int)
        assert tr['fidelidade_maxima'] is (k == niveis - 1)                           # só o topo mede validação-objetivo (e vira COMPLETE)
        assert tr['hp'] == c.espaco.para_hp(registro['vetor']) and list(tr['hp']) == list(H.ESPACO_MIMEPLANT)
        assert tr['impressao'] == H.impressao_digital(c.cab['id_execucao'], registro, [42], N, c.hash_git)
        assert os.path.isabs(tr['warmstart']) and os.path.samefile(tr['warmstart'], area['il'])
        assert os.path.isabs(tr['referencias']) and os.path.samefile(tr['referencias'], area['refs'])
        assert (tr['study'], tr['passos_max'], tr['seeds'], tr['inicializacao'], tr['smoke'], tr['dispositivo']) == ('s', N, [42], 'professor', False, 'cuda:0')
        assert all(tr[campo] == registro[campo] for campo in ('id', 'vetor', 'nivel', 'id_pai', 'bracket', 'degrau', 'origem', 'pais'))
        assert os.path.exists(os.path.join(d, 'avaliacao.log'))
    assert sorted(c.em_voo) == list(range(niveis)) and c.vagas() == 3 - niveis
    for v in c.em_voo.values():
        v['saida'].close()


def test_informar_marca_fidelidade_maxima_so_no_topo_da_escada_de_tres_niveis(area):
    c = coordenador(area, niveis=3)
    primeiro = pedir(c)
    c.dehb.pendentes.update({k: c.dehb.pendentes[0] for k in (1, 2)})                 # o núcleo só precisa achar o id; o assunto é o evento gravado
    for k in range(3):
        registro = dict(primeiro, id=k, nivel=k, fidelidade=c.fidelidades[k])
        c._informar(registro, resultado=resultado_de(c, registro, passos=c.fidelidades[k], objetivo=0.5))   # objetivo em TODO nível: o log não pode ser quem acusa
    inf = [e for e in c.jornal.eventos if e['ev'] == 'informar']
    assert [(e['nivel'], e['fidelidade'], e['fidelidade_maxima']) for e in inf] == [(0, N // 9, False), (1, N // 3, False), (2, N, True)]


def test_blocos_da_fidelidade_tem_sempre_o_bloco_de_um_nono_de_n():
    for n in (1_000_000, 900_000, 500_000, 250_000, 4 * P.PASSOS_MAXIMOS):
        assert [H.blocos_da_fidelidade(f, n) for f in NUC.escada_de_passos(n, 2)] == [3, 9], n
        assert [H.blocos_da_fidelidade(f, n) for f in NUC.escada_de_passos(n, 3)] == [1, 3, 9], n
    assert NUC.escada_de_passos(500_000, 3) == [55_555, 166_666, 500_000]             # N não divisível por 9: a fidelidade passa um pouco do múltiplo do bloco
    assert H.blocos_da_fidelidade(10, 900_000) == 1                                   # nunca zero blocos: o treino dividiria por zero
    # A escada do núcleo é por divisão inteira, então N // 3 nunca fica ABAIXO de 3 blocos e, nela, arredondar ou truncar dá no mesmo. O
    # arredondamento é o contrato com conferir_escada, que tolera 2 % para os dois lados: o que ela aceita tem de dar o MESMO nº de blocos.
    assert H.conferir_escada([299_999, 900_000], 900_000) is None and [H.blocos_da_fidelidade(f, 900_000) for f in (299_999, 300_001, 899_999)] == [3, 3, 9]


def test_conferir_escada_so_aceita_eta_3_com_dois_ou_tres_niveis():
    for n in (1_000_000, 900_000, 500_000, 4 * P.PASSOS_MAXIMOS):
        for niveis in (2, 3):
            assert H.conferir_escada(NUC.escada_de_passos(n, niveis, 3), n) is None
        with pytest.raises(SystemExit, match='não é múltiplo do bloco'):
            H.conferir_escada(NUC.escada_de_passos(n, 2, 2), n)                       # eta = 2: N // 2 são 4,5 blocos
        with pytest.raises(SystemExit, match='--eta 3 com --niveis 2 ou 3'):
            H.conferir_escada(NUC.escada_de_passos(n, 4, 3), n)                       # 4 níveis: N // 27 é um terço de bloco
    with pytest.raises(SystemExit, match='fidelidade 1000 '):
        H.conferir_escada([1000, 900_000], 900_000)                                   # "zero blocos" não é múltiplo do bloco
    with pytest.raises(SystemExit):
        H.conferir_escada([300_000, 850_000], 900_000)                                # 8,5 blocos


# ── main(): validação dos argumentos ─────────────────────────────────────────────────────────────
def test_main_recusa_argumentos_invalidos_antes_de_criar_qualquer_coisa(area, monkeypatch):
    vistos = []
    monkeypatch.setattr(H.Coordenador, 'rodar', lambda self: vistos.append(self) or 77)

    def main(*extra, study='v', n=N):
        return H.main(['--study', study, '--orcamento', str(n), '--warmstart', area['il'], '--referencias', area['refs'], *extra])
    assert main() == 77 and vistos[-1].args.paralelo == 3 and vistos[-1].args.max_tentativas == 2 and vistos[-1].fidelidades == [N // 3, N]
    for extra, mensagem in ((('--paralelo', '0'), '--paralelo'), (('--max-tentativas', '0'), '--max-tentativas'), (('--eta', '2'), 'múltiplo do bloco'),
                            (('--niveis', '4'), 'múltiplo do bloco'), (('--seeds-trial', '42', '7'), 'warm-start ausente.*seed_7'),
                            (('--referencias', area['refs'] + '.nao_existe'), 'referências inválidas'), (('--referencias', area['il']), 'referências inválidas')):
        with pytest.raises(SystemExit, match=mensagem):
            main(*extra)
    sem_objetivo = str(area['raiz'] / 'refs sem objetivo.json')
    json.dump({'oraculo_perfeito': {'validacao_rollback': {}}, 'aleatorio': {}}, open(sem_objetivo, 'w', encoding='utf-8'))
    with pytest.raises(SystemExit, match='referências inválidas'):
        main('--referencias', sem_objetivo)
    assert main('--paralelo', '1', '--max-tentativas', '1') == 77                     # 1 é o mínimo aceito
    assert main('--inicializacao', 'zero', '--warmstart', str(area['raiz'] / 'não existe')) == 77    # do zero não há clone para conferir
    assert len(vistos) == 3 and os.listdir(area['hpo']) == []                         # nenhuma recusa deixou pasta, jornal ou banco


def test_main_so_aceita_fidelidade_minima_que_cobre_o_aquecimento_do_critico(area, monkeypatch):
    vistos = []
    monkeypatch.setattr(H.Coordenador, 'rodar', lambda self: vistos.append(self) or 77)
    minimo = (H.ESPACO_MIMEPLANT['k_aquec'][2] + 1) * H.EPISODIOS_POR_COLETA * P.PASSOS_MAXIMOS
    assert minimo == 91_980

    def main(n, *extra):
        return H.main(['--study', 'v', '--orcamento', str(n), '--warmstart', area['il'], '--referencias', area['refs'], *extra])
    assert main(3 * minimo) == 77                                                     # exatamente o aquecimento + uma atualização: serve
    with pytest.raises(SystemExit, match=f'fidelidade mínima \\({minimo - 1} passos\\).*--aceitar-fidelidade-curta'):
        main(3 * minimo - 1)
    with pytest.raises(SystemExit, match='aquecimento'):
        main(9 * minimo - 9, '--niveis', '3')                                         # com 3 níveis é o N // 9 que tem de cobrir
    assert main(9 * minimo, '--niveis', '3') == 77
    assert main(3 * minimo - 1, '--aceitar-fidelidade-curta') == 77
    assert main(3 * minimo - 1, '--inicializacao', 'zero') == 77                      # o espaço do zero não tem k_aquec: basta uma coleta
    # --smoke: orçamento de brinquedo, 2 FRE, no máximo 2 em paralelo, e a regra do aquecimento não se aplica
    assert main(N, '--smoke', '--n-fre', '50') == 77
    c = vistos[-1]
    assert (c.N, c.args.n_fre, c.args.paralelo, c.fidelidades, c.args.smoke) == (4 * P.PASSOS_MAXIMOS, 2.0, 2, [486, 1460], True)
    assert main(N, '--smoke', '--paralelo', '1') == 77 and vistos[-1].args.paralelo == 1


def test_main_transforma_excecao_inesperada_em_failed_flag_e_saida_1(area, monkeypatch, capsys):
    monkeypatch.setattr(H.Coordenador, 'rodar', lambda self: (_ for _ in ()).throw(RuntimeError('defeito de programação ção')))
    assert H.main(argumentos(area)) == 1
    assert 'defeito de programação ção' in open(os.path.join(area['hpo'], 's', '_FAILED.flag'), encoding='utf-8').read()
    assert 'RuntimeError' in capsys.readouterr().err                                   # o traceback inteiro vai para o log do lançador
    monkeypatch.setattr(H.Coordenador, 'rodar', lambda self: (_ for _ in ()).throw(SystemExit('recusa deliberada')))
    with pytest.raises(SystemExit, match='recusa deliberada'):
        H.main(argumentos(area, study='t'))
    assert not os.path.exists(os.path.join(area['hpo'], 't'))                          # recusa não é falha do estudo: sem flag


# ── f) + i) INDICE.json, atributos do estudo e --reconstruir-espelho ─────────────────────────────
def estudo_de_duas_sessoes(area, monkeypatch):
    """Falha de configuração, repetição transitória, fidelidade baixa e máxima: todo tipo de trial. Duas sessões, com orçamentos diferentes."""
    controle = {'falha_config_ids': [1], 'transitoria_uma_vez_ids': [2]}
    assert rodar(area, monkeypatch, controle=controle, n_fre=1.0, k=2) == 0
    assert rodar(area, monkeypatch, controle=controle, n_fre=2.0, k=1) == 0
    return os.path.join(area['hpo'], 's')


def test_indice_liga_numero_do_trial_a_pasta_do_trabalho_e_bate_com_o_banco(area, monkeypatch, em_processo):
    d = estudo_de_duas_sessoes(area, monkeypatch)
    indice = json.load(open(os.path.join(d, 'INDICE.json'), encoding='utf-8'))
    ts, fim = trials(area), terminais(jornal(area))
    assert len(indice) == len(ts) == len(fim) > 4 and list(indice) == [str(k) for k in range(len(ts))]
    for t, e in zip(ts, fim):
        linha = indice[str(t.number)]
        assert linha == {'id_trabalho': e['id'], 'pasta': f"trabalho_{e['id']}", 'estado': t.state.name, 'valor': t.value}
        assert t.user_attrs['id_trabalho'] == e['id'] and t.user_attrs['evento_i'] == e['i'] and os.path.isdir(os.path.join(d, linha['pasta']))
    assert {v['estado'] for v in indice.values()} == {'COMPLETE', 'PRUNED', 'FAIL'}
    assert [v['id_trabalho'] for v in indice.values()].count(2) == 2                  # a tentativa falhada e a que valeu: dois trials, a MESMA pasta
    assert all((v['valor'] is not None) == (v['estado'] == 'COMPLETE') for v in indice.values())


def test_atributos_do_estudo_no_optuna_vem_do_cabecalho_e_as_sessoes_acumulam(area, monkeypatch, em_processo):
    estudo_de_duas_sessoes(area, monkeypatch)
    cab, attrs = jornal(area)[0], atributos(area)
    sessoes = attrs.pop('sessoes')
    assert attrs == json.loads(json.dumps(H.atributos_do_estudo(cab, 2.0, 1)))        # o orçamento e o K são os da ÚLTIMA sessão
    assert set(attrs) == {'sampler', 'poda', 'inicializacao', 'orcamento_passos', 'n_fre_alvo', 'seeds_trial', 'semente_dehb', 'versao_nucleo', 'espaco',
                          'paralelo', 'objetivo', 'aptidao_nas_fidelidades_menores', 'papel'}
    assert '520-549' in attrs['objetivo'] and '500-519' in attrs['aptidao_nas_fidelidades_menores'] and 'DIR_REAVALIACAO_DEHB' in attrs['papel']
    assert (attrs['sampler'], attrs['inicializacao'], attrs['orcamento_passos'], attrs['n_fre_alvo'], attrs['paralelo']) == ('dehb', 'professor', N, 2.0, 1)
    assert attrs['seeds_trial'] == [42] and attrs['semente_dehb'] == cab['semente'] and attrs['espaco'] == cab['espaco'] and attrs['versao_nucleo'] == NUC.VERSAO_NUCLEO
    assert '300000' in attrs['poda'] and 'eta=3' in attrs['poda']
    assert len(sessoes) == 2 and all(set(s) == {'data', 'hash_git', 'pid', 'n_eventos'} for s in sessoes)
    assert sessoes[0]['n_eventos'] == 2 and sessoes[1]['n_eventos'] > sessoes[0]['n_eventos']   # cabeçalho + sessão; depois, o jornal já andou


def test_reconstruir_espelho_recusa_coordenador_vivo_guarda_o_banco_velho_e_refaz_tudo_igual(area, monkeypatch, em_processo):
    d = estudo_de_duas_sessoes(area, monkeypatch)

    def foto():
        return [(t.number, t.state.name, t.value, t.params, t.user_attrs, sorted(t.intermediate_values.items()), t.datetime_start, t.datetime_complete)
                for t in trials(area)]
    trials_antes, attrs_antes, indice_antes = foto(), atributos(area), json.load(open(os.path.join(d, 'INDICE.json'), encoding='utf-8'))
    banco_antes = open(caminho_do_banco(area), 'rb').read()
    os.remove(os.path.join(d, 'INDICE.json'))
    trava = H.Trava(os.path.join(d, '_COORDENADOR.lock'))
    assert trava.tentar()
    with pytest.raises(SystemExit, match='coordenador vivo'):
        H.main(['--study', 's', '--reconstruir-espelho'])
    assert open(caminho_do_banco(area), 'rb').read() == banco_antes and not any('.anterior.' in f for f in os.listdir(area['hpo']))
    trava.soltar()
    with monkeypatch.context() as m:                                                 # defeito no meio da reconstrução: a trava volta assim mesmo
        m.setattr(H, 'Jornal', lambda caminho: (_ for _ in ()).throw(RuntimeError('jornal corrompido na linha 3')))
        with pytest.raises(RuntimeError, match='linha 3') as erro:
            H.main(['--study', 's', '--reconstruir-espelho'])
        outra = H.Trava(os.path.join(d, '_COORDENADOR.lock'))
        assert outra.tentar(), erro                                                  # com o traceback ainda vivo: foi o `finally`, não o coletor de lixo
        outra.soltar()
    assert open(caminho_do_banco(area), 'rb').read() == banco_antes
    jornal_antes = open(os.path.join(d, 'jornal.jsonl'), 'rb').read()
    assert H.main(['--study', 's', '--reconstruir-espelho']) == 0
    guardados = [f for f in os.listdir(area['hpo']) if f.startswith('s.db.anterior.')]
    assert len(guardados) == 1 and open(os.path.join(area['hpo'], guardados[0]), 'rb').read() == banco_antes
    assert foto() == trials_antes
    attrs = atributos(area)
    assert {k: v for k, v in attrs.items() if k != 'sessoes'} == {k: v for k, v in attrs_antes.items() if k != 'sessoes'}
    assert len(attrs['sessoes']) == 1 and attrs['sessoes'][0]['reconstruido_de'] == len(jornal(area))
    assert json.load(open(os.path.join(d, 'INDICE.json'), encoding='utf-8')) == indice_antes
    assert open(os.path.join(d, 'jornal.jsonl'), 'rb').read() == jornal_antes        # o espelho é projeção: refazê-lo não escreve no jornal
    outra = H.Trava(os.path.join(d, '_COORDENADOR.lock'))
    assert outra.tentar()                                                            # a trava foi devolvida
    outra.soltar()
    assert rodar(area, monkeypatch, n_fre=2.5) == 0 and len(trials(area)) == len(terminais(jornal(area)))   # e o estudo segue sobre o banco refeito
    with pytest.raises(SystemExit, match='sem jornal'):
        H.main(['--study', 'nunca existiu', '--reconstruir-espelho'])


def test_reconstruir_espelho_aceita_jornal_sem_sessao_e_devolve_1_quando_nao_reconcilia(area, monkeypatch, sem_esperas):
    c = coordenador(area)
    arq = os.path.join(c.dir, 'jornal.jsonl')
    so_o_cabecalho = open(arq, 'rb').read().split(b'\n')[0] + b'\n'
    open(arq, 'wb').write(so_o_cabecalho)                                             # caiu entre o cabeçalho e a primeira sessão
    assert H.main(['--study', 's', '--reconstruir-espelho']) == 0
    attrs = atributos(area)
    assert attrs['n_fre_alvo'] is None and attrs['paralelo'] is None and attrs['orcamento_passos'] == N and len(trials(area)) == 0
    monkeypatch.setattr(H.Espelho, 'reconciliar', lambda self, eventos: False)
    assert H.main(['--study', 's', '--reconstruir-espelho']) == 1                     # quem chama (e o lançador) tem de saber que o banco NÃO ficou bom
    assert open(arq, 'rb').read() == so_o_cabecalho


# ── g) espelho ───────────────────────────────────────────────────────────────────────────────────
HP = NUC.EspacoVetorial(H.ESPACO_MIMEPLANT).para_hp([0.5] * len(H.ESPACO_MIMEPLANT))


def evento(i, id_trabalho, ev='informar', **kw):
    base = {'i': i, 'ev': ev, 'id': id_trabalho, 'data': '2026-09-18T12:00:00.000', 'hp': HP, 'fidelidade': N // 3, 'nivel': 0, 'fidelidade_maxima': False,
            'bracket': 0, 'degrau': 0, 'origem': 'evolucao', 'falhou': False, 'aptidao': -0.4, 'objetivo': None, 'indicador': 0.4, 'passos': N // 3,
            'fre': 1 / 3, 'tempo_s': 10.0, 'hash_git': 'abc', 'curva': [{'passos': N // 9, 'proxy_norm': 0.1}, {'passos': N // 3, 'proxy_norm': 0.4}]}
    return dict(base, **kw)


def espelho(area, study='e'):
    mensagens = []
    esp = H.Espelho(study, H.ESPACO_MIMEPLANT, mensagens.append)
    esp.mensagens = mensagens
    return esp


def test_datas_do_espelho_sao_estritamente_crescentes_mesmo_com_fim_repetido_ou_voltando_no_tempo(area, sem_esperas):
    """analise_hpo.curva_melhor_ate_agora ordena por datetime_complete: a ordem das datas tem de ser a do JORNAL, inclusive com
    avaliações em paralelo que acabam no mesmo milissegundo, relógio acertado para trás e espelho reaberto noutra sessão."""
    eventos = [evento(0, 0, inicio='2026-09-18T10:00:00.000', fim='2026-09-18T10:30:00.000'),
               evento(1, 1, inicio='2026-09-18T10:00:00.000', fim='2026-09-18T10:30:00.000'),        # 'fim' repetido
               evento(2, 2, inicio='2026-09-18T09:00:00.000', fim='2026-09-18T09:20:00.000'),        # voltou no tempo
               evento(3, 3, inicio='2026-09-18T11:00:00.000', fim=None, data='2026-09-18T08:00:00.000')]   # sem 'fim': vale a data do evento
    esp = espelho(area)
    assert esp.reconciliar(eventos) is True
    esp.fechar()
    eventos += [evento(4, 4, inicio='2026-09-18T07:00:00.000', fim='2026-09-18T07:30:00.000'),        # outra sessão, relógio ainda atrás
                evento(5, 5, inicio='2026-09-18T13:00:00.000', fim='2026-09-18T13:30:00.000')]
    esp = espelho(area)
    assert esp.reconciliar(eventos) is True and esp.reconciliar(eventos) is True      # idempotente: nada a acrescentar na segunda
    esp.fechar()
    ts = trials(area, 'e')
    fins = [t.datetime_complete for t in ts]
    assert len(ts) == 6 and all(a < b for a, b in zip(fins, fins[1:])), fins
    ms = datetime.timedelta(milliseconds=1)
    base = datetime.datetime(2026, 9, 18, 10, 30)
    assert fins == [base, base + ms, base + 2 * ms, base + 3 * ms, base + 4 * ms, datetime.datetime(2026, 9, 18, 13, 30)]   # a data do PRÓPRIO evento quando ela serve
    assert all(t.datetime_start <= t.datetime_complete for t in ts)
    assert ts[0].datetime_start == datetime.datetime(2026, 9, 18, 10, 0) and ts[3].datetime_start == fins[3]              # início depois do fim é recortado


def test_espelho_projeta_cada_tipo_de_evento_no_estado_certo_do_optuna(area, sem_esperas):
    import optuna
    estados = optuna.trial.TrialState
    eventos = [evento(0, 0),
               evento(1, 1, ev='tentativa_falhou', tentativa=1, motivo='gpu_ou_sistema', fre=0.0625, passos=56_250, fidelidade_maxima=True, nivel=1, fidelidade=N),
               evento(2, 1, fidelidade_maxima=True, nivel=1, fidelidade=N, objetivo=0.75, indicador=0.6, fre=1.0, passos=N),
               evento(3, 2, falhou=True, aptidao=None, motivo='excecao_no_treino', fre=0.125, passos=112_500, fidelidade_maxima=True, objetivo=None),
               {'i': 4, 'ev': 'informar', 'id': 3, 'falhou': True, 'aptidao': None, 'motivo': 'transitoria_esgotada', 'data': '2026-09-18T12:00:00.000'},   # sem hp, sem fre
               evento(5, 4, curva=[{'passos': 1, 'proxy_norm': None}, {'passos': 2, 'proxy_norm': float('nan')}, {'passos': 3, 'proxy_norm': 0.25}, {'passos': 4}]),
               {'i': 6, 'ev': 'relancado_apos_queda', 'id': 9, 'fre_nao_cobrado': 0.1}, {'i': 7, 'ev': 'pedir', 'trabalho': {'id': 9}}]
    esp = espelho(area)
    assert esp.reconciliar(eventos) is True and esp.mensagens == []
    esp.fechar()
    t = trials(area, 'e')
    assert [x.state for x in t] == [estados.PRUNED, estados.FAIL, estados.COMPLETE, estados.FAIL, estados.FAIL, estados.PRUNED]
    assert [x.user_attrs['evento_i'] for x in t] == [0, 1, 2, 3, 4, 5] and [x.user_attrs['id_trabalho'] for x in t] == [0, 1, 1, 2, 3, 4]
    assert t[0].value is None and t[0].params == HP and t[0].user_attrs['fre'] == pytest.approx(1 / 3) and t[0].user_attrs['indicador'] == 0.4
    assert t[0].intermediate_values == {N // 9: 0.1, N // 3: 0.4} and t[0].user_attrs['pasta'] == 'trabalho_0'
    # tentativa que falhou na fidelidade MÁXIMA é FAIL (não COMPLETE), com o parcial cobrado: o FRE do estudo é a soma de fre e fre_parcial
    assert t[1].value is None and t[1].params == HP and t[1].user_attrs['fre_parcial'] == 0.0625 and 'fre' not in t[1].user_attrs
    assert (t[1].user_attrs['motivo'], t[1].user_attrs['tentativa'], t[1].user_attrs['passos']) == ('gpu_ou_sistema', 1, 56_250)
    assert t[2].value == 0.75 and t[2].user_attrs['fre'] == 1.0 and t[2].user_attrs['fidelidade'] == N
    assert t[3].user_attrs['fre_parcial'] == 0.125 and t[3].user_attrs['motivo'] == 'excecao_no_treino' and t[3].params == HP
    assert t[4].params == {} and t[4].user_attrs['fre_parcial'] == 0.0 and t[4].user_attrs['passos'] == 0                  # evento falho sem hp não derruba a projeção
    assert t[5].intermediate_values == {3: 0.25}                                      # o Optuna recusa NaN/None como valor intermediário
    assert sum(x.user_attrs.get('fre', 0.0) + x.user_attrs.get('fre_parcial', 0.0) for x in t) == pytest.approx(sum(e.get('fre', 0.0) for e in terminais(eventos)))


def test_banco_com_trial_a_mais_ou_fora_de_ordem_nao_e_reconciliado_nem_levanta_excecao(area, sem_esperas):
    import optuna
    eventos = [evento(0, 0), evento(1, 1), evento(2, 2)]
    esp = espelho(area, 'ordem')
    assert esp.reconciliar([eventos[1]]) is True                                      # o banco começa pelo evento 1: não é projeção DESTE jornal
    assert esp.reconciliar(eventos) is False
    assert len(esp.mensagens) == 1 and 'trial 0' in esp.mensagens[0] and '--reconstruir-espelho' in esp.mensagens[0]   # recusa direta: não é erro a repetir
    esp.fechar()
    assert len(trials(area, 'ordem')) == 1
    esp = espelho(area, 'sobra')
    assert esp.reconciliar(eventos) is True
    assert esp.reconciliar(eventos[:2]) is False and 'trial 2' in esp.mensagens[0]    # o banco tem um trial que o jornal não tem
    esp._abrir().add_trial(optuna.trial.create_trial(value=9.9, params={}, distributions={}))   # um trial que não veio do jornal (sem evento_i)
    assert esp.reconciliar(eventos + [evento(3, 3)]) is False and len(esp.mensagens) == 2
    esp.fechar()
    assert len(trials(area, 'sobra')) == 4                                            # nada foi acrescentado por cima do banco estranho


def test_erro_no_optuna_nunca_derruba_o_estudo_e_o_espelho_se_recupera_depois(area, monkeypatch, sem_esperas):
    import optuna
    real, estado = optuna.study.Study.add_trial, {'falhas': 10 ** 6, 'chamadas': 0}

    def add_trial(self, trial):
        estado['chamadas'] += 1
        if estado['falhas'] > 0:
            estado['falhas'] -= 1
            raise RuntimeError('database is locked')
        return real(self, trial)
    monkeypatch.setattr(optuna.study.Study, 'add_trial', add_trial)
    c = coordenador(area)
    c.espelho = H.Espelho('s', c.espaco_dict, c.mensagens.append)
    r0, r1 = pedir(c), pedir(c)
    c._informar(r0, resultado=resultado_de(c, r0, passos=N // 3))                     # não levanta
    assert estado['chamadas'] == 5 and len([m for m in c.mensagens if '[espelho] tentativa' in m and 'database is locked' in m]) == 5
    assert c.jornal.eventos[-1]['ev'] == 'informar' and not c.dehb.pendentes.get(0) and c.fre_consumidos() == pytest.approx(1 / 3)   # o jornal, que é a verdade, andou
    assert json.load(open(os.path.join(c.dir, 'PROGRESSO.json'), encoding='utf-8'))['podados'] == 1
    assert c.espelho.reconciliar(c.jornal.eventos) is False
    estado.update(falhas=1, chamadas=0)                                               # erro passageiro: uma falha, e a repetição não duplica o que já entrou
    c._informar(r1, resultado=resultado_de(c, r1, passos=N // 3))
    assert c.espelho.reconciliar(c.jornal.eventos) is True and estado['chamadas'] == 3
    c.espelho.fechar()
    assert [t.user_attrs['evento_i'] for t in trials(area)] == [e['i'] for e in terminais(c.jornal.eventos)]
    # os outros dois acessos ao banco também engolem o erro
    monkeypatch.setattr(H.Espelho, '_abrir', lambda self: (_ for _ in ()).throw(RuntimeError('banco ilegível')))
    assert c.espelho.indice() == {}
    c.espelho.atributos_do_estudo({'a': 1}, {'pid': 1})
    assert 'atributos do estudo não gravados' in c.mensagens[-1] and 'banco ilegível' in c.mensagens[-1]


def test_estudo_do_optuna_maximiza_e_os_trials_levam_as_distribuicoes_do_espaco_a_curva_e_a_posicao_na_escada(area, sem_esperas):
    """O que `reavaliar_topk` e `analise_hpo` leem do banco: best_trial é o de MAIOR objetivo; lr em escala log e n_epocas inteiro (é por
    elas que a importância de HP se orienta); COMPLETE e FAIL também levam a curva parcial."""
    import optuna
    eventos = [evento(0, 0, fidelidade_maxima=True, nivel=1, fidelidade=N, objetivo=0.25, fre=1.0, passos=N),
               evento(1, 1, fidelidade_maxima=True, nivel=1, fidelidade=N, objetivo=0.75, fre=1.0, passos=N),
               evento(2, 2, falhou=True, aptidao=None, motivo='excecao_no_treino', fre=0.125, passos=112_500),
               evento(3, 3, ev='tentativa_falhou', tentativa=1, motivo='gpu_ou_sistema', fre=0.0625, bracket=4, degrau=1, origem='promocao', hash_git='outro')]
    esp = espelho(area)
    assert esp.reconciliar(eventos) is True and esp.mensagens == []
    esp.fechar()
    st = estudo_do_optuna(area, 'e')
    assert st.direction == optuna.study.StudyDirection.MAXIMIZE and (st.best_value, st.best_trial.number) == (0.75, 1)
    H.liberar_banco(st)
    ts = trials(area, 'e')
    assert [t.intermediate_values for t in ts] == [{N // 9: 0.1, N // 3: 0.4}] * 4    # a curva parcial é o que sobra de uma falha
    posicao = [(t.user_attrs['bracket'], t.user_attrs['degrau'], t.user_attrs['origem'], t.user_attrs['hash_git']) for t in ts[2:]]
    assert posicao == [(0, 0, 'evolucao', 'abc'), (4, 1, 'promocao', 'outro')]        # a do EVENTO (e o hash_git de quem treinou, não o desta sessão)
    for t in ts:
        assert list(t.distributions) == list(H.ESPACO_MIMEPLANT)
        for nome, spec in H.ESPACO_MIMEPLANT.items():
            d = t.distributions[nome]
            if spec[0] in ('log', 'float'):
                assert isinstance(d, optuna.distributions.FloatDistribution) and d.log is (spec[0] == 'log') and (d.low, d.high) == (spec[1], spec[2]), nome
            elif spec[0] == 'int':
                assert isinstance(d, optuna.distributions.IntDistribution) and (d.low, d.high) == (spec[1], spec[2]) and isinstance(t.params[nome], int), nome
            else:
                assert isinstance(d, optuna.distributions.CategoricalDistribution) and list(d.choices) == list(spec[1]), nome
    assert {spec[0] for spec in H.ESPACO_MIMEPLANT.values()} == {'log', 'float', 'int', 'cat'}   # senão algum ramo acima não foi exercitado


def test_espelho_que_desistiu_nao_fica_segurando_o_banco(area, monkeypatch, sem_esperas):
    import optuna
    monkeypatch.setattr(optuna.study.Study, 'add_trial', lambda self, trial: (_ for _ in ()).throw(RuntimeError('database is locked')))
    esp = espelho(area, 'preso')
    assert esp.reconciliar([evento(0, 0)]) is False and len(esp.mensagens) == 5
    try:
        os.replace(esp.caminho, esp.caminho + '.movido')                             # é o que o --reconstruir-espelho faz; no Windows só dá com o .db fechado
    except PermissionError as e:
        pytest.fail(f'o espelho desistiu mas deixou o banco aberto: {e}')
    assert esp._study is None                                                         # fora do Windows renomear banco aberto funciona: confere-se a conexão


# ── h) jornal ────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def arq_jornal(tmp_path, sem_esperas):
    pasta = tmp_path / 'Área de Trabalho' / 'jornal ção'
    pasta.mkdir(parents=True)
    return str(pasta / 'jornal.jsonl')


def copias_do_corrompido(arq):
    return sorted(os.path.join(os.path.dirname(arq), f) for f in os.listdir(os.path.dirname(arq)) if '.corrompido.' in f)


def test_jornal_estragado_e_copiado_com_os_bytes_originais_antes_de_ser_truncado(arq_jornal):
    j = H.Jornal(arq_jornal)
    for k in range(3):
        j.acrescentar({'ev': 'x', 'k': k, 'texto': 'avaliação'})
    inteiro = open(arq_jornal, 'rb').read()
    assert j._tamanho == len(inteiro) and H.Jornal(arq_jornal)._tamanho == len(inteiro) and copias_do_corrompido(arq_jornal) == []   # íntegro: sem cópia
    duas = inteiro[:inteiro.rstrip(b'\n').rfind(b'\n') + 1]
    for estrago, sobra in ((inteiro[:-7], duas), (inteiro + b'\x00' * 64, inteiro), (inteiro + b'{"ev": "x", "i": 3', inteiro), (duas + b'{lixo}\n', duas)):
        open(arq_jornal, 'wb').write(estrago)
        j = H.Jornal(arq_jornal)
        copias = copias_do_corrompido(arq_jornal)
        assert len(copias) == 1 and open(copias[0], 'rb').read() == estrago           # a perícia precisa do arquivo como a queda o deixou
        assert open(arq_jornal, 'rb').read() == sobra and j._tamanho == len(sobra) and len(j.eventos) == sobra.count(b'\n')
        os.remove(copias[0])


def test_acrescimo_repetido_depois_de_erro_de_es_deixa_exatamente_uma_copia_da_linha(arq_jornal, monkeypatch):
    j = H.Jornal(arq_jornal)
    j.acrescentar({'ev': 'a'})
    real, estado = os.fsync, {'falhas': 1}

    def fsync(fd):
        if estado['falhas'] > 0:
            estado['falhas'] -= 1
            raise OSError('WinError 5')                                               # a linha JÁ foi escrita; só a confirmação falhou
        return real(fd)
    with monkeypatch.context() as m:
        m.setattr(H, 'os', ModuloCom(os, fsync=fsync))
        ev = j.acrescentar({'ev': 'b', 'texto': 'ção'})
    linhas = open(arq_jornal, 'rb').read().split(b'\n')
    assert [json.loads(ln) for ln in linhas[:-1]] == [{'ev': 'a', 'i': 0}, {'ev': 'b', 'texto': 'ção', 'i': 1}] and linhas[-1] == b''
    assert ev == j.eventos[1] and len(j.eventos) == 2 and j._tamanho == os.path.getsize(arq_jornal) and estado['falhas'] == 0
    j.acrescentar({'ev': 'c'})                                                        # a repetição não deixou o jornal "desconfiado" de si mesmo
    assert [e['i'] for e in H.Jornal(arq_jornal).eventos] == [0, 1, 2] and copias_do_corrompido(arq_jornal) == []


class _EscritaPelaMetade:
    """Arquivo cuja primeira escrita grava só a metade dos bytes e levanta OSError (disco cheio, antivírus no meio do caminho)."""

    def __init__(self, f, estado):
        self._f, self._estado = f, estado

    def write(self, dados):
        if self._estado['falhas'] > 0:
            self._estado['falhas'] -= 1
            self._f.write(dados[:len(dados) // 2])
            self._f.flush()
            raise OSError('escrita interrompida')
        return self._f.write(dados)

    def __getattr__(self, nome):
        return getattr(self._f, nome)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._f.close()


def test_escrita_cortada_pela_metade_e_refeita_por_cima_sem_colar_registros(arq_jornal, monkeypatch):
    j = H.Jornal(arq_jornal)
    j.acrescentar({'ev': 'a'})
    estado = {'falhas': 1}
    with monkeypatch.context() as m:
        m.setattr(H, 'open', lambda caminho, modo='r', *a, **kw: _EscritaPelaMetade(open(caminho, modo, *a, **kw), estado)
                  if 'r+' in modo or 'w' in modo else open(caminho, modo, *a, **kw), raising=False)
        j.acrescentar({'ev': 'b', 'longo': 'x' * 200})
    assert estado['falhas'] == 0 and j._tamanho == os.path.getsize(arq_jornal)
    assert [json.loads(ln) for ln in open(arq_jornal, 'rb').read().split(b'\n')[:-1]] == j.eventos == [{'ev': 'a', 'i': 0}, {'ev': 'b', 'longo': 'x' * 200, 'i': 1}]


def test_erro_de_es_persistente_levanta_e_o_acrescimo_seguinte_nao_acusa_outro_escritor(arq_jornal, monkeypatch):
    j = H.Jornal(arq_jornal)
    j.acrescentar({'ev': 'a'})
    tamanho, chamadas = j._tamanho, []
    with monkeypatch.context() as m:
        m.setattr(H, 'os', ModuloCom(os, fsync=lambda fd: chamadas.append(fd) or (_ for _ in ()).throw(OSError('WinError 1117'))))
        with pytest.raises(OSError, match='1117'):
            j.acrescentar({'ev': 'longo', 'texto': 'y' * 300}, tentativas=3)
    assert len(chamadas) == 3 and len(j.eventos) == 1 and j._tamanho == tamanho       # não gravou: nem em memória, nem no tamanho conhecido
    assert os.path.getsize(arq_jornal) > tamanho                                      # o resto da NOSSA escrita ficou no disco...
    ev = j.acrescentar({'ev': 'curto'})                                               # ...e não pode ser confundido com outro coordenador, nem sobrar atrás da linha nova
    assert ev['i'] == 1 and open(arq_jornal, 'rb').read() == b'{"ev": "a", "i": 0}\n{"ev": "curto", "i": 1}\n'
    assert j._tamanho == os.path.getsize(arq_jornal) and [e['ev'] for e in H.Jornal(arq_jornal).eventos] == ['a', 'curto']
    assert copias_do_corrompido(arq_jornal) == []


def test_bytes_de_outro_processo_no_jornal_abortam_o_acrescimo_sem_intercalar(arq_jornal):
    j = H.Jornal(arq_jornal)
    j.acrescentar({'ev': 'a'})
    j.acrescentar({'ev': 'b'})
    with open(arq_jornal, 'ab') as f:
        f.write(b'{"ev": "intruso", "i": 2}\n')
    com_intruso = open(arq_jornal, 'rb').read()
    with pytest.raises(RuntimeError, match='cresceu fora deste processo'):
        j.acrescentar({'ev': 'c'})
    assert open(arq_jornal, 'rb').read() == com_intruso and len(j.eventos) == 2       # nem truncou a linha do outro, nem escreveu a sua
    with pytest.raises(RuntimeError, match='cresceu fora deste processo'):
        j.acrescentar({'ev': 'c'})                                                    # e continua recusando: a primeira recusa não "legaliza" o resto
    assert open(arq_jornal, 'rb').read() == com_intruso


def test_linha_completa_estragada_so_e_cauda_cortada_se_nada_alem_de_nuls_vem_depois_dela(arq_jornal):
    """Uma queda estraga o FIM do arquivo. Linha completa ruim SEGUIDA de mais bytes é corrupção no meio: o jornal não pode ser
    'consertado' jogando fora o que vem depois. Já os NULs que o NTFS deixa na cauda não são 'mais bytes': a linha ruim antes deles é cauda."""
    j = H.Jornal(arq_jornal)
    for k in range(3):
        j.acrescentar({'ev': 'x', 'k': k})
    inteiro = open(arq_jornal, 'rb').read()
    estrago = inteiro + b'{lixo}\n' + b'{"ev": "x", "i": 4'
    open(arq_jornal, 'wb').write(estrago)
    with pytest.raises(RuntimeError, match='jornal corrompido na linha 4'):
        H.Jornal(arq_jornal)
    assert open(arq_jornal, 'rb').read() == estrago and copias_do_corrompido(arq_jornal) == []   # recusou sem tocar em nada
    um, dois, tres = inteiro.splitlines(keepends=True)
    open(arq_jornal, 'wb').write(um + tres + dois)                                    # JSON válido, índice fora de ordem: evento perdido ou arquivos misturados
    with pytest.raises(RuntimeError, match='jornal corrompido na linha 2 .*índice 2 onde se esperava 1'):
        H.Jornal(arq_jornal)
    assert open(arq_jornal, 'rb').read() == um + tres + dois and copias_do_corrompido(arq_jornal) == []
    estrago = inteiro + b'{"ev": "x", "i": 3, "k"\x00\x00\n' + b'\x00' * 32
    open(arq_jornal, 'wb').write(estrago)
    j = H.Jornal(arq_jornal)
    copias = copias_do_corrompido(arq_jornal)
    assert len(j.eventos) == 3 and j._tamanho == len(inteiro) and open(arq_jornal, 'rb').read() == inteiro
    assert len(copias) == 1 and open(copias[0], 'rb').read() == estrago


def test_jornal_que_encolheu_fora_deste_processo_aborta_o_acrescimo_em_vez_de_abrir_um_buraco_de_nuls(arq_jornal):
    """OneDrive restaurando uma versão anterior, alguém truncando o arquivo: os eventos que só existem em memória já não estão no disco,
    e escrever 'a partir do tamanho conhecido' deixa linha 1 + NULs + evento novo — que a próxima leitura descarta calada como cauda."""
    j = H.Jornal(arq_jornal)
    for k in range(3):
        j.acrescentar({'ev': 'x', 'k': k})
    inteiro = open(arq_jornal, 'rb').read()
    encolhido = inteiro[:inteiro.index(b'\n') + 1]
    open(arq_jornal, 'wb').write(encolhido)
    with pytest.raises(RuntimeError, match='fora deste processo'):
        j.acrescentar({'ev': 'y'})
    assert open(arq_jornal, 'rb').read() == encolhido and len(j.eventos) == 3


def test_valor_nao_finito_e_recusado_sem_tocar_no_arquivo_nem_na_memoria(arq_jornal):
    j = H.Jornal(arq_jornal)
    j.acrescentar({'ev': 'a'})
    antes = open(arq_jornal, 'rb').read()
    for ruim in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(ValueError):
            j.acrescentar({'ev': 'x', 'aninhado': {'v': [1.0, ruim]}})
    assert open(arq_jornal, 'rb').read() == antes and len(j.eventos) == 1 and j._tamanho == len(antes)
    assert j.acrescentar({'ev': 'b', 'i': 99})['i'] == 1                              # o índice é do jornal, não de quem chama
    assert [e['i'] for e in H.Jornal(arq_jornal).eventos] == [0, 1]


def test_escrita_atomica_nao_estraga_o_arquivo_bom_quando_falha_e_desiste_depois_das_tentativas(tmp_path, monkeypatch, sem_esperas):
    alvo = str(tmp_path / 'PROGRESSO ção.json')
    H.escrever_json_atomico(alvo, {'a': 'ção', 'n': 1})
    assert json.load(open(alvo, encoding='utf-8')) == {'a': 'ção', 'n': 1} and os.listdir(tmp_path) == ['PROGRESSO ção.json']   # sem .tmp esquecido
    with pytest.raises(ValueError):
        H.escrever_json_atomico(alvo, {'muita': 'coisa antes', 'v': float('nan')})    # JSON que o leitor estrito recusaria nunca chega ao destino
    assert json.load(open(alvo, encoding='utf-8')) == {'a': 'ção', 'n': 1}
    chamadas = []
    monkeypatch.setattr(H, 'os', ModuloCom(os, replace=lambda a, b: chamadas.append((a, b)) or (_ for _ in ()).throw(PermissionError('WinError 32'))))
    with pytest.raises(PermissionError):
        H.escrever_json_atomico(alvo, {'n': 2}, tentativas=3)
    assert len(chamadas) == 3 and all(b == alvo and a != alvo for a, b in chamadas) and json.load(open(alvo, encoding='utf-8')) == {'a': 'ção', 'n': 1}


# ── PROGRESSO.json ───────────────────────────────────────────────────────────────────────────────
def test_progresso_conta_cada_tipo_de_evento_e_o_melhor_e_o_maior_objetivo_na_fidelidade_maxima(area):
    c = coordenador(area, n_fre=8.0)
    hp = {k: {'lr': k} for k in range(8)}
    for k, (origem, pais) in enumerate([('evolucao', {'pool': 3, 'global': 0, 'uniforme': 1}), ('promocao', {}), ('evolucao', {'pool': 2, 'global': 2, 'uniforme': 0})]):
        c.jornal.acrescentar({'ev': 'pedir', 'trabalho': {'id': k, 'origem': origem, 'pais': pais}})
    for k, (topo, falhou, objetivo, fre) in enumerate([(True, False, 0.2, 1.0), (False, False, None, 0.25), (True, False, 0.7, 1.0), (True, True, None, 0.5),
                                                        (False, True, None, 0.125), (True, False, 0.5, 1.0)]):
        c.jornal.acrescentar({'ev': 'informar', 'id': k, 'fidelidade_maxima': topo, 'falhou': falhou, 'objetivo': objetivo, 'fre': fre, 'hp': hp[k]})
    c.jornal.acrescentar({'ev': 'tentativa_falhou', 'id': 6, 'tentativa': 1, 'fre': 0.0625, 'fidelidade_maxima': True})
    c.jornal.acrescentar({'ev': 'relancado_apos_queda', 'id': 7, 'passos_perdidos': 1, 'fre_nao_cobrado': 0.3004})
    c.jornal.acrescentar({'ev': 'relancado_apos_queda', 'id': 8, 'passos_perdidos': 1, 'fre_nao_cobrado': 0.2})
    c.em_voo, c.adotados, c._fila = {9: {}, 4: {}}, {6: object()}, [{'id': 11}, {'id': 10}]
    prog = c.progresso()
    assert (prog['concluidos'], prog['podados'], prog['falhas'], prog['tentativas_falhadas']) == (3, 1, 2, 1)
    assert prog['fre_consumidos'] == 3.938 and prog['fre_perdido_em_quedas_nao_cobrado'] == 0.5 and prog['fre_alvo'] == 8.0   # 3,9375 cobrados; o perdido em queda à parte
    assert prog['melhor'] == {'id_trabalho': 2, 'valor': 0.7, 'params': {'lr': 2}} and prog['melhor_ate_agora'] == [0.2, 0.7, 0.7]
    assert prog['em_voo'] == [4, 6, 9] and prog['na_fila'] == [11, 10]                 # a fila na ordem de lançamento; em voo inclui os órfãos adotados
    assert prog['promocoes'] == 1 and prog['pais_por_origem'] == {'pool': 5, 'global': 2, 'uniforme': 1}
    assert (prog['study'], prog['sampler'], prog['fidelidades'], prog['brackets']) == ('s', 'dehb', [N // 3, N], 0)
    assert json.load(open(os.path.join(c.dir, 'PROGRESSO.json'), encoding='utf-8')) == prog


def test_progresso_sem_avaliacao_completa_nao_tem_melhor_e_disco_preso_nao_derruba(area, monkeypatch):
    c = coordenador(area)
    c.jornal.acrescentar({'ev': 'informar', 'id': 0, 'fidelidade_maxima': True, 'falhou': True, 'objetivo': None, 'fre': 0.5})
    monkeypatch.setattr(H, 'escrever_json_atomico', lambda *a, **kw: (_ for _ in ()).throw(PermissionError('WinError 32')))
    prog = c.progresso()
    assert 'melhor' not in prog and 'melhor_ate_agora' not in prog and prog['falhas'] == 1
    assert 'PROGRESSO.json não gravado' in c.mensagens[-1] and not os.path.exists(os.path.join(c.dir, 'PROGRESSO.json'))


# ── j) manter a máquina acordada ─────────────────────────────────────────────────────────────────
def test_rodar_pede_a_maquina_acordada_e_devolve_o_pedido_mesmo_com_excecao_no_meio(area, monkeypatch):
    """Devolve o pedido ao Windows E a trava do estudo, com exceção ou sem. O quadro de rodar() é guardado VIVO de propósito: sem isso
    o coletor de lixo fecha o arquivo da Trava quando o quadro morre, e um `finally` sem o soltar() passaria despercebido aqui — e
    penduraria a retomada no mesmo processo em 'aguardando: outro coordenador deste estudo está vivo'."""
    chamadas, quadros, plano = [], [], {'levanta': True}
    monkeypatch.setattr(H, 'manter_a_maquina_acordada', chamadas.append)
    c = coordenador(area, abrir=False)

    def no_lugar_do_estudo():
        chamadas.append('rodando')
        quadros.append(sys._getframe(1))
        if plano['levanta']:
            raise RuntimeError('defeito no meio do estudo')
        return 5
    monkeypatch.setattr(c, '_rodar_com_trava', no_lugar_do_estudo)
    trava = H.Trava(os.path.join(c.dir, '_COORDENADOR.lock'))
    with pytest.raises(RuntimeError, match='defeito no meio') as erro:
        c.rodar()
    assert chamadas == [True, 'rodando', False] and quadros[-1].f_code.co_name == 'rodar'
    assert trava.tentar(), erro                                                      # a exceção também não pode deixar a trava do estudo presa
    trava.soltar()
    del chamadas[:]
    plano['levanta'] = False
    assert c.rodar() == 5 and chamadas == [True, 'rodando', False] and [q.f_code.co_name for q in quadros] == ['rodar', 'rodar']
    assert trava.tentar()                                                            # e o retorno normal também a devolve
    trava.soltar()


def test_manter_a_maquina_acordada_liga_e_desliga_o_pedido_e_nunca_levanta(monkeypatch):
    if os.name == 'nt':
        import ctypes
        chamadas = []
        monkeypatch.setattr(ctypes.windll.kernel32, 'SetThreadExecutionState', lambda flags: chamadas.append(flags), raising=False)
        H.manter_a_maquina_acordada(True)
        H.manter_a_maquina_acordada(False)
        assert chamadas == [0x80000001, 0x80000000]                                   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED; depois só ES_CONTINUOUS (devolve)
        monkeypatch.setattr(ctypes.windll.kernel32, 'SetThreadExecutionState', lambda flags: (_ for _ in ()).throw(OSError('API indisponível')), raising=False)
        assert H.manter_a_maquina_acordada(True) is None
        del chamadas[:]
        monkeypatch.setattr(ctypes.windll.kernel32, 'SetThreadExecutionState', lambda flags: chamadas.append(flags), raising=False)
    with monkeypatch.context() as m:
        m.setattr(H, 'os', ModuloCom(os, name='posix'))                               # fora do Windows não há API nenhuma: não faz nada e não falha
        assert H.manter_a_maquina_acordada(True) is None and H.manter_a_maquina_acordada(False) is None
    if os.name == 'nt':
        assert chamadas == []


# ── k) partida e aborto ──────────────────────────────────────────────────────────────────────────
def test_failed_flag_de_sessao_anterior_e_removida_na_partida(area, monkeypatch, em_processo):
    d = os.path.join(area['hpo'], 's')
    P.escrever_flag(d, '_FAILED.flag', 'erro de E/S persistente na sessão anterior')
    assert rodar(area, monkeypatch, n_fre=0.3) == 0
    assert not os.path.exists(os.path.join(d, '_FAILED.flag')) and os.path.exists(os.path.join(d, '_DONE.flag'))   # senão o validador reportaria a falha velha para sempre


def test_banco_do_optuna_sem_jornal_e_resto_de_outro_estudo_e_nada_e_gravado(area):
    """Sem isso um estudo novo projetaria os seus trials por cima do banco de um estudo alheio (mesmo nome, pasta apagada à mão)."""
    open(caminho_do_banco(area), 'wb').write(b'SQLite format 3\x00')
    with pytest.raises(SystemExit, match=r"restos de outro estudo \(\['s\.db'\]"):
        coordenador(area)
    assert os.listdir(os.path.join(area['hpo'], 's')) == []                           # nem jornal, nem cabeçalho


def test_jornal_vazio_nao_e_estudo_nem_para_exigir_retomar_nem_para_esconder_restos(area):
    d = os.path.join(area['hpo'], 's')
    os.makedirs(d)
    open(os.path.join(d, 'jornal.jsonl'), 'wb').close()                               # caiu entre criar o arquivo e gravar o cabeçalho
    c = coordenador(area, retomar=False)                                              # não há estudo nenhum ali: não exige --retomar
    assert [e['ev'] for e in c.jornal.eventos] == ['cabecalho', 'sessao'] == [e['ev'] for e in jornal(area)]
    d = os.path.join(area['hpo'], 't')
    os.makedirs(os.path.join(d, 'trabalho_0'))
    open(os.path.join(d, 'jornal.jsonl'), 'wb').close()
    with pytest.raises(SystemExit, match='restos de outro estudo'):
        coordenador(area, study='t')
    assert os.path.getsize(os.path.join(d, 'jornal.jsonl')) == 0                      # e não gravou cabeçalho por cima dos restos


def test_partida_poe_o_espelho_em_dia_antes_de_qualquer_outra_coisa(area, monkeypatch, em_processo):
    """Queda entre o jornal e o espelho: o jornal tem um evento terminal que o banco não tem. A sessão seguinte aborta no disjuntor JÁ na
    partida — nenhum _informar acontece nela —, e mesmo assim o banco tem de sair com o trial: quem o põe lá é a reconciliação da partida."""
    c = coordenador(area)                                                             # espelho MUDO: o jornal anda e o banco nem existe
    c._informar(pedir(c), falha={'passos': 1000, 'motivo': 'excecao_no_treino'})
    assert not os.path.exists(caminho_do_banco(area))
    assert rodar(area, monkeypatch, extra=('--max-falhas-seguidas', '1')) == 1
    assert [e['ev'] for e in jornal(area)].count('informar') == 1
    ts = trials(area)
    assert [(t.state.name, t.user_attrs['evento_i']) for t in ts] == [('FAIL', c.jornal.eventos[-1]['i'])]


def test_estudo_pronto_tambem_confere_o_cabecalho_antes_de_dizer_que_nao_ha_nada_a_fazer(area):
    c = coordenador(area, n_fre=0.0)
    P.escrever_flag(c.dir, '_DONE.flag', 'fre=0.0')
    antes = open(os.path.join(c.dir, 'jornal.jsonl'), 'rb').read()
    assert H.Coordenador(args_de(area, n_fre=0.0), log=lambda m: None).abrir() is False
    with pytest.raises(SystemExit, match='--retomar com semente diferente'):          # "pronto" vale para o MESMO estudo: outro, com o mesmo nome, não sai com rc 0
        coordenador(area, n_fre=0.0, seed=c.semente + 1)
    assert open(os.path.join(c.dir, 'jornal.jsonl'), 'rb').read() == antes


def test_estudo_que_termina_com_o_espelho_incompleto_nao_ganha_done_flag(area, monkeypatch, sem_esperas):
    """A _DONE.flag libera a reavaliação, que lê o BANCO: se a reconciliação final não fechou, o estudo acaba em _FAILED.flag pedindo o
    --reconstruir-espelho, e o INDICE.json e o PROGRESSO.json saem assim mesmo."""
    class EspelhoQueNaoFecha(EspelhoMudo):
        def reconciliar(self, eventos):
            super().reconciliar(eventos)
            return False
    espelhos = []
    monkeypatch.setattr(H.Coordenador, '_ciclo', lambda self: True)
    monkeypatch.setattr(H, 'Espelho', lambda *a: espelhos.append(EspelhoQueNaoFecha()) or espelhos[-1])
    c = H.Coordenador(args_de(area), log=lambda m: None)
    assert c.rodar() == 1 and espelhos[0].chamadas == [2, 2]                          # reconciliou na partida E no fim (cabeçalho + sessão)
    assert 'reconstruir-espelho' in open(os.path.join(c.dir, '_FAILED.flag'), encoding='utf-8').read() and not os.path.exists(os.path.join(c.dir, '_DONE.flag'))
    assert os.path.exists(os.path.join(c.dir, 'INDICE.json')) and os.path.exists(os.path.join(c.dir, 'PROGRESSO.json'))


def test_erro_de_es_no_ciclo_e_repetido_e_so_aborta_o_estudo_se_for_persistente(area, monkeypatch, sem_esperas):
    """WinError 5/32 (antivírus, OneDrive) dura segundos: o laço insiste, e a contagem ZERA a cada ciclo bom — 45 erros espalhados em
    dias de estudo não são defeito. 40 SEGUIDOS são: aborta com _FAILED.flag em vez de girar para sempre."""
    roteiro, mensagens = [], []

    def ciclo(self):
        passo = roteiro.pop(0)
        if passo == 'erro':
            raise PermissionError('WinError 32')
        return passo == 'fim'
    monkeypatch.setattr(H.Coordenador, '_ciclo', ciclo)
    monkeypatch.setattr(H, 'Espelho', lambda *a: EspelhoMudo())                       # o Optuna não é o assunto

    def estudo(study, passos):
        roteiro[:] = passos
        del mensagens[:]
        c = H.Coordenador(args_de(area, study=study), log=mensagens.append)
        return c.rodar(), c.dir
    rc, d = estudo('espalhados', ['erro', 'segue'] * 45 + ['fim'])
    assert rc == 0 and roteiro == [] and os.path.exists(os.path.join(d, '_DONE.flag')) and not os.path.exists(os.path.join(d, '_FAILED.flag'))
    assert sum('erro de E/S' in m for m in mensagens) == 45 and all('(1/40)' in m for m in mensagens if 'erro de E/S' in m)
    rc, d = estudo('quase', ['erro'] * 39 + ['fim'])
    assert rc == 0 and roteiro == [] and os.path.exists(os.path.join(d, '_DONE.flag'))
    rc, d = estudo('persistente', ['erro'] * 40 + ['fim'])
    assert rc == 1 and roteiro == ['fim']                                             # parou NO 40º erro seguido
    assert 'erro de E/S persistente' in open(os.path.join(d, '_FAILED.flag'), encoding='utf-8').read() and not os.path.exists(os.path.join(d, '_DONE.flag'))


def test_estudo_pronto_relancado_nao_escreve_nada_nem_com_alvo_igual_ao_gasto(area, monkeypatch, em_processo):
    assert rodar(area, monkeypatch, n_fre=1.0) == 0
    gasto = sum(e['fre'] for e in jornal(area) if e['ev'] == 'informar')
    antes = foto_da_pasta(area['hpo'])
    assert '_DONE.flag' in {os.path.basename(k) for k in antes}
    assert rodar(area, monkeypatch, n_fre=1.0) == 0 and rodar(area, monkeypatch, n_fre=gasto) == 0    # alvo IGUAL ao gasto também é "pronto"
    assert foto_da_pasta(area['hpo']) == antes                                        # nem sessão no jornal, nem atributo no banco, nem PROGRESSO


def test_abrir_devolve_false_no_estudo_pronto_e_remove_a_done_flag_quando_o_orcamento_cresce(area, monkeypatch, em_processo):
    assert rodar(area, monkeypatch, n_fre=1.0) == 0
    d = os.path.join(area['hpo'], 's')
    gasto = sum(e['fre'] for e in jornal(area) if e['ev'] == 'informar')
    n_eventos = len(jornal(area))
    c = H.Coordenador(args_de(area, n_fre=gasto), log=lambda m: None)
    assert c.abrir() is False and len(jornal(area)) == n_eventos and os.path.exists(os.path.join(d, '_DONE.flag'))
    c = H.Coordenador(args_de(area, n_fre=gasto + 0.01), log=lambda m: None)
    assert c.abrir() is True and not os.path.exists(os.path.join(d, '_DONE.flag'))    # sai JÁ na partida: estudo em andamento não pode parecer concluído
    assert jornal(area)[-1]['ev'] == 'sessao' and jornal(area)[-1]['n_fre'] == gasto + 0.01 and len(jornal(area)) == n_eventos + 1
    assert rodar(area, monkeypatch, n_fre=2.0) == 0                                   # e o estudo continua até o alvo novo, e a flag volta
    assert sum(e['fre'] for e in jornal(area) if e['ev'] == 'informar') >= 2.0 and os.path.exists(os.path.join(d, '_DONE.flag'))


def test_sessao_gravada_no_jornal_leva_orcamento_paralelismo_e_versao(area, monkeypatch):
    c = coordenador(area, n_fre=3.5, paralelo=2)
    s = c.jornal.eventos[-1]
    assert (s['ev'], s['n_fre'], s['paralelo'], s['pid'], s['hash_git'], s['versao_nucleo']) == ('sessao', 3.5, 2, os.getpid(), c.hash_git, NUC.VERSAO_NUCLEO)
    cab = c.jornal.eventos[0]
    assert (cab['ev'], cab['study'], cab['eta'], cab['F'], cab['CR'], cab['versao_nucleo']) == ('cabecalho', 's', 3, NUC.F_PADRAO, NUC.CR_PADRAO, NUC.VERSAO_NUCLEO)
    assert len(cab['id_execucao']) == 32 and coordenador(area, study='outro').cab['id_execucao'] != cab['id_execucao']   # cada estudo é uma encarnação: mesmo nome relançado do zero ≠ mesma impressão
    assert coordenador(area).cab['id_execucao'] == cab['id_execucao']                 # a retomada herda a do jornal
    monkeypatch.setattr(H, 'DIR_HPO', str(area['raiz'] / 'hpo de novo'))              # a pasta foi limpa e o MESMO nome recomeça do zero:
    assert coordenador(area).cab['id_execucao'] != cab['id_execucao']                 # resultado.json da encarnação anterior não pode servir


def test_abortar_mata_todos_os_filhos_em_voo_e_escreve_a_failed_flag(area):
    c = coordenador(area)
    filhos = [ProcessoMudo(), ProcessoMudo(quebra_no_kill=True), ProcessoMudo()]
    c.em_voo = {k: {'proc': p, 'registro': {'id': k}, 'saida': None} for k, p in enumerate(filhos)}
    assert c._abortar('defeito sistêmico ção ' + 'x' * 400) == 1
    assert filhos[0].morto and filhos[2].morto                                        # um filho que já morreu não poupa os outros
    flag = open(os.path.join(c.dir, '_FAILED.flag'), encoding='utf-8').read()
    assert 'defeito sistêmico ção' in flag and len(flag) < 340 and any(m.startswith('ABORTADO: defeito sistêmico') for m in c.mensagens)
    assert not os.path.exists(os.path.join(c.dir, '_DONE.flag'))


# ── correções de 19/09/2026 que a revisão do coordenador (interrompida) não chegou a travar ───────
@pytest.mark.parametrize('x, esperado', [(0.5, True), (3, True), (-2.0, True), (0, True), (float('nan'), False), (float('inf'), False),
                                         (-float('inf'), False), (None, False), ('0.5', False), (True, False), (False, False), ([1.0], False),
                                         ({'v': 1}, False), (10 ** 400, False), (-(10 ** 400), False)],
                         ids=['meio', 'tres', 'menos_dois', 'zero', 'nan', 'inf', 'menos_inf', 'None', 'texto', 'True', 'False', 'lista', 'dict',
                              'inteiro_gigante', 'menos_inteiro_gigante'])
def test_numero_finito_so_aceita_numero_real_finito_e_nunca_levanta(x, esperado):
    """O que vem de um JSON em disco pode ser null, texto, lista, bool ou um inteiro que não cabe em float (json.loads devolve int do
    Python): np.isfinite levantaria TypeError, que não é OSError — a sessão cairia, e de novo a cada retomada."""
    assert H.numero_finito(x) is esperado


def test_resultado_aproveitavel_exige_objetivo_no_topo_e_indicador_embaixo():
    base = {'aptidao': -0.5, 'passos': 300_000, 'fre': 1 / 3, 'objetivo': 0.5, 'indicador': 0.4}
    assert H.resultado_aproveitavel(base, True) and H.resultado_aproveitavel(base, False)
    assert not H.resultado_aproveitavel(dict(base, objetivo=None), True) and H.resultado_aproveitavel(dict(base, objetivo=None), False)
    assert not H.resultado_aproveitavel(dict(base, indicador=float('nan')), False) and H.resultado_aproveitavel(dict(base, indicador=None), True)
    for chave in ('aptidao', 'passos', 'fre'):
        assert not H.resultado_aproveitavel({k: v for k, v in base.items() if k != chave}, True), chave
    assert not H.resultado_aproveitavel(None, True) and not H.resultado_aproveitavel([base], True)


def test_jornal_que_sumiu_fora_deste_processo_aborta_o_acrescimo_sem_renascer_com_nuls(arq_jornal):
    """Apagado (ou movido pelo OneDrive) com o coordenador vivo: escrever 'a partir do tamanho conhecido' faria um arquivo NOVO começando
    com um buraco de NULs e sem cabeçalho — e o próximo abrir() gravaria um cabeçalho novo por cima da história."""
    j = H.Jornal(arq_jornal)
    j.acrescentar({'ev': 'a'})
    j.acrescentar({'ev': 'b'})
    os.remove(arq_jornal)
    with pytest.raises(RuntimeError, match=r'encolheu \(ou sumiu\)'):
        j.acrescentar({'ev': 'c'})
    assert not os.path.exists(arq_jornal) and len(j.eventos) == 2


def test_escrita_atomica_nao_deixa_temporario_nem_quando_o_json_e_recusado_nem_quando_o_destino_fica_preso(tmp_path, monkeypatch, sem_esperas):
    alvo = str(tmp_path / 'resultado ção.json')
    with pytest.raises(ValueError):
        H.escrever_json_atomico(alvo, {'curva': [1.0, float('nan')]})
    assert os.listdir(tmp_path) == [], 'meio JSON do valor recusado ficou na pasta'
    monkeypatch.setattr(H, 'os', ModuloCom(os, replace=lambda a, b: (_ for _ in ()).throw(PermissionError('WinError 32'))))
    with pytest.raises(PermissionError):
        H.escrever_json_atomico(alvo, {'n': 2}, tentativas=3)
    assert os.listdir(tmp_path) == [], 'o temporário sobrou depois de o destino ficar preso até o fim'
