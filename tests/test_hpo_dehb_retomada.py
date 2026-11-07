"""Testes de RETOMADA e do laço do coordenador do DEHB (`hpo/hpo_dehb.py`): write-ahead, adoção do estado, partida (cabeçalho,
flags, orçamento exato), órfãos, vigia de batimento em tempo ativo, fila de lançamento, disjuntor, FRE dos parciais, tolerância a
erro de E/S e os ramos de término de um avaliador.

Quase todos dirigem o objeto `Coordenador` À MÃO (um `_ciclo`/`_colher` por vez), com processos de mentira, espelho de mentira e
relógio controlado: nada de GPU, de Optuna nem de espera em tempo real. Só o Job Object e o órfão usam processos de verdade.
Todos os caminhos têm espaço e acento, como a pasta real do projeto."""
import os
import sys
import json
import time
import types
import threading
import subprocess

import numpy as np
import pytest

import protocolo as P

sys.path.insert(0, os.path.join(P.RAIZ, 'hpo'))
import hpo_dehb as H        # noqa: E402
import dehb_nucleo as NUC   # noqa: E402

FALSO = os.path.join(P.RAIZ, 'tests', '_avaliador_falso_dehb.py')
N = 900_000                 # fidelidades 300.000 (1/3 FRE) e 900.000 (1 FRE)
TERMINAIS = ('informar', 'tentativa_falhou')


# ── bancada: pasta do estudo, processos de mentira, espelho de mentira, relógio ──────────────────
@pytest.fixture
def area(tmp_path, monkeypatch):
    """DIR_HPO temporário + precondições mínimas, em caminho com acento e espaço. O hash do git vira constante: ele entra na
    impressão digital, e chamar o git a cada Coordenador custaria mais que o teste."""
    raiz = tmp_path / 'Área de Trabalho' / 'estudo dehb'
    hpo = raiz / 'hpo'
    hpo.mkdir(parents=True)
    (raiz / 'il' / 'seed_42').mkdir(parents=True)
    (raiz / 'il' / 'seed_42' / 'modelo_final.pt').write_bytes(b'x')
    refs = {k: {c: {r: {'media': m} for r in P.NOMES_REGIMES} for c in ('validacao_rollback', 'validacao_objetivo', 'teste')}
            for k, m in (('oraculo_perfeito', 5000.0), ('aleatorio', 600.0))}
    (raiz / 'referencias.json').write_text(json.dumps(refs), encoding='utf-8')
    monkeypatch.setattr(H, 'DIR_HPO', str(hpo))
    monkeypatch.setattr(P, 'hash_git', lambda: 'git-do-teste')
    monkeypatch.delenv('DEHB_FALSO', raising=False)
    return {'hpo': str(hpo), 'il': str(raiz / 'il'), 'refs': str(raiz / 'referencias.json'), 'raiz': raiz}


def argumentos(area, **troca):
    base = dict(study='s', orcamento=N, niveis=2, eta=3, seed=5, inicializacao='professor', seeds_trial=[42], smoke=False, retomar=True,
                n_fre=3.0, paralelo=2, warmstart=area['il'], referencias=area['refs'], dispositivo=None, max_tentativas=2,
                max_falhas_seguidas=4, ignorar_disjuntor=False, limite_sem_batimento_h=1.0, intervalo_s=5.0)
    return H.argparse.Namespace(**dict(base, **troca))


class ProcFalso:
    """Avaliador de mentira (no lugar do Popen): o ROTEIRO do teste decide o que ele deixa em disco e com que código termina."""

    def __init__(self, dir_trabalho, saida):
        self.dir, self.saida, self.id = dir_trabalho, saida, int(os.path.basename(dir_trabalho).split('_')[1])
        self.pid, self.rc, self.mortes, self.esperas_que_estouram = 40_000 + self.id, None, 0, 0

    def poll(self):
        return self.rc

    def kill(self):
        self.mortes += 1

    def wait(self, timeout=None):
        if self.esperas_que_estouram:
            self.esperas_que_estouram -= 1
            raise subprocess.TimeoutExpired('avaliador-de-mentira', timeout)
        self.rc = -9 if self.rc is None else self.rc
        return self.rc


class EspelhoFalso:
    """O espelho no Optuna é de outro arquivo de teste; aqui só interessa QUANDO ele é chamado e com o quê."""

    def __init__(self, *a, **k):
        self.chamadas = []

    def atributos_do_estudo(self, *a, **k):
        pass

    def reconciliar(self, eventos):
        self.chamadas.append(len(eventos))
        return True

    def indice(self):
        return {}

    def fechar(self):
        pass


def gravar_resultado(dir_trabalho, **troca):
    """O resultado.json que o avaliador gravaria (mesma função sintética do avaliador falso), com a impressão do trabalho.json."""
    tr = json.load(open(os.path.join(dir_trabalho, 'trabalho.json'), encoding='utf-8'))
    valor = -float(sum((v - 0.3) ** 2 for v in tr['vetor']))
    res = dict({'impressao': tr['impressao'], 'id': tr['id'], 'aptidao': -valor, 'objetivo': valor if tr['fidelidade_maxima'] else None,
                'indicador': valor, 'passos': int(tr['fidelidade']), 'fre': tr['fidelidade'] / tr['passos_max'], 'inicio': H.agora_iso(),
                'fim': H.agora_iso(), 'tempo_s': 0.0, 'curva': [], 'hash_git': 'falso'}, **troca)
    with open(os.path.join(dir_trabalho, 'resultado.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f)                                   # à mão: NaN não passa pelo escritor atômico (allow_nan=False)
    return res


def gravar_falha(dir_trabalho, motivo, passos, **troca):
    tr = json.load(open(os.path.join(dir_trabalho, 'trabalho.json'), encoding='utf-8'))
    H.escrever_json_atomico(os.path.join(dir_trabalho, 'falha.json'),
                            dict({'impressao': tr['impressao'], 'id': tr['id'], 'motivo': motivo, 'detalhe': 'de mentira', 'passos': passos,
                                  'inicio': H.agora_iso(), 'fim': H.agora_iso(), 'tempo_s': 0.0, 'hash_git': 'falso'}, **troca))


def gravar_batimento(dir_trabalho, passos, mtime):
    """_BATIMENTO.json com mtime EXPLÍCITO: dois batimentos no mesmo tique do sistema de arquivos pareceriam um só."""
    arq = os.path.join(dir_trabalho, '_BATIMENTO.json')
    H.escrever_json_atomico(arq, {'data': H.agora_iso(), 'pid': 1, 'passos': passos})
    os.utime(arq, (mtime, mtime))


def termina_bem(proc):
    gravar_resultado(proc.dir)
    proc.rc = 0


def pendurado(proc):
    pass


@pytest.fixture
def falsos(monkeypatch):
    """Troca, SÓ dentro do hpo_dehb, o subprocess por processos de mentira e o Espelho por um de mentira. `roteiro(proc)` roda
    no lançamento; `popen_levanta` é a fila de exceções que os próximos Popen levantam (arquivo preso, falta de memória...).
    O `sleep` do hpo_dehb também deixa de dormir de verdade: diante de OSError repetido o laço de produção dorme até 30 s, 40
    vezes (~13 min), e uma regressão viraria suíte PENDURADA no validador em vez de teste vermelho. O resto do relógio é o real."""
    estado = types.SimpleNamespace(lancados=[], roteiro=termina_bem, popen_levanta=[], saidas=[], sonecas=[])

    def dormir(s):
        estado.sonecas.append(s)
        assert len(estado.sonecas) < 20_000, 'o laço do coordenador não terminou'
        time.sleep(min(s, 0.01))                            # um respiro real e curto: as novas tentativas de E/S da produção contam com ele
    monkeypatch.setattr(H, 'time', types.SimpleNamespace(monotonic=time.monotonic, sleep=dormir, time=time.time, strftime=time.strftime))

    def popen(cmd, **kw):
        estado.saidas.append(kw.get('stdout'))
        if estado.popen_levanta:
            raise estado.popen_levanta.pop(0)
        proc = ProcFalso(cmd[-1], kw.get('stdout'))
        estado.lancados.append(proc)
        estado.roteiro(proc)
        return proc
    estado.ids = lambda: [p.id for p in estado.lancados]
    monkeypatch.setattr(H, 'subprocess', types.SimpleNamespace(Popen=popen, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT,
                                                               TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(H, 'prender_filhos_ao_coordenador', lambda: (lambda proc: None))
    monkeypatch.setattr(H, 'Espelho', EspelhoFalso)
    return estado


@pytest.fixture
def relogio(monkeypatch, falsos):
    """Troca, SÓ dentro do hpo_dehb, o módulo time: `monotonic` é do teste, `sleep` não dorme (e acusa laço que não termina).
    Depende de `falsos` para valer por cima do relógio de lá, seja qual for a ordem das fixtures na assinatura do teste."""
    r = types.SimpleNamespace(agora=1000.0, sonecas=[])

    def dormir(s):
        r.sonecas.append(s)
        assert len(r.sonecas) < 20_000, 'o laço do coordenador não terminou'
    monkeypatch.setattr(H, 'time', types.SimpleNamespace(monotonic=lambda: r.agora, sleep=dormir, time=time.time, strftime=time.strftime))
    return r


def sessao(area, **troca):
    """Uma sessão aberta à mão: `abrir()` (replay + evento 'sessao') e espelho de mentira. O laço é do teste."""
    c = H.Coordenador(argumentos(area, **troca), comando_avaliador=lambda d: ['avaliador-de-mentira', d], log=lambda m: None)
    assert c.abrir()
    c.espelho = EspelhoFalso()
    return c


def rodar_sessao(area, **troca):
    """A sessão inteira pelo caminho de produção (`_rodar_com_trava`), com os processos de mentira terminando na hora. Só com a
    fixture `falsos` (que também tira o sono do laço): sem ela o Popen e as sonecas de até 30 s seriam os de verdade."""
    assert not hasattr(H.subprocess, 'run'), 'rodar_sessao sem a fixture falsos lançaria processos e dormiria de verdade'
    c = H.Coordenador(argumentos(area, **dict({'intervalo_s': 0.0}, **troca)), comando_avaliador=lambda d: ['avaliador-de-mentira', d], log=lambda m: None)
    return c, c._rodar_com_trava()


def derrubar(c):
    """A sessão 'cai': o objeto é abandonado com o que estava em voo. Só fecha os logs, que no Windows prenderiam a pasta."""
    for v in c.em_voo.values():
        v['saida'].close()


def no_arquivo(area, study='s'):
    """Os eventos como estão NO ARQUIVO do jornal (a verdade que sobrevive a uma queda), não na memória do coordenador."""
    arq = os.path.join(area['hpo'], study, 'jornal.jsonl')
    return [json.loads(ln) for ln in open(arq, encoding='utf-8')] if os.path.exists(arq) else []


def do_tipo(area, tipo, study='s'):
    return [e for e in no_arquivo(area, study) if e['ev'] == tipo]


def terminais(area, study='s'):
    return [e for e in no_arquivo(area, study) if e['ev'] in TERMINAIS]


def unico(itens, o_que):
    """O único item da lista. Desempacotar direto, `(ev,) = ...`, falharia com 'not enough values to unpack', que não diz nada."""
    assert len(itens) == 1, f'esperava exatamente um {o_que}, veio {len(itens)}: {itens!r}'[:900]
    return itens[0]


def dir_trabalho(area, id_t, study='s'):
    return os.path.join(area['hpo'], study, f'trabalho_{id_t}')


def jornal_que_falha(monkeypatch, tipos, erro=None):
    """`Jornal.acrescentar` levanta OSError na próxima gravação de cada tipo de evento da lista `tipos` (consumida em ordem de
    chegada): é o que sobra para o coordenador quando as 8 tentativas internas do jornal se esgotam com o arquivo preso."""
    real, pendentes = H.Jornal.acrescentar, list(tipos)

    def acrescentar(self, evento, *a, **k):
        if evento.get('ev') in pendentes:
            pendentes.remove(evento['ev'])
            raise erro or OSError('[WinError 32] o arquivo está sendo usado por outro processo')
        return real(self, evento, *a, **k)
    monkeypatch.setattr(H.Jornal, 'acrescentar', acrescentar)
    return pendentes


def abrir_preso(monkeypatch, nome, vezes):
    """O open() do hpo_dehb levanta PermissionError nas `vezes` primeiras LEITURAS do arquivo `nome` (antivírus, OneDrive)."""
    estado = {'n': 0, 'vezes': vezes}

    def abrir(caminho, *a, **k):
        if os.path.basename(str(caminho)) == nome and not a:
            estado['n'] += 1
            if estado['n'] <= estado['vezes']:
                raise PermissionError(13, 'WinError 32: arquivo preso', str(caminho))
        return open(caminho, *a, **k)
    monkeypatch.setattr(H, 'open', abrir, raising=False)
    return estado


# ── a) write-ahead ───────────────────────────────────────────────────────────────────────────────
def test_pedir_ja_esta_no_arquivo_do_jornal_quando_o_comando_do_avaliador_e_montado(area, falsos):
    """Se o processo nascesse antes do 'pedir' gravado, uma queda nesse intervalo deixaria um treino de horas que o replay não
    conhece (e cujo resultado seria recusado). A fábrica do comando é o último ponto antes do Popen: lê o ARQUIVO de lá."""
    c = sessao(area, paralelo=3)
    falsos.roteiro, visto = pendurado, []

    def comando(d):
        visto.append((int(os.path.basename(d).split('_')[1]), [e['trabalho']['id'] for e in do_tipo(area, 'pedir')]))
        return ['avaliador-de-mentira', d]
    c.comando_avaliador = comando
    assert c._ciclo() is False
    assert [i for i, _ in visto] == [0, 1, 2]
    for id_t, gravados in visto:
        assert gravados == list(range(id_t + 1)), f'trabalho {id_t} lançado com o jornal em {gravados}: pedir → jornal → processo, um a um'
    derrubar(c)


def test_informar_ja_esta_no_arquivo_do_jornal_quando_o_nucleo_e_o_espelho_sao_avisados(area, falsos):
    """A ordem inversa (aplicar e depois gravar) faria uma queda no meio deixar o núcleo/espelho adiante do jornal, que é a verdade."""
    c = sessao(area, paralelo=1)
    c._ciclo()
    visto, informar_real = {'espelho': []}, c.dehb.informar

    def informar(id_t, aptidao):
        visto['nucleo'] = [e['id'] for e in do_tipo(area, 'informar')]
        return informar_real(id_t, aptidao)

    def reconciliar(eventos):
        visto['espelho'].append(([e['id'] for e in do_tipo(area, 'informar')], eventos[-1]['ev'], eventos[-1]['i']))
        return True
    c.dehb.informar, c.espelho.reconciliar = informar, reconciliar
    c._colher()
    assert visto['nucleo'] == [0], 'dehb.informar chamado antes de o evento estar no arquivo do jornal'
    assert visto['espelho'] == [([0], 'informar', no_arquivo(area)[-1]['i'])], 'espelho reconciliado antes do jornal, ou sem o evento novo (TODAS as chamadas contam)'
    assert [t.id for t in c.dehb.historico] == [0] and 0 not in c.dehb.pendentes


def test_tentativa_falhou_so_conta_e_so_volta_para_a_fila_depois_de_gravada(area, falsos, monkeypatch):
    """OSError ao gravar 'tentativa_falhou': nada pode ter sido aplicado (contador, fila), senão a repetição no ciclo seguinte contaria
    a mesma tentativa duas vezes — ou relançaria um trabalho que continua em voo."""
    c = sessao(area, paralelo=1)
    falsos.roteiro = lambda p: (gravar_falha(p.dir, 'gpu_ou_sistema', 75_000), setattr(p, 'rc', H.SAIDA_FALHA_TRANSITORIA))
    c._ciclo()
    jornal_que_falha(monkeypatch, ['tentativa_falhou'])
    with pytest.raises(OSError):
        c._colher()
    assert c.tentativas == {} and c._fila == [] and list(c.em_voo) == [0] and not do_tipo(area, 'tentativa_falhou')
    c._colher()                                                                      # o ciclo seguinte trata o MESMO término, uma vez
    assert [(e['id'], e['tentativa']) for e in do_tipo(area, 'tentativa_falhou')] == [(0, 1)]
    assert c.tentativas == {0: 1} and [r['id'] for r in c._fila] == [0] and not c.em_voo


# ── b) adoção do estado ──────────────────────────────────────────────────────────────────────────
def pedidos_e_informes(area, study):
    ev = no_arquivo(area, study)
    return ([(e['trabalho']['id'], e['trabalho']['vetor'], e['trabalho']['origem'], e['trabalho']['id_pai'], e['trabalho']['nivel']) for e in ev if e['ev'] == 'pedir'],
            [(e['id'], e['aptidao']) for e in ev if e['ev'] == 'informar'])


def reescrever_jornal(area, study, eventos):
    with open(os.path.join(area['hpo'], study, 'jornal.jsonl'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in eventos))


def test_adocao_restaura_o_gerador_e_os_pedidos_seguintes_sao_os_do_estudo_original(area, falsos):
    """A divergência é provocada num campo que é só RASTRO do pedido ('pais'): o estado adotado é idêntico ao original, então —
    se (e só se) o gerador for restaurado do último pedido — a busca continua bit a bit igual à de um estudo direto de mesma
    semente. Com 2 níveis a fila de promoção está sempre vazia entre pedidos, e a adoção é exata."""
    assert rodar_sessao(area, study='direto', paralelo=1, n_fre=5.0)[1] == 0
    assert rodar_sessao(area, study='partes', paralelo=1, n_fre=2.0)[1] == 0
    ev = no_arquivo(area, 'partes')
    alvo = next(e for e in ev if e['ev'] == 'pedir' and e['trabalho']['id'] == 1)
    alvo['trabalho']['pais'] = {'pool': 99, 'global': 0, 'uniforme': 0}                # como se outra versão do núcleo tivesse pedido
    reescrever_jornal(area, 'partes', ev)
    c, rc = rodar_sessao(area, study='partes', paralelo=1, n_fre=3.0)
    assert rc == 0 and [(e['no_evento'], e['versao_nucleo']) for e in do_tipo(area, 'divergencia', 'partes')] == [(alvo['i'], NUC.VERSAO_NUCLEO)]
    for n_fre in (4.0, 5.0):                                                           # as retomadas seguintes já entram adotando
        assert rodar_sessao(area, study='partes', paralelo=1, n_fre=n_fre)[1] == 0
    assert len(do_tipo(area, 'divergencia', 'partes')) == 1, 'uma divergência por OCORRÊNCIA, não uma por retomada'
    direto, partes = pedidos_e_informes(area, 'direto'), pedidos_e_informes(area, 'partes')
    assert len(partes[0]) > 4 and any(o == 'evolucao' and nivel == 1 for _, _, o, _, nivel in partes[0][4:])   # houve sorteio depois da adoção
    assert partes == direto, 'depois da adoção os pedidos deixaram de ser os do estudo original: o gerador não voltou ao último pedido'
    assert [e['i'] for e in no_arquivo(area, 'partes')] == list(range(len(no_arquivo(area, 'partes'))))


def test_jornal_de_outra_versao_do_nucleo_e_adotado_sem_sortear_e_a_busca_continua_igual(area, falsos, monkeypatch):
    """Outra VERSAO_NUCLEO no cabeçalho = a sequência de sorteios pode ter mudado: o replay nem tenta repetir os pedidos."""
    assert rodar_sessao(area, study='direto', paralelo=1, n_fre=4.0)[1] == 0
    assert rodar_sessao(area, study='velho', paralelo=1, n_fre=2.0)[1] == 0
    ev = no_arquivo(area, 'velho')
    ev[0]['versao_nucleo'] = '0.9'
    reescrever_jornal(area, 'velho', ev)
    os.remove(os.path.join(area['hpo'], 'velho', '_DONE.flag'))
    sorteios, pedir_real = [], NUC.Dehb.pedir
    monkeypatch.setattr(NUC.Dehb, 'pedir', lambda self: sorteios.append(self._proximo_id) or pedir_real(self))
    c = H.Coordenador(argumentos(area, study='velho', paralelo=1, n_fre=4.0), log=lambda m: None)
    assert c.abrir() and sorteios == [], f'o replay de um jornal de outra versão do núcleo sorteou os pedidos {sorteios}'
    assert c.dehb.assinatura()['pedidos'] == 4 and not c.dehb.pendentes
    assert rodar_sessao(area, study='velho', paralelo=1, n_fre=4.0)[1] == 0
    assert not do_tipo(area, 'divergencia', 'velho')                                   # não houve divergência: houve versão diferente
    assert pedidos_e_informes(area, 'velho') == pedidos_e_informes(area, 'direto')


# ── b2) partida: cabeçalho, flags e orçamento exato ──────────────────────────────────────────────
@pytest.mark.parametrize('chave, troca', [('semente', {'seed': 6}), ('espaco', {'inicializacao': 'zero'}), ('N', {'orcamento': N + 9}), ('fidelidades', {'niveis': 3}),
                                          ('seeds_trial', {'seeds_trial': [42, 43]}), ('smoke', {'smoke': True})])
def test_retomar_com_outro_cabecalho_e_recusado_dizendo_qual_chave_mudou(area, falsos, chave, troca):
    """Outra semente do DEHB = replay que diverge e estado ADOTADO em silêncio com o gerador de outro estudo; outras sementes de
    TREINO = avaliações incomparáveis no mesmo estudo (e outro denominador do FRE). 'eta' e 'inicializacao' não têm caso próprio:
    mudam sempre junto com 'fidelidades' e 'espaco', que são conferidos antes."""
    sessao(area)
    tamanho = len(no_arquivo(area))
    with pytest.raises(SystemExit, match=f'--retomar com {chave} diferente do estudo gravado.*outro estudo'):
        H.Coordenador(argumentos(area, **troca), log=lambda m: None).abrir()
    assert len(no_arquivo(area)) == tamanho, 'a retomada recusada escreveu no jornal do estudo alheio'


def test_orcamento_gasto_exato_nao_pede_mais_nada_nao_reabre_e_so_um_orcamento_maior_tira_a_done_flag(area, falsos):
    """K=1, n_fre=1.0: três avaliações de 1/3 FRE somam EXATAMENTE 1.0. As três fronteiras (pedir com '<', concluído com '>=',
    _DONE.flag removida quando o estudo continua) são de igualdade: cada uma com a sua frase."""
    assert rodar_sessao(area, paralelo=1, n_fre=1.0)[1] == 0
    feito = os.path.join(area['hpo'], 's', '_DONE.flag')
    assert len(do_tipo(area, 'pedir')) == 3, 'com o orçamento gasto EXATO (3 x 1/3 = 1.0 FRE) ainda se pediu mais um trabalho'
    assert sum(e['fre'] for e in do_tipo(area, 'informar')) == 1.0 and os.path.exists(feito)
    tamanho = len(no_arquivo(area))
    c = H.Coordenador(argumentos(area, paralelo=1, n_fre=1.0), log=lambda m: None)
    assert c.abrir() is False, 'estudo concluído com o FRE exato foi reaberto: relançar um estudo pronto tem de ser inofensivo'
    assert len(no_arquivo(area)) == tamanho and os.path.exists(feito)
    c = H.Coordenador(argumentos(area, paralelo=1, n_fre=2.0), log=lambda m: None)
    assert c.abrir() is True
    assert not os.path.exists(feito), 'estudo reaberto com a _DONE.flag de pé: o validador e a cadeia o dariam por concluído no meio'


def test_failed_flag_de_sessao_anterior_some_na_partida(area, falsos):
    d = os.path.join(area['hpo'], 's')
    P.escrever_flag(d, '_FAILED.flag', 'de ontem')
    assert rodar_sessao(area, paralelo=1, n_fre=0.3)[1] == 0 and os.path.exists(os.path.join(d, '_DONE.flag'))
    assert not os.path.exists(os.path.join(d, '_FAILED.flag')), 'a _FAILED.flag de ontem ficou: o validador reportaria falha num estudo concluído'


# ── c) Job Object e órfãos ───────────────────────────────────────────────────────────────────────
CODIGO_DO_COORDENADOR = ("import sys, os, json, time; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s'); import hpo_dehb as H; H.DIR_HPO = r'%s'; "
                         "c = H.Coordenador(H.argparse.Namespace(**json.loads(sys.argv[1])), comando_avaliador=lambda d: [sys.executable, '-X', 'utf8', r'%s', d], log=lambda m: None); "
                         "c.abrir(); c._ciclo(); print(json.dumps([v['proc'].pid for v in c.em_voo.values()]), flush=True); time.sleep(900)")


def esperar(condicao, prazo_s, o_que):
    """Espera por CONDIÇÃO, com prazo folgado (a máquina pode estar carregada); nunca um sleep fixo seguido de assert."""
    fim = time.monotonic() + prazo_s
    while not condicao():
        assert time.monotonic() < fim, f'{o_que} (prazo de {prazo_s} s)'
        time.sleep(0.05)


def primeira_linha(proc, prazo_s, o_que):
    """A 1ª linha do stdout de um processo auxiliar, com PRAZO: um readline() puro penduraria a suíte para sempre se o auxiliar
    travasse (não morresse: travasse) antes de falar. Estourado o prazo, o auxiliar é morto e o teste falha com o stderr dele."""
    caixa = []
    fio = threading.Thread(target=lambda: caixa.append(proc.stdout.readline()), daemon=True)
    fio.start()
    fio.join(prazo_s)
    if fio.is_alive() or not (caixa and caixa[0].strip()):
        proc.kill()                                          # fecha os canos: o readline acima e o read abaixo voltam
        fio.join(30)
        pytest.fail(f'{o_que} (prazo de {prazo_s} s); stderr do auxiliar: {proc.stderr.read()[-1500:]}')
    return caixa[0]


def trava_livre(caminho):
    t = H.Trava(caminho)
    livre = t.tentar()
    t.soltar()
    return livre


@pytest.mark.skipif(os.name != 'nt', reason='Job Object é do Windows; fora dele a defesa é só a trava _EXECUTANDO.lock')
def test_filhos_morrem_sozinhos_quando_o_processo_do_coordenador_e_morto(area):
    """Sem o Job Object, um coordenador morto deixa K treinos de horas na GPU e o relançamento dispara outros K. Os avaliadores
    (de verdade, o falso) dormiriam 15 min: só o KILL_ON_JOB_CLOSE explica a trava deles soltar em segundos."""
    env = dict(os.environ, DEHB_FALSO=json.dumps({'dorme_s': 900}), PYTHONUTF8='1')
    codigo = CODIGO_DO_COORDENADOR % (os.path.join(P.RAIZ, 'hpo'), P.RAIZ, area['hpo'], FALSO)
    coord = subprocess.Popen([sys.executable, '-X', 'utf8', '-c', codigo, json.dumps(vars(argumentos(area, paralelo=2)))], env=env, cwd=P.RAIZ,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    pids = []
    try:
        pids = json.loads(primeira_linha(coord, 120, 'o coordenador auxiliar não subiu nem lançou os filhos'))
        assert len(pids) == 2
        # cada avaliador fala no PRÓPRIO log depois de tomar a trava: daí em diante, trava livre = processo morto. (O 'contador' do
        # avaliador falso não serve aqui: dois processos que nascem juntos acrescentando ao mesmo arquivo perdem linha no Windows.)
        def comecou(i):
            log = os.path.join(dir_trabalho(area, i), 'avaliacao.log')
            return os.path.exists(log) and 'avaliador falso' in open(log, encoding='utf-8', errors='replace').read()
        esperar(lambda: comecou(0) and comecou(1), 120, 'os avaliadores não começaram')
        travas = [os.path.join(dir_trabalho(area, i), '_EXECUTANDO.lock') for i in (0, 1)]
        assert not any(trava_livre(t) for t in travas)
        coord.kill()
        coord.wait(timeout=60)
        esperar(lambda: all(trava_livre(t) for t in travas), 60, 'os filhos sobreviveram à morte do coordenador: não estavam presos ao Job Object')
    finally:
        coord.kill()
        coord.wait(timeout=30)
        for pid in pids:                                     # só sobra alguém aqui se o teste falhou: não deixar python dormindo 15 min
            subprocess.run(['taskkill', '/F', '/PID', str(pid)], capture_output=True)
        coord.stdout.close(); coord.stderr.close()


CODIGO_DO_ORFAO = ("import sys, os, json; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s'); import hpo_dehb as H; d = sys.argv[1]; "
                   "t = H.Trava(os.path.join(d, '_EXECUTANDO.lock')); assert t.tentar(); print('trava tomada', flush=True); sys.stdin.readline(); "
                   "tr = json.load(open(os.path.join(d, 'trabalho.json'), encoding='utf-8')); "
                   "H.escrever_json_atomico(os.path.join(d, 'resultado.json'), {'impressao': tr['impressao'], 'id': tr['id'], 'aptidao': 0.4242, 'objetivo': None, "
                   "'indicador': -0.4242, 'passos': tr['fidelidade'], 'fre': tr['fidelidade'] / tr['passos_max'], 'inicio': H.agora_iso(), 'fim': H.agora_iso(), "
                   "'tempo_s': 1.0, 'curva': [], 'hash_git': 'orfao'})")


def test_orfao_vivo_e_adotado_nao_duplicado_ocupa_vaga_e_o_resultado_dele_e_informado(area, falsos):
    """O Job Object falhou e um avaliador da sessão anterior continua vivo (um PROCESSO de verdade segura a trava). A sessão nova
    não o relança, conta a vaga dele, não dá o estudo por terminado enquanto ele vive e informa o resultado que ELE gravar."""
    falsos.roteiro = pendurado
    c1 = sessao(area, paralelo=2, n_fre=0.3)
    c1._ciclo()
    assert sorted(c1.em_voo) == [0, 1]
    derrubar(c1)
    orfao = subprocess.Popen([sys.executable, '-X', 'utf8', '-c', CODIGO_DO_ORFAO % (os.path.join(P.RAIZ, 'hpo'), P.RAIZ), dir_trabalho(area, 0)],
                             cwd=P.RAIZ, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
                             env=dict(os.environ, PYTHONUTF8='1'))
    try:
        assert primeira_linha(orfao, 120, 'o órfão auxiliar não tomou a trava').strip() == 'trava tomada'
        c2 = sessao(area, paralelo=2, n_fre=0.3)
        antes = len(falsos.lancados)
        c2._retomar_pendentes()
        assert sorted(c2.adotados) == [0] and [r['id'] for r in c2._fila] == [1], 'o órfão vivo tem de ser ADOTADO, e só o outro pendente relançado'
        assert [e['id'] for e in do_tipo(area, 'relancado_apos_queda')] == [1]
        assert c2.vagas() == 1, 'o adotado ocupa uma vaga da GPU'
        falsos.roteiro = termina_bem
        assert c2._ciclo() is False and c2._ciclo() is False, 'estudo dado por terminado com um órfão adotado ainda vivo'
        assert falsos.ids()[antes:] == [1] and len(do_tipo(area, 'pedir')) == 2         # nem duplicata do órfão, nem pedido além de K
        assert sorted(c2.adotados) == [0] and [e['id'] for e in no_arquivo(area) if e['ev'] in TERMINAIS] == [1], 'o órfão foi tratado ainda vivo'
        orfao.stdin.write('pode gravar\n'); orfao.stdin.flush()
        assert orfao.wait(timeout=90) == 0, orfao.stderr.read()[-1500:]
        assert c2._ciclo() is True
        assert [(e['id'], e['aptidao'], e['falhou']) for e in do_tipo(area, 'informar')][-1] == (0, 0.4242, False)
        assert falsos.ids()[antes:] == [1] and not c2.adotados and c2.vagas() == 2
    finally:
        orfao.kill()
        orfao.wait(timeout=30)
        for f in (orfao.stdin, orfao.stdout, orfao.stderr):
            f.close()


def test_orfao_que_morre_sem_resultado_vira_tentativa_falhada_cobrada_e_volta_para_a_fila(area, falsos):
    falsos.roteiro = pendurado
    c1 = sessao(area, paralelo=1)
    c1._ciclo()
    derrubar(c1)
    gravar_batimento(dir_trabalho(area, 0), 60_000, mtime=1_700_000_000)
    arq_trava = os.path.join(dir_trabalho(area, 0), '_EXECUTANDO.lock')
    do_orfao = H.Trava(arq_trava)                           # no Windows e no POSIX a trava vale entre dois descritores do MESMO processo
    assert do_orfao.tentar()
    c2 = sessao(area, paralelo=1)
    c2._retomar_pendentes()
    c2._colher()
    assert sorted(c2.adotados) == [0] and not terminais(area)
    assert c2.progresso()['em_voo'] == [0], 'PROGRESSO.json diz GPU livre com um órfão adotado treinando nela'
    do_orfao.soltar()                                       # o órfão morreu: o SO soltou a trava, e não há resultado.json
    c2._colher()
    ev = unico(do_tipo(area, 'tentativa_falhou'), "'tentativa_falhou' no jornal")
    assert (ev['id'], ev['motivo'], ev['codigo_saida'], ev['passos'], ev['tentativa']) == (0, 'processo_morreu', None, 60_000, 1)
    assert not c2.adotados and [r['id'] for r in c2._fila] == [0] and c2.vagas() == 1
    assert trava_livre(arq_trava), 'o coordenador ficou com a trava do trabalho: o avaliador relançado sairia com "outro processo já executa"'


def test_oserror_ao_tratar_o_termino_de_um_orfao_nao_o_deixa_adotado_para_sempre(area, falsos, monkeypatch):
    """O órfão terminou com resultado válido e o jornal falha ao gravar o 'informar'. O adotado continua adotado e o ciclo seguinte
    tem de conseguir sondar a trava de novo: se o coordenador tivesse ficado com ela, o estudo nunca mais terminaria."""
    falsos.roteiro = pendurado
    c1 = sessao(area, paralelo=1, n_fre=0.3)
    c1._ciclo()
    derrubar(c1)
    do_orfao = H.Trava(os.path.join(dir_trabalho(area, 0), '_EXECUTANDO.lock'))
    assert do_orfao.tentar()
    c2 = sessao(area, paralelo=1, n_fre=0.3)
    c2._retomar_pendentes()
    gravar_resultado(dir_trabalho(area, 0))
    do_orfao.soltar()
    jornal_que_falha(monkeypatch, ['informar'])
    with pytest.raises(OSError):
        c2._colher()
    assert sorted(c2.adotados) == [0] and not do_tipo(area, 'informar')
    assert c2._ciclo() is True, 'o órfão terminado nunca mais foi colhido depois de um erro de E/S'
    assert [(e['id'], e['falhou']) for e in do_tipo(area, 'informar')] == [(0, False)] and not c2.adotados and not falsos.lancados[1:]


def test_retomada_informa_o_resultado_integro_que_ja_esta_em_disco_sem_relancar_nem_registrar_queda(area, falsos):
    """O avaliador terminou DEPOIS de o coordenador cair (ou o coordenador caiu entre o resultado e o 'informar'): nada se perdeu."""
    falsos.roteiro = pendurado
    c1 = sessao(area, paralelo=2)
    c1._ciclo()
    derrubar(c1)
    res = gravar_resultado(dir_trabalho(area, 1))
    c2 = sessao(area, paralelo=2)
    c2._retomar_pendentes()
    assert [(e['id'], e['aptidao']) for e in do_tipo(area, 'informar')] == [(1, res['aptidao'])] and [r['id'] for r in c2._fila] == [0]
    assert [e['id'] for e in do_tipo(area, 'relancado_apos_queda')] == [0] and list(c2.dehb.pendentes) == [0]


def test_retomada_devolve_a_trava_de_cada_pendente_que_vai_relancar(area, falsos, monkeypatch):
    """As Travas da sondagem são mantidas VIVAS pelo teste: sem isso a contagem de referências do CPython fecha o arquivo (e o SO
    solta a trava) quando a variável local morre, e o teste passaria mesmo sem o soltar() — que é o contrato, e não o coletor."""
    falsos.roteiro = pendurado
    c1 = sessao(area, paralelo=2)
    c1._ciclo()
    derrubar(c1)
    c2 = sessao(area, paralelo=2)
    vivas = []

    class TravaGuardada(H.Trava):
        def __init__(self, caminho):
            super().__init__(caminho)
            vivas.append(self)
    with monkeypatch.context() as m:
        m.setattr(H, 'Trava', TravaGuardada)
        c2._retomar_pendentes()
    assert [r['id'] for r in c2._fila] == [0, 1] and not c2.adotados and len(vivas) == 2
    assert all(trava_livre(os.path.join(dir_trabalho(area, i), '_EXECUTANDO.lock')) for i in (0, 1)),         'a sondagem da retomada ficou com a trava: o avaliador relançado sairia com "outro processo já executa"'


def test_retomada_com_resultado_nao_finito_em_disco_relanca_em_vez_de_informar(area, falsos):
    """O ramo gêmeo do _tratar_termino. Aptidão NaN no 'informar' nem passa pelo jornal (allow_nan=False): a retomada cairia com
    ValueError, e cairia de novo a cada relançamento da cadeia, até alguém apagar o arquivo à mão."""
    falsos.roteiro = pendurado
    c1 = sessao(area, paralelo=1)
    c1._ciclo()
    derrubar(c1)
    gravar_resultado(dir_trabalho(area, 0), aptidao=float('nan'))
    c2 = sessao(area, paralelo=1)
    try:
        c2._retomar_pendentes()
    except ValueError as e:
        pytest.fail(f'a retomada levou a aptidão não finita ao informar, e o jornal a recusou derrubando a sessão ({e})')
    assert not do_tipo(area, 'informar'), 'resultado com aptidão não finita foi informado na retomada'
    assert [r['id'] for r in c2._fila] == [0] and [e['id'] for e in do_tipo(area, 'relancado_apos_queda')] == [0] and list(c2.dehb.pendentes) == [0]


# ── d) vigia de batimento em tempo ATIVO ─────────────────────────────────────────────────────────
def colher_ate_matar(c, relogio, proc, passo_s, teto=400):
    """Avança o relógio do coordenador de `passo_s` em `passo_s`, colhendo; devolve em qual colheita o processo foi morto."""
    for k in range(1, teto + 1):
        relogio.agora += passo_s
        c._colher()
        if proc.mortes:
            return k
    return None


def test_avaliador_calado_alem_do_limite_e_morto_e_os_passos_do_batimento_sao_cobrados(area, falsos, relogio):
    """Limite de 1 h com ciclos de 50 s de atividade: 72 ciclos somam exatamente 3.600 s (não passa), o 73º passa. O parcial
    treinado (passos do último _BATIMENTO.json) entra no orçamento, como o FAIL do hpo_rl."""
    falsos.roteiro = lambda p: gravar_batimento(p.dir, 123_456, mtime=1_700_000_000)
    c = sessao(area, paralelo=1, limite_sem_batimento_h=1.0)
    c._ciclo()
    proc = unico(falsos.lancados, 'avaliador lançado')
    c._colher()                                                                      # vê o batimento pela primeira vez: zera
    assert colher_ate_matar(c, relogio, proc, 50.0) == 73, 'o kill por falta de batimento não veio quando o silêncio ATIVO passou de 1 h'
    ev = unico(do_tipo(area, 'tentativa_falhou'), "'tentativa_falhou' no jornal")
    assert (ev['id'], ev['motivo'], ev['codigo_saida'], ev['tentativa']) == (0, 'processo_morreu', -9, 1)
    assert ev['passos'] == 123_456 and ev['fre'] == pytest.approx(123_456 / N), 'o parcial do batimento não foi cobrado'
    assert c.fre_consumidos() == pytest.approx(123_456 / N)
    assert not c.em_voo and [r['id'] for r in c._fila] == [0] and proc.saida.closed


def test_avaliador_que_nunca_bateu_tambem_e_morto_e_nao_ha_parcial_a_cobrar(area, falsos, relogio):
    falsos.roteiro = pendurado
    c = sessao(area, paralelo=1, limite_sem_batimento_h=1.0)
    c._ciclo()
    assert colher_ate_matar(c, relogio, falsos.lancados[0], 50.0) == 73
    ev = unico(do_tipo(area, 'tentativa_falhou'), "'tentativa_falhou' no jornal")
    assert (ev['motivo'], ev['passos'], ev['fre']) == ('processo_morreu', 0, 0.0)
    c._lancar_da_fila()                                                              # a repetição começa com o silêncio ZERADO, não com o do processo morto
    assert colher_ate_matar(c, relogio, falsos.lancados[1], 50.0) == 73, 'o avaliador relançado herdou o silêncio acumulado do anterior'


def test_primeira_colheita_da_sessao_nao_conta_silencio_nenhum(area, falsos, relogio):
    """Sem tique anterior não há intervalo a medir: o valor absoluto de time.monotonic (aqui 1000 s; na máquina, o tempo desde a
    partida do Windows) não é silêncio de ninguém. No laço de produção a 1ª colheita acontece com em_voo vazio; à mão, não."""
    falsos.roteiro = pendurado
    c = sessao(area, paralelo=1, limite_sem_batimento_h=1.0)
    c._fila.append(c.jornal.acrescentar({'ev': 'pedir', 'trabalho': c.dehb.pedir().registro(), 'gerador': c.dehb.estado_do_gerador(), 'data': H.agora_iso()})['trabalho'])
    c._lancar_da_fila()
    c._colher()
    assert c._silencio[0]['acumulado'] == 0.0, 'a 1ª colheita contou o relógio absoluto como silêncio do avaliador'
    relogio.agora += 50.0
    c._colher()
    assert c._silencio[0]['acumulado'] == 50.0
    derrubar(c)


def test_batimento_que_avanca_zera_o_silencio_acumulado(area, falsos, relogio):
    falsos.roteiro = lambda p: gravar_batimento(p.dir, 1_000, mtime=1_700_000_000)
    c = sessao(area, paralelo=1, limite_sem_batimento_h=1.0)
    c._ciclo()
    proc = unico(falsos.lancados, 'avaliador lançado')
    c._colher()
    for _ in range(70):
        relogio.agora += 50.0
        c._colher()
    assert c._silencio[0]['acumulado'] == 3500.0 and not proc.mortes
    gravar_batimento(proc.dir, 2_000, mtime=1_700_000_500)                           # o treino falou: o silêncio recomeça do zero
    assert colher_ate_matar(c, relogio, proc, 50.0) == 74, 'batimento novo não zerou o acumulado (74 = a colheita que vê o batimento + 73)'
    assert do_tipo(area, 'tentativa_falhou')[0]['passos'] == 2_000


@pytest.mark.parametrize('intervalo_s, teto_s', [(5.0, 60.0), (30.0, 90.0)])
def test_salto_do_relogio_entre_dois_ciclos_nao_mata_ninguem(area, falsos, relogio, intervalo_s, teto_s):
    """A máquina dormiu 8 h entre dois ciclos: time.monotonic salta, mas o avaliador também estava suspenso. O dt de um ciclo tem
    teto (max(60 s, 3 intervalos)); sem ele, todos os filhos seriam encerrados ao acordar."""
    falsos.roteiro = lambda p: gravar_batimento(p.dir, 1_000, mtime=1_700_000_000)
    c = sessao(area, paralelo=2, limite_sem_batimento_h=1.0, intervalo_s=intervalo_s)
    c._ciclo()
    c._colher()
    relogio.agora += 8 * 3600.0
    c._colher()
    assert [p.mortes for p in falsos.lancados] == [0, 0] and not do_tipo(area, 'tentativa_falhou'), 'a suspensão da máquina matou os avaliadores'
    assert [c._silencio[i]['acumulado'] for i in (0, 1)] == [teto_s, teto_s]
    relogio.agora += 50.0                                                            # abaixo do teto, o tempo ativo conta inteiro
    c._colher()
    assert c._silencio[0]['acumulado'] == teto_s + 50.0
    derrubar(c)


def test_processo_que_nao_morre_no_prazo_do_kill_continua_em_voo_e_nada_e_gravado(area, falsos, relogio):
    """kill() sem o processo terminar (preso em E/S do driver): tratar o término agora relançaria o trabalho com o primeiro ainda
    vivo na GPU. Fica em voo, e o ciclo seguinte insiste."""
    falsos.roteiro = pendurado
    c = sessao(area, paralelo=1, limite_sem_batimento_h=1.0)
    c._ciclo()
    proc = unico(falsos.lancados, 'avaliador lançado')
    proc.esperas_que_estouram = 1
    assert colher_ate_matar(c, relogio, proc, 50.0) == 73
    assert list(c.em_voo) == [0] and not do_tipo(area, 'tentativa_falhou') and not c._fila
    relogio.agora += 50.0
    c._colher()
    assert proc.mortes == 2 and not c.em_voo and [e['codigo_saida'] for e in do_tipo(area, 'tentativa_falhou')] == [-9]


# ── e) replay: falhas e contador de tentativas ───────────────────────────────────────────────────
def roteiro_por_id(casos, padrao=termina_bem):
    return lambda p: casos.get(p.id, padrao)(p)


def falha_da_configuracao(p):
    gravar_falha(p.dir, 'excecao_no_treino', 150_000)
    p.rc = H.SAIDA_FALHA_DA_CONFIGURACAO


def falha_transitoria(p):
    gravar_falha(p.dir, 'gpu_ou_sistema', 75_000)
    p.rc = H.SAIDA_FALHA_TRANSITORIA


def test_replay_reaplica_informar_falho_como_falha_e_chega_ao_mesmo_estado_da_sessao_viva(area, falsos):
    """O jornal guarda aptidao = null na falha. O replay tem de devolvê-la ao núcleo como +inf (nunca substitui, nunca é promovida) e
    NÃO pode passar 'tentativa_falhou' ao núcleo: a assinatura (populações, brackets, gerador) é a da sessão que gravou o jornal."""
    uma_vez = {'ja': False}

    def transitoria_uma_vez(p):
        (termina_bem if uma_vez['ja'] else falha_transitoria)(p)
        uma_vez['ja'] = True
    falsos.roteiro = roteiro_por_id({0: falha_da_configuracao, 2: transitoria_uma_vez})
    c1, rc = rodar_sessao(area, paralelo=2, n_fre=3.0)
    assert rc == 0 and [e['id'] for e in do_tipo(area, 'tentativa_falhou')] == [2]
    assert [(e['id'], e['aptidao']) for e in do_tipo(area, 'informar') if e['falhou']] == [(0, None)]
    c2 = H.Coordenador(argumentos(area, n_fre=9.0), log=lambda m: None)
    assert c2.abrir()
    falho = next(t for t in c2.dehb.historico if t.id == 0)
    assert np.isinf(falho.aptidao), f'o replay reaplicou a falha como aptidão {falho.aptidao}'
    assert not any(np.array_equal(falho.vetor, t.vetor) for t in c2.dehb.historico if t.origem == 'promocao'), 'configuração que FALHOU foi promovida'
    assert c2.dehb.assinatura() == c1.dehb.assinatura(), 'o replay não chegou ao estado da sessão que gravou o jornal'
    assert not do_tipo(area, 'divergencia')


def test_contador_de_tentativas_sobrevive_a_retomada(area, falsos):
    """--max-tentativas 2: falhou 1×, a sessão cai, retoma → só mais UMA tentativa antes de 'transitoria_esgotada'. Sem isso, uma
    máquina que reinicia a cada falha de GPU repetiria o mesmo trabalho para sempre, cobrando o parcial a cada volta."""
    falsos.roteiro = falha_transitoria
    c1 = sessao(area, paralelo=1, max_tentativas=2)
    c1._ciclo()
    c1._colher()
    assert c1.tentativas == {0: 1} and [r['id'] for r in c1._fila] == [0]
    c2 = sessao(area, paralelo=1, max_tentativas=2)                                  # a sessão caiu antes de relançar
    assert c2.tentativas == {0: 1}, 'o contador de tentativas não foi refeito do jornal'
    c2._retomar_pendentes()
    antes = len(falsos.lancados)
    for _ in range(4):
        c2._colher()
        c2._lancar_da_fila()
    assert falsos.ids()[antes:] == [0], 'depois da retomada cabia só mais UMA tentativa'
    assert [(e['ev'], e.get('tentativa'), e.get('motivo')) for e in no_arquivo(area) if e['ev'] in TERMINAIS and e['id'] == 0] == \
        [('tentativa_falhou', 1, 'gpu_ou_sistema'), ('tentativa_falhou', 2, 'gpu_ou_sistema'), ('informar', None, 'transitoria_esgotada')]
    assert c2.fre_consumidos() == pytest.approx(2 * 75_000 / N)


def test_disjuntor_conta_so_as_falhas_finais_do_FIM_do_jornal_e_tentativa_transitoria_nao_conta_nem_zera(area, falsos):
    """Falham 0 e 1, o 2 dá certo, falham 3 e 4, e o 5 falha de forma TRANSITÓRIA (vai repetir). Contar do começo do jornal, contar
    as falhas não seguidas, ou deixar a 'tentativa_falhou' quebrar a sequência dão, cada um, outra lista."""
    falsos.roteiro = roteiro_por_id({0: falha_da_configuracao, 1: falha_da_configuracao, 3: falha_da_configuracao, 4: falha_da_configuracao, 5: falha_transitoria})
    c = sessao(area, paralelo=1, n_fre=9.0, max_tentativas=5)
    visto = []
    for _ in range(6):
        c._ciclo()
        c._colher()
        visto.append(c.falhas_finais_seguidas())
    assert [(e['ev'], e['id']) for e in terminais(area)] == [('informar', k) for k in range(5)] + [('tentativa_falhou', 5)]
    assert visto == [1, 2, 0, 1, 2, 2], 'a sequência é a do FIM do jornal, zera no sucesso, e tentativa transitória não é falha final nem sucesso'


def test_disjuntor_no_laco_para_na_falha_de_numero_max_falhas_seguidas_e_mata_o_que_ja_tinha_lancado(area, falsos):
    """K=1 e todo mundo falha: o ciclo que colhe a 4ª falha ainda pede e lança o trabalho 4 ANTES da conferência — e o aborto o mata."""
    falsos.roteiro = falha_da_configuracao
    c, rc = rodar_sessao(area, paralelo=1, n_fre=9.0, max_falhas_seguidas=4)
    assert rc == 1 and [(e['id'], e['falhou']) for e in do_tipo(area, 'informar')] == [(k, True) for k in range(4)], 'o disjuntor não parou na 4ª falha final seguida'
    assert falsos.ids() == [0, 1, 2, 3, 4] and falsos.lancados[4].mortes == 1
    flag = open(os.path.join(area['hpo'], 's', '_FAILED.flag'), encoding='utf-8').read()
    assert '4 avaliações seguidas falharam' in flag and 'sistêmico' in flag


def test_disjuntor_na_partida_dispara_com_EXATAMENTE_o_limite_e_nada_e_lancado(area, falsos):
    """A cadeia relança o comando depois do aborto: com 4 falhas finais no fim do jornal (= limite) a retomada não pode queimar GPU."""
    falsos.roteiro = falha_da_configuracao
    c1 = sessao(area, paralelo=1, n_fre=9.0)
    for _ in range(4):
        c1._ciclo()
        c1._colher()
    assert c1.falhas_finais_seguidas() == 4 and not c1.em_voo
    c2, rc = rodar_sessao(area, paralelo=1, n_fre=9.0, max_falhas_seguidas=4)
    assert falsos.ids() == [0, 1, 2, 3] and len(do_tipo(area, 'pedir')) == 4, 'a retomada com exatamente max_falhas_seguidas falhas finais voltou a lançar avaliações'
    assert rc == 1 and 'as últimas 4 avaliações falharam' in open(os.path.join(area['hpo'], 's', '_FAILED.flag'), encoding='utf-8').read()


def test_fre_com_duas_sementes_de_treino_divide_os_passos_por_dois_N_nos_tres_lugares(area, falsos):
    """Com --seeds-trial 42 43 a avaliação completa treina 2N passos e vale 1 FRE (é o que o avaliador grava no resultado). Os três
    parciais que o COORDENADOR calcula têm de usar o mesmo denominador: tentativa falhada, falha final e parcial perdido em queda."""
    def morre_com_batimento(p):
        gravar_batimento(p.dir, 90_000, mtime=1_700_000_000)
        p.rc = -9
    falsos.roteiro = roteiro_por_id({2: falha_da_configuracao}, padrao=morre_com_batimento)
    c1 = sessao(area, paralelo=3, seeds_trial=[42, 43])
    c1._ciclo()
    c1._colher()
    assert [(e['id'], e['passos'], e['fre']) for e in do_tipo(area, 'tentativa_falhou')] == [(k, 90_000, pytest.approx(90_000 / (2 * N))) for k in (0, 1)], \
        'FRE da tentativa falhada: passos / (N x nº de sementes de treino)'
    assert [(e['id'], e['passos'], e['fre']) for e in do_tipo(area, 'informar')] == [(2, 150_000, pytest.approx(150_000 / (2 * N)))], \
        'FRE da falha final: passos / (N x nº de sementes de treino)'
    assert c1.fre_consumidos() == pytest.approx((90_000 + 90_000 + 150_000) / (2 * N))
    falsos.roteiro = lambda p: gravar_batimento(p.dir, 60_000, mtime=1_700_000_500)   # relançados, treinam 60.000 passos e o coordenador cai
    c1._lancar_da_fila()
    assert sorted(c1.em_voo) == [0, 1]
    derrubar(c1)
    c2 = sessao(area, paralelo=3, seeds_trial=[42, 43])
    c2._retomar_pendentes()
    assert [(e['id'], e['passos_perdidos'], e['fre_nao_cobrado']) for e in do_tipo(area, 'relancado_apos_queda')] == \
        [(k, 60_000, pytest.approx(60_000 / (2 * N))) for k in (0, 1)], 'FRE não cobrado: passos / (N x nº de sementes de treino)'


def test_espelho_e_reconciliado_ja_na_tentativa_falhada_e_nao_so_no_informar_seguinte(area, falsos):
    """O trial FAIL da tentativa tem de estar no .db assim que ela é gravada: pos_hpo e analise_hpo leem o espelho com o estudo vivo."""
    falsos.roteiro = falha_transitoria
    c = sessao(area, paralelo=1)
    c._ciclo()
    c._colher()
    ev = unico(do_tipo(area, 'tentativa_falhou'), "'tentativa_falhou' no jornal")
    assert c.espelho.chamadas == [ev['i'] + 1], 'o espelho não foi reconciliado com o jornal que já tem a tentativa_falhou'


# ── f) retomada com mais pendentes que K ─────────────────────────────────────────────────────────
def test_retomada_com_mais_pendentes_que_vagas_enfileira_todos_e_roda_cada_um_uma_vez_na_ordem(area, falsos):
    falsos.roteiro = pendurado
    c1 = sessao(area, paralelo=5, n_fre=3.0)
    c1._ciclo()
    assert sorted(c1.em_voo) == [0, 1, 2, 3, 4]
    derrubar(c1)
    gravar_batimento(dir_trabalho(area, 0), 30_000, mtime=1_700_000_000)
    gravar_batimento(dir_trabalho(area, 3), 90_000, mtime=1_700_000_000)
    c2 = sessao(area, paralelo=2, n_fre=3.0)                                         # a cadeia relançou com K menor
    antes = len(falsos.lancados)
    c2._retomar_pendentes()
    assert [r['id'] for r in c2._fila] == [0, 1, 2, 3, 4] and not c2.em_voo
    assert [(e['id'], e['passos_perdidos'], e['fre_nao_cobrado']) for e in do_tipo(area, 'relancado_apos_queda')] == \
        [(0, 30_000, pytest.approx(30_000 / N)), (1, 0, 0.0), (2, 0, 0.0), (3, 90_000, pytest.approx(90_000 / N)), (4, 0, 0.0)]
    pico = 0
    for _ in range(200):
        fim = c2._ciclo()
        pico = max(pico, len(c2.em_voo))
        assert len(c2.em_voo) <= 2 and (not c2._fila or c2.dehb._proximo_id == 5), 'passou de K em voo, ou pediu trabalho novo com a fila da retomada cheia'
        if fim:
            break
        vivos = [p for p in falsos.lancados[antes:] if p.rc is None]
        assert vivos, 'nada em voo e o estudo não terminou: a fila da retomada não anda'
        termina_bem(vivos[0])                                                        # um avaliador termina por ciclo
    assert fim and pico == 2, f'pico de {pico} avaliações em voo com --paralelo 2'
    lancados = falsos.ids()[antes:]
    assert lancados[:5] == [0, 1, 2, 3, 4] and len(lancados) == len(set(lancados)), f'ordem/duplicata no relançamento: {lancados}'
    informados = [e['id'] for e in do_tipo(area, 'informar')]
    assert sorted(informados) == sorted(lancados) == [e['trabalho']['id'] for e in do_tipo(area, 'pedir')]
    prog = json.load(open(os.path.join(area['hpo'], 's', 'PROGRESSO.json'), encoding='utf-8'))
    gasto = sum(e['fre'] for e in do_tipo(area, 'informar'))
    assert prog['fre_perdido_em_quedas_nao_cobrado'] == round(120_000 / N, 3)
    assert prog['fre_consumidos'] == pytest.approx(gasto, abs=1e-3) and c2.fre_consumidos() == pytest.approx(gasto), 'o parcial perdido na queda do COORDENADOR não é cobrado'


def test_trabalho_que_esperava_na_fila_na_queda_nao_tem_parcial_perdido(area, falsos):
    """A tentativa morreu, os 90.000 passos do batimento foram COBRADOS ('tentativa_falhou') e o trabalho esperava a vez na fila
    quando o coordenador caiu: nada estava em voo, nada se perdeu. O _BATIMENTO.json velho só é guardado no relançamento."""
    falsos.roteiro = lambda p: (gravar_batimento(p.dir, 90_000, mtime=1_700_000_000), setattr(p, 'rc', -9))
    c1 = sessao(area, paralelo=1)
    c1._ciclo()
    c1._colher()
    assert [(e['id'], e['passos']) for e in do_tipo(area, 'tentativa_falhou')] == [(0, 90_000)] and [r['id'] for r in c1._fila] == [0] and not c1.em_voo
    c2 = sessao(area, paralelo=1)
    c2._retomar_pendentes()
    assert [r['id'] for r in c2._fila] == [0]
    assert [(e['id'], e['passos_perdidos']) for e in do_tipo(area, 'relancado_apos_queda')] in ([], [(0, 0)]), 'passos já cobrados contados também como perdidos e não cobrados'
    assert c2.progresso()['fre_perdido_em_quedas_nao_cobrado'] == 0.0


def test_repeticao_de_falha_transitoria_passa_na_frente_da_fila(area, falsos):
    """A repetição já ocupava uma vaga: volta para a FRENTE, senão ficaria atrás de toda a fila de uma retomada."""
    falsos.roteiro = pendurado
    c1 = sessao(area, paralelo=3)
    c1._ciclo()
    derrubar(c1)
    c2 = sessao(area, paralelo=1)
    c2._retomar_pendentes()
    c2._lancar_da_fila()
    assert list(c2.em_voo) == [0] and [r['id'] for r in c2._fila] == [1, 2]
    falha_transitoria(c2.em_voo[0]['proc'])
    c2._colher()
    assert [r['id'] for r in c2._fila] == [0, 1, 2]
    derrubar(c2)


# ── g) erro de E/S no laço ───────────────────────────────────────────────────────────────────────
def test_erros_transitorios_de_es_no_laco_nao_derrubam_a_sessao_nem_duplicam_nada(area, falsos, relogio, monkeypatch):
    """Um erro de cada tipo, em trabalhos diferentes: Popen que falha (lançamento), resultado.json preso além da insistência
    (colheita), 'informar' e 'tentativa_falhou' que o jornal não conseguiu gravar. A sessão segue, e o jornal fica limpo."""
    uma_vez = {'ja': False}

    def transitoria_uma_vez(p):
        (termina_bem if uma_vez['ja'] else falha_transitoria)(p)
        uma_vez['ja'] = True
    falsos.roteiro = roteiro_por_id({2: transitoria_uma_vez})
    falsos.popen_levanta = [OSError('[WinError 1455] arquivo de paginação muito pequeno')]      # OSError puro, não PermissionError
    preso = abrir_preso(monkeypatch, 'resultado.json', 0)
    real = H.Coordenador._tratar_termino

    def tratar(self, registro, rc):
        if registro['id'] == 1 and not preso.get('armado'):
            preso.update(armado=True, n=0, vezes=6)                                  # o resultado do trabalho 1 fica preso por uma colheita inteira
        return real(self, registro, rc)
    monkeypatch.setattr(H.Coordenador, '_tratar_termino', tratar)
    faltam = jornal_que_falha(monkeypatch, ['informar', 'tentativa_falhou'])
    c, rc = rodar_sessao(area, paralelo=1, n_fre=3.0)
    d = os.path.join(area['hpo'], 's')
    assert rc == 0 and os.path.exists(os.path.join(d, '_DONE.flag')) and not os.path.exists(os.path.join(d, '_FAILED.flag'))
    assert faltam == [] and preso.get('armado') and preso['n'] > 6 and not falsos.popen_levanta   # todos os erros aconteceram de fato
    ev = no_arquivo(area)
    pedidos = [e['trabalho']['id'] for e in ev if e['ev'] == 'pedir']
    assert [e['i'] for e in ev] == list(range(len(ev))) and pedidos == list(range(len(pedidos)))
    assert sorted(e['id'] for e in ev if e['ev'] == 'informar') == pedidos, 'trabalho perdido ou informado duas vezes'
    assert [(e['id'], e['tentativa']) for e in ev if e['ev'] == 'tentativa_falhou'] == [(2, 1)]
    assert sorted(falsos.ids()) == sorted(pedidos + [2]), 'cada trabalho roda uma vez (o 2, duas: falhou de forma transitória)'
    assert [s for s in relogio.sonecas if s >= 1.5] == [2.0] * 4, 'cada erro isolado espera 1 + 1 s: o contador zera a cada ciclo bom'


def test_so_quarenta_erros_de_es_SEGUIDOS_abortam_com_flag_e_filhos_mortos(area, falsos, relogio, monkeypatch):
    falsos.roteiro = pendurado
    real, n = H.Coordenador._ciclo, {'ciclos': 0}

    def ciclo(self):
        n['ciclos'] += 1
        if n['ciclos'] == 1:
            return real(self)                                                        # o primeiro lança os K avaliadores
        raise OSError('[WinError 5] acesso negado')
    monkeypatch.setattr(H.Coordenador, '_ciclo', ciclo)
    c, rc = rodar_sessao(area, paralelo=2)
    d = os.path.join(area['hpo'], 's')
    assert rc == 1 and n['ciclos'] == 41, f"abortou depois de {n['ciclos'] - 1} erros seguidos, não 40"
    assert 'erro de E/S persistente' in open(os.path.join(d, '_FAILED.flag'), encoding='utf-8').read() and not os.path.exists(os.path.join(d, '_DONE.flag'))
    assert [p.mortes for p in falsos.lancados] == [1, 1], 'o aborto deixou avaliadores vivos na GPU'
    assert [s for s in relogio.sonecas if s >= 1.5] == [min(30.0, 1.0 + k) for k in range(1, 40)]


def test_erros_de_es_intercalados_com_ciclos_bons_nunca_abortam(area, falsos, relogio, monkeypatch):
    real, n = H.Coordenador._ciclo, {'ciclos': 0}

    def ciclo(self):
        n['ciclos'] += 1
        if n['ciclos'] % 40 == 0 or n['ciclos'] > 120:
            return real(self)
        raise OSError('[WinError 32] arquivo em uso')                                # 39 erros, 1 ciclo bom, 39 erros, ...
    monkeypatch.setattr(H.Coordenador, '_ciclo', ciclo)
    c, rc = rodar_sessao(area, paralelo=1, n_fre=1.0)
    assert rc == 0 and n['ciclos'] > 120 and not os.path.exists(os.path.join(area['hpo'], 's', '_FAILED.flag'))


def test_excecao_que_nao_e_de_es_sobe_na_hora_sem_insistir_nem_virar_erro_de_disco(area, falsos, monkeypatch):
    """Só OSError é 'arquivo preso'. Um defeito de programa (KeyError do núcleo) tratado como E/S viraria 40 sonecas e uma
    _FAILED.flag dizendo 'erro de E/S persistente': o rastro certo é o traceback, na hora (main() grava a flag com ele)."""
    n = {'ciclos': 0}

    def ciclo(self):
        n['ciclos'] += 1
        raise KeyError('defeito de programa, não de disco')
    monkeypatch.setattr(H.Coordenador, '_ciclo', ciclo)
    with pytest.raises(KeyError, match='defeito de programa'):
        rodar_sessao(area, paralelo=1)
    assert n['ciclos'] == 1 and not [x for x in falsos.sonecas if x >= 1.5] and not os.path.exists(os.path.join(area['hpo'], 's', '_FAILED.flag'))


@pytest.mark.parametrize('preso', ['_STATUS.txt', 'PROGRESSO.json'])
def test_arquivo_de_rastro_preso_depois_do_informar_nao_sobe_nem_faz_o_trabalho_ser_informado_de_novo(area, falsos, monkeypatch, preso):
    """_STATUS.txt e PROGRESSO.json são gravados DEPOIS de o 'informar' estar no jornal e no núcleo. Se o erro deles subisse, o
    trabalho ficaria em em_voo e o ciclo seguinte gravaria um 2º 'informar' (e o núcleo levantaria KeyError: a sessão cai)."""
    c = sessao(area, paralelo=1)
    c._ciclo()
    tentou = []

    def levanta(*a, **k):
        tentou.append(preso)
        raise PermissionError(13, f'WinError 32: {preso} preso pelo antivírus')
    if preso == '_STATUS.txt':
        monkeypatch.setattr(P, 'escrever_status', levanta)
    else:
        real = H.escrever_json_atomico
        monkeypatch.setattr(H, 'escrever_json_atomico', lambda cam, *a, **k: levanta() if cam.endswith('PROGRESSO.json') else real(cam, *a, **k))
    c._colher()
    assert tentou, f'o teste não prendeu o {preso}'
    assert not c.em_voo and [e['id'] for e in do_tipo(area, 'informar')] == [0] and [t.id for t in c.dehb.historico] == [0]
    c._colher()
    assert [e['id'] for e in do_tipo(area, 'informar')] == [0]


def test_abortar_mata_todos_os_avaliadores_e_grava_a_flag_mesmo_se_um_kill_levanta(area, falsos):
    falsos.roteiro = pendurado
    c = sessao(area, paralelo=3)
    c._ciclo()

    def kill_negado():
        raise PermissionError(5, 'acesso negado ao encerrar o 1º avaliador')
    falsos.lancados[0].kill = kill_negado
    assert c._abortar('parada de teste') == 1
    assert [p.mortes for p in falsos.lancados[1:]] == [1, 1], 'o aborto parou no kill que levantou e deixou avaliadores vivos na GPU'
    assert 'parada de teste' in open(os.path.join(area['hpo'], 's', '_FAILED.flag'), encoding='utf-8').read()
    derrubar(c)


def test_oserror_ao_gravar_o_informar_deixa_o_trabalho_em_voo_e_o_ciclo_seguinte_informa_uma_vez(area, falsos, monkeypatch):
    c = sessao(area, paralelo=1)
    c._ciclo()
    jornal_que_falha(monkeypatch, ['informar'])
    with pytest.raises(OSError):
        c._colher()
    assert list(c.em_voo) == [0] and list(c.dehb.pendentes) == [0] and not c.dehb.historico and not do_tipo(area, 'informar')
    assert os.path.exists(os.path.join(dir_trabalho(area, 0), 'resultado.json')) and c.espelho.chamadas == []
    c._colher()
    assert [e['id'] for e in do_tipo(area, 'informar')] == [0] and not c.em_voo and [t.id for t in c.dehb.historico] == [0]
    assert falsos.ids() == [0]


def test_oserror_ao_gravar_o_pedir_nao_deixa_pedido_fantasma_no_nucleo(area, falsos, monkeypatch):
    """O núcleo já sorteou o pedido quando o jornal falha. Se o ciclo seguinte pedir OUTRO, o primeiro fica pendente para sempre na
    memória (o degrau do bracket nunca fecha) e o jornal começa no id 1: a próxima retomada diverge e a adoção recusa o jornal."""
    c = sessao(area, paralelo=1, n_fre=1.0)
    jornal_que_falha(monkeypatch, ['pedir'])
    with pytest.raises(OSError):
        c._ciclo()
    assert not falsos.lancados and not do_tipo(area, 'pedir')                         # nada foi lançado sem estar no jornal
    for _ in range(50):
        if c._ciclo():
            break
    pedidos = [e['trabalho']['id'] for e in do_tipo(area, 'pedir')]
    assert pedidos == list(range(len(pedidos))), f'o jornal ficou com os pedidos {pedidos}: o que o núcleo sorteou e o jornal não gravou virou fantasma'
    assert not c.dehb.pendentes
    c2 = H.Coordenador(argumentos(area, paralelo=1, n_fre=9.0), log=lambda m: None)
    assert c2.abrir() and not do_tipo(area, 'divergencia') and c2.dehb.assinatura() == c.dehb.assinatura()


def test_oserror_ao_gravar_o_informar_da_transitoria_esgotada_nao_cobra_a_mesma_tentativa_duas_vezes(area, falsos, monkeypatch):
    """Última tentativa: 'tentativa_falhou' é gravado e, em seguida, o 'informar' (transitoria_esgotada) falha por E/S. O trabalho
    continua em voo; o ciclo seguinte tem de completar o 'informar', não gravar (e cobrar) a mesma tentativa outra vez."""
    falsos.roteiro = falha_transitoria
    c = sessao(area, paralelo=1, max_tentativas=1)
    c._ciclo()
    jornal_que_falha(monkeypatch, ['informar'])
    with pytest.raises(OSError):
        c._colher()
    c._colher()
    assert [(e['ev'], e.get('tentativa')) for e in no_arquivo(area) if e['ev'] in TERMINAIS] == [('tentativa_falhou', 1), ('informar', None)]
    assert c.fre_consumidos() == pytest.approx(75_000 / N)


# ── h) _ler_valido: arquivo preso não é arquivo estragado ────────────────────────────────────────
def test_resultado_preso_por_instantes_e_lido_na_insistencia_e_nao_e_renomeado(area, falsos, relogio, monkeypatch):
    c = sessao(area, paralelo=1)
    c._ciclo()
    preso = abrir_preso(monkeypatch, 'resultado.json', 2)
    c._colher()
    assert preso['n'] == 3 and relogio.sonecas == [0.5, 0.5]
    assert [(e['id'], e['falhou']) for e in do_tipo(area, 'informar')] == [(0, False)]
    assert sorted(f for f in os.listdir(dir_trabalho(area, 0)) if f.startswith('resultado')) == ['resultado.json'], 'resultado VÁLIDO renomeado: horas de GPU'


def test_resultado_preso_para_sempre_sobe_oserror_e_nada_se_perde(area, falsos, relogio, monkeypatch):
    c = sessao(area, paralelo=1)
    c._ciclo()
    preso = abrir_preso(monkeypatch, 'resultado.json', 10 ** 9)
    with pytest.raises(PermissionError):
        c._colher()
    assert preso['n'] == 6, 'a leitura insiste 6 vezes antes de devolver o problema ao laço'
    assert list(c.em_voo) == [0] and not [e for e in no_arquivo(area) if e['ev'] in TERMINAIS] and not c._fila
    assert sorted(f for f in os.listdir(dir_trabalho(area, 0)) if f.startswith('resultado')) == ['resultado.json']
    preso['vezes'] = 0                                                               # o antivírus soltou
    c._colher()
    assert [(e['id'], e['falhou']) for e in do_tipo(area, 'informar')] == [(0, False)] and falsos.ids() == [0] and not c.em_voo


@pytest.mark.parametrize('estrago', ['cortado', 'impressao_alheia', 'id_de_outro_trabalho', 'nao_e_objeto'])
def test_resultado_lido_e_imprestavel_e_renomeado_e_o_trabalho_reexecutado(area, falsos, estrago):
    def roteiro(p):
        if len(falsos.lancados) > 1:
            return termina_bem(p)
        res = gravar_resultado(p.dir, **({'impressao': 'de-outro-estudo'} if estrago == 'impressao_alheia' else {'id': 7} if estrago == 'id_de_outro_trabalho' else {}))
        if estrago in ('cortado', 'nao_e_objeto'):
            open(os.path.join(p.dir, 'resultado.json'), 'w', encoding='utf-8').write(json.dumps(res)[:-25] if estrago == 'cortado' else '[1, 2]')
        p.rc = 0
    falsos.roteiro = roteiro
    c = sessao(area, paralelo=1)
    c._ciclo()
    c._colher()
    arquivos = os.listdir(dir_trabalho(area, 0))
    assert 'resultado.json' not in arquivos and len([f for f in arquivos if f.startswith('resultado.json.rejeitado.')]) == 1
    assert [(e['id'], e['motivo'], e['passos']) for e in do_tipo(area, 'tentativa_falhou')] == [(0, 'sem_resultado', 0)] and not do_tipo(area, 'informar')
    c._lancar_da_fila()
    c._colher()
    assert falsos.ids() == [0, 0] and [(e['id'], e['falhou']) for e in do_tipo(area, 'informar')] == [(0, False)]


def test_recusado_que_nao_pode_ser_renomeado_ainda_vira_tentativa_falhada(area, falsos, monkeypatch):
    """O antivírus segura o arquivo RECUSADO na hora do rename: o rename é só arrumação (o _lancar guarda o arquivo de novo, como
    .anterior.*). Se o erro subisse, um resultado imprestável viraria 40 'erros de E/S' e aborto da sessão."""
    falsos.roteiro = lambda p: (gravar_resultado(p.dir, impressao='alheia'), setattr(p, 'rc', 0))
    c = sessao(area, paralelo=1)
    c._ciclo()
    real, negados = os.replace, []

    def replace(origem, destino):
        if '.rejeitado.' in str(destino):
            negados.append(os.path.basename(str(origem)))
            raise PermissionError(13, 'WinError 32: arquivo preso', str(origem))
        return real(origem, destino)
    monkeypatch.setattr(H.os, 'replace', replace)
    c._colher()
    assert negados == ['resultado.json'], 'o teste não chegou a negar o rename do arquivo recusado'
    assert [(e['id'], e['motivo']) for e in do_tipo(area, 'tentativa_falhou')] == [(0, 'sem_resultado')] and not c.em_voo and [r['id'] for r in c._fila] == [0]


def test_resultado_com_bytes_que_nao_sao_utf8_e_lido_e_imprestavel_como_o_json_cortado(area, falsos):
    """Setor devolvido com lixo depois de um desligamento abrupto, ou arquivo cortado no meio de um caractere acentuado: foi LIDO e
    não serve — tem de ser renomeado e o trabalho repetido, como no caso 'cortado' acima."""
    def roteiro(p):
        open(os.path.join(p.dir, 'resultado.json'), 'wb').write(b'{"impressao": "\xff\xfe lixo')
        p.rc = 0
    falsos.roteiro = roteiro
    c = sessao(area, paralelo=1)
    c._ciclo()
    c._colher()
    assert [(e['id'], e['motivo']) for e in do_tipo(area, 'tentativa_falhou')] == [(0, 'sem_resultado')]
    assert len([f for f in os.listdir(dir_trabalho(area, 0)) if f.startswith('resultado.json.rejeitado.')]) == 1


@pytest.mark.parametrize('onde', ['na_colheita', 'na_retomada'])
def test_resultado_com_aptidao_nula_e_tratado_como_aptidao_nao_finita(area, falsos, onde):
    """A chave AUSENTE já é tolerada (get com NaN por omissão); null, que é como o JSON escreve o valor ausente, derruba a sessão."""
    falsos.roteiro = (lambda p: (gravar_resultado(p.dir, aptidao=None), setattr(p, 'rc', 0))) if onde == 'na_colheita' else pendurado
    c = sessao(area, paralelo=1)
    c._ciclo()
    if onde == 'na_colheita':
        c._colher()
        assert [(e['id'], e['motivo']) for e in do_tipo(area, 'tentativa_falhou')] == [(0, 'sem_resultado')]
    else:
        derrubar(c)
        gravar_resultado(dir_trabalho(area, 0), aptidao=None)
        c2 = sessao(area, paralelo=1)
        c2._retomar_pendentes()
        assert [r['id'] for r in c2._fila] == [0] and not do_tipo(area, 'informar')


# ── i) _lancar ───────────────────────────────────────────────────────────────────────────────────
def test_relancamento_guarda_os_arquivos_da_tentativa_anterior_antes_de_o_processo_nascer(area, falsos):
    """A 1ª tentativa deixa falha.json (transitória), _BATIMENTO.json e um resultado.json não finito (impressão certa: não é
    rejeitado). Se sobrassem, a 2ª tentativa — que aqui morre sem gravar nada — seria cobrada pelos passos da 1ª."""
    no_lancamento = []

    def roteiro(p):
        no_lancamento.append(sorted(os.listdir(p.dir)))
        if len(falsos.lancados) == 1:
            gravar_resultado(p.dir, aptidao=float('nan'))
            gravar_batimento(p.dir, 50_000, mtime=1_700_000_000)
            falha_transitoria(p)
        else:
            p.rc = -9
    falsos.roteiro = roteiro
    c = sessao(area, paralelo=1, max_tentativas=3)
    c._ciclo()
    c._colher()
    c._lancar_da_fila()
    velhos = ('resultado.json', 'falha.json', '_BATIMENTO.json')
    assert not [f for f in no_lancamento[1] if f in velhos], f'o avaliador relançado nasceu com arquivos da tentativa anterior: {no_lancamento[1]}'
    for nome in velhos:
        assert len([f for f in no_lancamento[1] if f.startswith(nome + '.anterior.')]) == 1, f'{nome} não foi guardado como .anterior.*'
    assert 'trabalho.json' in no_lancamento[0] and 'trabalho.json' in no_lancamento[1]
    c._colher()
    assert [(e['tentativa'], e['motivo'], e['passos']) for e in do_tipo(area, 'tentativa_falhou')] == [(1, 'gpu_ou_sistema', 75_000), (2, 'processo_morreu', 0)]


def test_trabalho_json_deste_pedido_ja_esta_em_disco_quando_o_avaliador_nasce(area, falsos):
    """O avaliador lê DIR/trabalho.json ao nascer. Gravado depois do Popen, ele cairia sem o arquivo — ou, num relançamento, leria
    o da tentativa anterior. Olha-se a pasta de DENTRO do Popen, no 1º lançamento (pasta recém-criada)."""
    visto = []

    def roteiro(p):
        arq = os.path.join(p.dir, 'trabalho.json')
        visto.append(json.load(open(arq, encoding='utf-8')) if os.path.exists(arq) else None)
    falsos.roteiro = roteiro
    c = sessao(area, paralelo=1)
    c._ciclo()
    pedido = unico(do_tipo(area, 'pedir'), "'pedir' no jornal")['trabalho']
    assert visto != [None], 'o avaliador nasceu sem trabalho.json na pasta: o arquivo tem de ser gravado ANTES do Popen'
    tr = unico(visto, 'lançamento')
    assert (tr['id'], tr['vetor'], tr['fidelidade'], tr['fidelidade_maxima'], tr['seeds'], tr['passos_max']) == (0, pedido['vetor'], 300_000, False, [42], N)
    assert tr['impressao'] == H.impressao_digital(c.cab['id_execucao'], pedido, [42], N, 'git-do-teste'), 'a impressão do trabalho.json não é a DESTE pedido'
    derrubar(c)


def test_popen_que_levanta_fecha_o_log_e_o_registro_continua_na_fila(area, falsos):
    c = sessao(area, paralelo=1)
    falsos.popen_levanta = [OSError('[WinError 5] acesso negado')]
    with pytest.raises(OSError):
        c._ciclo()
    assert falsos.saidas[0].closed, 'avaliacao.log ficou aberto: no Windows ele prenderia a pasta do trabalho'
    assert [r['id'] for r in c._fila] == [0] and not c.em_voo
    assert c._ciclo() is False
    assert falsos.ids() == [0] and list(c.em_voo) == [0] and not c._fila
    assert [e['trabalho']['id'] for e in do_tipo(area, 'pedir')] == [0], 'o pedido que não pôde ser lançado foi pedido de novo'
    derrubar(c)


# ── j) a trava _EXECUTANDO.lock no avaliar() de PRODUÇÃO ─────────────────────────────────────────
def test_avaliar_de_producao_com_a_trava_tomada_sai_como_transitoria_sem_tocar_na_pasta(area, monkeypatch):
    """É a segunda defesa contra duplicata (a primeira é o Job Object). Com torch bloqueado, o avaliar() que passa da trava pára na
    importação deixando batimento e falha.json: é o controle que prova que o 4 do primeiro caso veio da TRAVA."""
    monkeypatch.setitem(sys.modules, 'torch', None)                                  # `import torch` levanta ImportError: barato e sem GPU
    d = str(area['raiz'] / 'trabalho_0')
    os.makedirs(d)
    json.dump({'impressao': 'x', 'id': 0}, open(os.path.join(d, 'trabalho.json'), 'w', encoding='utf-8'))
    de_outro = H.Trava(os.path.join(d, '_EXECUTANDO.lock'))
    assert de_outro.tentar()
    assert H.avaliar(d) == H.SAIDA_FALHA_TRANSITORIA
    assert sorted(os.listdir(d)) == ['_EXECUTANDO.lock', 'trabalho.json'], 'avaliar() ignorou a trava de quem já executa este trabalho'
    de_outro.soltar()
    assert H.avaliar(d) == H.SAIDA_FALHA_TRANSITORIA
    falha = json.load(open(os.path.join(d, 'falha.json'), encoding='utf-8'))
    assert falha['motivo'] == 'importacao' and falha['impressao'] == 'x' and os.path.exists(os.path.join(d, '_BATIMENTO.json'))


# ── k) os ramos de _tratar_termino ───────────────────────────────────────────────────────────────
def so_batimento(p):
    gravar_batimento(p.dir, 50_000, mtime=1_700_000_000)


def batimento_cortado(p):
    so_batimento(p)
    open(os.path.join(p.dir, '_BATIMENTO.json'), 'w', encoding='utf-8').write('{"data": "2026-09-18T10:00:00", "pas')


def falha_transitoria_com_batimento(p):
    so_batimento(p)
    falha_transitoria(p)


CASOS_DE_TERMINO = {
    # nome: (roteiro, código de saída, evento esperado, motivo, passos cobrados)
    'saida_0_sem_resultado': (so_batimento, 0, 'tentativa_falhou', 'sem_resultado', 50_000),
    'saida_0_com_resultado_nao_finito': (lambda p: gravar_resultado(p.dir, aptidao=float('nan')), 0, 'tentativa_falhou', 'sem_resultado', 0),
    'saida_3_com_falha_json': (falha_da_configuracao, 3, 'informar', 'excecao_no_treino', 150_000),
    'saida_3_sem_falha_json': (so_batimento, 3, 'tentativa_falhou', 'processo_morreu', 50_000),
    'saida_3_com_falha_json_de_outro_pedido': (lambda p: gravar_falha(p.dir, 'excecao_no_treino', 150_000, impressao='alheia'), 3, 'tentativa_falhou', 'processo_morreu', 0),
    'saida_4_com_falha_json': (falha_transitoria_com_batimento, 4, 'tentativa_falhou', 'gpu_ou_sistema', 75_000),
    'saida_4_sem_falha_json': (so_batimento, 4, 'tentativa_falhou', 'processo_morreu', 50_000),
    'morto_por_sinal': (so_batimento, -9, 'tentativa_falhou', 'processo_morreu', 50_000),
    'morto_no_meio_da_escrita_do_batimento': (batimento_cortado, -9, 'tentativa_falhou', 'processo_morreu', 0),   # ilegível = 0 passos, não exceção
    'violacao_de_acesso_no_windows': (pendurado, 0xC0000005, 'tentativa_falhou', 'processo_morreu', 0),
}


@pytest.mark.parametrize('caso', list(CASOS_DE_TERMINO))
def test_cada_termino_do_avaliador_cai_no_ramo_certo(area, falsos, caso):
    roteiro, rc, evento, motivo, passos = CASOS_DE_TERMINO[caso]
    falsos.roteiro = lambda p: (roteiro(p), setattr(p, 'rc', rc))
    c = sessao(area, paralelo=1)
    c._ciclo()
    c._colher()
    ev = unico(terminais(area), 'evento terminal no jornal')
    assert (ev['ev'], ev['id'], ev['motivo'], ev['passos']) == (evento, 0, motivo, passos)
    assert ev['fre'] == pytest.approx(passos / N) and ev['fidelidade'] == 300_000 and ev['fidelidade_maxima'] is False
    assert not c.em_voo and falsos.lancados[0].saida.closed
    if evento == 'informar':                                                         # falha DETERMINÍSTICA: final, não repete, +inf no núcleo
        assert ev['falhou'] and ev['aptidao'] is None and not c._fila and c.tentativas == {}
        assert np.isinf(c.dehb.historico[0].aptidao) and not c.dehb.pendentes
    else:                                                                            # falha TRANSITÓRIA: cobrada, repetida, e o núcleo não fica sabendo
        assert (ev['tentativa'], ev['codigo_saida']) == (1, rc) and [r['id'] for r in c._fila] == [0]
        assert list(c.dehb.pendentes) == [0] and not c.dehb.historico


def test_resultado_valido_em_disco_vale_mesmo_com_codigo_de_saida_ruim(area, falsos):
    """O avaliador gravou o resultado (atômico, com impressão) e caiu ao desmontar o CUDA: o treino de horas está feito."""
    falsos.roteiro = lambda p: (gravar_resultado(p.dir), so_batimento(p), setattr(p, 'rc', 0xC0000005))
    c = sessao(area, paralelo=1)
    c._ciclo()
    c._colher()
    ev = unico(terminais(area), 'evento terminal no jornal')
    assert ev['ev'] == 'informar', f"resultado válido em disco descartado por causa do código de saída: virou {ev['ev']} ({ev.get('motivo')})"
    assert (ev['falhou'], ev['passos']) == (False, 300_000) and ev['fre'] == pytest.approx(1 / 3)
    assert np.isfinite(c.dehb.historico[0].aptidao) and c.dehb.historico[0].aptidao == ev['aptidao']


# ── j) correções de 19/09/2026 que a revisão do coordenador (interrompida) não chegou a travar ─────
def test_pedido_que_ficou_por_gravar_entra_no_jornal_antes_do_informar_do_ciclo_seguinte(area, falsos, monkeypatch):
    """O núcleo sorteou o pedido 1 ANTES de ver o resultado do 0, e o jornal não gravou o pedido. Se no ciclo seguinte o 'informar' do 0
    entrasse no arquivo antes do 'pedir' do 1, o replay faria as duas coisas na ordem trocada e sortearia OUTRO pedido 1: divergência."""
    falsos.roteiro = pendurado
    c = sessao(area, paralelo=1)
    c._ciclo()                                                                       # o trabalho 0 está em voo
    c.args.paralelo = 2
    jornal_que_falha(monkeypatch, ['pedir'])
    with pytest.raises(OSError):
        c._ciclo()                                                                   # o núcleo pede o 1, o jornal falha
    assert list(c.dehb.pendentes) == [0, 1] and [e['trabalho']['id'] for e in do_tipo(area, 'pedir')] == [0]
    termina_bem(falsos.lancados[0])                                                  # o 0 termina ANTES do ciclo que regrava o pedido 1
    c._ciclo()
    ordem = [(e['ev'], e['id'] if e['ev'] == 'informar' else e['trabalho']['id']) for e in no_arquivo(area) if e['ev'] in ('pedir', 'informar')]
    assert ordem.index(('pedir', 1)) < ordem.index(('informar', 0)), f'o pedido regravado entrou DEPOIS do informar: {ordem}'
    c2 = H.Coordenador(argumentos(area, paralelo=2), log=lambda m: None)
    assert c2.abrir() and not do_tipo(area, 'divergencia') and c2.dehb.assinatura() == c.dehb.assinatura()
    derrubar(c)


@pytest.mark.parametrize('max_na_retomada, esperado', [(1, 'informar'), (2, 'relancar')])
def test_queda_entre_a_ultima_tentativa_e_o_informar_e_completada_na_retomada_sem_tentativa_extra(area, falsos, monkeypatch, max_na_retomada, esperado):
    """A sessão cai ENTRE o 'tentativa_falhou' da última tentativa e o 'informar' (transitoria_esgotada). Na retomada com o mesmo
    --max-tentativas o trabalho vai direto ao 'informar': relançá-lo daria uma tentativa a mais do que o limite. Com um limite MAIOR, ele
    tem direito a mais uma — e o batimento da tentativa já cobrada não é contado de novo como 'perdido na queda'."""
    falsos.roteiro = lambda p: (so_batimento(p), falha_transitoria(p))
    c = sessao(area, paralelo=1, max_tentativas=1)
    c._ciclo()
    jornal_que_falha(monkeypatch, ['informar'])
    with pytest.raises(OSError):
        c._colher()
    derrubar(c)
    assert [(e['ev'], e.get('tentativa')) for e in terminais(area)] == [('tentativa_falhou', 1)]
    c2 = sessao(area, paralelo=1, max_tentativas=max_na_retomada)
    c2._retomar_pendentes()
    if esperado == 'informar':
        ev = unico(do_tipo(area, 'informar'), 'informar')
        assert (ev['id'], ev['falhou'], ev['motivo']) == (0, True, 'transitoria_esgotada') and not c2._fila and not c2.dehb.pendentes
        assert not do_tipo(area, 'relancado_apos_queda')
    else:
        assert not do_tipo(area, 'informar') and [r['id'] for r in c2._fila] == [0]
        assert [e['passos_perdidos'] for e in do_tipo(area, 'relancado_apos_queda')] == [0], 'o batimento da tentativa COBRADA foi dado também como perdido'
    assert len(falsos.lancados) == 1 and c2.fre_consumidos() == pytest.approx(75_000 / N)


RESULTADOS_SEM_O_NUMERO_DA_FIDELIDADE = {
    # nome: (id do trabalho: com 4 em voo o 3 é o primeiro no topo e o 0 está embaixo; campos trocados no resultado; aproveitável?)
    'topo_valido': (3, {}, True),
    'topo_sem_objetivo': (3, {'objetivo': None}, False),
    'baixo_valido': (0, {}, True),
    'baixo_sem_indicador': (0, {'indicador': None}, False),
    'passos_em_texto': (0, {'passos': '300000'}, False),
    'fre_nulo': (0, {'fre': None}, False),
    'aptidao_bool': (0, {'aptidao': True}, False),
    'aptidao_inteiro_que_nao_cabe_em_float': (0, {'aptidao': 10 ** 400}, False),
}


@pytest.mark.parametrize('caso', list(RESULTADOS_SEM_O_NUMERO_DA_FIDELIDADE))
def test_resultado_sem_os_numeros_da_sua_fidelidade_vira_tentativa_falhada_e_nunca_excecao(area, falsos, caso):
    """Com a impressão certa, mas sem o número que o espelho usa (objetivo no topo, indicador embaixo) ou com passos/fre/aptidão que não
    são números finitos: antes, TypeError/KeyError derrubava a sessão — e de novo a cada retomada, porque o arquivo continua lá."""
    alvo, troca, aproveitavel = RESULTADOS_SEM_O_NUMERO_DA_FIDELIDADE[caso]

    def roteiro(p):
        if p.id == alvo:
            gravar_resultado(p.dir, **troca)
            p.rc = 0
    falsos.roteiro = roteiro
    c = sessao(area, paralelo=4)
    c._ciclo()
    assert c.dehb.pendentes[alvo].nivel == (1 if alvo == 3 else 0)                   # pré-condição: o 3 está mesmo no topo
    c._colher()
    ev = unico([e for e in terminais(area) if e['id'] == alvo], f'evento terminal do trabalho {alvo}')
    if aproveitavel:
        assert ev['ev'] == 'informar' and not ev['falhou']
    else:
        assert (ev['ev'], ev['motivo']) == ('tentativa_falhou', 'sem_resultado') and alvo in c.dehb.pendentes and c._fila[0]['id'] == alvo
    derrubar(c)


def test_erro_ao_fechar_o_log_do_avaliador_nao_deixa_no_ar_um_trabalho_ja_informado(area, falsos):
    """O término já está no jornal quando o log é fechado. Se o OSError do close subisse, o trabalho ficaria em em_voo e o ciclo seguinte o
    trataria de novo: 'informar' duplicado (e o núcleo levantaria KeyError, que não é erro de E/S e derruba a sessão)."""
    c = sessao(area, paralelo=1)
    c._ciclo()

    class LogQueNaoFecha:
        def __init__(self, real):
            self.real = real

        def close(self):
            self.real.close()
            raise OSError('[WinError 5] acesso negado')
    c.em_voo[0]['saida'] = LogQueNaoFecha(c.em_voo[0]['saida'])
    c._colher()
    assert not c.em_voo and [e['id'] for e in do_tipo(area, 'informar')] == [0]
    c._colher()
    assert [e['id'] for e in do_tipo(area, 'informar')] == [0]


def test_progresso_do_fim_do_estudo_insiste_mais_que_o_de_cada_evento(area, falsos, monkeypatch):
    """O PROGRESSO.json de cada evento é de melhor esforço (4 tentativas). O do FIM fica para sempre ao lado da _DONE.flag, e relançar um
    estudo pronto não o regrava: preso naquele instante, sobraria o da gravação anterior (visto no ensaio de caos)."""
    real, gravados = H.escrever_json_atomico, []

    def gravar(caminho, obj, tentativas=12):
        gravados.append((os.path.basename(caminho), tentativas))
        return real(caminho, obj, tentativas=tentativas)
    monkeypatch.setattr(H, 'escrever_json_atomico', gravar)
    c, rc = rodar_sessao(area, paralelo=1, n_fre=1.0)
    progresso = [t for nome, t in gravados if nome == 'PROGRESSO.json']
    assert rc == 0 and progresso[-1] == 12 and set(progresso[:-1]) == {4}, progresso
