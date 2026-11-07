"""Testes do AVALIADOR do DEHB (`hpo_dehb.avaliar`, o que roda em cada subprocesso). Nenhum usa GPU: o trabalho pede 'cpu'. Dois grupos:
  reais    `avaliar()` de ponta a ponta com o PPO e o ambiente de verdade, partindo de um clone MINÚSCULO (LSTM 32, encoder 16), no
           smoke do projeto (N = 4 x 365 -> fidelidades 486 e 1460; HP_SMOKE): uma avaliação de fidelidade baixa num SUBPROCESSO lançado
           e colhido pelo próprio Coordenador e a PROMOÇÃO dela à fidelidade máxima (que de quebra repete o treino da primeira: é o
           teste de determinismo). São 24 episódios do ambiente, e o custo é o ambiente, não a rede: por isso as duas corridas são UMA
           fixture de módulo e o arquivo inteiro é `lento`;
  roteiro  a LÓGICA de `avaliar()` com `ppo.treinar_rl` e `avaliacao.avaliar` trocados por dublês de roteiro fixo (< 1 s cada): que bloco
           vira indicador, em que estado da rede o objetivo é medido, que sementes entram, FRE real, classificação das falhas, trava.
A pasta de cada trabalho é montada pelo PRÓPRIO Coordenador (`_lancar`); a única mexida é acrescentar o tamanho da rede aos hp do
trabalho.json (o espaço de busca não tem arquitetura, e o padrão do ppo é 512/256). Caminhos com espaço e acento, como a pasta real."""
import os
import sys
import json
import types
import hashlib

import numpy as np
import pytest
import torch

import protocolo as P
import avaliacao as A

sys.path.insert(0, os.path.join(P.RAIZ, 'hpo'))
import hpo_dehb as H        # noqa: E402
import ppo as PPO           # noqa: E402
import envs_lote            # noqa: E402

pytestmark = pytest.mark.lento

N_SMOKE = 4 * P.PASSOS_MAXIMOS          # o N do --smoke: fidelidades 486 e 1460, bloco de 162 passos
N_CHEIO = 900_000                       # sem smoke (só com dublês): fidelidades 300.000 e 900.000
COLETA_SMOKE = 2 * P.PASSOS_MAXIMOS     # HP_SMOKE tem V = 2: o treino real anda de 730 em 730 passos
REDE_MINIMA = {'dim_lstm': 32, 'dim_enc': 16}
SOBRA = 40                              # o treino de roteiro passa 40 passos de cada fronteira: passos REAIS != fidelidade nominal
TravaDoSO = H.Trava                     # a classe verdadeira, para conferir a trava mesmo quando o teste troca H.Trava


# ── oficina: checkpoint minúsculo, referências, coordenador ──────────────────────────────────────
@pytest.fixture(scope='module', autouse=True)
def sem_gpu():
    """CUDA_VISIBLE_DEVICES VAZIA não esconde a placa no Windows (medido: torch.cuda.is_available() continua True; só device_count()
    vira 0). '-1' esconde, neste processo (se o CUDA ainda não foi iniciado) e nos subprocessos. A garantia de verdade é o trabalho.json
    pedir 'cpu'; isto é o cinto além do suspensório."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('CUDA_VISIBLE_DEVICES', '-1')
        yield


def referencias_distintas():
    """Médias DIFERENTES por conjunto e por regime: normalizar com a referência do conjunto (ou regime) errado muda o número."""
    desloc, fator = {'validacao_rollback': 0.0, 'validacao_objetivo': 10.0, 'teste': 20.0}, {'in_dist': 1.0, 'dry': 0.6, 'humid': 0.8}
    return {k: {c: {r: {'media': (m + desloc[c]) * fator[r]} for r in P.NOMES_REGIMES} for c in desloc}
            for k, m in (('oraculo_perfeito', 5000.0), ('aleatorio', 600.0))}


def norma(media, refs, conjunto, regime):
    a, p = refs['aleatorio'][conjunto][regime]['media'], refs['oraculo_perfeito'][conjunto][regime]['media']
    return (media - a) / (p - a)


@pytest.fixture(scope='module')
def oficina(tmp_path_factory):
    raiz = tmp_path_factory.mktemp('avaliador') / 'Área de Trabalho' / 'estudo dehb'
    (raiz / 'hpo').mkdir(parents=True)
    from bc import AgenteBC
    torch.manual_seed(0)
    ag = AgenteBC(dim_lstm=REDE_MINIMA['dim_lstm'], dim_encoder=REDE_MINIMA['dim_enc'], dispositivo='cpu')
    ag.obs_media, ag.obs_std = np.full(P.OBS_DIM, 0.3, np.float32), np.full(P.OBS_DIM, 0.2, np.float32)
    for seed in (42, 43):
        ag.salvar(str(raiz / 'il' / f'seed_{seed}' / 'modelo_final.pt'))
    refs = referencias_distintas()
    (raiz / 'referencias.json').write_text(json.dumps(refs), encoding='utf-8')
    return {'raiz': raiz, 'hpo': str(raiz / 'hpo'), 'il': str(raiz / 'il'), 'refs': str(raiz / 'referencias.json'), 'referencias': refs}


def ler(pasta, nome):
    with open(os.path.join(pasta, nome), encoding='utf-8') as f:
        return json.load(f)


def alterar_trabalho(pasta, hp=None, **campos):
    tr = ler(pasta, 'trabalho.json')
    tr['hp'].update(hp or {})
    tr.update(campos)
    H.escrever_json_atomico(os.path.join(pasta, 'trabalho.json'), tr)


def comando_nulo(pasta):
    """`_lancar` chama isto DEPOIS de gravar o trabalho.json e antes de criar o processo: é a hora de encolher a rede."""
    alterar_trabalho(pasta, hp=REDE_MINIMA)
    return [sys.executable, '-c', 'pass']


def comando_real(pasta):
    alterar_trabalho(pasta, hp=REDE_MINIMA)
    return [sys.executable, '-X', 'utf8', os.path.join(P.RAIZ, 'hpo', 'hpo_dehb.py'), '--avaliar', pasta]


def coordenador(oficina, study, comando=comando_nulo, seeds=(42,), smoke=True):
    args = H.argparse.Namespace(study=study, orcamento=N_SMOKE if smoke else N_CHEIO, niveis=2, eta=3, seed=7, inicializacao='professor',
                                seeds_trial=list(seeds), smoke=smoke, retomar=True, n_fre=2.0, paralelo=2, warmstart=oficina['il'],
                                referencias=oficina['refs'], dispositivo='cpu', max_tentativas=2, max_falhas_seguidas=4,
                                ignorar_disjuntor=False, limite_sem_batimento_h=3.0, intervalo_s=0.05)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(H, 'DIR_HPO', oficina['hpo'])
        c = H.Coordenador(args, comando_avaliador=comando, log=lambda m: None)
        c.abrir()
    c.espelho = types.SimpleNamespace(reconciliar=lambda eventos: True, fechar=lambda: None)   # o espelho no Optuna tem testes próprios
    return c


def pedir(c, nivel):
    """Pede ao DEHB até sair um trabalho do nível desejado, com o pedido no jornal antes de tudo (como `_ciclo`). Os pedidos que não
    interessam são informados como falha (+inf: nunca promovidos nem pais), para o bracket andar."""
    for _ in range(12):
        t = c.dehb.pedir()
        registro = c.jornal.acrescentar({'ev': 'pedir', 'trabalho': t.registro(), 'gerador': c.dehb.estado_do_gerador(), 'data': H.agora_iso()})['trabalho']
        if registro['nivel'] == nivel:
            return registro
        c._informar(registro, motivo='fora_do_teste')
    raise AssertionError(f'o DEHB não pediu nenhum trabalho do nível {nivel}')


def montar(c, registro):
    """A pasta do trabalho como o coordenador a monta (`_lancar`), com um processo que não faz nada no lugar do avaliador."""
    c._lancar(registro)
    v = c.em_voo.pop(registro['id'])
    v['proc'].wait(timeout=300)
    v['saida'].close()
    pasta = c._dir_trabalho(registro['id'])
    return pasta, ler(pasta, 'trabalho.json')


def trabalho_montado(oficina, study, nivel, **kw):
    c = coordenador(oficina, study, **kw)
    registro = pedir(c, nivel)
    pasta, tr = montar(c, registro)
    return c, registro, pasta, tr


def digital(rede):
    h = hashlib.sha1()
    for v in rede.state_dict().values():
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def trava_livre(pasta):
    t = TravaDoSO(os.path.join(pasta, '_EXECUTANDO.lock'))
    livre = t.tentar()
    t.soltar()
    return livre


def conferir_higiene(pasta):
    """Nada não finito no que o coordenador lê (o escritor atômico usa allow_nan=False) e nenhum temporário esquecido."""
    def recusar(constante):
        raise ValueError(f'{constante} num JSON do avaliador')
    assert [f for f in os.listdir(pasta) if '.tmp' in f] == []
    for nome in ('resultado.json', 'falha.json', '_BATIMENTO.json', 'trabalho.json'):
        if os.path.exists(os.path.join(pasta, nome)):
            with open(os.path.join(pasta, nome), encoding='utf-8') as f:
                json.loads(f.read(), parse_constant=recusar)


# ── corridas REAIS (uma vez por módulo) ──────────────────────────────────────────────────────────
@pytest.fixture(scope='module')
def corridas(oficina):
    """Duas execuções reais de `avaliar()`. Só EXECUTA e guarda; quem confere são os testes.
      baixa  trabalho 0 (fidelidade 486) num subprocesso `--avaliar`, lançado por `_lancar` e colhido por `_colher`;
      alta   os outros dois do degrau são informados como falha, então o DEHB PROMOVE o trabalho 0: a MESMA configuração, agora na
             fidelidade máxima, neste processo, com `avaliacao.avaliar` espionada (a verdadeira roda; a espiã só anota).
    Um fio só no torch (aqui e no subprocesso): com a rede de 19 mil parâmetros os 10 fios padrão só disputam a CPU com o ambiente, e a
    máquina é o notebook de trabalho do dono."""
    mp = pytest.MonkeyPatch()
    mp.setenv('OMP_NUM_THREADS', '1')                         # o subprocesso herda (e herda o CUDA_VISIBLE_DEVICES=-1 de `sem_gpu`)
    mp.setenv('MKL_NUM_THREADS', '1')
    fios = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        c = coordenador(oficina, 'real', comando=comando_real)
        baixa = pedir(c, 0)
        c._lancar(baixa)
        proc = c.em_voo[baixa['id']]['proc']
        rc_baixa = proc.wait(timeout=3600)
        c._colher()

        anotado, verdadeira = [], A.avaliar

        def espia(politica, env, sementes, regime='in_dist'):
            res = verdadeira(politica, env, sementes, regime)
            anotado.append({'sementes': list(sementes), 'regime': regime, 'digital': digital(politica.agente.rede), 'media': float(np.mean(res['recompensas']))})
            return res

        mp.setattr(A, 'avaliar', espia)
        c.comando_avaliador = comando_nulo
        alta = pedir(c, 1)
        pasta_alta, _ = montar(c, alta)
        rc_alta = H.avaliar(pasta_alta)
        c._tratar_termino(alta, rc_alta)
    finally:
        torch.set_num_threads(fios)
        mp.undo()
    return {'c': c, 'baixa': baixa, 'alta': alta, 'pasta_baixa': c._dir_trabalho(baixa['id']), 'pasta_alta': pasta_alta,
            'rc': {'baixa': rc_baixa, 'alta': rc_alta}, 'pid_baixa': proc.pid, 'anotado': anotado}


def test_avaliador_real_na_fidelidade_baixa_entrega_o_indicador_com_passos_reais_e_o_coordenador_aceita(corridas, oficina):
    c, registro, pasta = corridas['c'], corridas['baixa'], corridas['pasta_baixa']
    assert corridas['rc']['baixa'] == 0, open(os.path.join(pasta, 'avaliacao.log'), encoding='utf-8', errors='replace').read()[-2000:]
    tr, res, met, cfg, bat = (ler(pasta, n) for n in ('trabalho.json', 'resultado.json', 'metricas.json', 'config.json', '_BATIMENTO.json'))
    assert tr['fidelidade'] == 486 and not tr['fidelidade_maxima'] and tr['blocos'] == 3 and tr['dispositivo'] == 'cpu' and tr['smoke']
    assert res['impressao'] == tr['impressao'] == H.impressao_digital(c.cab['id_execucao'], registro, [42], N_SMOKE, c.hash_git) and res['id'] == registro['id']
    # o treino anda em coletas inteiras (V x 365 = 730): 486 pedidos viram 730 treinados, e é isso que se cobra
    assert res['passos'] == met['por_seed'][0]['passos'] == met['por_seed'][0]['curva_treino'][-1]['passos'] == COLETA_SMOKE
    assert res['fre'] == met['fre'] == COLETA_SMOKE / N_SMOKE and res['fre'] != tr['fidelidade'] / N_SMOKE
    assert res['aptidao'] == -res['indicador'] and res['indicador'] != 0.0, 'o DEHB minimiza: aptidão = -indicador'
    assert res['objetivo'] is None and met['objetivo'] is None
    curva = met['por_seed'][0]['curva_validacao_rollback']
    assert [p['passos'] for p in curva] == [COLETA_SMOKE] and not met['por_seed'][0]['indicador_pos_treino']
    assert res['indicador'] == met['indicador'] == pytest.approx(norma(curva[-1]['media'], oficina['referencias'], 'validacao_rollback', 'in_dist'), rel=1e-12)
    assert res['curva'] == [{'passos': COLETA_SMOKE, 'proxy_norm': res['indicador']}]
    assert 'validacao_objetivo' not in met['por_seed'][0] and 'objetivo' not in met['por_seed'][0]     # validação-objetivo só na fidelidade máxima
    assert cfg['experimento'] == 'hpo_dehb' and cfg['id_trabalho'] == registro['id'] and cfg['fidelidade'] == 486 and cfg['orcamento_passos'] == N_SMOKE
    assert cfg['seeds_trial'] == [42] and cfg['hp']['V'] == 2 and cfg['hp']['blocos'] == 3 and cfg['hp']['lr'] == tr['hp']['lr'] and cfg['dehb']['nivel'] == 0
    assert cfg['dispositivo'] == 'cpu' and res['dispositivo'] == 'cpu'
    assert bat['pid'] == corridas['pid_baixa'] != os.getpid() and bat['passos'] == COLETA_SMOKE and bat['bloco'] == 1
    assert trava_livre(pasta) and os.path.exists(os.path.join(pasta, 'avaliacao.log'))
    conferir_higiene(pasta)
    inf = next(e for e in c.jornal.eventos if e['ev'] == 'informar' and e['id'] == registro['id'])
    assert not inf['falhou'] and inf['aptidao'] == res['aptidao'] and inf['passos'] == COLETA_SMOKE and inf['fre'] == 0.5 and inf['objetivo'] is None
    assert not inf['fidelidade_maxima'] and inf['curva'] == res['curva'] and not any(e['ev'] == 'tentativa_falhou' for e in c.jornal.eventos)


def test_avaliador_real_na_fidelidade_maxima_mede_o_objetivo_nos_tres_regimes_no_melhor_bloco_de_rollback(corridas, oficina):
    c, registro, pasta, refs = corridas['c'], corridas['alta'], corridas['pasta_alta'], oficina['referencias']
    assert corridas['rc']['alta'] == 0, ler(pasta, 'falha.json') if os.path.exists(os.path.join(pasta, 'falha.json')) else 'sem falha.json'
    tr, res, met, bat = (ler(pasta, n) for n in ('trabalho.json', 'resultado.json', 'metricas.json', '_BATIMENTO.json'))
    assert tr['fidelidade'] == N_SMOKE and tr['fidelidade_maxima'] and tr['blocos'] == 9 and res['impressao'] == tr['impressao']
    seed = met['por_seed'][0]
    por_regime = seed['validacao_objetivo']
    assert list(por_regime) == P.NOMES_REGIMES
    for regime, r in por_regime.items():
        assert r['normalizada'] == pytest.approx(norma(r['media'], refs, 'validacao_objetivo', regime), rel=1e-12)
    assert res['objetivo'] == met['objetivo'] == seed['objetivo'] == pytest.approx(np.mean([r['normalizada'] for r in por_regime.values()]), rel=1e-12)
    assert res['aptidao'] == -res['objetivo'] and res['objetivo'] != 0.0, 'o DEHB minimiza: aptidão = -objetivo'
    assert [p['passos'] for p in res['curva']] == [COLETA_SMOKE, 2 * COLETA_SMOKE] and all(np.isfinite(p['proxy_norm']) for p in res['curva'])
    assert res['indicador'] == res['curva'][-1]['proxy_norm'] and res['passos'] == N_SMOKE and res['fre'] == 1.0
    # o que foi avaliado, com que sementes e em que estado da rede
    rollback, objetivo = corridas['anotado'][:2], corridas['anotado'][2:]
    assert [(a['sementes'], a['regime']) for a in rollback] == [([500, 501, 502], 'in_dist')] * 2
    assert [(a['sementes'], a['regime']) for a in objetivo] == [([520, 521, 522], regime) for regime in P.NOMES_REGIMES]
    assert [a['media'] for a in rollback] == [p['media'] for p in seed['curva_validacao_rollback']]
    assert [a['media'] for a in objetivo] == [por_regime[regime]['media'] for regime in P.NOMES_REGIMES]
    k = int(np.argmax([a['media'] for a in rollback]))                         # o primeiro dos melhores
    assert seed['melhor'] == {'valor': rollback[k]['media'], 'passos': (k + 1) * COLETA_SMOKE}
    assert {a['digital'] for a in objetivo} == {rollback[k]['digital']}           # o objetivo é medido no estado do MELHOR bloco (restaurado se não for o último)
    assert rollback[0]['digital'] != rollback[1]['digital']                       # o treino mexeu na rede entre os blocos: a conferência acima distingue
    assert bat['pid'] == os.getpid() and bat['passos'] == N_SMOKE and bat['regime'] == 'humid'
    assert trava_livre(pasta)
    conferir_higiene(pasta)
    inf = next(e for e in c.jornal.eventos if e['ev'] == 'informar' and e['id'] == registro['id'])
    assert inf['fidelidade_maxima'] and not inf['falhou'] and inf['aptidao'] == -inf['objetivo'] == res['aptidao'] and inf['fre'] == 1.0


def test_avaliador_real_e_deterministico_em_cpu_a_promocao_reproduz_bit_a_bit_o_indicador_da_fidelidade_baixa(corridas):
    """Mesma configuração + mesma semente, treinada DUAS vezes em processos diferentes (subprocesso x este): o 1º bloco da fidelidade
    máxima É o indicador da fidelidade baixa, bit a bit — 'o mesmo número, no mesmo passo' (N // 3 com 3 blocos tem as fronteiras dos 3
    primeiros blocos de N com 9). É o que deixa o DEHB re-treinar do clone em cada fidelidade sem mudar a régua da promoção."""
    assert corridas['rc'] == {'baixa': 0, 'alta': 0}
    assert corridas['alta']['origem'] == 'promocao' and corridas['alta']['vetor'] == corridas['baixa']['vetor']
    baixa, alta = ler(corridas['pasta_baixa'], 'resultado.json'), ler(corridas['pasta_alta'], 'resultado.json')
    assert alta['curva'][0] == baixa['curva'][0] and alta['curva'][0]['proxy_norm'] == baixa['indicador'] == -baixa['aptidao']
    treino_baixa, treino_alta = (ler(corridas[p], 'metricas.json')['por_seed'][0]['curva_treino'] for p in ('pasta_baixa', 'pasta_alta'))
    assert treino_alta[:len(treino_baixa)] == treino_baixa and len(treino_alta) == 2 * len(treino_baixa)      # perdas, entropia, KL: o treino inteiro


# ── dublês de roteiro: a lógica de avaliar() sem treinar nem simular ─────────────────────────────
@pytest.fixture
def palco(monkeypatch):
    """`ppo.treinar_rl` de roteiro: carrega o MESMO checkpoint, soma 1 a um viés IN-PLACE antes de cada bloco (como o otimizador: cada
    bloco tem uma impressão digital própria), chama log e callback, e pode explodir num bloco. `avaliacao.avaliar` de roteiro: devolve
    as recompensas de `palco['recompensas']` na ordem das chamadas e anota sementes, regime e o estado da rede avaliada."""
    p = {'recompensas': [], 'avaliacoes': [], 'treinos': [], 'sem_callback': False, 'erro': None, 'erro_no_treino': 0, 'erro_no_bloco': 2, 'durante': None}

    def treinar_rl(politica_il, hp, seed, orcamento_passos, callback=None, log=print, dispositivo=None):
        from politica import AgenteRL
        p['treinos'].append({'politica_il': politica_il, 'hp': dict(hp), 'seed': seed, 'orcamento': orcamento_passos, 'dispositivo': dispositivo})
        agente = AgenteRL.carregar_de(politica_il, 'cpu')
        bloco, passos = orcamento_passos // int(hp['blocos']), 0
        for k in range(1, (0 if p['sem_callback'] else int(hp['blocos'])) + 1):
            if p['erro'] is not None and len(p['treinos']) - 1 == p['erro_no_treino'] and k == p['erro_no_bloco']:
                raise p['erro']
            with torch.no_grad():
                agente.rede.head[-1].bias.add_(1.0)
            passos = k * bloco + SOBRA
            log(f'bloco {k}')
            if p['durante'] is not None:
                p['durante']()
            callback(passos, agente)
        return {'agente': agente, 'passos': passos or orcamento_passos + SOBRA, 'curva': [{'passos': passos, 'fase': 'ppo'}]}

    def avaliar(politica, env, sementes, regime='in_dist'):
        k = len(p['avaliacoes'])
        m = p['recompensas'][k] if k < len(p['recompensas']) else 0.0
        p['avaliacoes'].append({'sementes': list(sementes), 'regime': regime, 'digital': digital(politica.agente.rede), 'media': m})
        return {'recompensas': [m] * len(sementes), 'contagem_acoes': {'esperar': 3, 'plantar': 1}, 'comp': {'colheita': [m] * len(sementes)}}

    monkeypatch.setattr(PPO, 'treinar_rl', treinar_rl)
    monkeypatch.setattr(A, 'avaliar', avaliar)
    monkeypatch.setattr(envs_lote, 'fabrica_env_padrao', lambda: 'ambiente de roteiro')
    return p


OBJETIVO_ROTEIRO = [3010.0, 1686.0, 2248.0]      # in_dist, dry, humid -> normalizadas 6/11, 1/2, 1/2 com as referências de validação-objetivo
OBJETIVO_NORMALIZADO = (6 / 11 + 0.5 + 0.5) / 3


def test_fidelidade_baixa_informa_o_ultimo_bloco_com_os_passos_reais_e_nao_toca_a_validacao_objetivo(oficina, palco):
    c, registro, pasta, tr = trabalho_montado(oficina, 'baixa', 0)
    palco['recompensas'] = [3000.0, 1000.0, 2000.0]                                # o MELHOR bloco é o 1º; o indicador é o ÚLTIMO (o que o podador vê)
    assert H.avaliar(pasta) == 0
    res, met, bat = ler(pasta, 'resultado.json'), ler(pasta, 'metricas.json'), ler(pasta, '_BATIMENTO.json')
    passos = [k * 162 + SOBRA for k in (1, 2, 3)]
    assert res['indicador'] == pytest.approx((2000.0 - 600.0) / 4400.0) and res['objetivo'] is None
    assert res['aptidao'] == -res['indicador'], 'o DEHB minimiza: aptidão = -indicador'
    assert res['impressao'] == tr['impressao'] and res['id'] == registro['id']
    assert res['passos'] == passos[-1] and res['fre'] == passos[-1] / N_SMOKE != 486 / N_SMOKE         # cobra-se o treinado, não o pedido
    assert [p['passos'] for p in res['curva']] == passos
    assert [p['proxy_norm'] for p in res['curva']] == pytest.approx([(m - 600.0) / 4400.0 for m in palco['recompensas']])
    assert [(a['sementes'], a['regime']) for a in palco['avaliacoes']] == [([500, 501, 502], 'in_dist')] * 3   # só validação-rollback, in-dist
    assert 'validacao_objetivo' not in met['por_seed'][0] and met['objetivo'] is None
    assert met['por_seed'][0]['melhor'] == {'valor': 3000.0, 'passos': passos[0]} and not met['por_seed'][0]['indicador_pos_treino']
    assert bat == {'data': bat['data'], 'pid': os.getpid(), 'passos': passos[-1], 'bloco': 3}
    treino = palco['treinos'][0]
    assert treino['politica_il'] == os.path.join(oficina['il'], 'seed_42', 'modelo_final.pt') and treino['seed'] == 42
    assert treino['orcamento'] == 486 and treino['dispositivo'] == 'cpu' and treino['hp']['blocos'] == 3 and treino['hp']['inicializacao'] == 'professor'
    assert {k: treino['hp'].get(k) for k in H.HP_SMOKE} == H.HP_SMOKE and treino['hp']['lr'] == tr['hp']['lr']
    assert trava_livre(pasta)
    conferir_higiene(pasta)


def test_fidelidade_maxima_mede_o_objetivo_no_melhor_bloco_restaurado_e_o_indicador_no_ultimo(oficina, palco):
    c, registro, pasta, tr = trabalho_montado(oficina, 'melhor_no_meio', 1)
    palco['recompensas'] = [1000.0, 4000.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 2500.0] + OBJETIVO_ROTEIRO
    assert H.avaliar(pasta) == 0
    res, met, bat = ler(pasta, 'resultado.json'), ler(pasta, 'metricas.json'), ler(pasta, '_BATIMENTO.json')
    rollback, objetivo = palco['avaliacoes'][:9], palco['avaliacoes'][9:]
    assert [(a['sementes'], a['regime']) for a in objetivo] == [([520, 521, 522], regime) for regime in P.NOMES_REGIMES]
    assert len({a['digital'] for a in rollback}) == 9
    assert {a['digital'] for a in objetivo} == {rollback[1]['digital']}, 'o objetivo tem de ser medido no estado do melhor bloco de rollback (o 2º), não no último'
    assert res['objetivo'] == pytest.approx(OBJETIVO_NORMALIZADO)
    assert res['aptidao'] == -res['objetivo'], 'o DEHB minimiza: aptidão = -objetivo (e não -indicador) na fidelidade máxima'
    assert res['indicador'] == pytest.approx((2500.0 - 600.0) / 4400.0)               # o indicador continua sendo o ÚLTIMO bloco
    por_regime = met['por_seed'][0]['validacao_objetivo']
    assert [por_regime[r]['normalizada'] for r in P.NOMES_REGIMES] == pytest.approx([6 / 11, 0.5, 0.5])
    assert [por_regime[r]['media'] for r in P.NOMES_REGIMES] == OBJETIVO_ROTEIRO
    assert met['por_seed'][0]['melhor'] == {'valor': 4000.0, 'passos': 2 * 162 + SOBRA}
    assert res['passos'] == N_SMOKE - 2 + SOBRA and res['fre'] == res['passos'] / N_SMOKE and len(res['curva']) == 9   # 9 blocos de 162 = 1458
    assert bat['regime'] == 'humid' and bat['passos'] == res['passos'] and bat['pid'] == os.getpid()
    assert trava_livre(pasta)
    conferir_higiene(pasta)


def test_quando_o_ultimo_bloco_e_o_melhor_o_objetivo_e_medido_no_estado_final(oficina, palco):
    c, registro, pasta, tr = trabalho_montado(oficina, 'melhor_no_fim', 1)
    palco['recompensas'] = [1000.0 + 100.0 * k for k in range(9)] + OBJETIVO_ROTEIRO
    assert H.avaliar(pasta) == 0
    assert {a['digital'] for a in palco['avaliacoes'][9:]} == {palco['avaliacoes'][8]['digital']}
    assert ler(pasta, 'metricas.json')['por_seed'][0]['melhor'] == {'valor': 1800.0, 'passos': 9 * 162 + SOBRA}


def test_empate_entre_blocos_fica_com_o_primeiro_que_e_o_que_metricas_declara(oficina, palco):
    """O 2º bloco e o último empatam. `melhor` (gravado em metricas.json) é o 2º — só troca com ganho ESTRITO —, então é nele que o
    objetivo tem de ser medido: senão o arquivo diria um bloco e o número viria de outro."""
    c, registro, pasta, tr = trabalho_montado(oficina, 'empate', 1)
    palco['recompensas'] = [1000.0, 3000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 3000.0] + OBJETIVO_ROTEIRO
    assert H.avaliar(pasta) == 0
    assert ler(pasta, 'metricas.json')['por_seed'][0]['melhor'] == {'valor': 3000.0, 'passos': 2 * 162 + SOBRA}
    assert {a['digital'] for a in palco['avaliacoes'][9:]} == {palco['avaliacoes'][1]['digital']}


def test_sem_smoke_entram_as_sementes_de_validacao_do_protocolo_e_os_hp_do_pedido_sem_reducao(oficina, palco):
    c, registro, pasta, tr = trabalho_montado(oficina, 'cheio', 1, smoke=False)
    palco['recompensas'] = [1000.0 + 100.0 * k for k in range(9)] + OBJETIVO_ROTEIRO
    assert H.avaliar(pasta) == 0
    rollback, objetivo = palco['avaliacoes'][:9], palco['avaliacoes'][9:]
    assert all(a['sementes'] == list(range(500, 520)) and a['regime'] == 'in_dist' for a in rollback)       # validação-rollback: 20 sementes
    assert [(a['sementes'], a['regime']) for a in objetivo] == [(list(range(520, 550)), r) for r in P.NOMES_REGIMES]   # validação-objetivo: 30
    assert not any(s >= P.SEED_TESTE for a in palco['avaliacoes'] for s in a['sementes'])                   # teste (999+) nunca entra na HPO
    hp = palco['treinos'][0]['hp']
    assert hp == dict(tr['hp'], inicializacao='professor', blocos=9) and 'V' not in hp                      # HP_SMOKE só no smoke
    assert palco['treinos'][0]['orcamento'] == N_CHEIO
    res, cfg = ler(pasta, 'resultado.json'), ler(pasta, 'config.json')
    assert res['passos'] == N_CHEIO + SOBRA and res['fre'] == (N_CHEIO + SOBRA) / N_CHEIO and res['objetivo'] == pytest.approx(OBJETIVO_NORMALIZADO)
    assert cfg['orcamento_passos'] == N_CHEIO and cfg['fidelidade'] == N_CHEIO and cfg['hp'] == hp and cfg['study'] == 'cheio'
    assert cfg['dehb'] == {k: registro[k] for k in ('nivel', 'bracket', 'degrau', 'origem', 'pais')}


def test_varias_sementes_de_treino_dividem_o_fre_e_tiram_a_media_do_indicador_e_do_objetivo(oficina, palco):
    c, registro, pasta, tr = trabalho_montado(oficina, 'duas_sementes', 1, seeds=(42, 43))
    palco['recompensas'] = ([1000.0] * 8 + [2800.0] + OBJETIVO_ROTEIRO) + ([1000.0] * 8 + [1920.0] + [5010.0, 3006.0, 4008.0])   # 2ª semente: objetivo 1,0
    assert H.avaliar(pasta) == 0
    res, met = ler(pasta, 'resultado.json'), ler(pasta, 'metricas.json')
    assert [(t['seed'], os.path.basename(os.path.dirname(t['politica_il']))) for t in palco['treinos']] == [(42, 'seed_42'), (43, 'seed_43')]
    por_semente = N_SMOKE - 2 + SOBRA
    assert res['passos'] == 2 * por_semente and res['fre'] == 2 * por_semente / (N_SMOKE * 2)               # FRE por execução completa, não por semente
    assert res['indicador'] == pytest.approx(((2800.0 - 600.0) / 4400.0 + (1920.0 - 600.0) / 4400.0) / 2)
    assert res['objetivo'] == pytest.approx((OBJETIVO_NORMALIZADO + 1.0) / 2) and res['aptidao'] == -res['objetivo']
    assert [s['seed'] for s in met['por_seed']] == [42, 43] and [s['objetivo'] for s in met['por_seed']] == pytest.approx([OBJETIVO_NORMALIZADO, 1.0])
    assert res['curva'][-1]['proxy_norm'] == pytest.approx((2800.0 - 600.0) / 4400.0) and len(res['curva']) == 9   # a curva do espelho é a da 1ª semente
    assert ler(pasta, '_BATIMENTO.json')['passos'] == 2 * por_semente
    inf_antes = len(c.jornal.eventos)
    c._tratar_termino(registro, 0)
    inf = c.jornal.eventos[inf_antes]
    assert inf['ev'] == 'informar' and inf['fre'] == res['fre'] and inf['aptidao'] == res['aptidao'] and c.fre_consumidos() == pytest.approx(res['fre'])


def test_orcamento_menor_que_um_bloco_mede_uma_vez_no_estado_final(oficina, palco):
    c, registro, pasta, tr = trabalho_montado(oficina, 'sem_bloco', 0)
    palco['sem_callback'], palco['recompensas'] = True, [1700.0]
    assert H.avaliar(pasta) == 0
    res, met = ler(pasta, 'resultado.json'), ler(pasta, 'metricas.json')
    assert met['por_seed'][0]['indicador_pos_treino'] and res['curva'] == [{'passos': 486 + SOBRA, 'proxy_norm': pytest.approx(0.25)}]
    assert res['indicador'] == pytest.approx(0.25) and res['passos'] == 486 + SOBRA and len(palco['avaliacoes']) == 1


# ── falhas: da configuração (saída 3) x transitória (saída 4) ────────────────────────────────────
FALHAS_DE_ROTEIRO = [
    # da CONFIGURAÇÃO: o DEHB é informado (+inf) e o trabalho não se repete
    (ValueError('nan nos logits da política'), 3, 'excecao_no_treino'),
    (RuntimeError('size mismatch for lstm.weight_ih_l0'), 3, 'excecao_no_treino'),       # RuntimeError comum NÃO é GPU
    (AssertionError('len_seq deve dividir 365'), 3, 'excecao_no_treino'),
    (KeyError('lr'), 3, 'excecao_no_treino'),
    # TRANSITÓRIA: a máquina, não a configuração — o coordenador repete. Cada caso isola UM termo da regra
    (torch.cuda.OutOfMemoryError('faltou memória na placa'), 4, 'gpu_ou_sistema'),        # pelo TIPO (a mensagem não diz a sigla)
    (RuntimeError('CUDA error: an illegal memory access was encountered'), 4, 'gpu_ou_sistema'),
    (RuntimeError('cuDNN error: falha de execução'), 4, 'gpu_ou_sistema'),
    (MemoryError('sem RAM'), 4, 'gpu_ou_sistema'),
    (PermissionError(13, 'arquivo preso pelo antivírus'), 4, 'gpu_ou_sistema'),           # OSError: disco, OneDrive, antivírus
]


@pytest.mark.parametrize('erro, saida, motivo', FALHAS_DE_ROTEIRO, ids=[type(e).__name__ + '_' + str(k) for k, (e, _, _) in enumerate(FALHAS_DE_ROTEIRO)])
def test_excecao_no_treino_vira_falha_json_com_impressao_passos_gastos_e_a_classe_certa(oficina, palco, erro, saida, motivo, request):
    c, registro, pasta, tr = trabalho_montado(oficina, 'falha_' + request.node.callspec.id, 0)
    palco['erro'], palco['erro_no_bloco'] = erro, 2                                 # explode no 2º bloco: o 1º (162 + sobra) já foi treinado
    assert H.avaliar(pasta) == saida
    falha = ler(pasta, 'falha.json')
    assert falha['motivo'] == motivo and falha.get('impressao') == tr['impressao'] and falha['id'] == registro['id']
    assert falha['passos'] == 162 + SOBRA                                            # o parcial treinado entra no orçamento
    assert type(erro).__name__ in falha['detalhe'] and 'Traceback' in falha['detalhe'] and falha['hash_git'] == c.hash_git
    assert not os.path.exists(os.path.join(pasta, 'resultado.json')) and not os.path.exists(os.path.join(pasta, 'metricas.json'))
    assert trava_livre(pasta)
    conferir_higiene(pasta)
    # o coordenador ACEITA a falha (impressão do pedido) e cobra os passos: informa se é da configuração, repete se é transitória
    c._tratar_termino(registro, saida)
    ev = c.jornal.eventos[-1]
    assert ev['id'] == registro['id'] and ev['passos'] == 162 + SOBRA and ev['fre'] == (162 + SOBRA) / N_SMOKE and ev['motivo'] == motivo
    assert (ev['ev'], [r['id'] for r in c._fila]) == (('informar', []) if saida == 3 else ('tentativa_falhou', [registro['id']]))


def test_falha_na_segunda_semente_cobra_os_passos_das_duas(oficina, palco):
    c, registro, pasta, tr = trabalho_montado(oficina, 'falha_na_2a', 0, seeds=(42, 43))
    palco['erro'], palco['erro_no_treino'], palco['erro_no_bloco'] = ValueError('explodiu na 2ª semente'), 1, 3
    assert H.avaliar(pasta) == 3
    assert ler(pasta, 'falha.json')['passos'] == (3 * 162 + SOBRA) + (2 * 162 + SOBRA)   # a 1ª semente inteira + 2 blocos da 2ª
    assert ler(pasta, '_BATIMENTO.json')['passos'] == (3 * 162 + SOBRA) + (2 * 162 + SOBRA)


@pytest.mark.parametrize('nivel, recompensas', [(0, [1000.0, 1000.0, float('nan')]),
                                               (1, [1000.0] * 9 + [3010.0, float('inf'), 2248.0]),
                                               (1, [1000.0] * 9 + [3010.0, float('nan'), 2248.0])], ids=['indicador_nan', 'objetivo_inf', 'objetivo_nan'])
def test_aptidao_nao_finita_e_falha_da_configuracao_e_nunca_vira_resultado(oficina, palco, nivel, recompensas, request):
    c, registro, pasta, tr = trabalho_montado(oficina, 'nao_finita_' + request.node.callspec.id, nivel)
    palco['recompensas'] = recompensas
    assert H.avaliar(pasta) == 3
    falha = ler(pasta, 'falha.json')
    assert falha['motivo'] == 'aptidao_nao_finita' and falha.get('impressao') == tr['impressao']
    assert falha['passos'] == tr['blocos'] * 162 + SOBRA                             # o treino inteiro foi gasto: cobra-se inteiro
    assert not os.path.exists(os.path.join(pasta, 'resultado.json'))
    assert trava_livre(pasta)
    conferir_higiene(pasta)


def test_indicador_nao_finito_no_topo_e_falha_com_nome_e_nao_deixa_temporario_na_pasta(oficina, palco):
    """Fidelidade máxima com o rollback NaN e o objetivo finito. Antes (defeito corrigido em 19/09/2026) a aptidão (= -objetivo) passava
    pela conferência de finitude, o escritor atômico (allow_nan=False, e está certo) recusava o resultado.json com o indicador NaN, a
    falha saía como 'excecao_no_treino' e sobrava `resultado.json.tmp<pid>` com meio JSON na pasta do trabalho."""
    c, registro, pasta, tr = trabalho_montado(oficina, 'tmp_esquecido', 1)
    palco['recompensas'] = [float('nan')] * 9 + OBJETIVO_ROTEIRO
    assert H.avaliar(pasta) == 3
    falha = ler(pasta, 'falha.json')
    assert falha.get('impressao') == tr['impressao'] and not os.path.exists(os.path.join(pasta, 'resultado.json'))
    # o indicador conta também no topo (é ele que escolhe o bloco): a falha tem de sair com o NOME certo, não como exceção do escritor
    assert falha['motivo'] == 'aptidao_nao_finita' and 'indicador=nan' in falha['detalhe']
    conferir_higiene(pasta)


def test_trava_ocupada_sai_como_transitoria_sem_treinar_nem_gravar_nada(oficina, palco):
    """Um avaliador órfão de outra sessão ainda segura a pasta: o segundo NÃO pode treinar por cima nem apagar o rastro do primeiro."""
    c, registro, pasta, tr = trabalho_montado(oficina, 'ocupada', 0)
    dono = TravaDoSO(os.path.join(pasta, '_EXECUTANDO.lock'))
    assert dono.tentar()
    try:
        antes = sorted(os.listdir(pasta))
        assert H.avaliar(pasta) == H.SAIDA_FALHA_TRANSITORIA
        assert palco['treinos'] == [] and sorted(os.listdir(pasta)) == antes
        assert not trava_livre(pasta)                                              # e não soltou a trava que não era dele
    finally:
        dono.soltar()


@pytest.mark.parametrize('erro', [None, ValueError('explodiu')], ids=['sucesso', 'falha'])
def test_trava_fica_presa_durante_o_treino_e_e_solta_no_fim_mesmo_com_o_objeto_vivo(oficina, palco, monkeypatch, erro):
    """A trava tem de ser solta por `soltar()`, não pelo coletor de lixo: o teste mantém o objeto vivo, como faria um chamador que
    guardasse o quadro (depurador, traceback retido)."""
    vivas, durante = [], []

    class TravaViva(TravaDoSO):
        def __init__(self, caminho):
            super().__init__(caminho)
            vivas.append(self)

    c, registro, pasta, tr = trabalho_montado(oficina, 'trava_' + ('falha' if erro else 'sucesso'), 0)
    monkeypatch.setattr(H, 'Trava', TravaViva)
    palco['recompensas'], palco['erro'], palco['erro_no_bloco'] = [1000.0, 1100.0, 1200.0], erro, 3
    palco['durante'] = lambda: durante.append(trava_livre(pasta))
    try:
        assert H.avaliar(pasta) == (3 if erro else 0)
        assert durante and not any(durante), 'a trava tem de estar presa enquanto o treino roda'
        assert len(vivas) == 1 and trava_livre(pasta), 'avaliar() voltou e a trava _EXECUTANDO.lock continua presa'
    finally:
        for t in vivas:
            t.soltar()


def test_importacao_que_falha_e_transitoria_e_deixa_falha_json_com_a_impressao(oficina, monkeypatch):
    c, registro, pasta, tr = trabalho_montado(oficina, 'importacao', 0)
    monkeypatch.setitem(sys.modules, 'ppo', None)                                   # `import ppo` passa a levantar ImportError
    assert H.avaliar(pasta) == H.SAIDA_FALHA_TRANSITORIA
    falha = ler(pasta, 'falha.json')
    assert falha['motivo'] == 'importacao' and falha.get('impressao') == tr['impressao'] and falha['passos'] == 0
    assert 'ImportError' in falha['detalhe'] or 'ModuleNotFoundError' in falha['detalhe']
    assert ler(pasta, '_BATIMENTO.json')['pid'] == os.getpid()                       # o batimento sai ANTES de importar torch (que demora)


# ── falhas com o treinar_rl VERDADEIRO (morrem antes do primeiro passo de ambiente: são rápidas) ──
def test_checkpoint_corrompido_arquitetura_divergente_e_hp_invalido_sao_falhas_da_configuracao(oficina):
    lixo = oficina['raiz'] / 'il corrompido'
    (lixo / 'seed_42').mkdir(parents=True, exist_ok=True)
    (lixo / 'seed_42' / 'modelo_final.pt').write_bytes(b'isto nao e um checkpoint')
    for caso, mexida in (('corrompido', {'warmstart': str(lixo)}), ('len_seq', {'hp': {'len_seq': 7}}), ('arquitetura', {'hp': {'dim_lstm': 48}})):
        c, registro, pasta, tr = trabalho_montado(oficina, 'real_' + caso, 0)
        alterar_trabalho(pasta, **mexida)
        assert H.avaliar(pasta) == H.SAIDA_FALHA_DA_CONFIGURACAO, caso
        falha = ler(pasta, 'falha.json')
        assert falha['motivo'] == 'excecao_no_treino' and falha.get('impressao') == tr['impressao'] and falha['passos'] == 0, caso
        assert c._ler_valido(registro['id'], 'falha.json', registro) is not None and trava_livre(pasta)
        conferir_higiene(pasta)


def test_checkpoint_que_sumiu_e_erro_de_sistema_e_nao_da_configuracao(oficina):
    """FileNotFoundError é OSError: transitória. `main` confere o checkpoint antes de começar, então um arquivo que some no meio do
    estudo é a máquina (OneDrive, disco) — se fosse 'da configuração', toda a população viraria +inf por um defeito que não é dela;
    como transitória, repete e, se persistir, o disjuntor para o estudo."""
    c, registro, pasta, tr = trabalho_montado(oficina, 'real_sumiu', 0)
    alterar_trabalho(pasta, warmstart=str(oficina['raiz'] / 'pasta que não existe'))
    assert H.avaliar(pasta) == H.SAIDA_FALHA_TRANSITORIA
    falha = ler(pasta, 'falha.json')
    assert falha['motivo'] == 'gpu_ou_sistema' and falha.get('impressao') == tr['impressao'] and 'FileNotFoundError' in falha['detalhe']
    assert trava_livre(pasta)
