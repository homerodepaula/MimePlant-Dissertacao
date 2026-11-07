"""RL do zero com HPO por processo gaussiano (GP) — extensão do controle D-32 (D-38).

Pergunta: o otimizador bayesiano por GP, que deu a melhor configuração do IL + RL, também melhora o RL treinado do zero?
O estudo espelha o `tpe_zero_r3` (TPE + Hyperband do zero) em TUDO, menos no amostrador:
  mesmo orçamento (32 FRE, 1 FRE = 1.000.000 de passos), mesmo podador Hyperband(N/3, N, 3), n_startup = 5, semente 1001,
  3 trabalhadores no mesmo estudo, espaço de 8 hiperparâmetros (os 4 ligados ao professor não existem sem clone),
  incumbente (k = 1) reavaliado com as 5 sementes de treino. A escolha usa só a validação-objetivo; teste só é reportado.

  Z1  gp_zero_r3: hpo_sampler_extra.py --sampler-extra gp --inicializacao zero --n-fre 32 (3 processos)
  Z2  reavaliar_topk.py --k 1 x 5 sementes, 4 processos -> results/rodada3/hpo_reavaliacao_zero_gp
  Z3  analise_hpo.py -> ../experimentos/analise_hpo_rodada3_zero_gp
  Z4  comparar_gp_zero.py -> results/rodada3/controle_zero_gp/comparacao.json (recompensa bruta em teste)

Nada do que já existe é sobrescrito (pastas e estudo novos). Cada fase é retomável: se a máquina cair, relance este script.
Com `--execucoes 3` rodam gp_zero_r3_a|b|c (sementes 1001/1002/1003, 2 trabalhadores cada), o desenho usado no IL + RL.

Uso:  python -X utf8 execucao/encadear_gp_zero.py [--so-plano] [--execucoes 1|3]
"""
import os
import sys
import json
import time
import ctypes
import argparse
import subprocess

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)                                      # este script fica em <repositório>/execucao
EXPERIMENTOS = os.path.join(os.path.dirname(REPO), 'experimentos')  # pasta irmã do repositório (a mesma convenção de hpo/*.py)
JOBS = os.path.join(EXPERIMENTOS, '_jobs')
DIR_HPO = os.path.join(EXPERIMENTOS, 'hpo')
sys.path.insert(0, REPO)
import protocolo as P  # noqa: E402

PY = sys.executable
N = P.ORCAMENTO_RL_PASSOS
N_FRE = 32                 # o mesmo do tpe_zero_r3 (TPE + Hyperband do zero)
N_STARTUP = 5
INVOLUCRO = os.path.join(AQUI, 'hpo_sampler_extra.py')
R = os.path.relpath(P.DIR_RODADA, REPO)
REFS = os.path.join(R, 'referencias', 'referencias.json')
WS = os.path.join(R, 'il_warmstart')
REAV = os.path.join(R, 'hpo_reavaliacao_zero_gp')
SAIDA_COMP = os.path.join(R, 'controle_zero_gp')
DIR_ANALISE = os.path.join(EXPERIMENTOS, 'analise_hpo_rodada3_zero_gp')
ESTUDOS = []               # [(estudo, semente, n_trabalhadores)], preenchido em main()
MAX_LANCAMENTOS = 3
BELOW_NORMAL = 0x00004000
CREATE_NO_WINDOW = 0x08000000


def cmd(*p):
    return [PY, '-X', 'utf8'] + [str(x) for x in p]


def definir_estudos(execucoes):
    if execucoes == 1:      # espelho exato do tpe_zero_r3: 1 estudo, semente 1001, 3 trabalhadores
        return [('gp_zero_r3', 1001, 3)]
    return [(f'gp_zero_r3_{s}', semente, 2) for s, semente in (('a', 1001), ('b', 1002), ('c', 1003))]


def jobs_hpo():
    js = []
    for estudo, semente, trab in ESTUDOS:
        for w in range(trab):
            js.append({'nome': f'gpz_{estudo}_w{w}',
                       'cmd': cmd(INVOLUCRO, '--sampler-extra', 'gp', '--n-startup', N_STARTUP,
                                  '--study', estudo, '--n-fre', N_FRE, '--orcamento', N, '--inicializacao', 'zero',
                                  '--seed-sampler', semente, '--paralelo', '--trabalhador', w,
                                  '--warmstart', WS, '--referencias', REFS, '--retomar'),
                       'flags': [os.path.join(DIR_HPO, estudo, '_DONE.flag')], 'abs': True})
    return js


def jobs_reavaliacao():
    return [{'nome': 'gpz_reavaliacao',
             'cmd': cmd('hpo/reavaliar_topk.py', '--study', *[e for e, _, _ in ESTUDOS], '--k', 1, '--orcamento', N,
                        '--saida', REAV, '--paralelo', 4, '--retomar'),
             'flags': [os.path.join(REAV, '_DONE.flag')]}]


def jobs_analise():
    return [{'nome': 'gpz_analise',
             'cmd': cmd('hpo/analise_hpo.py', '--studies', *[e for e, _, _ in ESTUDOS], '--reavaliacao', REAV,
                        '--saida', os.path.join('..', 'experimentos', 'analise_hpo_rodada3_zero_gp')),
             'flags': [os.path.join(DIR_ANALISE, 'analise_hpo.json')], 'abs': True}]


def jobs_comparacao():
    return [{'nome': 'gpz_comparacao',
             'cmd': cmd(os.path.join(AQUI, 'comparar_gp_zero.py')),
             'flags': [os.path.join(SAIDA_COMP, 'comparacao.json')]}]


def log(msg):
    linha = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(linha, flush=True)
    with open(os.path.join(JOBS, 'encadeador_gp_zero.log'), 'a', encoding='utf-8') as f:
        f.write(linha + '\n')


def pronto(j):
    return all(os.path.exists(f if j.get('abs') else os.path.join(REPO, f)) for f in j['flags'])


def gravar_estado(e):
    tmp = os.path.join(JOBS, 'estado_gp_zero.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(e, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(JOBS, 'estado_gp_zero.json'))


def rodar_fase(nome_fase, jobs, estado):
    pendentes = [j for j in jobs if not pronto(j)]
    for j in jobs:
        if j not in pendentes:
            estado[j['nome']] = {'estado': 'pronto (flags já existiam)'}
    if not pendentes:
        log(f'{nome_fase}: tudo pronto, pulando')
        return True
    vivos, fila = {}, list(pendentes)
    for j in pendentes:
        j['lancamentos'] = 0
    while fila or vivos:
        while fila:
            j = fila.pop(0)
            j['lancamentos'] += 1
            saida = open(os.path.join(JOBS, j['nome'] + '.log'), 'a', encoding='utf-8')
            saida.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} lançamento {j['lancamentos']}: {' '.join(j['cmd'])}\n")
            saida.flush()
            proc = subprocess.Popen(j['cmd'], cwd=REPO, stdout=saida, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    creationflags=BELOW_NORMAL | CREATE_NO_WINDOW)
            vivos[j['nome']] = (j, proc, saida)
            estado[j['nome']] = {'estado': 'rodando', 'pid': proc.pid, 'lancamento': j['lancamentos'],
                                 'inicio': time.strftime('%Y-%m-%d %H:%M:%S')}
            gravar_estado(estado)
            log(f"{nome_fase}: lançado {j['nome']} (pid {proc.pid}, lançamento {j['lancamentos']})")
            if fila:
                time.sleep(20)     # dois trabalhadores criando o mesmo SQLite ao mesmo tempo derrubam um deles
        time.sleep(60)
        for nome, (j, proc, saida) in list(vivos.items()):
            rc = proc.poll()
            if rc is None:
                continue
            saida.close()
            del vivos[nome]
            ok = rc == 0 and pronto(j)
            estado[nome] = {'estado': 'pronto' if ok else 'falhou', 'exit': rc, 'lancamento': j['lancamentos'],
                            'fim': time.strftime('%Y-%m-%d %H:%M:%S')}
            gravar_estado(estado)
            if ok:
                log(f'{nome_fase}: {nome} concluído')
            elif j['lancamentos'] < MAX_LANCAMENTOS:
                log(f'{nome_fase}: {nome} saiu com exit={rc} sem as flags; relançando com --retomar')
                fila.append(j)
            else:
                log(f'{nome_fase}: {nome} falhou {MAX_LANCAMENTOS} vezes (exit={rc}); ver _jobs/{nome}.log')
    return all(pronto(j) for j in jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--so-plano', action='store_true')
    ap.add_argument('--execucoes', type=int, choices=(1, 3), default=1,
                    help='1 = espelho do tpe_zero_r3 (padrão); 3 = três execuções independentes, como no IL + RL')
    args = ap.parse_args()
    os.makedirs(JOBS, exist_ok=True)
    ESTUDOS.extend(definir_estudos(args.execucoes))
    fases = [('Z1 hpo gp do zero', jobs_hpo()), ('Z2 reavaliação', jobs_reavaliacao()),
             ('Z3 análise', jobs_analise()), ('Z4 comparação', jobs_comparacao())]
    log(f'plano GP do zero (D-38): orcamento={N} n_fre={N_FRE} por estudo n_startup={N_STARTUP} '
        f'estudos={ESTUDOS} k=1 | espaço de 8 hiperparâmetros | git={P.hash_git()}')
    for nome, js in fases:
        for j in js:
            log(f"  {nome}: {j['nome']}: {' '.join(j['cmd'][3:])}{'  [PRONTO]' if pronto(j) else ''}")
    if args.so_plano:
        return 0
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception as e:
        log(f'AVISO: SetThreadExecutionState falhou: {e}')
    estado = {'pid_encadeador': os.getpid(), 'inicio': time.strftime('%Y-%m-%d %H:%M:%S'),
              'parametros': {'orcamento': N, 'n_fre': N_FRE, 'n_startup': N_STARTUP, 'amostrador': 'gp',
                             'estudos': [list(x) for x in ESTUDOS], 'k': 1, 'espaco': '8 hiperparâmetros'}}
    gravar_estado(estado)
    for nome, js in fases:
        log(f'--- {nome}')
        if not rodar_fase(nome, js, estado):
            log(f'PARADO: a fase {nome} não completou. Corrija e relance (ele pula o que já tem flag).')
            estado['fim'] = 'falhou em ' + nome
            gravar_estado(estado)
            return 1
    log('FIM: GP do zero concluído. Comparação em results/rodada3/controle_zero_gp/comparacao.json')
    estado['fim'] = 'ok ' + time.strftime('%Y-%m-%d %H:%M:%S')
    gravar_estado(estado)
    return 0


if __name__ == '__main__':
    sys.exit(main())
