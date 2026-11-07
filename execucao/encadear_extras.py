"""Otimizadores EXTRA sobre o IL + RL (o método da dissertação) — enfileirado para depois do controle D-32.

Acrescenta duas famílias de otimizador ao estudo de AutoRL, com paridade total de protocolo com TPE, busca aleatória e DEHB:
mesmo espaço de 12 hiperparâmetros, mesmo objetivo, mesmo podador Hyperband(N/3, N, 3), 32 FRE por execução, 3 execuções por
otimizador (sementes 1001/1002/1003), k = 1 na reavaliação com 5 sementes, e **sempre partindo do clone do professor**
(`inicializacao = professor`, D-30) — é o algoritmo IL + RL que a pesquisa quer melhorar, não o controle do zero.

  G  gp_r3_a|b|c    otimização bayesiana com processo gaussiano (`optuna.samplers.GPSampler`).
                    Família diferente do TPE: modela a função objetivo e a incerteza, em vez da densidade de bons/ruins.
                    Sem dependência nova (usa torch e scipy, já instalados). É o regime em que GP costuma ir bem: poucas
                    dezenas de trials.
  Q  qmc_r3_a|b|c   Sobol embaralhado (`optuna.samplers.QMCSampler`). Cobertura de baixa discrepância do espaço; é a
                    amostragem que o ARLBench usa para gerar as paisagens de hiperparâmetros. Serve de baseline mais
                    forte que a busca aleatória independente.

Os dois rodam pelo invólucro `hpo_sampler_extra.py`, que reaproveita o `hpo/hpo_rl.py` inteiro e só troca o amostrador, sem
modificar nenhum arquivo do repositório.

Ordem: G (3 estudos × 2 trabalhadores = 6 processos) → reavaliação do GP → Q → reavaliação do Q → análises.
Cada fase é retomável; se a máquina cair, basta relançar este script, que pula o que já tem flag.
Uso:  python -X utf8 execucao/encadear_extras.py [--so-plano] [--sem-esperar] [--so-gp]
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
INVOLUCRO = os.path.join(AQUI, 'hpo_sampler_extra.py')
sys.path.insert(0, REPO)
import protocolo as P  # noqa: E402

PY = sys.executable
N = P.ORCAMENTO_RL_PASSOS
N_FRE = 32
N_STARTUP = 5
TRAB = 2
SEMENTES = {'a': 1001, 'b': 1002, 'c': 1003}
R = os.path.relpath(P.DIR_RODADA, REPO)
REFS = os.path.join(R, 'referencias', 'referencias.json')
WS = os.path.join(R, 'il_warmstart')
FAMILIAS = {'gp': [f'gp_r3_{s}' for s in SEMENTES], 'qmc': [f'qmc_r3_{s}' for s in SEMENTES]}
REAV = {'gp': os.path.join(R, 'hpo_reavaliacao_gp'), 'qmc': os.path.join(R, 'hpo_reavaliacao_qmc')}
# o controle D-32 termina quando esta tabela existir
ESPERAR = [os.path.join(REPO, R, 'controle_zero', 'comparacao.json')]
MAX_LANCAMENTOS = 3
BELOW_NORMAL = 0x00004000
CREATE_NO_WINDOW = 0x08000000


def cmd(*p):
    return [PY, '-X', 'utf8'] + [str(x) for x in p]


def jobs_hpo(familia):
    js = []
    for estudo in FAMILIAS[familia]:
        for w in range(TRAB):
            js.append({'nome': f'x_{estudo}_w{w}',
                       'cmd': cmd(INVOLUCRO, '--sampler-extra', familia, '--n-startup', N_STARTUP,
                                  '--study', estudo, '--n-fre', N_FRE, '--orcamento', N,
                                  '--seed-sampler', SEMENTES[estudo[-1]], '--paralelo', '--trabalhador', w,
                                  '--warmstart', WS, '--referencias', REFS, '--retomar'),
                       'flags': [os.path.join(DIR_HPO, estudo, '_DONE.flag')], 'abs': True})
    return js


def jobs_reavaliacao(familia):
    return [{'nome': f'x_reavaliacao_{familia}',
             'cmd': cmd('hpo/reavaliar_topk.py', '--study', *FAMILIAS[familia], '--k', 1, '--orcamento', N,
                        '--saida', REAV[familia], '--paralelo', 4, '--retomar'),
             'flags': [os.path.join(REAV[familia], '_DONE.flag')]}]


def jobs_analise(familia):
    saida = os.path.join('..', 'experimentos', f'analise_hpo_rodada3_{familia}')
    return [{'nome': f'x_analise_{familia}',
             'cmd': cmd('hpo/analise_hpo.py', '--studies', *FAMILIAS[familia], '--reavaliacao', REAV[familia],
                        '--saida', saida),
             'flags': [os.path.join(EXPERIMENTOS, f'analise_hpo_rodada3_{familia}', 'analise_hpo.json')], 'abs': True}]


def log(msg):
    linha = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(linha, flush=True)
    with open(os.path.join(JOBS, 'encadeador_extras.log'), 'a', encoding='utf-8') as f:
        f.write(linha + '\n')


def pronto(j):
    return all(os.path.exists(f if j.get('abs') else os.path.join(REPO, f)) for f in j['flags'])


def gravar_estado(e):
    tmp = os.path.join(JOBS, 'estado_extras.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(e, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(JOBS, 'estado_extras.json'))


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
    ap.add_argument('--sem-esperar', action='store_true')
    ap.add_argument('--so-gp', action='store_true', help='não roda a família Sobol/QMC')
    args = ap.parse_args()
    os.makedirs(JOBS, exist_ok=True)
    fases = [('G1 gp', jobs_hpo('gp')), ('G2 reavaliação gp', jobs_reavaliacao('gp')), ('G3 análise gp', jobs_analise('gp'))]
    if not args.so_gp:
        fases += [('Q1 qmc', jobs_hpo('qmc')), ('Q2 reavaliação qmc', jobs_reavaliacao('qmc')),
                  ('Q3 análise qmc', jobs_analise('qmc'))]
    log(f'plano extras: IL+RL (inicializacao=professor) | orcamento={N} n_fre={N_FRE} n_startup={N_STARTUP} '
        f'| 3 execuções por otimizador, sementes {list(SEMENTES.values())} | k=1 | git={P.hash_git()}')
    for nome, js in fases:
        for j in js:
            log(f"  {nome}: {j['nome']}: {' '.join(str(x) for x in j['cmd'][3:])}{'  [PRONTO]' if pronto(j) else ''}")
    if args.so_plano:
        return 0
    if not args.sem_esperar:
        log('esperando o controle D-32 terminar: ' + ESPERAR[0])
        t0 = time.time()
        while not all(os.path.exists(f) for f in ESPERAR):
            if time.time() - t0 > 72 * 3600:
                log('PARADO: 72 h esperando o controle. Use --sem-esperar para começar assim mesmo.')
                return 1
            time.sleep(120)
        log(f'controle concluído; começando os extras ({(time.time() - t0) / 3600:.1f} h de espera)')
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception as e:
        log(f'AVISO: SetThreadExecutionState falhou: {e}')
    estado = {'pid_encadeador': os.getpid(), 'inicio': time.strftime('%Y-%m-%d %H:%M:%S'),
              'parametros': {'orcamento': N, 'n_fre': N_FRE, 'n_startup': N_STARTUP, 'trabalhadores': TRAB,
                             'sementes_hpo': SEMENTES, 'k': 1, 'inicializacao': 'professor',
                             'familias': list(FAMILIAS) if not args.so_gp else ['gp']}}
    gravar_estado(estado)
    for nome, js in fases:
        log(f'--- {nome}')
        if not rodar_fase(nome, js, estado):
            log(f'PARADO: a fase {nome} não completou. Corrija e relance (ele pula o que já tem flag).')
            estado['fim'] = 'falhou em ' + nome
            gravar_estado(estado)
            return 1
    log('FIM: otimizadores extras concluídos.')
    estado['fim'] = 'ok ' + time.strftime('%Y-%m-%d %H:%M:%S')
    gravar_estado(estado)
    return 0


if __name__ == '__main__':
    sys.exit(main())
