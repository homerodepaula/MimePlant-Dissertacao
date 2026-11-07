"""Encadeador do ESTUDO DE AUTORL da rodada 3: comparação entre otimizadores de HPO (TPE+Hyperband × busca aleatória × DEHB),
com 3 execuções independentes por otimizador (D-35, opção C) — o foco da dissertação.

Padrões (D-35, 19/09/2026): 1.000.000 de passos por execução de RL (1 FRE) · **32 FRE por execução de HPO** · 3 execuções por
otimizador (sementes de otimizador 1001, 1002, 1003) · k = 1 na reavaliação (o incumbente de cada execução) · n_startup = 5
no TPE (**D-36**: com 32 FRE, os 10 da D-22 fariam ~40 % da busca ser aleatória, o que apaga a diferença entre TPE e RS).

Estudos e fases (6 processos na GPU por fase, como na rodada 2):
  F1 tpe      tpe_r3_a|b|c        hpo_rl.py --sampler tpe     3 estudos × 2 trabalhadores
  F2 rs       rs_r3_a|b|c         hpo_rl.py --sampler random  3 estudos × 2 trabalhadores
  F3 dehb     dehb_r3_a|b|c       hpo_dehb.py --paralelo 3    2 coordenadores por vez, depois o terceiro
  F4 pos_hpo  incumbente dos 6 estudos Optuna × 5 sementes (validação-objetivo e teste), ablações e análise → define o MimePlant
  F5 pos_dehb incumbente dos 3 estudos DEHB × 5 sementes em pasta própria + curvas dos 9 estudos
O controle do zero (D-32) NÃO entra aqui: decisão de 19/09, roda depois, ao ver o estudo de AutoRL (`--com-controle` o inclui).

Robustez: relança com --retomar até 3 vezes, prioridade abaixo do normal, sem suspensão por inatividade. Log em ../_jobs/.
Uso:  python -X utf8 execucao/encadear_hpo.py [--so-plano] [--com-controle]
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
N = P.ORCAMENTO_RL_PASSOS          # 1.000.000 (D-35)
N_FRE = 32                         # por execução de HPO (D-35, opção C)
N_STARTUP = 5                      # D-36
K = 1                              # incumbente de cada execução
TRAB_OPTUNA = 2                    # trabalhadores por estudo Optuna (3 estudos por fase = 6 processos)
PAR_DEHB = 3                       # avaliações simultâneas por coordenador DEHB
SEMENTES_HPO = {'a': 1001, 'b': 1002, 'c': 1003}
R = os.path.relpath(P.DIR_RODADA, REPO)
REFS = os.path.join(R, 'referencias', 'referencias.json')
WS = os.path.join(R, 'il_warmstart')
TPE = [f'tpe_r3_{s}' for s in SEMENTES_HPO]
RS = [f'rs_r3_{s}' for s in SEMENTES_HPO]
DEHB = [f'dehb_r3_{s}' for s in SEMENTES_HPO]
MAX_LANCAMENTOS = 3
BELOW_NORMAL = 0x00004000
CREATE_NO_WINDOW = 0x08000000


def cmd(*partes):
    return [PY, '-X', 'utf8'] + [str(p) for p in partes]


def flag_study(study):
    return os.path.join(DIR_HPO, study, '_DONE.flag')


def jobs_optuna(studies, sampler):
    js = []
    for study in studies:
        seed = SEMENTES_HPO[study[-1]]
        for w in range(TRAB_OPTUNA):
            extra = ['--n-startup', N_STARTUP] if sampler == 'tpe' else []
            js.append({'nome': f'r3_hpo_{study}_w{w}',
                       'cmd': cmd('hpo/hpo_rl.py', '--study', study, '--sampler', sampler, '--n-fre', N_FRE,
                                  '--orcamento', N, '--seed-sampler', seed, *extra,
                                  '--paralelo', '--trabalhador', w, '--warmstart', WS, '--referencias', REFS, '--retomar'),
                       'flags': [flag_study(study)], 'abs': True})
    return js


def jobs_dehb(studies):
    return [{'nome': f'r3_hpo_{study}',
             'cmd': cmd('hpo/hpo_dehb.py', '--study', study, '--n-fre', N_FRE, '--orcamento', N,
                        '--paralelo', PAR_DEHB, '--niveis', 2, '--seed', SEMENTES_HPO[study[-1]],
                        '--warmstart', WS, '--referencias', REFS, '--retomar'),
             'flags': [flag_study(study)], 'abs': True} for study in studies]


def job_pos_hpo():
    return {'nome': 'r3_pos_hpo',
            'cmd': cmd('hpo/pos_hpo.py', '--studies', *TPE, *RS, '--k', K, '--orcamento', N,
                       '--padroes', 'ppo_il', 'ppo_zero', '--max-horas', 120, '--retomar'),
            'flags': [os.path.join(R, 'pos_hpo', '_DONE.flag')]}


def jobs_pos_dehb():
    return [
        {'nome': 'r3_pos_dehb_reavaliacao',
         'cmd': cmd('hpo/reavaliar_topk.py', '--study', *DEHB, '--k', K, '--orcamento', N,
                    '--saida', os.path.join(R, 'hpo_reavaliacao_dehb'), '--paralelo', 4, '--retomar'),
         'flags': [os.path.join(R, 'hpo_reavaliacao_dehb', '_DONE.flag')]},
        {'nome': 'r3_analise_autorl',
         'cmd': cmd('hpo/analise_hpo.py', '--studies', *TPE, *RS, *DEHB,
                    '--reavaliacao', os.path.join(R, 'hpo_reavaliacao_dehb'),
                    '--saida', os.path.join('..', 'experimentos', 'analise_hpo_rodada3_dehb')),
         'flags': [os.path.join(EXPERIMENTOS, 'analise_hpo_rodada3_dehb', 'analise_hpo.json')], 'abs': True},
    ]


def job_controle():
    return {'nome': 'r3_controle_zero',
            'cmd': cmd('hpo/controle_zero.py', '--n-fre', N_FRE, '--orcamento', N, '--n-startup', N_STARTUP,
                       '--paralelo', 3, '--retomar'),
            'flags': [os.path.join(R, 'controle_zero', 'comparacao.json')]}


def log(msg):
    linha = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(linha, flush=True)
    with open(os.path.join(JOBS, 'encadeador_hpo.log'), 'a', encoding='utf-8') as f:
        f.write(linha + '\n')


def pronto(job):
    return all(os.path.exists(f if job.get('abs') else os.path.join(REPO, f)) for f in job['flags'])


def gravar_estado(estado):
    tmp = os.path.join(JOBS, 'estado_hpo.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(estado, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(JOBS, 'estado_hpo.json'))


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
            # Escalonado: dois trabalhadores do MESMO study criando o SQLite ao mesmo tempo dão
            # "table alembic_version already exists" e um deles morre (visto no lançamento de 19/09 20:08).
            if fila:
                time.sleep(20)
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
    ap.add_argument('--com-controle', action='store_true', help='inclui o controle do zero (D-32) ao final')
    args = ap.parse_args()
    os.makedirs(JOBS, exist_ok=True)
    # Ordem igual à do Passo 3 do README: o pós-HPO vem ANTES do DEHB. O MimePlant sai só dos estudos Optuna, então adiantá-lo
    # entrega o resultado principal ~27 h mais cedo, e o DEHB não divide a GPU com a reavaliação.
    fases = [('F1 tpe', jobs_optuna(TPE, 'tpe')),
             ('F2 rs', jobs_optuna(RS, 'random')),
             ('F3 pos_hpo', [job_pos_hpo()]),
             ('F4a dehb', jobs_dehb(DEHB[:2])),
             ('F4b dehb', jobs_dehb(DEHB[2:])),
             # a análise LÊ a reavaliação: tem de ser uma fase DEPOIS, não junto. Em 24/09 as duas foram lançadas
             # juntas, a análise terminou em 1 min com `reavaliacao: null` e precisou ser refeita à mão.
             ('F5a pos_dehb reavaliação', jobs_pos_dehb()[:1]),
             ('F5b análise', jobs_pos_dehb()[1:])]
    if args.com_controle:
        fases += [('F6 controle_zero', [job_controle()])]
    log(f'plano AutoRL (D-35 opção C + D-36): orcamento={N} n_fre={N_FRE} por execução · 3 execuções por otimizador '
        f'(sementes {list(SEMENTES_HPO.values())}) · k={K} · n_startup={N_STARTUP} · git={P.hash_git()}')
    for nome, js in fases:
        for j in js:
            log(f"  {nome}: {j['nome']}: {' '.join(j['cmd'][3:])}{'  [PRONTO]' if pronto(j) else ''}")
    if args.so_plano:
        return 0
    faltando = [f for f in (os.path.join(REPO, REFS), os.path.join(REPO, WS, '_DONE.flag'),
                            os.path.join(REPO, R, 'rl_padrao', 'resumo.json')) if not os.path.exists(f)]
    if faltando:
        log('PARADO: etapas 1-3 incompletas: ' + '; '.join(faltando))
        return 1
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception as e:
        log(f'AVISO: SetThreadExecutionState falhou: {e}')
    estado = {'pid_encadeador': os.getpid(), 'inicio': time.strftime('%Y-%m-%d %H:%M:%S'),
              'd35': {'orcamento': N, 'n_fre': N_FRE, 'execucoes_por_otimizador': len(SEMENTES_HPO),
                      'sementes_hpo': SEMENTES_HPO, 'k': K, 'n_startup': N_STARTUP,
                      'otimizadores': ['tpe+hyperband', 'random+hyperband', 'dehb'],
                      'controle_zero': bool(args.com_controle)}}
    gravar_estado(estado)
    for nome, js in fases:
        log(f'--- {nome}')
        if not rodar_fase(nome, js, estado):
            log(f'PARADO: a fase {nome} não completou. Corrija e relance este script (ele pula o que já tem flag).')
            estado['fim'] = 'falhou em ' + nome
            gravar_estado(estado)
            return 1
    log('FIM: estudo de AutoRL concluído.')
    estado['fim'] = 'ok ' + time.strftime('%Y-%m-%d %H:%M:%S')
    gravar_estado(estado)
    return 0


if __name__ == '__main__':
    sys.exit(main())
