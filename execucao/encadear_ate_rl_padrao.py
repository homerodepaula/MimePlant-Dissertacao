"""Encadeador das etapas 1-3 da rodada 3 (referências, warm-start IL, RL sem HPO), para esta máquina (sem o _setup da defesa).

Reproduz EXATAMENTE os comandos do Passo 2 do README (etapas 1 a 3), com orçamento = protocolo.ORCAMENTO_RL_PASSOS:
  E1 (em paralelo: não dependem uma da outra)
     referencias.py --saida results/rodada3/referencias --retomar                                  (CPU)
     treinar_warmstart_il.py --seeds 42 123 2024 7 1337 --config lr_alto --saida results/rodada3/il_warmstart --retomar  (GPU)
  E2 (depois das duas flags de E1; 4 processos, como a automação da defesa: braço × {A: 42 123 2024, B: 7 1337})
     bakeoff.py --bracos <braço> --seeds <...> --orcamento N --saida results/rodada3/rl_padrao --warmstart ... --referencias ... --retomar
  E3 bakeoff.py --retomar com os 2 braços e as 5 sementes: nada treina (flags existem); só regrava resumo.json lendo todos os runs
     (4 processos escrevendo o mesmo resumo.json em paralelo podem deixar um resumo parcial).

Robustez: cada job é relançado com --retomar até 3 lançamentos; prioridade abaixo do normal; SetThreadExecutionState impede a
suspensão por inatividade enquanto este processo vive (não altera configuração do Windows). Log em ../_jobs/.
Fica FORA do repositório de propósito: não suja a árvore do git (config.json grava hash e git_sujo).
Uso:  python -X utf8 execucao/encadear_ate_rl_padrao.py [--so-plano]
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
sys.path.insert(0, REPO)
import protocolo as P  # noqa: E402

PY = sys.executable
N = P.ORCAMENTO_RL_PASSOS
R = os.path.relpath(P.DIR_RODADA, REPO)                     # results\rodada3
REFS = os.path.join(R, 'referencias', 'referencias.json')
WS = os.path.join(R, 'il_warmstart')
RLP = os.path.join(R, 'rl_padrao')
SEMENTES = [str(s) for s in P.SEMENTES_TREINO]              # 42 123 2024 7 1337
GRUPOS = {'A': SEMENTES[:3], 'B': SEMENTES[3:]}
BRACOS = ['ppo_il', 'ppo_zero']
MAX_LANCAMENTOS = 3
BELOW_NORMAL = 0x00004000
CREATE_NO_WINDOW = 0x08000000


def cmd(*partes):
    return [PY, '-X', 'utf8'] + [str(p) for p in partes]


def jobs_e1():
    return [
        {'nome': 'r3_referencias', 'cmd': cmd('referencias.py', '--saida', os.path.join(R, 'referencias'), '--retomar'),
         'flags': [os.path.join(R, 'referencias', '_DONE.flag')]},
        {'nome': 'r3_warmstart', 'cmd': cmd('treinar_warmstart_il.py', '--seeds', *SEMENTES, '--config', 'lr_alto', '--saida', WS, '--retomar'),
         'flags': [os.path.join(WS, '_DONE.flag')]},
    ]


def jobs_e2():
    js = []
    for braco in BRACOS:
        for g, seeds in GRUPOS.items():
            js.append({'nome': f'r3_rl_padrao_{braco}_{g}',
                       'cmd': cmd('bakeoff.py', '--bracos', braco, '--seeds', *seeds, '--orcamento', N, '--saida', RLP,
                                  '--warmstart', WS, '--referencias', REFS, '--retomar'),
                       'flags': [os.path.join(RLP, braco, f'seed_{s}', '_DONE.flag') for s in seeds]})
    return js


def job_e3():
    return {'nome': 'r3_rl_padrao_resumo',
            'cmd': cmd('bakeoff.py', '--bracos', *BRACOS, '--seeds', *SEMENTES, '--orcamento', N, '--saida', RLP,
                       '--warmstart', WS, '--referencias', REFS, '--retomar'),
            'flags': [os.path.join(RLP, 'resumo.json')], 'sempre': True}


def log(msg):
    linha = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(linha, flush=True)
    with open(os.path.join(JOBS, 'encadeador_ate_rl_padrao.log'), 'a', encoding='utf-8') as f:
        f.write(linha + '\n')


def pronto(job):
    return all(os.path.exists(os.path.join(REPO, f)) for f in job['flags'])


def gravar_estado(estado):
    tmp = os.path.join(JOBS, 'estado.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(estado, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(JOBS, 'estado.json'))


def rodar_etapa(nome_etapa, jobs, estado):
    """Roda os jobs da etapa em paralelo; relança os que saírem sem flag (até MAX_LANCAMENTOS). True se todos ficaram prontos."""
    pendentes = [j for j in jobs if j.get('sempre') or not pronto(j)]
    for j in jobs:
        if j not in pendentes:
            estado[j['nome']] = {'estado': 'pronto (flags já existiam)'}
    if not pendentes:
        log(f'{nome_etapa}: tudo pronto, pulando')
        return True
    vivos = {}
    for j in pendentes:
        j['lancamentos'] = 0
    fila = list(pendentes)
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
            estado[j['nome']] = {'estado': 'rodando', 'pid': proc.pid, 'lancamento': j['lancamentos'], 'inicio': time.strftime('%Y-%m-%d %H:%M:%S')}
            gravar_estado(estado)
            log(f"{nome_etapa}: lançado {j['nome']} (pid {proc.pid}, lançamento {j['lancamentos']})")
        time.sleep(30)
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
                log(f'{nome_etapa}: {nome} concluído')
            elif j['lancamentos'] < MAX_LANCAMENTOS:
                log(f'{nome_etapa}: {nome} saiu com exit={rc} sem as flags; relançando com --retomar')
                fila.append(j)
            else:
                log(f'{nome_etapa}: {nome} falhou {MAX_LANCAMENTOS} vezes (exit={rc}); ver _jobs/{nome}.log')
    return all(pronto(j) for j in jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--so-plano', action='store_true')
    args = ap.parse_args()
    os.makedirs(JOBS, exist_ok=True)
    etapas = [('E1 referencias+warmstart', jobs_e1()), ('E2 rl_padrao', jobs_e2()), ('E3 resumo', [job_e3()])]
    log(f'plano: repo={REPO} | python={PY} | git={P.hash_git()} sujo={P.git_sujo()} | orcamento={N} | rodada={P.RODADA}')
    for nome, js in etapas:
        for j in js:
            log(f"  {nome}: {j['nome']}: {' '.join(j['cmd'][3:])}{'  [PRONTO]' if pronto(j) and not j.get('sempre') else ''}")
    if args.so_plano:
        return 0
    if P.git_sujo():
        log('AVISO: árvore do git suja — o config.json de cada execução vai registrar git_sujo=True')
    # impede a suspensão por inatividade enquanto este processo viver (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception as e:
        log(f'AVISO: SetThreadExecutionState falhou: {e}')
    estado = {'pid_encadeador': os.getpid(), 'inicio': time.strftime('%Y-%m-%d %H:%M:%S')}
    gravar_estado(estado)
    for nome, js in etapas:
        log(f'--- {nome}')
        if not rodar_etapa(nome, js, estado):
            log(f'PARADO: a etapa {nome} não completou. Corrija e relance este script (ele pula o que já tem flag).')
            estado['fim'] = 'falhou em ' + nome
            gravar_estado(estado)
            return 1
    log('FIM: etapas 1-3 concluídas. Resumo em ' + os.path.join(RLP, 'resumo.json'))
    estado['fim'] = 'ok ' + time.strftime('%Y-%m-%d %H:%M:%S')
    gravar_estado(estado)
    return 0


if __name__ == '__main__':
    sys.exit(main())
