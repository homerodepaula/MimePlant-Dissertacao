"""Controle D-32 — "RL do zero com HPO", o 4º quadrante — ENFILEIRADO: espera a F5 do estudo de AutoRL terminar e então roda.

Por que não usar `hpo/controle_zero.py` como lançador: os padrões dele são da rodada 2 (64 FRE, n_startup 10, reavaliação
do top-3 EM SÉRIE ≈ 38 h) e ele só passa `--paralelo` aos trabalhadores i > 0, o que tira o `constant_liar` do trabalhador 0
e quebra a paridade com os estudos do método. Aqui as mesmas peças são chamadas com os parâmetros da D-35/D-36:
  C1  tpe_zero_r3 e rs_zero_r3: hpo_rl.py --inicializacao zero --n-fre 32 --n-startup 5 --paralelo --trabalhador i (6 processos)
  C2  reavaliar_topk.py --k 1 (incumbente de cada estudo) × 5 sementes, 4 processos → results/rodada3/hpo_reavaliacao_zero
  C3  controle_zero.py --so-comparar → results/rodada3/controle_zero/comparacao.json (tabela 2×2 em teste)
O espaço do controle tem 8 hiperparâmetros (os 4 ligados ao professor não têm efeito sem clone); a ressalva é obrigatória no texto.

Uso:  python -X utf8 execucao/encadear_controle_zero.py [--so-plano] [--sem-esperar]
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
N_FRE = 32
N_STARTUP = 5
SEED_HPO = 1001
TRAB = 3
R = os.path.relpath(P.DIR_RODADA, REPO)
REFS = os.path.join(R, 'referencias', 'referencias.json')
WS = os.path.join(R, 'il_warmstart')
ESTUDOS = [('tpe_zero_r3', 'tpe'), ('rs_zero_r3', 'random')]
REAV_ZERO = os.path.relpath(P.DIR_REAVALIACAO_ZERO, REPO)
# a F5 do estudo de AutoRL termina quando estes dois existirem
ESPERAR = [os.path.join(REPO, R, 'hpo_reavaliacao_dehb', '_DONE.flag'),
           os.path.join(EXPERIMENTOS, 'analise_hpo_rodada3_dehb', 'analise_hpo.json')]
MAX_LANCAMENTOS = 3
BELOW_NORMAL = 0x00004000
CREATE_NO_WINDOW = 0x08000000


def cmd(*p):
    return [PY, '-X', 'utf8'] + [str(x) for x in p]


def jobs_hpo():
    js = []
    for estudo, sampler in ESTUDOS:
        for w in range(TRAB):
            extra = ['--n-startup', N_STARTUP] if sampler == 'tpe' else []
            js.append({'nome': f'ctrl_{estudo}_w{w}',
                       'cmd': cmd('hpo/hpo_rl.py', '--study', estudo, '--sampler', sampler, '--n-fre', N_FRE,
                                  '--orcamento', N, '--inicializacao', 'zero', '--seed-sampler', SEED_HPO, *extra,
                                  '--paralelo', '--trabalhador', w, '--warmstart', WS, '--referencias', REFS, '--retomar'),
                       'flags': [os.path.join(DIR_HPO, estudo, '_DONE.flag')], 'abs': True})
    return js


def jobs_reavaliacao():
    return [{'nome': 'ctrl_reavaliacao',
             'cmd': cmd('hpo/reavaliar_topk.py', '--study', *[e for e, _ in ESTUDOS], '--k', 1, '--orcamento', N,
                        '--saida', REAV_ZERO, '--paralelo', 4, '--retomar'),
             'flags': [os.path.join(REAV_ZERO, '_DONE.flag')]}]


def jobs_comparacao():
    return [{'nome': 'ctrl_comparacao',
             'cmd': cmd('hpo/controle_zero.py', '--so-comparar', '--n-fre', N_FRE, '--orcamento', N),
             'flags': [os.path.join(R, 'controle_zero', 'comparacao.json')]}]


def log(msg):
    linha = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(linha, flush=True)
    with open(os.path.join(JOBS, 'encadeador_controle_zero.log'), 'a', encoding='utf-8') as f:
        f.write(linha + '\n')


def pronto(j):
    return all(os.path.exists(f if j.get('abs') else os.path.join(REPO, f)) for f in j['flags'])


def gravar_estado(e):
    tmp = os.path.join(JOBS, 'estado_controle.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(e, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(JOBS, 'estado_controle.json'))


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
    ap.add_argument('--sem-esperar', action='store_true', help='não espera a F5 do estudo de AutoRL')
    args = ap.parse_args()
    os.makedirs(JOBS, exist_ok=True)
    fases = [('C1 hpo do zero', jobs_hpo()), ('C2 reavaliação', jobs_reavaliacao()), ('C3 tabela 2x2', jobs_comparacao())]
    log(f'plano controle D-32: orcamento={N} n_fre={N_FRE} n_startup={N_STARTUP} seed_hpo={SEED_HPO} '
        f'trabalhadores={TRAB} k=1 | espaço de 8 hiperparâmetros | git={P.hash_git()}')
    for nome, js in fases:
        for j in js:
            log(f"  {nome}: {j['nome']}: {' '.join(j['cmd'][3:])}{'  [PRONTO]' if pronto(j) else ''}")
    if args.so_plano:
        return 0
    if not args.sem_esperar:
        log('esperando a F5 do estudo de AutoRL: ' + ' ; '.join(ESPERAR))
        t0 = time.time()
        while not all(os.path.exists(f) for f in ESPERAR):
            if time.time() - t0 > 48 * 3600:
                log('PARADO: 48 h esperando a F5. Rode com --sem-esperar se quiser começar assim mesmo.')
                return 1
            time.sleep(120)
        log(f'F5 concluída; começando o controle ({(time.time() - t0) / 3600:.1f} h de espera)')
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception as e:
        log(f'AVISO: SetThreadExecutionState falhou: {e}')
    estado = {'pid_encadeador': os.getpid(), 'inicio': time.strftime('%Y-%m-%d %H:%M:%S'),
              'parametros': {'orcamento': N, 'n_fre': N_FRE, 'n_startup': N_STARTUP, 'seed_hpo': SEED_HPO,
                             'trabalhadores': TRAB, 'k': 1, 'espaco': '8 hiperparâmetros'}}
    gravar_estado(estado)
    for nome, js in fases:
        log(f'--- {nome}')
        if not rodar_fase(nome, js, estado):
            log(f'PARADO: a fase {nome} não completou. Corrija e relance (ele pula o que já tem flag).')
            estado['fim'] = 'falhou em ' + nome
            gravar_estado(estado)
            return 1
    log('FIM: controle do zero concluído. Tabela 2x2 em results/rodada3/controle_zero/comparacao.json')
    estado['fim'] = 'ok ' + time.strftime('%Y-%m-%d %H:%M:%S')
    gravar_estado(estado)
    return 0


if __name__ == '__main__':
    sys.exit(main())
