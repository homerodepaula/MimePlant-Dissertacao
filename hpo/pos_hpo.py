"""hpo/pos_hpo.py — encadeia o que vem depois dos estudos de HPO (Fase 4.6–4.8), de forma resiliente:
espera os studies terminarem (_DONE.flag), depois roda em sequência: reavaliação do top-k em teste (+ configurações
padrão, reaproveitando o bake-off), ablações da melhor configuração, análise (curvas, importância, tabelas).
Cada etapa é resumível (--retomar) e grava flags; se uma falhar, o script para com _FAILED.flag e o validador relança.

Uso:  python -X utf8 hpo/pos_hpo.py [--studies tpe_r3 rs_r3] [--k 3] [--orcamento N] [--max-horas 72] [--retomar]
      Studies, pastas e orçamento padrão vêm de `protocolo.py`; o orçamento é REPASSADO à reavaliação e às ablações
      (até a rodada 2 não era: as duas caíam no próprio padrão, e bastaria mudar o orçamento da HPO para reavaliar
      os vencedores com outro orçamento, em silêncio).
"""
import os
import sys
import time
import argparse
import subprocess

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import protocolo as P                                  # noqa: E402

VF = os.path.dirname(RAIZ)
DIR_HPO = os.path.join(VF, 'experimentos', 'hpo')

TOLERANCIA_ZUMBI_H = 3.0   # um trial em fidelidade máxima dura ~20-60 min; RUNNING há mais de 3 h é processo morto
ORCAMENTO_DA_TOLERANCIA = 250_000   # ... com 250.000 passos por trial (rodada 2), que é onde as 3 h foram calibradas


def tolerancia_zumbi_h(orcamento: int) -> float:
    """A tolerância acompanha o orçamento por trial (19/09/2026). Na rodada 2 um trial em fidelidade máxima levou até
    ~50 min com 6 processos dividindo a GPU (`experimentos/_jobs/hpo_tpe_v1.log`); com 1 milhão de passos um trial
    SAUDÁVEL passa de 3 h, e a tolerância fixa o daria por morto — a reavaliação sairia sem os últimos trials, que é
    justamente o que esta espera existe para evitar. Nunca menos que as 3 h originais."""
    return TOLERANCIA_ZUMBI_H * max(1.0, float(orcamento) / ORCAMENTO_DA_TOLERANCIA)


def trials_em_voo(study, tolerancia_h=TOLERANCIA_ZUMBI_H):
    """Quantos trials do study estão RUNNING de verdade (com menos de `tolerancia_h` horas). Lê o SQLite do Optuna em
    modo somente leitura, sem importar o optuna: o banco está sendo escrito pelos trabalhadores e abrir um Study
    aqui sobrescreveria user_attrs e disputaria lock à toa."""
    import sqlite3
    from datetime import datetime
    caminho = os.path.join(DIR_HPO, study + '.db')
    if not os.path.exists(caminho):
        return 0
    con = sqlite3.connect(f"file:{caminho.replace(os.sep, '/')}?mode=ro", uri=True)
    try:
        inicios = [r[0] for r in con.execute("select datetime_start from trials where state = 'RUNNING'")]
    finally:
        con.close()
    agora = datetime.now()
    vivos = 0
    for ini in inicios:
        try:
            idade_h = (agora - datetime.fromisoformat(str(ini))).total_seconds() / 3600
        except ValueError:
            idade_h = 0.0        # formato inesperado: trata como vivo, o --max-horas segura o pior caso
        if idade_h <= tolerancia_h:
            vivos += 1
    return vivos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--studies', nargs='+', default=list(P.STUDIES_HPO))
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--orcamento', type=int, default=P.ORCAMENTO_RL_PASSOS, help='passos por execução; o MESMO da HPO')
    ap.add_argument('--sem-ablacoes', action='store_true')
    ap.add_argument('--paralelo-reavaliacao', type=int, default=4, help='processos simultâneos na reavaliação (as ablações já usam 4)')
    ap.add_argument('--padroes', nargs='*', default=['ppo_il', 'ppo_zero'],
                    help='configurações padrão reavaliadas junto (reaproveitadas de protocolo.DIR_RL_PADRAO quando existem)')
    ap.add_argument('--max-horas', type=float, default=72.0)
    ap.add_argument('--retomar', action='store_true')
    args = ap.parse_args()
    saida = P.DIR_POS_HPO
    os.makedirs(saida, exist_ok=True)
    P.limpar_flags(saida)

    def log(msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    flags = [os.path.join(DIR_HPO, s, '_DONE.flag') for s in args.studies]
    t0 = time.time()
    log(f'esperando {flags}')
    # A flag NÃO basta quando o study roda em vários trabalhadores (07/09/2026, 6 processos): o primeiro que vê o
    # orçamento estourado grava _DONE.flag e faz study.stop(), que é local ao processo — os outros continuam com um
    # trial RUNNING cada, que `fre_consumidos()` não conta e que será concluído e contado depois. Se a reavaliação
    # lesse o banco nesse instante, o top-k sairia sem os últimos trials (justamente os do TPE mais informado) e a
    # ablação e a análise, que leem o banco horas depois, veriam um "melhor" diferente. Então, depois das flags,
    # espera-se também que nenhum study tenha trial RUNNING. Um RUNNING com mais de `TOLERANCIA_ZUMBI_H` é um
    # processo morto que o Optuna não fechou: não se espera por ele. Tudo limitado por --max-horas.
    tolerancia = tolerancia_zumbi_h(args.orcamento)
    log(f'trial RUNNING há mais de {tolerancia:.1f} h conta como processo morto (orçamento {args.orcamento} passos)')
    ultimo_log = 0.0
    while True:
        faltam = [f for f in flags if not os.path.exists(f)]
        em_voo = {} if faltam else {s: trials_em_voo(s, tolerancia) for s in args.studies}
        if not faltam and not any(em_voo.values()):
            break
        if (time.time() - t0) / 3600 > args.max_horas:
            log('tempo máximo excedido'); P.escrever_flag(saida, '_FAILED.flag', 'timeout esperando a HPO'); return 1
        if time.time() - ultimo_log > 1800:   # o validador trata log parado como aviso: dizer que está vivo
            log(f'esperando: flags faltando {len(faltam)}, trials em voo {em_voo}')
            ultimo_log = time.time()
        time.sleep(300)
    log('studies concluídos e sem trial em voo; iniciando a reavaliação')
    py = sys.executable
    env = dict(os.environ, PYTHONUTF8='1')
    etapas = [
        ('reavaliacao', [py, '-X', 'utf8', os.path.join(RAIZ, 'hpo', 'reavaliar_topk.py'), '--study', *args.studies, '--k', str(args.k),
                         '--padrao', *args.padroes, '--orcamento', str(args.orcamento), '--paralelo', str(args.paralelo_reavaliacao), '--retomar']),
        ('ablacoes', [py, '-X', 'utf8', os.path.join(RAIZ, 'hpo', 'ablacoes.py'), '--study', args.studies[0],
                      '--orcamento', str(args.orcamento), '--retomar']),
        ('analise', [py, '-X', 'utf8', os.path.join(RAIZ, 'hpo', 'analise_hpo.py'), '--studies', *args.studies]),
    ]
    if args.sem_ablacoes:
        etapas = [e for e in etapas if e[0] != 'ablacoes']
    for nome, cmd in etapas:
        log(f'=== {nome}: {" ".join(cmd)}')
        P.escrever_status(saida, nome, 'inicio')
        r = subprocess.run(cmd, cwd=RAIZ, env=env)
        P.escrever_status(saida, nome, f'exit={r.returncode}')
        if r.returncode != 0:
            log(f'{nome} falhou (exit {r.returncode})'); P.escrever_flag(saida, '_FAILED.flag', f'{nome} exit={r.returncode}'); return 1
    P.escrever_flag(saida, '_DONE.flag', 'reavaliacao + ablacoes + analise')
    log('pós-HPO concluído')
    return 0


if __name__ == '__main__':
    sys.exit(main())
