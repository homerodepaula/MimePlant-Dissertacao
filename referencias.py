"""referencias.py — professor (oráculo sem pausas), variante com pausas e agente aleatório nos 3 regimes (Fase 2.2, FarmGym v4).

As chaves do JSON são históricas e NÃO mudam (`bakeoff.normalizado` e `hpo/hpo_rl.py` dependem delas):
  `oraculo_perfeito`  = o PROFESSOR dos experimentos (p = 0, D-27; manejo intensivo, D-33) e a âncora 1 da normalizada;
  `oraculo_subotimo`  = a variante com pausas (p = 0,14), reportada como contexto; nenhum experimento a usa;
  `aleatorio`         = a âncora 0;
  `professor_v4`      = o professor da rodada 2 (manejo reativo), só para comparar rodadas: a âncora 1 mudou na D-33, e
                        com esta linha um retorno normalizado da rodada 2 converte para a escala nova sem re-medir nada.

Mede em cada conjunto de sementes EXATAMENTE como ele é usado (06/09/2026; antes um único bloco 500–549 normalizava
agentes medidos em 520–549 e podava agentes medidos em 500–519 — criticas.md, §7.3):
  - validacao_rollback  500–519  → normaliza o proxy da poda (Hyperband) e as curvas de seleção;
  - validacao_objetivo  520–549  → normaliza o objetivo da HPO e o critério do bake-off (D-18);
  - teste               999–1048 → só para reportar.
Pela função única `avaliacao.avaliar`. Resumível por (política, conjunto, regime): cada bloco vai para
<saida>/parciais/<politica>_<conjunto>_<regime>.json; ao final escreve referencias.json, config.json e _DONE.flag.

Uso:  python -X utf8 referencias.py [--saida results/rodada3/referencias  (padrão: protocolo.DIR_REFERENCIAS)] [--conjuntos validacao_rollback validacao_objetivo teste] [--retomar]
"""
import os
import sys
import json
import time
import argparse
import traceback

import numpy as np

import protocolo as P

P.preparar_sys_path()
import avaliacao as A                             # noqa: E402
from env import AmbienteFazendaGym                # noqa: E402
from mascara_acoes import InvolucroMascaraAcoes   # noqa: E402

CONJUNTOS = {'validacao_rollback': P.sementes_validacao_rollback,
             'validacao_objetivo': P.sementes_validacao_objetivo,
             'teste': P.sementes_teste}


def politicas():
    return {
        'oraculo_subotimo': lambda: A.PoliticaOraculo(P.novo_oraculo_variante()),   # variante com pausas (contexto)
        'oraculo_perfeito': lambda: A.PoliticaOraculo(P.novo_professor()),          # o PROFESSOR e a âncora 1 (D-27)
        'aleatorio': lambda: A.PoliticaAleatoria(0),
        'professor_v4': lambda: A.PoliticaOraculo(P.novo_professor_v4()),           # âncora da rodada 2 (comparação)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--saida', default=P.DIR_REFERENCIAS)
    ap.add_argument('--conjuntos', nargs='+', default=list(CONJUNTOS), choices=list(CONJUNTOS))
    ap.add_argument('--retomar', action='store_true')
    args = ap.parse_args()

    saida = os.path.abspath(args.saida)
    parciais = os.path.join(saida, 'parciais')
    os.makedirs(parciais, exist_ok=True)
    P.limpar_flags(saida)
    t0 = time.time()
    P.escrever_status(saida, 'inicio', ' '.join(args.conjuntos))
    try:
        env = InvolucroMascaraAcoes(AmbienteFazendaGym(passos_maximos=P.PASSOS_MAXIMOS))
        tudo = {}
        for nome, fabrica in politicas().items():
            for conjunto in args.conjuntos:
                sementes = CONJUNTOS[conjunto]()
                for regime in P.NOMES_REGIMES:
                    chave = f'{nome}_{conjunto}_{regime}'
                    arq = os.path.join(parciais, chave + '.json')
                    if args.retomar and os.path.exists(arq):
                        res = json.load(open(arq, encoding='utf-8'))
                    else:
                        t1 = time.time()
                        res = A.avaliar(fabrica(), env, sementes, regime)
                        res['tempo_s'] = time.time() - t1
                        with open(arq, 'w', encoding='utf-8') as f:
                            json.dump(res, f)
                    r = A.resumo(res)
                    tudo.setdefault(nome, {}).setdefault(conjunto, {})[regime] = {**r, 'recompensas': res['recompensas'],
                                                                                   'contagem_acoes': res['contagem_acoes'],
                                                                                   'acoes_pct': A.acoes_pct(res),
                                                                                   'ociosidade_pct': A.ociosidade_pct(res),
                                                                                   'comp_medio': {k: float(np.mean(v)) for k, v in res['comp'].items()}}
                    print(f"  {nome:18s} {conjunto:9s} {regime:8s} n={r['n']:3d} média {r['media']:7.1f} ± {r['std']:6.1f} "
                          f"IQM {r['iqm']:7.1f} EP {r['erro_padrao']:5.1f} | {time.time() - t0:5.0f}s", flush=True)
                    P.escrever_status(saida, 'bloco', chave)
        with open(os.path.join(saida, 'referencias.json'), 'w', encoding='utf-8') as f:
            json.dump(tudo, f, indent=1, ensure_ascii=False)
        P.registrar_config(saida, experimento='referencias', conjuntos=args.conjuntos, regimes=P.NOMES_REGIMES,
                           politicas=list(politicas()), tempo_s=time.time() - t0)
        print('\n=== resumo (média [IQM]) ===')
        for nome in tudo:
            for conjunto in tudo[nome]:
                linha = ' | '.join(f"{reg} {v['media']:6.0f} [{v['iqm']:6.0f}]" for reg, v in tudo[nome][conjunto].items())
                print(f'  {nome:18s} {conjunto:9s} {linha}')
        P.escrever_flag(saida, '_DONE.flag', 'ok')
        return 0
    except Exception:
        traceback.print_exc()
        P.escrever_flag(saida, '_FAILED.flag', traceback.format_exc().splitlines()[-1])
        return 1


if __name__ == '__main__':
    sys.exit(main())
