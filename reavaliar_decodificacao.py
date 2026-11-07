"""reavaliar_decodificacao.py — re-avalia checkpoints já treinados com a regra de decodificação da D-26 (06/09/2026):
política AMOSTRADA (τ = 1) em validação-objetivo e teste, e argmax como análise secundária só no teste (`avaliacao['teste_argmax']`).

Uso:  python -X utf8 reavaliar_decodificacao.py --pastas results\\il_warmstart_v2 results\\bakeoff
          [--saida results\\_reavaliacao_decodificacao] [--referencias results\\referencias_v2\\referencias.json] [--retomar] [--forcar]

Para cada <pasta>/seed_*/ ou <pasta>/<braco>/seed_*/ com modelo_final.pt, metricas.json e _DONE.flag:
  - guarda o bloco antigo (argmax, regra D-05) em metricas['avaliacao_argmax_pre_D26'] (uma única vez);
  - substitui metricas['avaliacao'] pelo de `bakeoff.avaliar_final` (amostrada + teste_argmax) e anota metricas['decodificacao'];
  - o treino, o checkpoint e a seleção do melhor bloco (validação-rollback, argmax na rodada 2a) ficam como estão — só a avaliação muda.
Ao final regenera resumo.json da pasta (bake-off: IQM entre sementes por braço; warm-start: tabela conjunto × regime), guardando o antigo
em resumo_argmax_pre_D26.json. Idempotente (pula runs já re-avaliados, a menos de --forcar); grava _DONE.flag/_FAILED.flag em --saida.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
import protocolo as P  # noqa: E402

P.preparar_sys_path()
import avaliacao as A  # noqa: E402
from bakeoff import carregar_referencias, avaliar_final  # noqa: E402
from agents.rl.envs_lote import fabrica_env_padrao  # noqa: E402
from agents.rl.politica import AgenteRL  # noqa: E402

REGRA = 'amostrada tau=1 (D-26, 06/09/2026); argmax so no teste (teste_argmax)'


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def runs_em(pasta):
    """Diretórios de run (seed_*) diretamente na pasta ou um nível abaixo (bake-off: <braco>/seed_*)."""
    out = []
    for padrao in (os.path.join(pasta, 'seed_*'), os.path.join(pasta, '*', 'seed_*')):
        for d in sorted(glob.glob(padrao)):
            if all(os.path.exists(os.path.join(d, f)) for f in ('modelo_final.pt', 'metricas.json', '_DONE.flag')):
                out.append(d)
    return out


def reavaliar_run(d, env, refs, forcar):
    caminho = os.path.join(d, 'metricas.json')
    m = json.load(open(caminho, encoding='utf-8'))
    if not forcar and (m.get('decodificacao') or {}).get('regra') == REGRA:
        log(f'{os.path.relpath(d, RAIZ)}: já re-avaliado; pulando')
        return False
    t0 = time.time()
    agente = AgenteRL.carregar_de(os.path.join(d, 'modelo_final.pt'))
    agente.rede.eval()
    aval = avaliar_final(agente, env, refs, P.N_VALIDACAO_OBJETIVO, P.N_TESTE, log)
    if 'avaliacao_argmax_pre_D26' not in m:
        m['avaliacao_argmax_pre_D26'] = m.get('avaliacao')
    m['avaliacao'] = aval
    m['decodificacao'] = {'regra': REGRA, 'data': time.strftime('%Y-%m-%d %H:%M:%S'), 'git': P.hash_git(), 'tempo_s': round(time.time() - t0, 1)}
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=1, ensure_ascii=False)
    va, ta = aval['validacao_objetivo'], aval['teste_argmax']
    log(f"{os.path.relpath(d, RAIZ)}: val-obj norm3 {va['media_normalizada_3_regimes']:.3f} "
        f"(in {va['in_dist']['media']:.0f} / dry {va['dry']['media']:.0f} / humid {va['humid']['media']:.0f}) | teste argmax norm3 {ta['media_normalizada_3_regimes']:.3f} | {time.time() - t0:.0f}s")
    return True


def resumo_bakeoff(pasta):
    bracos = {}
    for d in runs_em(pasta):
        braco = os.path.basename(os.path.dirname(d))
        m = json.load(open(os.path.join(d, 'metricas.json'), encoding='utf-8'))
        bracos.setdefault(braco, {})[int(m['seed'])] = m
    if not bracos:
        return
    antigo = os.path.join(pasta, 'resumo.json')
    if os.path.exists(antigo) and not os.path.exists(os.path.join(pasta, 'resumo_argmax_pre_D26.json')):
        shutil.copy2(antigo, os.path.join(pasta, 'resumo_argmax_pre_D26.json'))
    resumo, resumo_argmax = {}, {}
    for braco, por_seed in bracos.items():
        vals = [m['avaliacao']['validacao_objetivo']['media_normalizada_3_regimes'] for m in por_seed.values()]
        vals = [v for v in vals if v is not None]
        resumo[braco] = {'n_sementes': len(vals), 'iqm_validacao_objetivo_norm': A.iqm(vals) if vals else None,
                         'media': float(np.mean(vals)) if vals else None, 'std_entre_sementes': float(np.std(vals)) if vals else None,
                         'erro_padrao': float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None, 'por_semente': vals,
                         'teste_norm3_por_semente': [m['avaliacao']['teste']['media_normalizada_3_regimes'] for m in por_seed.values()],
                         'teste_argmax_norm3_por_semente': [m['avaliacao'].get('teste_argmax', {}).get('media_normalizada_3_regimes') for m in por_seed.values()]}
    orc = next(iter(next(iter(bracos.values())).values())).get('orcamento')
    seeds = sorted({s for por_seed in bracos.values() for s in por_seed})
    with open(antigo, 'w', encoding='utf-8') as f:
        json.dump({'regra': 'maior IQM do retorno normalizado (validacao-objetivo, 3 regimes); empate (<1 EP) -> menor variancia; C abaixo de A e B',
                   'decodificacao': REGRA, 'bracos': resumo, 'orcamento': orc, 'seeds': seeds}, f, indent=1, ensure_ascii=False)
    for b, r in resumo.items():
        log(f"  resumo bake-off {b:9s} IQM {r['iqm_validacao_objetivo_norm']} | média {r['media']} ± {r['std_entre_sementes']} (n={r['n_sementes']})")


def resumo_warmstart(pasta):
    runs = [d for d in runs_em(pasta) if os.path.dirname(d) == os.path.abspath(pasta) or os.path.dirname(d) == pasta]
    if not runs:
        return
    antigo = os.path.join(pasta, 'resumo.json')
    if os.path.exists(antigo) and not os.path.exists(os.path.join(pasta, 'resumo_argmax_pre_D26.json')):
        shutil.copy2(antigo, os.path.join(pasta, 'resumo_argmax_pre_D26.json'))
    resumo, seeds, hp, receita = {}, [], None, None
    for d in runs:
        m = json.load(open(os.path.join(d, 'metricas.json'), encoding='utf-8'))
        seeds.append(int(m['seed'])); hp = hp or m.get('config'); receita = receita or m.get('receita')
        for conj in m['avaliacao']:
            for reg, v in m['avaliacao'][conj].items():
                if isinstance(v, dict) and 'media' in v:
                    resumo.setdefault(conj, {}).setdefault(reg, []).append(v['media'])
    tabela = {conj: {reg: {'media_entre_sementes': float(np.mean(v)), 'std_entre_sementes': float(np.std(v)), 'por_semente': v}
                     for reg, v in regs.items()} for conj, regs in resumo.items()}
    with open(antigo, 'w', encoding='utf-8') as f:
        json.dump({'config': hp, 'receita': receita, 'seeds': seeds, 'decodificacao': REGRA, 'tabela': tabela}, f, indent=1, ensure_ascii=False)
    for conj in tabela:
        log('  resumo warm-start ' + conj + ': ' + ' | '.join(f"{reg} {v['media_entre_sementes']:.0f} ± {v['std_entre_sementes']:.0f}" for reg, v in tabela[conj].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pastas', nargs='+', required=True)
    ap.add_argument('--saida', default=os.path.join('results', '_reavaliacao_decodificacao'))
    ap.add_argument('--referencias', default=P.REFERENCIAS_PADRAO)
    ap.add_argument('--retomar', action='store_true')
    ap.add_argument('--forcar', action='store_true')
    args = ap.parse_args()
    saida = os.path.abspath(args.saida)
    os.makedirs(saida, exist_ok=True)
    P.limpar_flags(saida)
    try:
        refs = carregar_referencias(args.referencias)
        env = fabrica_env_padrao()
        P.fixar_sementes(0) if hasattr(P, 'fixar_sementes') else None
        n = 0
        for pasta in args.pastas:
            pasta = os.path.abspath(pasta)
            runs = runs_em(pasta)
            log(f'{os.path.relpath(pasta, RAIZ)}: {len(runs)} runs')
            for d in runs:
                P.escrever_status(saida, os.path.relpath(d, RAIZ), 'inicio')
                n += int(reavaliar_run(d, env, refs, args.forcar))
            if glob.glob(os.path.join(pasta, '*', 'seed_*')):
                resumo_bakeoff(pasta)
            else:
                resumo_warmstart(pasta)
        P.registrar_config(saida, experimento='reavaliacao_decodificacao', pastas=[os.path.relpath(os.path.abspath(p), RAIZ) for p in args.pastas],
                           regra=REGRA, referencias=args.referencias, n_reavaliados=n)
        P.escrever_flag(saida, '_DONE.flag', f'{n} runs re-avaliados')
        log(f'concluído: {n} runs re-avaliados')
        return 0
    except Exception:
        traceback.print_exc()
        P.escrever_flag(saida, '_FAILED.flag', traceback.format_exc().splitlines()[-1])
        return 1


if __name__ == '__main__':
    sys.exit(main())
