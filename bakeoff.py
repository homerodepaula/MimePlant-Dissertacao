"""bakeoff.py — harness da Fase 3 (D-19): roda os braços da fase de RL com a interface única, avalia pelo protocolo e aplica
a regra pré-registrada (experimentos/DESENHO_FASE3_BAKEOFF_RL.md).

Uso:  python -X utf8 bakeoff.py --bracos ppo_il ppo_zero --seeds 42 123 2024 7 1337 [--orcamento N]
        [--saida results/rodada3/rl_padrao] [--warmstart results/rodada3/il_warmstart] [--retomar] [--smoke] [--hp-json arquivo.json]
      Caminhos e orçamento padrão vêm de `protocolo.py` (DIR_RL_PADRAO, DIR_WARMSTART, REFERENCIAS_PADRAO, ORCAMENTO_RL_PASSOS).
      Na rodada 3 este é o RL SEM HPO: `ppo_il` parte do clone do professor na configuração padrão; `ppo_zero` é o controle.
Por (braço, semente): treina com `callback` a cada bloco (validação-rollback 500–519, in-dist; guarda o melhor estado),
carrega o melhor estado, avalia em validação-objetivo (520–549 × 3 regimes) e em teste (999–1048 × 3 regimes; só reportar) com a
política amostrada (D-26) e, só no teste, também com argmax (`avaliacao['teste_argmax']`, análise secundária),
normaliza pelas referências v2 e grava metricas.json, config.json, modelo_final.pt, _DONE.flag. Resumível (--retomar).
Ao final, resumo.json com IQM entre sementes por braço e a regra de decisão aplicada (a D-20 é registrada à mão).
"""
import os
import sys
import glob
import json
import copy
import time
import argparse
import traceback

import numpy as np
import torch

import protocolo as P

P.preparar_sys_path()
import avaliacao as A                        # noqa: E402
from envs_lote import fabrica_env_padrao     # noqa: E402
import ppo as PPO                            # noqa: E402
import elite as ELITE                        # noqa: E402

BRACOS = {
    'ppo_il': dict(modulo=PPO, warmstart=True),
    'ppo_zero': dict(modulo=PPO, warmstart=False),
    'elite': dict(modulo=ELITE, warmstart=True),
}
HP_SMOKE = {'ppo_il': {'V': 2, 'k_aquec': 1, 'n_epocas': 1, 'minilote_seqs': 4, 'blocos': 2},
            'ppo_zero': {'V': 2, 'n_epocas': 1, 'minilote_seqs': 4, 'blocos': 2},
            'elite': {'n_eps_coleta': 2, 'tam_elite': 4, 'updates_por_iter': 2, 'blocos': 2}}


def carregar_referencias(caminho):
    with open(caminho, encoding='utf-8') as f:
        return json.load(f)


def normalizado(media, refs, conjunto, regime):
    """`conjunto` é a chave de `referencias.json` com as MESMAS sementes em que `media` foi medida
    ('validacao_rollback', 'validacao_objetivo' ou 'teste'); referências antigas com só 'validacao' não servem (06/09)."""
    if conjunto not in refs['oraculo_perfeito']:
        raise KeyError(f"referencias.json sem o conjunto {conjunto!r}: regenerar com referencias.py (06/09/2026)")
    return A.normalizar(media, refs['aleatorio'][conjunto][regime]['media'], refs['oraculo_perfeito'][conjunto][regime]['media'])


def avaliar_final(agente, env, refs, n_val, n_teste, log):
    pol = A.PoliticaRede(agente)
    out = {}
    for conjunto, sementes in (('validacao_objetivo', P.sementes_validacao_objetivo()[:n_val]), ('teste', P.sementes_teste()[:n_teste])):
        out[conjunto] = {}
        for regime in P.NOMES_REGIMES:
            res = A.avaliar(pol, env, sementes, regime)
            r = A.resumo(res)
            r['normalizado'] = normalizado(r['media'], refs, conjunto, regime) if refs else None
            r.update(recompensas=res['recompensas'], contagem_acoes=res['contagem_acoes'],
                     acoes_pct=A.acoes_pct(res), ociosidade_pct=A.ociosidade_pct(res),
                     comp_medio={k: float(np.mean(v)) for k, v in res['comp'].items()},
                     kpis={k: float(np.nanmean(v)) for k, v in res['kpis'].items()})
            out[conjunto][regime] = r
            log(f"    [{conjunto:18s}] {regime:8s} média {r['media']:7.1f} ± {r['std']:6.1f} IQM {r['iqm']:7.1f} norm {r['normalizado'] if r['normalizado'] is None else round(r['normalizado'], 3)} | ocioso {r['ociosidade_pct']:.0f}%")
        out[conjunto]['media_normalizada_3_regimes'] = float(np.mean([out[conjunto][reg]['normalizado'] for reg in P.NOMES_REGIMES])) if refs else None
    # D-26 (06/09/2026): a regra e a politica amostrada (tau = 1, acima); o argmax fica como analise secundaria, so no teste
    pol0 = A.PoliticaRede(agente, tau=0.0)
    out['teste_argmax'] = {}
    for regime in P.NOMES_REGIMES:
        res = A.avaliar(pol0, env, P.sementes_teste()[:n_teste], regime)
        r = A.resumo(res)
        r['normalizado'] = normalizado(r['media'], refs, 'teste', regime) if refs else None
        r.update(recompensas=res['recompensas'], acoes_pct=A.acoes_pct(res), ociosidade_pct=A.ociosidade_pct(res),
                 comp_medio={k: float(np.mean(v)) for k, v in res['comp'].items()})
        out['teste_argmax'][regime] = r
        log(f"    [teste (argmax)     ] {regime:8s} média {r['media']:7.1f} ± {r['std']:6.1f} IQM {r['iqm']:7.1f} norm {r['normalizado'] if r['normalizado'] is None else round(r['normalizado'], 3)} | ocioso {r['ociosidade_pct']:.0f}%")
    out['teste_argmax']['media_normalizada_3_regimes'] = float(np.mean([out['teste_argmax'][reg]['normalizado'] for reg in P.NOMES_REGIMES])) if refs else None
    return out


def rodar_um(braco, seed, args, refs, log):
    dir_out = os.path.join(os.path.abspath(args.saida), braco, f'seed_{seed}')
    if args.retomar and os.path.exists(os.path.join(dir_out, '_DONE.flag')):
        log(f'{braco} seed {seed}: já concluído'); return json.load(open(os.path.join(dir_out, 'metricas.json'), encoding='utf-8'))
    os.makedirs(dir_out, exist_ok=True)
    P.escrever_status(dir_out, 'inicio', f'{braco} seed={seed} orcamento={args.orcamento}')
    info = BRACOS[braco]
    politica_il = os.path.join(os.path.abspath(args.warmstart), f'seed_{seed}', 'modelo_final.pt') if info['warmstart'] else None
    if politica_il and not os.path.exists(politica_il):
        raise FileNotFoundError(f'warm-start ausente: {politica_il}')
    hp = dict(HP_SMOKE[braco]) if args.smoke else {}
    if args.hp_json:
        hp.update(json.load(open(args.hp_json, encoding='utf-8')).get(braco, {}))
    if not info['warmstart']:
        # o braço do zero gravava `inicializacao = 'professor'` (o padrão do dicionário) embora treinasse sem política de
        # partida — defeito só de registro, declarado no STATUS de 10/09; agora o registro diz o que o treino fez
        hp['inicializacao'] = 'zero'
    n_rb, n_val, n_teste = (3, 3, 3) if args.smoke else (P.N_VALIDACAO_ROLLBACK, P.N_VALIDACAO_OBJETIVO, P.N_TESTE)
    sementes_rb = P.sementes_validacao_rollback()[:n_rb]
    env_aval = fabrica_env_padrao()
    curva_val, melhor = [], {'valor': -np.inf, 'passos': 0, 'estado': None}

    nivel_il = None
    if politica_il:
        from politica import AgenteRL
        ag_il = AgenteRL.carregar_de(politica_il)
        r_il = A.avaliar(A.PoliticaRede(ag_il), env_aval, sementes_rb, 'in_dist')
        nivel_il = float(np.mean(r_il['recompensas']))
        log(f"  IL de partida em validação-rollback: {nivel_il:.1f}")
        curva_val.append({'passos': 0, 'media': nivel_il, 'iqm': A.iqm(r_il['recompensas']), 'delta_il': 0.0})
        melhor.update(valor=nivel_il, passos=0, estado=copy.deepcopy(ag_il.rede.state_dict()))
        del ag_il

    def callback(passos, agente):
        res = A.avaliar(A.PoliticaRede(agente), env_aval, sementes_rb, 'in_dist')
        m = float(np.mean(res['recompensas']))
        ponto = {'passos': int(passos), 'media': m, 'iqm': A.iqm(res['recompensas']),
                 'delta_il': (m - nivel_il) if nivel_il is not None else None,
                 'esperar_pct': 100.0 * res['contagem_acoes']['esperar'] / (len(sementes_rb) * P.PASSOS_MAXIMOS),
                 'ociosidade_pct': A.ociosidade_pct(res)}
        curva_val.append(ponto)
        if m > melhor['valor']:
            melhor.update(valor=m, passos=int(passos), estado=copy.deepcopy(agente.rede.state_dict()))
        log(f"    [validação-rollback] passos {passos}: {m:.1f} (Δ_IL {ponto['delta_il'] if ponto['delta_il'] is None else round(ponto['delta_il'], 1)}) | melhor {melhor['valor']:.1f} @ {melhor['passos']}")
        P.escrever_status(dir_out, f'bloco_{passos}', f'val={m:.1f}')
        return False

    t0 = time.time()
    res = info['modulo'].treinar_rl(politica_il, hp, seed, args.orcamento, callback=callback, log=log, dispositivo=args.dispositivo)
    agente = res['agente']
    if melhor['estado'] is not None and melhor['valor'] >= curva_val[-1]['media']:
        agente.rede.load_state_dict(melhor['estado'])
    agente.rede.eval()
    aval = avaliar_final(agente, env_aval, refs, n_val, n_teste, log)
    agente.salvar(os.path.join(dir_out, 'modelo_final.pt'))
    eps = res['episodios_treino']
    metricas = {
        'braco': braco, 'seed': seed, 'orcamento': args.orcamento, 'passos': res['passos'], 'hp': res['hp'],
        'nivel_il_rollback': nivel_il, 'curva_validacao_rollback': curva_val, 'melhor': {'valor': melhor['valor'], 'passos': melhor['passos']},
        'curva_treino': res['curva'],
        'treino_episodios': {'n': len(eps), 'retorno_medio_ultimos_20': float(np.mean([e['retorno'] for e in eps[-20:]])) if eps else None,
                             'retornos': [e['retorno'] for e in eps], 'esperar_pct_ultimos_20': float(np.mean([e['acoes'][P.ACAO_ESPERAR] for e in eps[-20:]]) / P.PASSOS_MAXIMOS * 100) if eps else None,
                             'ociosidade_pct_ultimos_20': float(np.mean([e['acoes'][P.ACAO_ESPERAR] + e['acoes'][P.NOMES_ACOES.index('remover_espantalho')] for e in eps[-20:]]) / P.PASSOS_MAXIMOS * 100) if eps else None},
        'avaliacao': aval, 'tempo_treino_s': res['tempo_s'], 'tempo_total_s': time.time() - t0,
    }
    with open(os.path.join(dir_out, 'metricas.json'), 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=1, ensure_ascii=False)
    P.registrar_config(dir_out, experimento='bakeoff', braco=braco, seed_treino=seed, hp=res['hp'], orcamento_passos=args.orcamento,
                       warmstart=politica_il, referencias=args.referencias, selecao='melhor bloco em validacao-rollback (in-dist)')
    P.escrever_flag(dir_out, '_DONE.flag', f"val_obj_norm={aval['validacao_objetivo']['media_normalizada_3_regimes']}")
    return metricas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bracos', nargs='+', default=['ppo_il', 'elite', 'ppo_zero'], choices=list(BRACOS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2024])
    ap.add_argument('--orcamento', type=int, default=P.ORCAMENTO_RL_PASSOS)
    ap.add_argument('--saida', default=P.DIR_RL_PADRAO)
    ap.add_argument('--warmstart', default=P.DIR_WARMSTART)
    ap.add_argument('--referencias', default=P.REFERENCIAS_PADRAO)
    ap.add_argument('--hp-json', default=None)
    ap.add_argument('--dispositivo', default=None, help="'cpu' ou 'cuda' (padrão: cuda se disponível)")
    ap.add_argument('--retomar', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.orcamento = 4 * P.PASSOS_MAXIMOS
    saida = os.path.abspath(args.saida)
    os.makedirs(saida, exist_ok=True)
    P.limpar_flags(saida)
    refs = carregar_referencias(args.referencias) if os.path.exists(args.referencias) else None

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"bake-off: braços {args.bracos} | sementes {args.seeds} | orçamento {args.orcamento} | dispositivo {P.dispositivo()} | refs {'ok' if refs else 'AUSENTES'}")
    try:
        tudo = {}
        for braco in args.bracos:
            for seed in args.seeds:
                log(f'=== {braco} | seed {seed} ===')
                tudo.setdefault(braco, {})[seed] = rodar_um(braco, seed, args, refs, log)
        # O resumo le TODOS os runs concluidos no disco, e nao so os bracos deste processo: o bake-off roda em dois
        # jobs paralelos (ppo_il+elite e ppo_zero) que escrevem o mesmo resumo.json, e ate 06/09 o ultimo a terminar
        # apagava o braco do outro da tabela (auditoria de 06/09).
        for d in sorted(glob.glob(os.path.join(saida, '*', 'seed_*', 'metricas.json'))):
            b, s = os.path.basename(os.path.dirname(os.path.dirname(d))), int(os.path.basename(os.path.dirname(d))[5:])
            if s not in tudo.get(b, {}):
                try:
                    tudo.setdefault(b, {})[s] = json.load(open(d, encoding='utf-8'))
                except Exception as e:
                    log(f'  aviso: {d} ilegivel ({e})')
        resumo = {}
        for braco, por_seed in tudo.items():
            vals = [m['avaliacao']['validacao_objetivo']['media_normalizada_3_regimes'] for m in por_seed.values()]
            vals = [v for v in vals if v is not None]
            resumo[braco] = {'n_sementes': len(vals), 'iqm_validacao_objetivo_norm': A.iqm(vals) if vals else None,
                             'media': float(np.mean(vals)) if vals else None, 'std_entre_sementes': float(np.std(vals)) if vals else None,
                             'erro_padrao': float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None, 'por_semente': vals}
        with open(os.path.join(saida, 'resumo.json'), 'w', encoding='utf-8') as f:
            json.dump({'regra': 'maior IQM do retorno normalizado (validacao-objetivo, 3 regimes); empate (<1 EP) -> menor variancia; C abaixo de A e B',
                       'bracos': resumo, 'orcamento': args.orcamento, 'seeds': args.seeds}, f, indent=1, ensure_ascii=False)
        for b, r in resumo.items():
            log(f"  {b:9s} IQM {r['iqm_validacao_objetivo_norm']} | média {r['media']} ± {r['std_entre_sementes']} (n={r['n_sementes']})")
        if all(os.path.exists(os.path.join(saida, b, f'seed_{s}', '_DONE.flag')) for b in args.bracos for s in args.seeds):
            P.escrever_flag(saida, '_DONE.flag', 'bake-off completo')
        return 0
    except Exception:
        traceback.print_exc()
        P.escrever_flag(saida, '_FAILED.flag', traceback.format_exc().splitlines()[-1])
        return 1


if __name__ == '__main__':
    sys.exit(main())
