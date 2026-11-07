"""treinar_warmstart_il.py — warm-start de imitação (BC + DAgger) do MimePlant, FarmGym v3 (Fase 2.3).

Receita única (a do cap. 4, pelo `AgenteDAgger`): BC 100 épocas (Adam + cosseno, clip 1,0, lotes de 64 janelas de
`len_seq`) + DAgger 7 rodadas × 3.650 passos, 10 épocas/rodada, lr 1e-4, β linear 0,5 → 0, rótulo sempre do PROFESSOR
(oráculo sem pausas, D-27); fronteiras de episódio preservadas. Novidades da defesa (D-10/D-18): a melhor rodada do DAgger é escolhida
em VALIDAÇÃO-rollback (500–519), a avaliação final é em validação-objetivo (520–549) e em TESTE (999–1048), nos 3
regimes, pela função única `avaliacao.avaliar`; demos por caminho fixo com hash; config.json completo.

Uso:  python -X utf8 treinar_warmstart_il.py --seeds 42 123 2024 7 1337 [--config lr_alto] [--saida results/rodada3/il_warmstart]
      [--demos results/demos_v5/trajetorias_especialista.json] [--retomar] [--smoke]
Resumível por semente (pula as que têm _DONE.flag); _DONE.flag global ao terminar todas; _FAILED.flag em erro.
"""
import os
import sys
import json
import time
import copy
import argparse
import traceback

import numpy as np

import protocolo as P

P.preparar_sys_path()
import torch                                       # noqa: E402
import avaliacao as A                              # noqa: E402
from env import AmbienteFazendaGym                 # noqa: E402
from mascara_acoes import InvolucroMascaraAcoes    # noqa: E402
from bot_perfeito import BotPerfeito               # noqa: E402
from dagger import AgenteDAgger                    # noqa: E402

# Configurações de IL da qualificação (Tab. 4.2). O warm-start usa UMA configuração fixa (D-15): padrão lr_alto.
CONFIGS_IL = {
    'baseline':   {'lr': 3e-4, 'dim_lstm': 512, 'len_seq': 32},
    'lr_alto':    {'lr': 1e-3, 'dim_lstm': 512, 'len_seq': 32},
    'lr_baixo':   {'lr': 5e-5, 'dim_lstm': 512, 'len_seq': 32},
    'lstm_menor': {'lr': 3e-4, 'dim_lstm': 256, 'len_seq': 32},
    'seq_longa':  {'lr': 3e-4, 'dim_lstm': 512, 'len_seq': 64},
}
RECEITA = {'n_epocas_bc': 100, 'batch': 64, 'n_rounds_dagger': 7, 'passos_por_round': 3650,
           'epocas_por_round': 10, 'lr_dagger': 1e-4, 'beta_inicial': 0.5, 'beta_final': 0.0}


def carregar_demos(caminho):
    with open(caminho, encoding='utf-8') as f:
        raw = json.load(f)
    return [[{'obs': np.asarray(t['obs'], np.float32), 'acao': int(t['acao']),
              'mascara': np.asarray(t['mascara'], np.float32)} for t in ep] for ep in raw]


def novo_env():
    return InvolucroMascaraAcoes(AmbienteFazendaGym(passos_maximos=P.PASSOS_MAXIMOS))


def avaliar_tudo(agente, env, n_val, n_teste, log):
    """Validação-objetivo e teste nos 3 regimes (dicionários serializáveis)."""
    pol = A.PoliticaRede(agente)
    out = {}
    for nome, sementes in (('validacao_objetivo', P.sementes_validacao_objetivo()[:n_val]),
                           ('teste', P.sementes_teste()[:n_teste])):
        out[nome] = {}
        for regime in P.NOMES_REGIMES:
            res = A.avaliar(pol, env, sementes, regime)
            out[nome][regime] = {**A.resumo(res), 'recompensas': res['recompensas'], 'contagem_acoes': res['contagem_acoes'],
                                 'acoes_pct': A.acoes_pct(res), 'ociosidade_pct': A.ociosidade_pct(res),
                                 'kpis': {k: float(np.nanmean(v)) for k, v in res['kpis'].items()},
                                 'comp_medio': {k: float(np.mean(v)) for k, v in res['comp'].items()}}
            log(f"    [{nome:18s}] {regime:8s} média {out[nome][regime]['media']:7.1f} ± {out[nome][regime]['std']:6.1f} "
                f"IQM {out[nome][regime]['iqm']:7.1f} | esperar {100 * res['contagem_acoes']['esperar'] / (len(sementes) * P.PASSOS_MAXIMOS):.0f}%")
    return out


def treinar_semente(seed, hp, receita, demos, hash_demos, caminho_demos, dir_seed, n_val_rollback, n_val, n_teste, log):
    t0 = time.time()
    os.makedirs(dir_seed, exist_ok=True)
    P.escrever_status(dir_seed, 'inicio', f'seed={seed} hp={hp}')
    P.fixar_sementes(seed)
    env = novo_env()
    bot = P.novo_professor()   # D-27: rotulador do DAgger = o mesmo professor das demonstrações
    agente = AgenteDAgger(dim_lstm=hp['dim_lstm'])
    agente.calcular_normalizacao(demos)
    sementes_rollback = P.sementes_validacao_rollback()[:n_val_rollback]

    # ── BC ──
    t_bc0 = time.time()
    hist_bc = agente.treinar_bc(demos, n_epocas=receita['n_epocas_bc'], batch_size=receita['batch'], lr=hp['lr'], len_seq=hp['len_seq'])
    t_bc = time.time() - t_bc0
    val_bc = A.avaliar(A.PoliticaRede(agente), env, sementes_rollback, 'in_dist')
    log(f"  [BC] {t_bc:.0f}s | acurácia {hist_bc['acuracia'][-1]:.1%} | validação-rollback {np.mean(val_bc['recompensas']):.1f}")
    agente.salvar(os.path.join(dir_seed, 'modelo_bc.pt'))
    P.escrever_status(dir_seed, 'bc', f"acur={hist_bc['acuracia'][-1]:.4f} val={np.mean(val_bc['recompensas']):.1f}")

    # ── DAgger com seleção da melhor rodada em validação-rollback ──
    curva = [{'rodada': 0, 'validacao_rollback': float(np.mean(val_bc['recompensas']))}]
    melhor = {'rodada': 0, 'valor': curva[0]['validacao_rollback'], 'estado': copy.deepcopy(agente.rede.state_dict())}

    def callback(rodada, ag):
        res = A.avaliar(A.PoliticaRede(ag), env, sementes_rollback, 'in_dist')
        v = float(np.mean(res['recompensas']))
        curva.append({'rodada': rodada, 'validacao_rollback': v})
        if v > melhor['valor']:
            melhor.update(rodada=rodada, valor=v, estado=copy.deepcopy(ag.rede.state_dict()))
        log(f"    rodada {rodada}: validação-rollback {v:.1f} (melhor: rodada {melhor['rodada']} = {melhor['valor']:.1f})")
        P.escrever_status(dir_seed, f'dagger_r{rodada}', f'val={v:.1f}')

    t_d0 = time.time()
    hist_dag = agente.treinar_dagger(demos, env, bot, n_rounds=receita['n_rounds_dagger'], passos_por_round=receita['passos_por_round'],
                                     epocas_por_round=receita['epocas_por_round'], batch_size=receita['batch'], lr=receita['lr_dagger'],
                                     beta_inicial=receita['beta_inicial'], beta_final=receita['beta_final'],
                                     len_seq=hp['len_seq'], callback_rodada=callback)
    t_dag = time.time() - t_d0
    agente.rede.load_state_dict(melhor['estado'])
    agente.rede.eval()
    log(f"  [DAgger] {t_dag:.0f}s | melhor rodada = {melhor['rodada']} ({melhor['valor']:.1f} em validação-rollback)")

    # ── Avaliação final (validação-objetivo + teste, 3 regimes) ──
    t_a0 = time.time()
    aval = avaliar_tudo(agente, env, n_val, n_teste, log)
    t_aval = time.time() - t_a0
    agente.salvar(os.path.join(dir_seed, 'modelo_final.pt'))

    metricas = {
        'nome_agente': 'IL-warmstart', 'config': hp, 'receita': receita, 'seed': seed,
        'acuracia_bc_final': float(hist_bc['acuracia'][-1]), 'perda_bc_final': float(hist_bc['perda'][-1]),
        'bc_perdas': [float(x) for x in hist_bc['perda']], 'bc_acuracias': [float(x) for x in hist_bc['acuracia']],
        'dagger_historico': {k: [float(x) for x in v] for k, v in hist_dag.items()},
        'curva_validacao_rollback': curva, 'melhor_rodada': melhor['rodada'],
        'avaliacao': aval,
        'tempos_s': {'bc': t_bc, 'dagger': t_dag, 'avaliacao': t_aval, 'total': time.time() - t0},
    }
    with open(os.path.join(dir_seed, 'metricas.json'), 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=1, ensure_ascii=False)
    P.registrar_config(dir_seed, experimento='il_warmstart_v2', seed_treino=seed, hp=hp, receita=receita,
                       demos=caminho_demos, hash_demos=hash_demos, n_demos=len(demos),
                       sementes={'validacao_rollback': [sementes_rollback[0], sementes_rollback[-1]],
                                 'validacao_objetivo': [P.SEED_VALIDACAO_OBJETIVO, P.SEED_VALIDACAO_OBJETIVO + n_val - 1],
                                 'teste': [P.SEED_TESTE, P.SEED_TESTE + n_teste - 1]},
                       selecao='melhor rodada do DAgger em validacao-rollback (in-dist)', tempo_s=time.time() - t0)
    P.escrever_flag(dir_seed, '_DONE.flag', f"melhor_rodada={melhor['rodada']}")
    return metricas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=P.SEMENTES_TREINO)
    ap.add_argument('--config', default='lr_alto', choices=list(CONFIGS_IL))
    ap.add_argument('--saida', default=P.DIR_WARMSTART)
    ap.add_argument('--demos', default=os.path.relpath(P.DEMOS_PADRAO, P.RAIZ))
    ap.add_argument('--retomar', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='2 épocas, 1 rodada de 730 passos, 3 episódios de avaliação')
    args = ap.parse_args()

    saida = os.path.abspath(args.saida)
    os.makedirs(saida, exist_ok=True)
    P.limpar_flags(saida)
    hp = dict(CONFIGS_IL[args.config], nome=args.config)
    receita = dict(RECEITA)
    n_val_rollback, n_val, n_teste = P.N_VALIDACAO_ROLLBACK, P.N_VALIDACAO_OBJETIVO, P.N_TESTE
    if args.smoke:
        receita.update(n_epocas_bc=2, n_rounds_dagger=1, passos_por_round=730, epocas_por_round=1)
        n_val_rollback, n_val, n_teste = 3, 3, 3

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    caminho_demos = os.path.abspath(args.demos)
    if not os.path.exists(caminho_demos):
        log(f'demos não encontradas: {caminho_demos}'); P.escrever_flag(saida, '_FAILED.flag', 'demos ausentes'); return 1
    hash_demos = P.hash_arquivo(caminho_demos)
    if P.HASH_DEMOS_PADRAO and hash_demos != P.HASH_DEMOS_PADRAO and not args.smoke:
        # D-27: abortar, não avisar. Treinar com demos de outro professor (ou de outra versão do ambiente) produz um
        # agente de partida silenciosamente incompatível com o resto da rodada, e o erro só apareceria nas tabelas.
        log(f'ERRO: hash das demos ({hash_demos[:16]}) difere do protocolo ({P.HASH_DEMOS_PADRAO[:16]}); '
            f'regenere as demos ou atualize protocolo.HASH_DEMOS_PADRAO')
        P.escrever_flag(saida, '_FAILED.flag', f'hash das demos: {hash_demos[:16]} != {P.HASH_DEMOS_PADRAO[:16]}')
        return 1
    demos = carregar_demos(caminho_demos)
    log(f"{len(demos)} demos ({sum(len(e) for e in demos)} transições) | config {args.config} {hp} | dispositivo {P.dispositivo()}")
    P.escrever_status(saida, 'inicio', f'seeds={args.seeds} config={args.config} smoke={args.smoke}')

    try:
        for seed in args.seeds:
            dir_seed = os.path.join(saida, f'seed_{seed}')
            if args.retomar and os.path.exists(os.path.join(dir_seed, '_DONE.flag')):
                log(f'seed {seed}: já concluída, pulando'); continue
            log(f'=== seed {seed} ===')
            treinar_semente(seed, hp, receita, demos, hash_demos, caminho_demos, dir_seed, n_val_rollback, n_val, n_teste, log)
        # resumo entre sementes
        resumo = {}
        for seed in args.seeds:
            m = os.path.join(saida, f'seed_{seed}', 'metricas.json')
            if os.path.exists(m):
                d = json.load(open(m, encoding='utf-8'))
                for conj in d['avaliacao']:
                    for reg, v in d['avaliacao'][conj].items():
                        resumo.setdefault(conj, {}).setdefault(reg, []).append(v['media'])
        tabela = {conj: {reg: {'media_entre_sementes': float(np.mean(v)), 'std_entre_sementes': float(np.std(v)), 'por_semente': v}
                         for reg, v in regs.items()} for conj, regs in resumo.items()}
        with open(os.path.join(saida, 'resumo.json'), 'w', encoding='utf-8') as f:
            json.dump({'config': hp, 'receita': receita, 'seeds': args.seeds, 'tabela': tabela}, f, indent=1, ensure_ascii=False)
        for conj in tabela:
            log(conj + ': ' + ' | '.join(f"{reg} {v['media_entre_sementes']:.0f} ± {v['std_entre_sementes']:.0f}" for reg, v in tabela[conj].items()))
        if all(os.path.exists(os.path.join(saida, f'seed_{s}', '_DONE.flag')) for s in args.seeds):
            P.escrever_flag(saida, '_DONE.flag', f'seeds={args.seeds}')
        return 0
    except Exception:
        traceback.print_exc()
        P.escrever_flag(saida, '_FAILED.flag', traceback.format_exc().splitlines()[-1])
        return 1


if __name__ == '__main__':
    sys.exit(main())
