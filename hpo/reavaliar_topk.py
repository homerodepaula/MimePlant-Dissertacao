"""hpo/reavaliar_topk.py — reavaliação em TESTE das melhores configurações da HPO e das configurações padrão (Fase 4.6).

Para cada candidato (top-k de um study por valor de tuning, mais configurações padrão nomeadas), treina com as 5 sementes
de treino, escolhe o melhor bloco em validação-rollback, e avalia em validação-objetivo (30 × 3 regimes) e em TESTE
(999–1048, 50 × 3 regimes) pela função única `avaliacao.avaliar`. Execuções já feitas no bake-off (mesma configuração e
semente) são reaproveitadas por cópia dos `metricas.json` (`--reusar-bakeoff`). Resumível por (candidato, semente).

Uso:
  python -X utf8 hpo/reavaliar_topk.py --study tpe_r3 rs_r3 --k 3 [--seeds 42 123 2024 7 1337] [--orcamento N] [--saida ...]
        [--padrao ppo_il ppo_zero] [--reusar-bakeoff results/bakeoff] [--retomar]
Saída: <saida>/<candidato>/seed_<s>/{metricas.json, config.json, modelo_final.pt}, <saida>/resumo.json (IQM e IC 95 % por bootstrap
estratificado por semente), _DONE.flag.
"""
import os
import sys
import json
import copy
import time
import shutil
import argparse
import subprocess
import traceback

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import protocolo as P                                  # noqa: E402

P.preparar_sys_path()
import optuna                                          # noqa: E402
import avaliacao as A                                  # noqa: E402
import ppo as PPO                                      # noqa: E402
from envs_lote import fabrica_env_padrao               # noqa: E402
from bakeoff import carregar_referencias, avaliar_final  # noqa: E402

VF = os.path.dirname(RAIZ)
DIR_HPO = os.path.join(VF, 'experimentos', 'hpo')
PADROES = {'ppo_il': {'inicializacao': 'professor'}, 'ppo_zero': {'inicializacao': 'zero'}}
BAKEOFF_EQUIV = {'padrao_ppo_il': 'ppo_il', 'padrao_ppo_zero': 'ppo_zero'}


def bootstrap_iqm(por_seed, n_boot=2000, seed=0):
    """IQM sobre os retornos por episódio de todas as sementes e IC 95 % por bootstrap estratificado (reamostra sementes e,
    dentro de cada uma, episódios). `por_seed`: lista de listas de recompensas."""
    rng = np.random.default_rng(seed)
    todos = np.concatenate([np.asarray(s, float) for s in por_seed])
    ponto = A.iqm(todos)
    n = len(por_seed)
    amostras = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        pool = np.concatenate([rng.choice(np.asarray(por_seed[i], float), len(por_seed[i]), replace=True) for i in idx])
        amostras.append(A.iqm(pool))
    return float(ponto), float(np.percentile(amostras, 2.5)), float(np.percentile(amostras, 97.5))


def candidatos_do_study(nome, k):
    """Top-k trials concluídos, por valor de ajuste.

    A inicialização NÃO está em `t.params` (ela é fixada por argumento, não sorteada — D-30), então vem do
    `user_attr` que o próprio estudo gravou. Sem isso, um candidato de um estudo do zero (D-32) chegaria ao
    `rodar_candidato` sem a chave, cairia no padrão 'professor' e seria retreinado A PARTIR DA IMITAÇÃO —
    exatamente o oposto do que o estudo mediu. Estudos anteriores à D-30 não têm o atributo; para eles o padrão
    'professor' é o correto."""
    st = optuna.load_study(study_name=nome, storage=f"sqlite:///{os.path.join(DIR_HPO, nome + '.db').replace(os.sep, '/')}")
    ini = st.user_attrs.get('inicializacao', 'professor')
    conc = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    conc.sort(key=lambda t: -t.value)
    return [(f'{nome}_trial{t.number}', dict(t.params, inicializacao=ini), float(t.value)) for t in conc[:k]]


def rodar_candidato(nome, hp, seed, args, refs, log):
    dir_out = os.path.join(os.path.abspath(args.saida), nome, f'seed_{seed}')
    if args.retomar and os.path.exists(os.path.join(dir_out, '_DONE.flag')):
        return json.load(open(os.path.join(dir_out, 'metricas.json'), encoding='utf-8'))
    # reaproveitamento do bake-off (configuração padrão idêntica, mesma semente, mesmo orçamento)
    if nome in BAKEOFF_EQUIV and args.reusar_bakeoff:
        origem = os.path.join(os.path.abspath(args.reusar_bakeoff), BAKEOFF_EQUIV[nome], f'seed_{seed}')
        if os.path.exists(os.path.join(origem, '_DONE.flag')):
            m = json.load(open(os.path.join(origem, 'metricas.json'), encoding='utf-8'))
            if int(m.get('orcamento', -1)) == int(args.orcamento):
                os.makedirs(dir_out, exist_ok=True)
                for f in ('metricas.json', 'config.json'):
                    shutil.copy(os.path.join(origem, f), os.path.join(dir_out, f))
                m['reaproveitado_de'] = origem
                with open(os.path.join(dir_out, 'metricas.json'), 'w', encoding='utf-8') as f:
                    json.dump(m, f, indent=1, ensure_ascii=False)
                P.escrever_flag(dir_out, '_DONE.flag', f'copiado de {origem}')
                log(f'  {nome} seed {seed}: reaproveitado do bake-off')
                return m
    os.makedirs(dir_out, exist_ok=True)
    P.escrever_status(dir_out, 'inicio', f'{nome} seed={seed}')
    env_aval = fabrica_env_padrao()
    sementes_rb = P.sementes_validacao_rollback()
    politica_il = os.path.join(os.path.abspath(args.warmstart), f'seed_{seed}', 'modelo_final.pt')
    curva_val, melhor = [], {'valor': -np.inf, 'passos': 0, 'estado': None}

    def callback(passos, agente):
        res = A.avaliar(A.PoliticaRede(agente), env_aval, sementes_rb, 'in_dist')
        m = float(np.mean(res['recompensas']))
        curva_val.append({'passos': int(passos), 'media': m})
        if m > melhor['valor']:
            melhor.update(valor=m, passos=int(passos), estado=copy.deepcopy(agente.rede.state_dict()))
        return False

    t0 = time.time()
    res = PPO.treinar_rl(politica_il, hp, seed, args.orcamento, callback=callback, log=lambda *a: None)
    agente = res['agente']
    if melhor['estado'] is not None and melhor['valor'] >= curva_val[-1]['media']:
        agente.rede.load_state_dict(melhor['estado'])
    agente.rede.eval()
    aval = avaliar_final(agente, env_aval, refs, P.N_VALIDACAO_OBJETIVO, P.N_TESTE, log)
    agente.salvar(os.path.join(dir_out, 'modelo_final.pt'))
    m = {'candidato': nome, 'hp': res['hp'], 'seed': seed, 'orcamento': args.orcamento, 'passos': res['passos'],
         'curva_validacao_rollback': curva_val, 'melhor': {'valor': melhor['valor'], 'passos': melhor['passos']},
         'avaliacao': aval, 'tempo_treino_s': res['tempo_s'], 'tempo_total_s': time.time() - t0}
    with open(os.path.join(dir_out, 'metricas.json'), 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=1, ensure_ascii=False)
    P.registrar_config(dir_out, experimento='hpo_reavaliacao', candidato=nome, seed_treino=seed, hp=res['hp'], orcamento_passos=args.orcamento,
                       warmstart=politica_il if res['hp'].get('inicializacao', 'professor') == 'professor' else None)
    P.escrever_flag(dir_out, '_DONE.flag', f"val_obj_norm={aval['validacao_objetivo']['media_normalizada_3_regimes']}")
    return m


def escolher_melhor_candidato(res, n_sementes=None):
    """O MimePlant: entre os candidatos da HPO, o de maior MÉDIA, entre sementes, da recompensa normalizada nos 3 regimes
    em VALIDAÇÃO-OBJETIVO (520-549). Nunca em teste. Os padrões (`padrao_*`) são termos de comparação, não candidatos.

    Definição ÚNICA (08/09/2026): a análise final (`analise_hpo.py`, testes pareados) e as ablações (`ablacoes.py`, base
    das variantes) chamam esta função. Antes, as ablações usavam o maior valor de ajuste do estudo TPE, medido com UMA
    semente, e a análise usava a média com cinco — e as duas escolhas divergiam (trial 34 x trial 70), de modo que a
    tabela de ablações falaria de uma configuração e a comparação final de outra.

    `res` é o conteúdo de `<reavaliacao>/resumo.json`. Com `n_sementes`, falha se algum candidato não tiver exatamente
    esse número de sementes: escolher sobre uma reavaliação incompleta seria escolher sobre metade dos dados."""
    cands = {n: v for n, v in res.items()
             if not n.startswith('padrao') and isinstance(v, dict) and 'validacao_objetivo' in v}
    if n_sementes is not None:
        incompletos = sorted(n for n, v in cands.items() if v.get('n_sementes') != n_sementes)
        if incompletos:
            raise ValueError(f'candidatos sem {n_sementes} sementes na reavaliação: {incompletos}')
    if not cands:
        return None
    return max(cands, key=lambda n: cands[n]['validacao_objetivo']['normalizada_media_3_regimes']['media'])


def resumir(nome, valor_tuning, por_seed):
    """Agrega as sementes: IQM + IC 95 % por regime (validação-objetivo e teste) e média normalizada."""
    out = {'candidato': nome, 'valor_tuning': valor_tuning, 'n_sementes': len(por_seed)}
    for conjunto in ('validacao_objetivo', 'teste'):
        out[conjunto] = {}
        for regime in P.NOMES_REGIMES:
            recs = [m['avaliacao'][conjunto][regime]['recompensas'] for m in por_seed]
            ponto, lo, hi = bootstrap_iqm(recs)
            medias = [m['avaliacao'][conjunto][regime]['media'] for m in por_seed]
            out[conjunto][regime] = {'iqm': ponto, 'ic95': [lo, hi], 'media_entre_sementes': float(np.mean(medias)),
                                     'std_entre_sementes': float(np.std(medias)), 'por_semente': medias,
                                     'normalizada_por_semente': [m['avaliacao'][conjunto][regime]['normalizado'] for m in por_seed]}
        norm = [m['avaliacao'][conjunto]['media_normalizada_3_regimes'] for m in por_seed]
        out[conjunto]['normalizada_media_3_regimes'] = {'media': float(np.mean(norm)), 'std': float(np.std(norm)), 'iqm': A.iqm(norm), 'por_semente': norm}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', nargs='*', default=[])
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--padrao', nargs='*', default=[], choices=list(PADROES))
    ap.add_argument('--seeds', type=int, nargs='+', default=P.SEMENTES_TREINO)
    ap.add_argument('--orcamento', type=int, default=P.ORCAMENTO_RL_PASSOS)
    ap.add_argument('--saida', default=P.DIR_REAVALIACAO)
    ap.add_argument('--warmstart', default=P.DIR_WARMSTART)
    ap.add_argument('--referencias', default=P.REFERENCIAS_PADRAO)
    ap.add_argument('--reusar-bakeoff', default=P.DIR_RL_PADRAO)
    ap.add_argument('--retomar', action='store_true')
    ap.add_argument('--paralelo', type=int, default=1,
                    help='processos simultâneos. Cada um treina uma FATIA das execuções (candidato, semente); ao fim, este '
                         'processo agrega tudo pelas flags. Sequencial, a reavaliação era a etapa mais longa da rodada 2.')
    ap.add_argument('--fatia', default=None, help='(interno) i/n: roda só as execuções de índice i módulo n; não agrega nem grava flag da raiz')
    args = ap.parse_args()

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    saida = os.path.abspath(args.saida)
    os.makedirs(saida, exist_ok=True)
    if not args.fatia:
        P.limpar_flags(saida)
    refs = carregar_referencias(args.referencias)
    candidatos = []
    for st in args.study:
        candidatos.extend(candidatos_do_study(st, args.k))
    for p in args.padrao:
        candidatos.append((f'padrao_{p}', dict(PADROES[p]), None))
    log(f"{len(candidatos)} candidatos × {len(args.seeds)} sementes | orçamento {args.orcamento} | dispositivo {P.dispositivo()}")
    execucoes = [(nome, hp, seed) for nome, hp, _ in candidatos for seed in args.seeds]
    if args.fatia:
        # trabalhador: só treina a sua fatia (resumível pelas flags por execução); quem agrega é o processo principal
        i, n = (int(x) for x in args.fatia.split('/'))
        try:
            for j, (nome, hp, seed) in enumerate(execucoes):
                if j % n == i:
                    log(f'=== [fatia {i}/{n}] {nome} | seed {seed}')
                    rodar_candidato(nome, hp, seed, args, refs, log)
            return 0
        except Exception:
            traceback.print_exc()
            return 1
    if args.paralelo > 1:
        base = [sys.executable, '-X', 'utf8', os.path.abspath(__file__), '--study', *args.study, '--k', str(args.k), '--padrao', *args.padrao,
                '--seeds', *[str(s) for s in args.seeds], '--orcamento', str(args.orcamento), '--saida', saida,
                '--warmstart', os.path.abspath(args.warmstart), '--referencias', os.path.abspath(args.referencias),
                '--reusar-bakeoff', (os.path.abspath(args.reusar_bakeoff) if args.reusar_bakeoff else ''), '--retomar']
        vivos = []
        for i in range(args.paralelo):
            f = open(os.path.join(saida, f'_fatia_{i}.log'), 'a', encoding='utf-8')
            vivos.append((i, subprocess.Popen(base + ['--fatia', f'{i}/{args.paralelo}'], cwd=RAIZ,
                                              env=dict(os.environ, PYTHONUTF8='1'), stdout=f, stderr=subprocess.STDOUT), f))
            log(f'  fatia {i}/{args.paralelo} lançada (pid {vivos[-1][1].pid})')
        falhas = []
        for i, p, f in vivos:
            p.wait(); f.close()
            log(f'  fatia {i} terminou (exit {p.returncode})')
            if p.returncode != 0:
                falhas.append(i)
        if falhas:
            P.escrever_flag(saida, '_FAILED.flag', f'fatias com erro: {falhas} (ver _fatia_<i>.log)')
            return 1
        args.retomar = True   # tudo já tem flag: o laço abaixo só lê os metricas.json e agrega
    try:
        resumo = {}
        for nome, hp, vt in candidatos:
            por_seed = []
            for seed in args.seeds:
                log(f'=== {nome} | seed {seed} | hp {hp}')
                por_seed.append(rodar_candidato(nome, hp, seed, args, refs, log))
            resumo[nome] = resumir(nome, vt, por_seed)
            r = resumo[nome]
            log(f"  {nome}: tuning {vt} | teste norm {r['teste']['normalizada_media_3_regimes']['media']:.3f} ± {r['teste']['normalizada_media_3_regimes']['std']:.3f} "
                f"| in-dist IQM {r['teste']['in_dist']['iqm']:.0f} [{r['teste']['in_dist']['ic95'][0]:.0f}, {r['teste']['in_dist']['ic95'][1]:.0f}]")
            with open(os.path.join(saida, 'resumo.json'), 'w', encoding='utf-8') as f:
                json.dump(resumo, f, indent=1, ensure_ascii=False)
        P.escrever_flag(saida, '_DONE.flag', f'{len(candidatos)} candidatos')
        return 0
    except Exception:
        traceback.print_exc()
        P.escrever_flag(saida, '_FAILED.flag', traceback.format_exc().splitlines()[-1])
        return 1


if __name__ == '__main__':
    sys.exit(main())
