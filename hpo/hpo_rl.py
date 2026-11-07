"""hpo/hpo_rl.py — Otimização de Hiperparâmetros da fase de RL do MimePlant (Fase 4; D-16, D-18, D-21).

Substrato: `agents/rl/ppo.treinar_rl` sobre `hp.ESPACO_MIMEPLANT` (PPO recorrente partindo SEMPRE da imitação do
professor — D-30 — com a temperatura inicial da política clonada, a penalidade KL e seu decaimento, que são o
mecanismo de "soltar a mão", e os demais hiperparâmetros do PPO). Treinar do zero não é configuração do método: é o
braço de CONTROLE `ppo_zero` do bake-off. O espaço deixou de ser condicional, então os 12 hiperparâmetros existem em
todo trial e a análise de importância não descarta nenhum.
Métodos: TPE (multivariado, com grupos) + HyperbandPruner; random search com a
MESMA poda; o DEHB, estudo de comparação na mesma escada e no mesmo orçamento, tem coordenador próprio em `hpo_dehb.py`
(D-34). Orçamento em FRE (1 FRE = um refinamento de `--orcamento` passos +
avaliação). Objetivo de cada trial = média nos 3 regimes da recompensa normalizada em VALIDAÇÃO-OBJETIVO (520–549).
Poda: a cada bloco de N/9 passos o trial reporta a recompensa normalizada em validação-rollback (in-dist, 20 episódios),
a mesma quantidade para todos os trials no mesmo passo. Sementes de teste (999+i) não entram aqui: a reavaliação do top-k
em teste é feita por `hpo/reavaliar_topk.py`.

Uso (a partir da raiz do código):
  python -X utf8 hpo/hpo_rl.py --study tpe_r3 --sampler tpe --n-fre 64 [--orcamento N] [--seeds-trial 42] [--retomar]
  python -X utf8 hpo/hpo_rl.py --study rs_r3  --sampler random --n-fre 64
  vários processos no mesmo study: --paralelo em todos e --trabalhador 0, 1, 2... (semente própria por processo)
  (orçamento, warm-start e referências padrão vêm de protocolo.py: ORCAMENTO_RL_PASSOS, DIR_WARMSTART, REFERENCIAS_PADRAO)
  python -X utf8 hpo/hpo_rl.py --study smoke --smoke
Retomável: storage SQLite em ../experimentos/hpo/<study>.db (load_if_exists); FRE consumidos são recontados dos trials.
Saída: ../experimentos/hpo/<study>/{trial_<n>/{config.json, metricas.json}, PROGRESSO.json, _DONE.flag}.
"""
import os
import sys
import json
import time
import copy
import zlib
import argparse
import traceback

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import protocolo as P                                  # noqa: E402

P.preparar_sys_path()
import optuna                                          # noqa: E402
import avaliacao as A                                  # noqa: E402
from hp import ESPACO_MIMEPLANT, ESPACOS, CONDICIONAIS_PROFESSOR, INICIALIZACAO_DO_METODO  # noqa: E402
import ppo as PPO                                      # noqa: E402
from envs_lote import fabrica_env_padrao               # noqa: E402
from bakeoff import carregar_referencias, normalizado  # noqa: E402

VF = os.path.dirname(RAIZ)
DIR_HPO = os.path.join(VF, 'experimentos', 'hpo')


def espaco_de(inicializacao, completo=False):
    """O espaço de busca de cada inicialização.

    Com `professor`, os 12 hiperparâmetros do método (D-30). Com `zero`, os 8 que de fato agem: `ppo.treinar_rl`
    anula `temperatura_inicial`, `beta_kl0`, `beta_kl_decay` e `k_aquec` quando não há política de partida
    (ppo.py:73-82), de modo que sorteá-los gastaria orçamento e poluiria o modelo de densidade do TPE com
    dimensões inertes. `completo=True` força as 12 dimensões também no controle, para quem preferir igualar a
    DIMENSÃO da busca em vez de igualar o conjunto de hiperparâmetros que produzem efeito.

    A assimetria é declarada de propósito, porque não há escolha neutra: com 8 dimensões o controle resolve um
    problema de busca mais fácil sob o mesmo orçamento; com 12, gasta parte do orçamento em dimensões que não
    mudam nada. O padrão é o espaço efetivo, e a alternativa fica disponível."""
    if inicializacao == 'professor' or completo:
        return ESPACO_MIMEPLANT
    return ESPACOS['ppo_zero']


def sugerir(trial, smoke=False, inicializacao=INICIALIZACAO_DO_METODO, espaco_completo=False):
    """Espaço do MÉTODO: o agente parte da imitação do professor e solta a mão pelo decaimento da âncora
    (D-30, 07/09/2026). `inicializacao` não é sorteada — é fixada por argumento, e o padrão é `professor`.

    `inicializacao='zero'` é o BRAÇO DE CONTROLE (D-32, 11/09/2026): mesma busca, mesmo orçamento, mesmo podador,
    mesmo objetivo e mesmas sementes, treinando do zero. Ele existe para fechar o quadrante que faltava — com
    hiperparâmetros otimizados dos DOIS lados —, e não para ser escolhido pela busca. Nenhum estudo mistura as duas
    inicializações: cada uma tem o seu, e a comparação é entre estudos."""
    hp = {'inicializacao': inicializacao}
    for nome, spec in espaco_de(inicializacao, espaco_completo).items():
        tipo = spec[0]
        if tipo == 'log':
            hp[nome] = trial.suggest_float(nome, spec[1], spec[2], log=True)
        elif tipo == 'float':
            hp[nome] = trial.suggest_float(nome, spec[1], spec[2])
        elif tipo == 'int':
            hp[nome] = trial.suggest_int(nome, spec[1], spec[2])
        elif tipo == 'cat':
            hp[nome] = trial.suggest_categorical(nome, spec[1])
    if smoke:
        hp.update({'V': 2, 'k_aquec': min(1, hp.get('k_aquec', 0)), 'n_epocas': 1, 'minilote_seqs': 4, 'blocos': 2})
    return hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default=P.STUDIES_HPO[0])
    ap.add_argument('--sampler', choices=['tpe', 'random'], default='tpe')
    ap.add_argument('--n-fre', type=float, default=64.0, help='orçamento total em execuções completas equivalentes')
    ap.add_argument('--orcamento', type=int, default=P.ORCAMENTO_RL_PASSOS, help='passos de ambiente de um trial em fidelidade máxima (N)')
    ap.add_argument('--seeds-trial', type=int, nargs='+', default=[42], help='sementes de treino por trial (média)')
    ap.add_argument('--seed-sampler', type=int, default=0)
    ap.add_argument('--n-startup', type=int, default=20)
    ap.add_argument('--warmstart', default=P.DIR_WARMSTART)
    ap.add_argument('--referencias', default=P.REFERENCIAS_PADRAO)
    ap.add_argument('--sem-poda', action='store_true')
    ap.add_argument('--paralelo', action='store_true', help='mais de um processo no mesmo study (constant_liar)')
    ap.add_argument('--trabalhador', type=int, default=0,
                    help='índice deste processo quando o study roda em vários (0 = o primeiro). Com i > 0 a semente do '
                         'amostrador é própria (200000·i + crc32 do study): sem isso todos sorteariam a mesma sequência')
    ap.add_argument('--retomar', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--dispositivo', default=None)
    ap.add_argument('--inicializacao', choices=['professor', 'zero'], default=INICIALIZACAO_DO_METODO,
                    help="'professor' (padrao) e o metodo; 'zero' e o braco de CONTROLE da D-32, em estudo proprio")
    ap.add_argument('--espaco-completo', action='store_true',
                    help='no controle, sortear tambem os 4 hiperparametros inertes, igualando a DIMENSAO da busca')
    ap.add_argument('--forcar-inicializacao', choices=['professor', 'zero'], default=None,
                    help='(depuração) enfileira um primeiro trial com esta inicialização')
    args = ap.parse_args()

    if args.smoke:
        args.orcamento = 4 * P.PASSOS_MAXIMOS
        args.n_fre = 2.0
        args.n_startup = 1
    N = int(args.orcamento)
    n_rb = 3 if args.smoke else P.N_VALIDACAO_ROLLBACK
    n_obj = 3 if args.smoke else P.N_VALIDACAO_OBJETIVO
    dir_study = os.path.join(DIR_HPO, args.study)
    os.makedirs(dir_study, exist_ok=True)
    if not args.retomar:
        P.limpar_flags(dir_study)
    elif os.path.exists(os.path.join(dir_study, '_FAILED.flag')):
        # falha de uma sessão ANTERIOR (19/09/2026): o validador reportaria FALHA para sempre num estudo que se recuperou
        os.remove(os.path.join(dir_study, '_FAILED.flag'))
    storage =f"sqlite:///{os.path.join(DIR_HPO, args.study + '.db').replace(os.sep, '/')}"
    refs = carregar_referencias(args.referencias)
    # proxy da poda: medido em validação-rollback (500–519) → referências do MESMO conjunto (06/09/2026)
    if 'validacao_rollback' not in refs['oraculo_perfeito']:
        raise KeyError('referencias.json antigo (só "validacao"): regenerar com referencias.py antes da HPO (06/09/2026)')
    r_alea = refs['aleatorio']['validacao_rollback']['in_dist']['media']
    r_perf = refs['oraculo_perfeito']['validacao_rollback']['in_dist']['media']

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # A semente do amostrador é derivada do study E do número de trials já existentes (07/09/2026). Dois defeitos
    # medidos na rodada abortada: (i) `tpe_v1` e `rs_v1` recebiam `--seed-sampler 0` os dois, e os trials 0 a 3
    # saíam BIT A BIT idênticos nos dois bancos, de modo que a comparação TPE x busca aleatória comparava um study
    # com uma cópia de si mesmo durante toda a fase de partida; (ii) o amostrador era reconstruído com a mesma
    # semente a cada `--retomar`, e como a cadeia relança jobs mortos, a busca aleatória repetia as configurações já
    # avaliadas. Somar `1000 * n_trials_previos` continua determinístico e reprodutível (a semente efetiva fica
    # gravada nos user_attrs), mas garante um fluxo novo a cada retomada.
    # `zlib.crc32` e não `hash()`: o hash de str em Python é aleatorizado por processo (PYTHONHASHSEED), o que
    # tornaria a semente irreprodutível entre execuções — exatamente o oposto do que este bloco conserta.
    semente_study = int(args.seed_sampler) if args.seed_sampler else zlib.crc32(args.study.encode('utf-8')) % 100_000
    if args.trabalhador > 0 and not args.seed_sampler:
        semente_study += 200_000 * int(args.trabalhador)   # mesma regra de controle_zero.py e dos trabalhadores w2/w3 da rodada 2
    # `min_resource = N/3` (era N/9). Motivo (auditoria de 06/09, medido na rodada 1): no primeiro rung com N/9
    # (~28 mil passos, 12 % do orcamento) uma politica treinada do zero ainda esta perto do aleatorio (proxy mediano
    # 0,03) enquanto a clonada do professor ja esta em 0,62. O Hyperband promove 1 em 3, entao matava 54 % dos trials
    # `inicializacao = zero` contra 21-24 % dos `professor`. Com a D-30 a inicializacao saiu do espaco e todo trial
    # parte do professor, entao esse vies especifico deixou de existir; o N/3 fica porque a razao de fundo continua
    # valendo -- no primeiro rung com N/9 a politica ainda nao teve atualizacoes suficientes para separar
    # hiperparametro bom de hiperparametro ruim, sobretudo com `k_aquec` alto consumindo o inicio do orcamento.
    pruner = optuna.pruners.NopPruner() if args.sem_poda else optuna.pruners.HyperbandPruner(
        min_resource=max(1, N // 3), max_resource=N, reduction_factor=3)
    study = optuna.create_study(study_name=args.study, storage=storage, direction='maximize',
                                pruner=pruner, load_if_exists=True)
    # o amostrador só pode ser construído depois do study, porque a semente depende de quantos trials já existem
    semente_efetiva = semente_study + 1_000 * len(study.trials)
    if args.sampler == 'tpe':
        study.sampler = optuna.samplers.TPESampler(seed=semente_efetiva, multivariate=True, group=True,
                                                   n_startup_trials=args.n_startup, constant_liar=args.paralelo)
    else:
        study.sampler = optuna.samplers.RandomSampler(seed=semente_efetiva)
    log(f'amostrador {args.sampler} com semente {semente_efetiva} '
        f'(base {semente_study} + 1000 x {len(study.trials)} trials já no banco)')
    for k, v in {'hash_git': P.hash_git(), 'orcamento_passos': N, 'n_fre_alvo': args.n_fre, 'seeds_trial': args.seeds_trial,
                 'sampler': args.sampler, 'poda': 'nop' if args.sem_poda else 'hyperband(N/3,N,3)',
                 'semente_sampler_efetiva': semente_efetiva, 'inicializacao': args.inicializacao,
                 'espaco_completo': bool(args.espaco_completo),
                 'versoes': P.versoes(), 'espaco': {k: list(v) if isinstance(v, tuple) else v
                           for k, v in espaco_de(args.inicializacao, args.espaco_completo).items()},
                 'objetivo': 'media dos 3 regimes da recompensa normalizada em validacao-objetivo 520-549 (referencias do mesmo conjunto)',
                 'proxy_poda': 'recompensa normalizada em validacao-rollback 500-519 (in-dist; referencias do mesmo conjunto)'}.items():
        study.set_user_attr(k, json.loads(json.dumps(v, default=str)))

    if args.forcar_inicializacao:
        raise SystemExit("--forcar-inicializacao saiu com a D-30: 'inicializacao' nao e sorteada. Para um estudo "
                         "inteiro do zero, use --inicializacao zero em um study PROPRIO (D-32); para uma unica "
                         "execucao, bakeoff.py --bracos ppo_zero.")
    env_aval = fabrica_env_padrao()
    sementes_rb = P.sementes_validacao_rollback()[:n_rb]
    sementes_obj = P.sementes_validacao_objetivo()[:n_obj]

    def fre_consumidos():
        """Soma o FRE dos trials concluídos e podados, e o FRE parcial dos que morreram (FAIL) — antes um trial
        interrompido não contava e o método ficava com mais orçamento do que o declarado (criticas.md, §6.5; 06/09)."""
        total = 0.0
        for t in study.trials:
            if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED):
                total += float(t.user_attrs.get('fre', t.user_attrs.get('fre_parcial', 0.0)))
            elif t.state == optuna.trial.TrialState.FAIL:
                total += float(t.user_attrs.get('fre_parcial', 0.0))
        return total

    def escrever_progresso():
        conc = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pod = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        prog = {'study': args.study, 'sampler': args.sampler, 'fre_alvo': args.n_fre, 'fre_consumidos': round(fre_consumidos(), 3),
                'concluidos': len(conc), 'podados': len(pod), 'atualizado': time.strftime('%Y-%m-%d %H:%M:%S')}
        if conc:
            bt = max(conc, key=lambda t: t.value)
            prog['melhor'] = {'numero': bt.number, 'valor': bt.value, 'params': bt.params}
            prog['melhor_ate_agora'] = [max(t.value for t in conc[:i + 1]) for i in range(len(conc))]
        with open(os.path.join(dir_study, 'PROGRESSO.json'), 'w', encoding='utf-8') as f:
            json.dump(prog, f, indent=1, ensure_ascii=False)
        return prog

    def objetivo(trial):
        hp = sugerir(trial, smoke=args.smoke, inicializacao=args.inicializacao,
                     espaco_completo=args.espaco_completo)
        dir_t = os.path.join(dir_study, f'trial_{trial.number}')
        os.makedirs(dir_t, exist_ok=True)
        t0 = time.time()
        valores, detalhes, passos_total = [], [], 0
        podado = False
        for seed in args.seeds_trial:
            politica_il = os.path.join(args.warmstart, f'seed_{seed}', 'modelo_final.pt')
            curva_val, melhor = [], {'valor': -np.inf, 'passos': 0, 'estado': None}

            def callback(passos, agente):
                res = A.avaliar(A.PoliticaRede(agente), env_aval, sementes_rb, 'in_dist')
                m = float(np.mean(res['recompensas']))
                proxy = A.normalizar(m, r_alea, r_perf)
                curva_val.append({'passos': int(passos), 'media': m, 'proxy_norm': proxy, 'ociosidade_pct': A.ociosidade_pct(res)})
                trial.set_user_attr('fre_parcial', (passos_total + int(passos)) / (N * max(1, len(args.seeds_trial))))
                if m > melhor['valor']:
                    melhor.update(valor=m, passos=int(passos), estado=copy.deepcopy(agente.rede.state_dict()))
                if len(args.seeds_trial) == 1:          # poda só com uma semente por trial (a curva é do trial inteiro)
                    trial.report(proxy, step=int(passos))
                    # O ÚLTIMO bloco não pode podar (07/09/2026). O SuccessiveHalving do Optuna não conhece o
                    # recurso máximo e cria rungs indefinidamente: com min_resource = N/3 e eta = 3, um dos brackets
                    # do Hyperband tem o primeiro rung em 249.999, ABAIXO do passo terminal real (254.040 com V=12,
                    # porque o laço é `while passos < orcamento`). O trial então treinava o orçamento inteiro e só
                    # aí era julgado, virando PODADO sem produzir objetivo: medido nos logs da rodada abortada,
                    # 12 trials e 12,24 FRE no tpe_v1 (36 % do consumido) e 17 trials e 17,34 FRE no rs_v1 (48 %) —
                    # e o desperdício maior na busca aleatória enviesava justamente a comparação TPE x aleatória.
                    # A checagem `passos < N` vem ANTES do `should_prune()` de propósito: o `should_prune` tem efeito
                    # colateral (grava `completed_rung_*` no trial e contamina os valores competidores dos outros),
                    # e o curto-circuito do `and` garante que ele nem seja chamado no bloco terminal.
                    if int(passos) < N and trial.should_prune():
                        return True
                return False

            res = PPO.treinar_rl(politica_il, hp, seed, N, callback=callback, log=lambda *a: None, dispositivo=args.dispositivo)
            passos_total += res['passos']
            if res['podado']:
                podado = True
                detalhes.append({'seed': seed, 'passos': res['passos'], 'curva_validacao_rollback': curva_val, 'podado': True})
                break
            agente = res['agente']
            if melhor['estado'] is not None and melhor['valor'] >= curva_val[-1]['media']:
                agente.rede.load_state_dict(melhor['estado'])
            agente.rede.eval()
            pol = A.PoliticaRede(agente)
            por_regime = {}
            for regime in P.NOMES_REGIMES:
                r = A.avaliar(pol, env_aval, sementes_obj, regime)
                media = float(np.mean(r['recompensas']))
                por_regime[regime] = {'media': media, 'iqm': A.iqm(r['recompensas']),
                                      'normalizada': normalizado(media, refs, 'validacao_objetivo', regime),
                                      'esperar_pct': 100.0 * r['contagem_acoes']['esperar'] / (len(sementes_obj) * P.PASSOS_MAXIMOS),
                                      'ociosidade_pct': A.ociosidade_pct(r), 'acoes_pct': A.acoes_pct(r),
                                      'contagem_acoes': r['contagem_acoes'],
                                      'comp_medio': {k: float(np.mean(v)) for k, v in r['comp'].items()}}
            obj = float(np.mean([por_regime[reg]['normalizada'] for reg in P.NOMES_REGIMES]))
            valores.append(obj)
            detalhes.append({'seed': seed, 'passos': res['passos'], 'curva_validacao_rollback': curva_val, 'melhor': {'valor': melhor['valor'], 'passos': melhor['passos']},
                             'validacao_objetivo': por_regime, 'objetivo': obj, 'curva_treino': res['curva'],
                             'retorno_treino_ultimos_20': float(np.mean([e['retorno'] for e in res['episodios_treino'][-20:]])),
                             'esperar_pct_treino': float(np.mean([e['acoes'][P.ACAO_ESPERAR] for e in res['episodios_treino'][-20:]]) / P.PASSOS_MAXIMOS * 100)})
        fre = passos_total / (N * max(1, len(args.seeds_trial)))
        trial.set_user_attr('fre', fre)
        trial.set_user_attr('passos', passos_total)
        trial.set_user_attr('tempo_s', time.time() - t0)
        with open(os.path.join(dir_t, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump({'trial': trial.number, 'hp': hp, 'seeds_trial': args.seeds_trial, 'orcamento': N, 'hash_git': study.user_attrs['hash_git'],
                       'data': time.strftime('%Y-%m-%d %H:%M:%S')}, f, ensure_ascii=False, indent=1)
        with open(os.path.join(dir_t, 'metricas.json'), 'w', encoding='utf-8') as f:
            json.dump({'trial': trial.number, 'podado': podado, 'fre': fre, 'tempo_s': time.time() - t0, 'objetivo': (float(np.mean(valores)) if valores else None),
                       'por_seed': detalhes}, f, ensure_ascii=False, indent=1, default=float)
        if podado:
            log(f"trial {trial.number}: PODADO em {passos_total} passos ({fre:.2f} FRE) | {hp['inicializacao']} | {time.time() - t0:.0f}s")
            raise optuna.TrialPruned()
        log(f"trial {trial.number}: objetivo {np.mean(valores):.3f} | {hp['inicializacao']} T={hp.get('temperatura_inicial', '-')} "
            f"β={hp.get('beta_kl0', '-')} lr={hp['lr']:.1e} ent={hp['c_ent']:.1e} | {fre:.2f} FRE | {time.time() - t0:.0f}s")
        return float(np.mean(valores))

    def parar_por_orcamento(study_, trial_):
        prog = escrever_progresso()
        P.escrever_status(dir_study, f'trial_{trial_.number}', f"estado={trial_.state.name} fre={prog['fre_consumidos']}")
        if prog['fre_consumidos'] >= args.n_fre:
            study_.stop()

    log(f"study {args.study} | sampler {args.sampler} | poda {'não' if args.sem_poda else 'hyperband'} | N={N} | alvo {args.n_fre} FRE "
        f"| já consumidos {fre_consumidos():.2f} FRE em {len(study.trials)} trials | dispositivo {P.dispositivo()}")
    P.escrever_status(dir_study, 'inicio', f'sampler={args.sampler} n_fre={args.n_fre} N={N}')
    try:
        if fre_consumidos() < args.n_fre:
            study.optimize(objetivo, callbacks=[parar_por_orcamento], gc_after_trial=True)
        prog = escrever_progresso()
        log(f"fim: {prog['concluidos']} concluídos, {prog['podados']} podados, {prog['fre_consumidos']:.2f} FRE | melhor: {prog.get('melhor')}")
        P.escrever_flag(dir_study, '_DONE.flag', f"fre={prog['fre_consumidos']}")
        return 0
    except Exception:
        traceback.print_exc()
        P.escrever_flag(dir_study, '_FAILED.flag', traceback.format_exc().splitlines()[-1])
        return 1


if __name__ == '__main__':
    sys.exit(main())
