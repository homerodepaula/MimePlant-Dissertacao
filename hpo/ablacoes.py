"""hpo/ablacoes.py — ablações da configuração ótima da HPO (Fase 4.7), com 5 sementes e avaliação em teste.

Leitura sob a D-27 (professor = oráculo íntegro): a pergunta das ablações deixa de ser "quanto a imitação recupera de um
professor degradado" e passa a ser "quanto a imitação de uma heurística competente ajuda o reforço a ir ALÉM dela".
A ablação `zero` responde isso diretamente. Desde a D-30 a inicialização não é hiperparâmetro da HPO, então esta
ablação é o ÚNICO lugar em que o treino do zero é medido com a configuração ótima.

BASE (10/09/2026): a configuração ablada é o MimePlant — o melhor candidato da HPO pela validação-objetivo com as 5
sementes da reavaliação, escolhido por `reavaliar_topk.escolher_melhor_candidato`, a mesma função que a análise final
usa. Antes era o maior valor de ajuste do estudo TPE (uma semente), que divergia da escolha da análise.

PARALELISMO (10/09/2026): sem `--apenas`, o script é o ORQUESTRADOR — escolhe a base, grava `_base.json`, lança um
trabalhador por ablação (até `--paralelo`, padrão 4), espera todos e consolida. Cada trabalhador (`--apenas NOME`) roda
uma ablação em todas as sementes, lendo a base do arquivo, e grava por (ablação, semente) com `_DONE.flag` própria, de
modo que trabalhadores não disputam arquivo. A consolidação relê tudo pelo cache e treina em série o que algum
trabalhador não concluiu.

Variantes da configuração base:
  - se a melhor usa `inicializacao = professor`: `zero` com os demais hiperparâmetros iguais (o que a imitação acrescenta),
    `beta_kl0 = 0` (sem âncora ao professor — desde 06/09 o piso `beta_kl_min` não religa o KL quando β_0 = 0),
    `temperatura_inicial = 1` (sem suavizar a política clonada) e `abl_kl_forte` (β_0 = 0,1, τ = 1: o "IL→RL clássico",
    ancorado ao professor de verdade, com os demais hiperparâmetros da configuração ótima);
  - se a melhor usa `zero`: `professor` com os demais iguais e os condicionais padrão (o que a imitação acrescenta).
"Sem RL" é o agente de partida, já medido em `protocolo.DIR_WARMSTART`. Reaproveita `rodar_candidato`/`resumir` de reavaliar_topk.py.

Uso:  python -X utf8 hpo/ablacoes.py [--seeds 42 123 2024 7 1337] [--orcamento N] [--saida ...] [--retomar]   (padrões: protocolo.py)
"""
import os
import sys
import json
import time
import argparse
import subprocess
import traceback

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocolo as P                                  # noqa: E402

P.preparar_sys_path()
import optuna                                          # noqa: E402
from bakeoff import carregar_referencias               # noqa: E402
from reavaliar_topk import rodar_candidato, resumir, escolher_melhor_candidato, DIR_HPO  # noqa: E402
from ppo import HP_PADRAO                              # noqa: E402


def variantes(params):
    base = dict(params)
    out = {}
    if base.get('inicializacao', 'professor') == 'professor':
        v = {k: x for k, x in base.items() if k not in ('temperatura_inicial', 'beta_kl0', 'beta_kl_decay', 'k_aquec')}
        v['inicializacao'] = 'zero'; out['abl_zero'] = v
        v = dict(base); v['beta_kl0'] = 0.0; out['abl_sem_kl'] = v
        v = dict(base); v['temperatura_inicial'] = 1.0; out['abl_sem_temperatura'] = v
        v = dict(base); v['beta_kl0'] = HP_PADRAO['beta_kl0']; v['beta_kl_decay'] = HP_PADRAO['beta_kl_decay']
        v['temperatura_inicial'] = 1.0; out['abl_kl_forte'] = v
    else:
        v = dict(base); v['inicializacao'] = 'professor'
        for k in ('temperatura_inicial', 'beta_kl0', 'beta_kl_decay', 'k_aquec'):
            v[k] = HP_PADRAO[k]
        out['abl_professor'] = v
    return out


def escolher_base(args, log):
    """A configuração ablada é o MimePlant: o melhor candidato da HPO pela validação-objetivo com as 5 sementes da
    reavaliação, pela MESMA função que a análise final usa (`reavaliar_topk.escolher_melhor_candidato`). Os
    hiperparâmetros não estão no `resumo.json` da reavaliação, então vêm do estudo Optuna pelo número do trial.

    Até 08/09 a base era o maior valor de ajuste do estudo TPE, medido com uma semente: o trial 34 (1,742), enquanto a
    reavaliação com cinco sementes elegia o trial 70 (1,661 contra 1,655 do 34). A tabela de ablações falaria de uma
    configuração e a comparação final de outra. Decisão do Homero, 10/09/2026."""
    dir_rea = os.path.abspath(args.reavaliacao)
    if not os.path.exists(os.path.join(dir_rea, '_DONE.flag')):
        raise SystemExit(f'reavaliação sem _DONE.flag em {dir_rea}: a base das ablações só pode ser escolhida sobre '
                         f'a reavaliação COMPLETA, senão a escolha dependeria de quais candidatos já terminaram')
    res = json.load(open(os.path.join(dir_rea, 'resumo.json'), encoding='utf-8'))
    # A reavaliação é conferida contra o PRÓPRIO protocolo (as 5 sementes de treino), e não contra `--seeds` das
    # ablações: são quantidades independentes, e confundi-las faria as ablações recusarem uma reavaliação correta
    # sempre que rodassem com outro número de sementes (achado pelo teste de 10/09/2026).
    nome = escolher_melhor_candidato(res, n_sementes=len(P.SEMENTES_TREINO))
    if nome is None:
        raise SystemExit('reavaliação sem candidato da HPO')
    study, numero = nome.rsplit('_trial', 1)
    st = optuna.load_study(study_name=study, storage=f"sqlite:///{os.path.join(DIR_HPO, study + '.db').replace(os.sep, '/')}")
    trial = next(t for t in st.trials if t.number == int(numero))
    val = res[nome]['validacao_objetivo']['normalizada_media_3_regimes']
    log(f"base das ablações: {nome} | validação-objetivo 5 sementes {val['media']:.4f} ± {val['std']:.4f} | "
        f"ajuste (1 semente) {trial.value:.4f}")
    return {'candidato': nome, 'study': study, 'numero': trial.number, 'params': dict(trial.params),
            'valor_ajuste_1_semente': trial.value, 'validacao_objetivo_5_sementes': val,
            'criterio': 'reavaliar_topk.escolher_melhor_candidato: maior média entre sementes da normalizada nos 3 '
                        'regimes em validação-objetivo 520-549, excluídos os padrões; nunca em teste'}


def _comando_trabalhador(args, nome, base_json):
    """Linha de comando de um trabalhador: roda UMA ablação em todas as sementes, com a base já fixada pelo
    orquestrador (lida de `base_json`), para que os trabalhadores nunca escolham bases diferentes."""
    cmd = [sys.executable, '-X', 'utf8', os.path.abspath(__file__), '--apenas', nome, '--base-json', base_json,
           '--saida', args.saida, '--warmstart', args.warmstart, '--referencias', args.referencias,
           '--orcamento', str(args.orcamento), '--seeds', *[str(s) for s in args.seeds], '--retomar']
    if args.reusar_bakeoff:
        cmd += ['--reusar-bakeoff', args.reusar_bakeoff]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default=P.STUDIES_HPO[0], help='(mantido por compatibilidade; a base vem da reavaliação)')
    ap.add_argument('--seeds', type=int, nargs='+', default=P.SEMENTES_TREINO)
    ap.add_argument('--orcamento', type=int, default=P.ORCAMENTO_RL_PASSOS)
    ap.add_argument('--saida', default=P.DIR_ABLACOES)
    ap.add_argument('--warmstart', default=P.DIR_WARMSTART)
    ap.add_argument('--referencias', default=P.REFERENCIAS_PADRAO)
    ap.add_argument('--reavaliacao', default=P.DIR_REAVALIACAO)
    ap.add_argument('--reusar-bakeoff', default='')
    ap.add_argument('--paralelo', type=int, default=4,
                    help='ablações simultâneas, uma por processo (10/09/2026: decisão do Homero; 1 = sequencial)')
    ap.add_argument('--apenas', default=None, help='(interno) trabalhador: roda só esta ablação')
    ap.add_argument('--base-json', default=None, help='(interno) base já escolhida pelo orquestrador')
    ap.add_argument('--retomar', action='store_true')
    args = ap.parse_args()
    args.saida = os.path.abspath(args.saida)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    refs = carregar_referencias(args.referencias)

    # ── trabalhador: uma ablação, todas as sementes; nunca mexe nas flags do topo da pasta ──
    if args.apenas:
        base = json.load(open(args.base_json, encoding='utf-8'))
        hp = variantes(base['params'])[args.apenas]
        log(f"trabalhador {args.apenas} | base {base['candidato']} | sementes {args.seeds}")
        for s in args.seeds:
            rodar_candidato(args.apenas, hp, s, args, refs, log)
        log(f"trabalhador {args.apenas} concluído")
        return 0

    # ── orquestrador ──
    os.makedirs(args.saida, exist_ok=True)
    P.limpar_flags(args.saida)
    try:
        base = escolher_base(args, log)
        caminho_base = os.path.join(args.saida, '_base.json')
        # Numa retomada, a base tem de ser a MESMA da execução anterior: uma tabela de ablações com variantes de duas
        # bases diferentes seria pior do que falhar. A escolha é determinística, então isso só dispara se a
        # reavaliação mudou entre as execuções.
        if args.retomar and os.path.exists(caminho_base):
            anterior = json.load(open(caminho_base, encoding='utf-8'))
            if anterior.get('candidato') != base['candidato']:
                raise SystemExit(f"base mudou entre execuções ({anterior.get('candidato')} -> {base['candidato']}): "
                                 f"apague {args.saida} para refazer as ablações sobre a base nova")
        with open(caminho_base, 'w', encoding='utf-8') as f:
            json.dump(base, f, indent=1, ensure_ascii=False)
        vars_ = variantes(base['params'])
        log(f"ablações: {list(vars_)} | {len(args.seeds)} sementes | até {args.paralelo} em paralelo")

        if args.paralelo > 1:
            env = dict(os.environ, PYTHONUTF8='1')
            pendentes, vivos = list(vars_), {}
            while pendentes or vivos:
                while pendentes and len(vivos) < args.paralelo:
                    nome = pendentes.pop(0)
                    saida_log = open(os.path.join(args.saida, f'_trabalhador_{nome}.log'), 'a', encoding='utf-8')
                    vivos[nome] = (subprocess.Popen(_comando_trabalhador(args, nome, caminho_base), cwd=RAIZ, env=env,
                                                    stdout=saida_log, stderr=subprocess.STDOUT), saida_log)
                    log(f"  trabalhador {nome} lançado (pid {vivos[nome][0].pid})")
                time.sleep(30)
                for nome in [n for n, (p, _) in vivos.items() if p.poll() is not None]:
                    p, saida_log = vivos.pop(nome)
                    saida_log.close()
                    log(f"  trabalhador {nome} terminou (exit {p.returncode})")

        # Consolidação: reler cada (ablação, semente). O que um trabalhador concluiu vem do cache (_DONE.flag da
        # semente); o que falhou é treinado aqui, em série — a falha de um trabalhador não deixa buraco na tabela.
        args.retomar = True
        resumo = {'base': base,
                  'melhor_trial': {'numero': base['numero'], 'valor': base['valor_ajuste_1_semente'], 'params': base['params']}}
        for nome, hp in vars_.items():
            por_seed = [rodar_candidato(nome, hp, s, args, refs, log) for s in args.seeds]
            resumo[nome] = resumir(nome, None, por_seed)
            resumo[nome]['hp'] = hp
            r = resumo[nome]['teste']
            log(f"  {nome}: teste norm {r['normalizada_media_3_regimes']['media']:.3f} ± {r['normalizada_media_3_regimes']['std']:.3f} | in-dist IQM {r['in_dist']['iqm']:.0f}")
        with open(os.path.join(args.saida, 'resumo.json'), 'w', encoding='utf-8') as f:
            json.dump(resumo, f, indent=1, ensure_ascii=False)
        P.escrever_flag(args.saida, '_DONE.flag', f"{len(vars_)} ablações sobre {base['candidato']}")
        return 0
    except BaseException as e:
        if isinstance(e, SystemExit) and e.code in (0, None):
            raise
        traceback.print_exc()
        P.escrever_flag(args.saida, '_FAILED.flag', traceback.format_exc().splitlines()[-1])
        return 1


if __name__ == '__main__':
    sys.exit(main())
