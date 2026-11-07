"""Compara, em RECOMPENSA BRUTA no conjunto de teste, o RL do zero com HPO por GP com as demais condições.

Lê só os `resumo.json` das reavaliações (5 sementes de treino) e as referências; não treina nem avalia nada.
Linhas: bot instrutor, RL puro, RL + HPO (TPE, busca aleatória, GP), IL + RL puro, IL + RL + HPO (os 5 otimizadores).
Em cada família com mais de uma execução, a que representa o otimizador é a de maior validação-objetivo (teste só reportado).
O teste t pareado por semente de treino usa a média bruta dos 3 regimes de cada semente.

Uso:  python -X utf8 execucao/comparar_gp_zero.py   ->  results/rodada3/controle_zero_gp/comparacao.json
"""
import os
import sys
import json
import time

import numpy as np
from scipy import stats

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)                                      # este script fica em <repositório>/execucao
EXPERIMENTOS = os.path.join(os.path.dirname(REPO), 'experimentos')  # pasta irmã do repositório (a mesma convenção de hpo/*.py)
sys.path.insert(0, REPO)
import protocolo as P  # noqa: E402

R3 = P.DIR_RODADA
REGIMES = ('in_dist', 'dry', 'humid')
SAIDA = os.path.join(R3, 'controle_zero_gp')
sys.stdout.reconfigure(encoding='utf-8')


def ler(*caminho):
    with open(os.path.join(R3, *caminho), encoding='utf-8') as f:
        return json.load(f)


def candidatos(pasta):
    d = ler(pasta, 'resumo.json')
    return {k: v for k, v in d.items() if isinstance(v, dict) and 'teste' in v}


def media3(bloco, campo='iqm'):
    return float(np.mean([bloco[g][campo] for g in REGIMES]))


def por_semente(c):
    return np.mean([c['teste'][g]['por_semente'] for g in REGIMES], axis=0)


def linha(rotulo, nome, c):
    return {'condicao': rotulo, 'candidato': nome, **{g: c['teste'][g]['iqm'] for g in REGIMES},
            'media': media3(c['teste']), 'validacao': media3(c['validacao_objetivo']),
            'por_semente': [float(x) for x in por_semente(c)]}


def melhor_em_validacao(cands, prefixo):
    do = {k: v for k, v in cands.items() if k.startswith(prefixo)}
    nome = max(do, key=lambda k: media3(do[k]['validacao_objetivo']))
    return nome, do[nome], do


def pareado(a, b):
    """a − b por semente de treino (recompensa bruta, média dos 3 regimes)."""
    d = np.asarray(a['por_semente']) - np.asarray(b['por_semente'])
    t, p = stats.ttest_rel(a['por_semente'], b['por_semente'])
    return {'diferenca_iqm': a['media'] - b['media'], 'diferenca_media_por_semente': float(d.mean()),
            'vence_em_sementes': int((d > 0).sum()), 'n': int(len(d)), 't': float(t), 'p': float(p)}


def professor(refs):
    t = refs['oraculo_perfeito']['teste']
    campo = 'iqm' if 'iqm' in t[REGIMES[0]] else 'media'
    p = {g: float(t[g][campo]) for g in REGIMES}
    p['media'] = float(np.mean([p[g] for g in REGIMES]))
    return p


def main():
    base = candidatos('hpo_reavaliacao')
    zero = candidatos('hpo_reavaliacao_zero')
    zgp = candidatos('hpo_reavaliacao_zero_gp')
    fontes_il = {'TPE': ('hpo_reavaliacao', 'tpe_r3'), 'Busca aleatória': ('hpo_reavaliacao', 'rs_r3'),
                 'DEHB': ('hpo_reavaliacao_dehb', 'dehb_r3'), 'GP': ('hpo_reavaliacao_gp', 'gp_r3'),
                 'Sobol': ('hpo_reavaliacao_qmc', 'qmc_r3')}

    L = {'rl_puro': linha('RL puro', 'padrao_ppo_zero', base['padrao_ppo_zero'])}
    for rot, pref in (('TPE', 'tpe_zero'), ('Busca aleatória', 'rs_zero')):
        n, c, _ = melhor_em_validacao(zero, pref)
        L[f'rl_hpo_{pref}'] = linha(f'RL + HPO ({rot})', n, c)
    n, c, todos_gp = melhor_em_validacao(zgp, 'gp_zero')
    L['rl_hpo_gp_zero'] = linha('RL + HPO (GP)', n, c)
    execucoes_gp = [linha('RL + HPO (GP)', k, v) for k, v in sorted(todos_gp.items())]
    L['il_rl_puro'] = linha('IL + RL puro', 'padrao_ppo_il', base['padrao_ppo_il'])
    for rot, (pasta, pref) in fontes_il.items():
        n, c, _ = melhor_em_validacao(candidatos(pasta), pref)
        L[f'il_rl_hpo_{pref}'] = linha(f'IL + RL + HPO ({rot})', n, c)

    comp = {'GP do zero − RL puro': pareado(L['rl_hpo_gp_zero'], L['rl_puro']),
            'GP do zero − TPE do zero': pareado(L['rl_hpo_gp_zero'], L['rl_hpo_tpe_zero']),
            'TPE do zero − RL puro': pareado(L['rl_hpo_tpe_zero'], L['rl_puro']),
            'IL + RL com GP − IL + RL puro': pareado(L['il_rl_hpo_gp_r3'], L['il_rl_puro']),
            'IL + RL com GP − GP do zero': pareado(L['il_rl_hpo_gp_r3'], L['rl_hpo_gp_zero'])}
    saida = {'gerado_em': time.strftime('%Y-%m-%d %H:%M:%S'),
             'escala': 'recompensa bruta (IQM entre as 5 sementes de treino), conjunto de teste 999-1048',
             'selecao': 'por família, a execução de maior validação-objetivo (520-549); teste só reportado',
             'professor': professor(ler('referencias', 'referencias.json')),
             'condicoes': L, 'execucoes_gp_zero': execucoes_gp, 'comparacoes_pareadas': comp}
    os.makedirs(SAIDA, exist_ok=True)
    with open(os.path.join(SAIDA, 'comparacao.json'), 'w', encoding='utf-8') as f:
        json.dump(saida, f, ensure_ascii=False, indent=1)

    print(f"{'condição':34s} {'normal':>8s} {'seco':>8s} {'úmido':>8s} {'média':>8s}  candidato")
    p = saida['professor']
    print(f"{'Bot instrutor':34s} {p['in_dist']:8.0f} {p['dry']:8.0f} {p['humid']:8.0f} {p['media']:8.0f}")
    for v in L.values():
        print(f"{v['condicao']:34s} {v['in_dist']:8.0f} {v['dry']:8.0f} {v['humid']:8.0f} {v['media']:8.0f}  {v['candidato']}")
    print()
    for k, v in comp.items():
        print(f"{k:34s} {v['diferenca_iqm']:+8.0f}  vence em {v['vence_em_sementes']}/{v['n']} sementes  p = {v['p']:.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
