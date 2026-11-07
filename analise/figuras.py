"""analise/figuras.py — figuras da dissertação a partir dos JSON de resultados (nunca à mão).

Uso:  python -X utf8 analise/figuras.py [--saida ../dissertacao] [--bakeoff results/bakeoff] [--warmstart results/il_warmstart_v2]
Gera: fig_bakeoff_curvas.png (curvas em validação-rollback por braço e semente, linha do agente de partida).
"""
import os
import sys
import json
import glob
import argparse

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import protocolo as P  # noqa: E402

import matplotlib      # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

NOMES = {'ppo_il': '(A) PPO a partir da imitação', 'elite': '(B) Auto-imitação de elite', 'ppo_zero': '(C) PPO do zero'}


def fig_bakeoff_curvas(dir_bakeoff, saida):
    bracos = [b for b in ('ppo_il', 'elite', 'ppo_zero') if os.path.isdir(os.path.join(dir_bakeoff, b))]
    fig, eixos = plt.subplots(1, len(bracos), figsize=(4.2 * len(bracos), 3.6), sharey=True)
    if len(bracos) == 1:
        eixos = [eixos]
    # nível do agente de partida por semente (registrado nos braços que partem da imitação); desenhado em todos os painéis
    niveis = {}
    for f in glob.glob(os.path.join(dir_bakeoff, '*', 'seed_*', 'metricas.json')):
        d = json.load(open(f, encoding='utf-8'))
        if d.get('nivel_il_rollback') is not None:
            niveis[d['seed']] = d['nivel_il_rollback']
    for ax, b in zip(eixos, bracos):
        for f in sorted(glob.glob(os.path.join(dir_bakeoff, b, 'seed_*', 'metricas.json'))):
            d = json.load(open(f, encoding='utf-8'))
            curva = d['curva_validacao_rollback']
            xs = [c['passos'] / 1000 for c in curva]; ys = [c['media'] for c in curva]
            ax.plot(xs, ys, marker='o', ms=3, lw=1.2, label=f"semente {d['seed']}")
        for seed, nivel in niveis.items():
            ax.axhline(nivel, color='gray', ls='--', lw=0.8)
        ax.set_title(NOMES.get(b, b), fontsize=10)
        ax.set_xlabel('passos de ambiente (mil)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    eixos[0].set_ylabel('recompensa em validação de seleção')
    fig.tight_layout()
    caminho = os.path.join(saida, 'fig_bakeoff_curvas.png')
    fig.savefig(caminho, dpi=160)
    plt.close(fig)
    return caminho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--saida', default=os.path.join(os.path.dirname(RAIZ), 'dissertacao'))
    ap.add_argument('--bakeoff', default=os.path.join(RAIZ, 'results', 'bakeoff'))
    args = ap.parse_args()
    os.makedirs(args.saida, exist_ok=True)
    print('[ok]', fig_bakeoff_curvas(args.bakeoff, args.saida))
    return 0


if __name__ == '__main__':
    sys.exit(main())
