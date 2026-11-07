"""Curvas de validação do RL sem HPO (ppo_il, ppo_zero) da rodada 3 — insumo da decisão D-35 (orçamento por execução).

Lê results/rodada3/rl_padrao/<braço>/seed_<s>/metricas.json (execuções concluídas) ou, se ainda não houver, o _STATUS.txt
(linhas `bloco_<passos> | val=<média>`), então funciona COM A RODADA EM ANDAMENTO. As médias são o retorno bruto em
validação-rollback (500–519, in-dist, política amostrada) — o mesmo critério de seleção de checkpoint do bake-off.
Se referencias.json existir, também normaliza (0 = aleatório, 1 = professor v5, conjunto validacao_rollback, in_dist).

Diagnóstico do orçamento do reforço (README, Passo 2): se as curvas ainda sobem no último bloco, o orçamento é curto; se achatam cedo, dá
para reduzir. Por execução: bloco do melhor valor, se o melhor é o último, e ganho do último terço (em % do ganho total).
Saídas em ../analise_rl_padrao_rodada3/: curvas_rl_padrao.png/.pdf e diagnostico.json. Não escreve nada no repositório.
Uso:  python -X utf8 execucao/curvas_rl_padrao.py
"""
import os
import re
import sys
import glob
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)                                      # este script fica em <repositório>/execucao
EXPERIMENTOS = os.path.join(os.path.dirname(REPO), 'experimentos')  # pasta irmã do repositório (a mesma convenção de hpo/*.py)
sys.path.insert(0, REPO)
import protocolo as P  # noqa: E402

RLP = os.path.join(P.DIR_RODADA, 'rl_padrao')
SAIDA = os.path.join(EXPERIMENTOS, 'analise_rl_padrao_rodada3')
CORES = {'ppo_il': '#0072B2', 'ppo_zero': '#D55E00'}


def iqm(v):
    v = np.sort(np.asarray(v, float))
    k = int(np.floor(0.25 * len(v)))
    return float(v[k:len(v) - k].mean()) if len(v) - 2 * k > 0 else float(v.mean())


def ler_run(pasta):
    """(pontos [(passos, media)], concluido, nivel_il)"""
    m = os.path.join(pasta, 'metricas.json')
    if os.path.exists(m):
        d = json.load(open(m, encoding='utf-8'))
        return [(p['passos'], p['media']) for p in d['curva_validacao_rollback']], True, d.get('nivel_il_rollback')
    pts, nivel = [], None
    st = os.path.join(pasta, '_STATUS.txt')
    if os.path.exists(st):
        for linha in open(st, encoding='utf-8'):
            mm = re.search(r'\|\s*bloco_(\d+)\s*\|\s*val=([-\d.]+)', linha)
            if mm:
                pts.append((int(mm.group(1)), float(mm.group(2))))
    return pts, False, nivel


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.makedirs(SAIDA, exist_ok=True)
    refs = None
    rp = os.path.join(P.DIR_REFERENCIAS, 'referencias.json')
    if os.path.exists(rp):
        r = json.load(open(rp, encoding='utf-8'))
        a, o = r['aleatorio']['validacao_rollback']['in_dist']['media'], r['oraculo_perfeito']['validacao_rollback']['in_dist']['media']
        refs = (a, o)
    norm = (lambda x: (x - refs[0]) / (refs[1] - refs[0])) if refs else (lambda x: x)

    diag, fig, axs = {}, *plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, braco in zip(axs, ['ppo_il', 'ppo_zero']):
        runs = {}
        for pasta in sorted(glob.glob(os.path.join(RLP, braco, 'seed_*'))):
            pts, fim, nivel = ler_run(pasta)
            if pts:
                runs[int(os.path.basename(pasta)[5:])] = (pts, fim)
        diag[braco] = {}
        for s, (pts, fim) in runs.items():
            x = np.array([p for p, _ in pts]) / 1e6
            y = np.array([norm(v) for _, v in pts])
            ax.plot(x, y, color=CORES[braco], alpha=0.35, lw=1, marker='o', ms=2.5)
            blocos = [(p, v) for p, v in pts if p > 0]
            if blocos:
                vals = [v for _, v in pts]
                i_best = int(np.argmax(vals))
                ini, fim_v = vals[0], vals[-1]
                terco = max(1, len(vals) // 3)
                ganho_total = max(vals) - ini
                diag[braco][s] = {
                    'concluido': fim, 'n_blocos': len(blocos), 'passos_ultimo': pts[-1][0],
                    'melhor_passos': pts[i_best][0], 'melhor_e_ultimo': i_best == len(vals) - 1,
                    'melhor_valor': max(vals), 'inicio': ini, 'ultimo': fim_v,
                    'ganho_ultimo_terco_pct_do_total': (100.0 * (fim_v - vals[-1 - terco]) / ganho_total) if ganho_total > 0 else None,
                }
        # IQM entre sementes nos passos comuns
        comuns = sorted(set.intersection(*[set(p for p, _ in pts) for pts, _ in runs.values()])) if runs else []
        if len(runs) >= 2 and comuns:
            mat = np.array([[dict(pts)[p] for p in comuns] for pts, _ in runs.values()])
            ax.plot(np.array(comuns) / 1e6, [norm(iqm(mat[:, j])) for j in range(len(comuns))], color=CORES[braco], lw=2.5,
                    label=f'{braco}: IQM de {len(runs)} sementes')
        if refs:
            ax.axhline(1.0, color='k', ls='--', lw=1, label='professor v5 (1)')
            ax.axhline(0.0, color='gray', ls=':', lw=1, label='aleatório (0)')
        ax.set_title(f'{braco} — {sum(1 for _, f in runs.values() if f)}/{len(runs)} concluídas')
        ax.set_xlabel('passos de ambiente (milhões)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
    axs[0].set_ylabel('retorno normalizado, validação-rollback in-dist' if refs else 'retorno bruto, validação-rollback in-dist')
    fig.suptitle(f'RL sem HPO, rodada 3 (orçamento {P.ORCAMENTO_RL_PASSOS:,} passos)'.replace(',', '.'))
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(SAIDA, f'curvas_rl_padrao.{ext}'), dpi=150)

    todos = [d for b in diag.values() for d in b.values() if d['n_blocos'] > 0]
    resumo = {'n_execucoes_com_blocos': len(todos),
              'melhor_e_ultimo': sum(d['melhor_e_ultimo'] for d in todos),
              'normalizado': bool(refs), 'referencias_rollback_in_dist': refs, 'por_braco': diag}
    json.dump(resumo, open(os.path.join(SAIDA, 'diagnostico.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f"figura e diagnóstico em {SAIDA}")
    for braco, runs in diag.items():
        for s, d in sorted(runs.items()):
            g = d['ganho_ultimo_terco_pct_do_total']
            print(f"  {braco:8s} seed {s:5d} | {'fim' if d['concluido'] else 'andando'} | blocos {d['n_blocos']} | "
                  f"melhor @ {d['melhor_passos']:>9,} {'(ÚLTIMO)' if d['melhor_e_ultimo'] else '        '} | "
                  f"ganho no último terço: {'-' if g is None else f'{g:5.1f}%'}".replace(',', '.'))
    print(f"melhor bloco = último em {resumo['melhor_e_ultimo']} de {len(todos)} execuções (rodada 2: 12 de 15 → orçamento curto)")


if __name__ == '__main__':
    main()
