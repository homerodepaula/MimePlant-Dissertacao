"""hpo/analise_hpo.py — análise dos estudos de HPO e da reavaliação (Fase 4.8): curva melhor-até-agora × FRE, importância
dos hiperparâmetros, tabela dos melhores trials (tuning × teste), comparação final com IQM + IC 95 % e testes pareados.

Uso:  python -X utf8 hpo/analise_hpo.py [--studies tpe_r3 rs_r3] [--reavaliacao ...] [--saida ../experimentos/analise_hpo_rodada3]   (padrões: protocolo.py)
Saída: <saida>/analise_hpo.json (todos os números, com origem), <saida>/tab_*.tex (tabelas LaTeX), <saida>/fig_*.png (figuras).
"""
import os
import sys
import json
import argparse

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import protocolo as P                                  # noqa: E402

P.preparar_sys_path()
import optuna                                          # noqa: E402
import matplotlib                                      # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                        # noqa: E402

VF = os.path.dirname(RAIZ)
DIR_HPO = os.path.join(VF, 'experimentos', 'hpo')


def carregar(nome):
    return optuna.load_study(study_name=nome, storage=f"sqlite:///{os.path.join(DIR_HPO, nome + '.db').replace(os.sep, '/')}")


def curva_melhor_ate_agora(study):
    """Pontos (FRE acumulados, melhor valor até então) na ordem de conclusão dos trials (podados consomem FRE sem valor)."""
    ts = sorted([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)],
                key=lambda t: t.datetime_complete or t.datetime_start)
    fre, melhor, xs, ys = 0.0, -np.inf, [], []
    for t in ts:
        fre += float(t.user_attrs.get('fre', 0.0))
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
            melhor = max(melhor, t.value)
        xs.append(fre); ys.append(melhor if np.isfinite(melhor) else np.nan)
    return xs, ys


def importancia(study):
    """Importância dos hiperparâmetros, com o MÉTODO declarado na saída (07/09/2026).

    Antes, o `except Exception` mudo escondia que o fANOVA nunca rodava: `optuna.importance._fanova` importa
    `sklearn.ensemble.RandomForestRegressor`, o interpretador dos jobs não tinha scikit-learn, e a função caía
    silenciosamente no PED-ANOVA — que é outro método (razão de densidades entre trials bons e todos, não
    decomposição de variância). O JSON saía só com `{nome: valor}`, sem dizer qual avaliador produziu os números,
    e a legenda da Tabela 5.1 da dissertação afirma "(fANOVA)". Agora o método vai junto e o fallback é anunciado.

    A lista de parâmetros é explícita: com `params=None` o Optuna usa a INTERSEÇÃO dos espaços dos trials, o que
    descartava em silêncio qualquer hiperparâmetro condicional. Desde a D-30 o espaço não é mais condicional, mas
    passar a lista trava a propriedade em vez de depender dela.
    """
    conc = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(conc) < 5:
        return {'metodo': None, 'n_concluidos': len(conc),
                'aviso': 'menos de 5 trials concluídos: importância não calculada', 'valores': {}}
    nomes = sorted({p for t in conc for p in t.params})
    for metodo, construir in (('fanova', lambda: optuna.importance.FanovaImportanceEvaluator(seed=0)),
                              ('ped_anova', optuna.importance.PedAnovaImportanceEvaluator)):
        try:
            vals = optuna.importance.get_param_importances(study, evaluator=construir(), params=nomes)
            return {'metodo': metodo, 'n_concluidos': len(conc), 'parametros_avaliados': nomes,
                    'valores': {k: float(v) for k, v in vals.items()}}
        except Exception as e:
            ultimo = f'{type(e).__name__}: {e}'
            print(f'  [importancia] {metodo} indisponível ({ultimo}); tentando o próximo', flush=True)
    return {'metodo': None, 'n_concluidos': len(conc), 'erro': ultimo, 'valores': {}}


def fmt(x, casas=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '---'
    s = f'{x:,.{casas}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--studies', nargs='+', default=list(P.STUDIES_HPO))
    ap.add_argument('--reavaliacao', default=P.DIR_REAVALIACAO)
    ap.add_argument('--saida', default=os.path.join(VF, 'experimentos', 'analise_hpo_' + P.RODADA))
    args = ap.parse_args()
    os.makedirs(args.saida, exist_ok=True)
    out = {'studies': {}, 'origem': {'hpo': DIR_HPO, 'reavaliacao': args.reavaliacao}}

    # ── estudos: curva, contagens, melhor, importância ──
    fig, ax = plt.subplots(figsize=(7, 4))
    for nome in args.studies:
        try:
            st = carregar(nome)
        except Exception as e:
            print(f'[aviso] study {nome} não carregado: {e}'); continue
        conc = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pod = [t for t in st.trials if t.state == optuna.trial.TrialState.PRUNED]
        xs, ys = curva_melhor_ate_agora(st)
        melhor = max(conc, key=lambda t: t.value) if conc else None
        top = sorted(conc, key=lambda t: -t.value)[:5]
        out['studies'][nome] = {
            'sampler': st.user_attrs.get('sampler'), 'poda': st.user_attrs.get('poda'), 'n_concluidos': len(conc), 'n_podados': len(pod),
            'fre_consumidos': xs[-1] if xs else 0.0, 'melhor': ({'numero': melhor.number, 'valor': melhor.value, 'params': melhor.params} if melhor else None),
            'top5': [{'numero': t.number, 'valor': t.value, 'params': t.params} for t in top],
            'curva_fre': xs, 'curva_melhor': ys, 'importancia': importancia(st),
            'fre_ate_superar': {},
        }
        ax.step(xs, ys, where='post', label=f'{nome} ({len(conc)} concl., {len(pod)} podados)')
        print(f"{nome}: {len(conc)} concluídos, {len(pod)} podados, {xs[-1] if xs else 0:.1f} FRE | melhor {melhor.value if melhor else None} | importância {out['studies'][nome]['importancia']}")
    ax.set_xlabel('Orçamento consumido (FRE)'); ax.set_ylabel('Melhor objetivo até então (recompensa normalizada, validação)')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.saida, 'fig_hpo_curvas.png'), dpi=150); plt.close(fig)

    # ── reavaliação em teste ──
    caminho_res = os.path.join(args.reavaliacao, 'resumo.json')
    if os.path.exists(caminho_res):
        res = json.load(open(caminho_res, encoding='utf-8'))
        out['reavaliacao'] = res
        linhas = []
        for nome, r in res.items():
            t = r['teste']
            nome_tex = nome.replace('_', '\\_')
            tuning = fmt(r.get('valor_tuning'), 3) if r.get('valor_tuning') is not None else '---'
            celulas = [f"{fmt(t[reg]['iqm'])} [{fmt(t[reg]['ic95'][0])}, {fmt(t[reg]['ic95'][1])}]" for reg in P.NOMES_REGIMES]
            norm = fmt(t['normalizada_media_3_regimes']['media'], 3)
            linhas.append(nome_tex + ' & ' + tuning + ' & ' + ' & '.join(celulas) + ' & ' + norm + ' \\\\')
        tab = ('\\begin{tabular}{l c c c c c}\n\\hline\n\\textbf{Candidato} & \\textbf{Tuning} & \\textbf{In-dist} & \\textbf{Dry} & \\textbf{Humid} & \\textbf{Norm.} \\\\ \\hline\n'
               + '\n'.join(linhas) + '\n\\hline\n\\end{tabular}\n')
        with open(os.path.join(args.saida, 'tab_reavaliacao_teste.tex'), 'w', encoding='utf-8') as f:
            f.write(tab)
        # testes pareados por semente entre o melhor candidato e cada padrão (normalizada média, teste)
        try:
            from scipy import stats
            nomes = list(res)
            # A escolha do "melhor candidato" tem de sair de VALIDACAO-OBJETIVO (520-549). Ate 06/09 ela usava
            # `res[n]['teste']`, ou seja, selecionava nas sementes de teste (999-1048) e depois fazia os testes
            # pareados sobre essa escolha -- selecao no conjunto de teste, o que o protocolo proibe (auditoria de 06/09).
            # Mesma função que escolhe a base das ablações (08/09/2026): a configuração ablada e a da comparação final
            # são, por construção, a mesma.
            from reavaliar_topk import escolher_melhor_candidato
            melhor_nome = escolher_melhor_candidato(res)
            comp = {}
            if melhor_nome:
                a = np.asarray(res[melhor_nome]['teste']['normalizada_media_3_regimes']['por_semente'])
                for n in nomes:
                    if n == melhor_nome:
                        continue
                    b = np.asarray(res[n]['teste']['normalizada_media_3_regimes']['por_semente'])
                    if len(a) == len(b) and len(a) >= 3:
                        d = a - b
                        tt = stats.ttest_rel(a, b)
                        comp[n] = {'diferenca_media': float(d.mean()), 't': float(tt.statistic), 'p': float(tt.pvalue),
                                   'd_cohen_pareado': float(d.mean() / (d.std(ddof=1) + 1e-12)), 'prob_melhoria': float(np.mean([x > y for x in a for y in b]))}
            out['testes_pareados'] = {'melhor': melhor_nome, 'escolhido_em': 'validacao_objetivo (520-549)',
                                      'reportado_em': 'teste (999-1048)', 'contra': comp}
        except Exception as e:
            out['testes_pareados'] = {'erro': str(e)}

    with open(os.path.join(args.saida, 'analise_hpo.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)
    print('[ok]', os.path.join(args.saida, 'analise_hpo.json'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
