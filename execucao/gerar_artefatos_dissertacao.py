"""Gera as TABELAS (.tex) e FIGURAS (.pdf) da dissertação a partir dos JSON de resultados da rodada 3.

Regra do projeto: nenhum número do texto é digitado à mão. Tudo aqui sai de `results/rodada3/**` e de
`experimentos/hpo/**`, em **retorno bruto** (decisão de 24/09), no conjunto de teste (sementes 999–1048), com IQM.
(Versão anterior do gerador: o atual é `gerar_artefatos_final.py`, que importa daqui só as funções auxiliares.)

Saídas em `Texto/dissertacao/`:
  tabelas/tab_referencias.tex      professor, variante, aleatório e clone IL nos 3 regimes
  tabelas/tab_rl_sem_hpo.tex       clone, ppo_il e ppo_zero contra o professor
  tabelas/tab_otimizadores.tex     os 15 incumbentes (5 otimizadores × 3 execuções)
  tabelas/tab_agregado.tex         agregado por otimizador + teste estatístico
  tabelas/tab_quadrantes.tex       tabela 2×2 do controle D-32
  tabelas/tab_ablacoes.tex         ablações sobre o MimePlant
  tabelas/tab_custos.tex           custo computacional por camada
  fig_curvas_rl.pdf                curvas de validação por bloco (ppo_il × ppo_zero)
  fig_otimizadores.pdf             dispersão das 3 execuções de cada otimizador
  fig_melhor_ate_agora.pdf         melhor-até-agora × FRE, por otimizador
  fig_quadrantes.pdf               o 2×2 em barras
  numeros.json                     todos os números citados no texto, para conferência

Uso:  python -X utf8 execucao/gerar_artefatos_dissertacao.py
"""
import os
import re
import sys
import glob
import json
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats          # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)                                      # este script fica em <repositório>/execucao
EXPERIMENTOS = os.path.join(os.path.dirname(REPO), 'experimentos')  # pasta irmã do repositório (a mesma convenção de hpo/*.py)
R3 = os.path.join(REPO, 'results', 'rodada3')
DIR_HPO = os.path.join(EXPERIMENTOS, 'hpo')
SAIDA = os.path.join(os.path.dirname(REPO), 'dissertacao')
TAB = os.path.join(SAIDA, 'tabelas')
REGIMES = ('in_dist', 'dry', 'humid')
NOMES_REGIME = {'in_dist': 'Normal', 'dry': 'Seco', 'humid': 'Úmido'}
CORES = {'TPE': '#0072B2', 'RS': '#E69F00', 'DEHB': '#009E73', 'GP': '#D55E00', 'QMC': '#CC79A7'}
ROTULO = {'TPE': 'TPE+Hyperband', 'RS': 'Busca aleatória', 'DEHB': 'DEHB', 'GP': 'GP-BO', 'QMC': 'Sobol/QMC'}

plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.bbox': 'tight', 'figure.dpi': 150})


def br(x, casas=1):
    """Número no padrão brasileiro: milhar com ponto, decimal com vírgula."""
    if x is None:
        return '---'
    s = f'{x:,.{casas}f}'
    return s.replace(',', '#').replace('.', ',').replace('#', '.')


def ler(caminho):
    with open(caminho, encoding='utf-8') as f:
        return json.load(f)


def media3(d, campo='iqm'):
    return sum(d[g][campo] for g in REGIMES) / 3


def escrever(nome, linhas, legenda, rotulo, colunas, nota=None, tam=None, tabcol='4pt', curta=None):
    """Grava uma tabela. `curta` é a legenda da Lista de Tabelas (o `\\caption[curta]{legenda}` do LaTeX)."""
    os.makedirs(TAB, exist_ok=True)
    corpo = '\n'.join(linhas)
    # Pedido do autor (19/10/2026): o nome da tabela é só o título curto; a explicação vai para a nota abaixo da tabela.
    if curta:
        explic = legenda.strip()
        # tira um título inicial redundante ("HPO no IL + RL, por otimizador:" / "As cinco condições comparadas.")
        m = re.match(r'^([^:.]{3,80}):\s+', explic) or re.match(r'^([^.]{3,60})\.\s+', explic)
        if m:
            palavras = {w.lower() for w in re.findall(r'[A-Za-zÀ-ÿ]{4,}', m.group(1))}
            comuns = {w.lower() for w in re.findall(r'[A-Za-zÀ-ÿ]{4,}', curta)}
            if palavras and len(palavras & comuns) / len(palavras) >= 0.5:
                explic = explic[m.end():]
        explic = explic[:1].upper() + explic[1:]
        nota = 'Fonte: elaborada pelo autor. Nota: ' + explic + ('' if nota is None else ' ' + nota)
        cap = f'\\caption{{{curta}}}'
    else:
        cap = f'\\caption{{{legenda}}}'
    tex = f"""% GERADO POR gerar_artefatos_final.py (via gerar_artefatos_dissertacao.escrever) — não editar à mão
\\begin{{table}}[htbp]
\\centering
{(tam + chr(92) + 'setlength{' + chr(92) + 'tabcolsep}{' + tabcol + '}') if tam else ''}
{cap}
\\label{{{rotulo}}}
\\begin{{tabular}}{{{colunas}}}
\\hline
{corpo}
\\hline
\\end{{tabular}}

{{\\footnotesize {nota or 'Fonte: elaborada pelo autor.'}}}
\\end{{table}}
"""
    with open(os.path.join(TAB, nome), 'w', encoding='utf-8') as f:
        f.write(tex)
    return tex


def coletar_incumbentes():
    """(otimizador, candidato, {regime: iqm}, média, valor de ajuste) para os 15 incumbentes do método.

    `media_val` é a média dos 3 regimes do IQM bruto em validação-objetivo; `val_norm` é a média, entre as sementes, da
    recompensa normalizada (0 = aleatório, 1 = professor), que é a regra de escolha do código (`escolher_melhor_candidato`)."""
    fontes = [('hpo_reavaliacao', None), ('hpo_reavaliacao_dehb', 'DEHB'),
              ('hpo_reavaliacao_gp', 'GP'), ('hpo_reavaliacao_qmc', 'QMC')]
    out = []
    for pasta, rot in fontes:
        arq = os.path.join(R3, pasta, 'resumo.json')
        if not os.path.exists(arq):
            continue
        for k, v in ler(arq).items():
            if not (isinstance(v, dict) and 'teste' in v) or k.startswith('padrao'):
                continue
            o = rot or ('TPE' if k.startswith('tpe') else 'RS')
            out.append({'otim': o, 'cand': k, 'teste': {g: v['teste'][g]['iqm'] for g in REGIMES},
                        'media_teste': media3(v['teste']), 'media_val': media3(v['validacao_objetivo']),
                        'val_norm': v['validacao_objetivo']['normalizada_media_3_regimes']['media'],
                        'ajuste': v.get('valor_tuning'),
                        'ic': {g: v['teste'][g].get('ic95') for g in REGIMES}})
    return out


def padroes():
    r = ler(os.path.join(R3, 'hpo_reavaliacao', 'resumo.json'))
    return {k: {'teste': {g: r[k]['teste'][g]['iqm'] for g in REGIMES}, 'media': media3(r[k]['teste']),
                'ic': {g: r[k]['teste'][g].get('ic95') for g in REGIMES}}
            for k in ('padrao_ppo_il', 'padrao_ppo_zero') if k in r}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.makedirs(TAB, exist_ok=True)
    N = {}

    # ── referências e clone ─────────────────────────────────────────────────────────────────────
    ref = ler(os.path.join(R3, 'referencias', 'referencias.json'))
    clone = ler(os.path.join(R3, 'il_warmstart', 'resumo.json'))['tabela']['teste']
    N['professor'] = {g: ref['oraculo_perfeito']['teste'][g]['iqm'] for g in REGIMES}
    N['professor_media'] = media3(ref['oraculo_perfeito']['teste'])
    N['aleatorio'] = {g: ref['aleatorio']['teste'][g]['iqm'] for g in REGIMES}
    N['aleatorio_media'] = media3(ref['aleatorio']['teste'])
    N['variante'] = {g: ref['oraculo_subotimo']['teste'][g]['iqm'] for g in REGIMES}
    N['clone'] = {g: clone[g]['media_entre_sementes'] for g in REGIMES}
    N['clone_std'] = {g: clone[g]['std_entre_sementes'] for g in REGIMES}
    N['clone_media'] = sum(N['clone'].values()) / 3

    linhas = ['\\textbf{Política} & \\textbf{Normal} & \\textbf{Seco} & \\textbf{Úmido} & \\textbf{Média} \\\\ \\hline']
    for rot, vals, m in (('Professor (\\textit{BotIntensivo})', N['professor'], N['professor_media']),
                         ('Professor com pausas ($p=0{,}14$)', N['variante'], sum(N['variante'].values()) / 3),
                         ('Clone por imitação (BC + DAgger)', N['clone'], N['clone_media']),
                         ('Agente aleatório', N['aleatorio'], N['aleatorio_media'])):
        linhas.append(' & '.join([rot] + [br(vals[g], 0) for g in REGIMES] + [br(m, 0)]) + ' \\\\')
    escrever('tab_referencias.tex', linhas,
             'Recompensa bruta das políticas de referência e do clone por imitação, no conjunto de teste '
             '(sementes 999--1048, 50 episódios por regime; o clone é a média entre as 5 sementes de treino).',
             'tab:referencias', 'l c c c c')

    # ── RL sem HPO ──────────────────────────────────────────────────────────────────────────────
    pad = padroes()
    N['ppo_il'] = pad['padrao_ppo_il']['teste']
    N['ppo_il_media'] = pad['padrao_ppo_il']['media']
    N['ppo_zero'] = pad['padrao_ppo_zero']['teste']
    N['ppo_zero_media'] = pad['padrao_ppo_zero']['media']
    linhas = ['\\textbf{Agente} & \\textbf{Normal} & \\textbf{Seco} & \\textbf{Úmido} & \\textbf{Média} \\\\ \\hline']
    for rot, vals, m in (('Clone por imitação (sem RL)', N['clone'], N['clone_media']),
                         ('IL + RL, configuração padrão', N['ppo_il'], N['ppo_il_media']),
                         ('RL do zero, configuração padrão', N['ppo_zero'], N['ppo_zero_media']),
                         ('Professor', N['professor'], N['professor_media'])):
        linhas.append(' & '.join([rot] + [br(vals[g], 0) for g in REGIMES] + [br(m, 0)]) + ' \\\\')
    escrever('tab_rl_sem_hpo.tex', linhas,
             'Recompensa bruta (IQM) do clone e das duas configurações padrão de RL, no conjunto de teste, '
             'média de 5 sementes de treino.', 'tab:rl_sem_hpo', 'l c c c c')

    # ── incumbentes ─────────────────────────────────────────────────────────────────────────────
    inc = coletar_incumbentes()
    N['n_incumbentes'] = len(inc)
    linhas = ['\\textbf{Otimizador} & \\textbf{Execução} & \\textbf{Normal} & \\textbf{Seco} & \\textbf{Úmido} & \\textbf{Média} \\\\ \\hline']
    for d in sorted(inc, key=lambda x: -x['media_teste']):
        linhas.append(' & '.join([ROTULO[d['otim']], '\\texttt{' + d['cand'].replace('_', '\\_') + '}']
                                 + [br(d['teste'][g], 0) for g in REGIMES] + ['\\textbf{' + br(d['media_teste'], 0) + '}']) + ' \\\\')
    escrever('tab_otimizadores.tex', linhas,
             'Incumbente de cada execução de HPO, reavaliado com as 5 sementes de treino. Recompensa bruta (IQM) '
             'no conjunto de teste.', 'tab:otimizadores', 'l l c c c c')

    # ── agregado + teste estatístico ────────────────────────────────────────────────────────────
    por = {}
    for d in inc:
        por.setdefault(d['otim'], []).append(d['media_teste'])
    grupos = [np.array(por[o]) for o in por]
    F, p_anova = stats.f_oneway(*grupos)
    H, p_kw = stats.kruskal(*grupos)
    todos = np.concatenate(grupos)
    ss_ent = sum(len(g) * (g.mean() - todos.mean()) ** 2 for g in grupos)
    eta2 = ss_ent / ((todos - todos.mean()) ** 2).sum()
    N['anova'] = {'F': float(F), 'p': float(p_anova)}
    N['kruskal'] = {'H': float(H), 'p': float(p_kw)}
    N['eta2'] = float(eta2)
    N['desvio_15'] = float(todos.std(ddof=1))
    N['amplitude_15'] = float(todos.max() - todos.min())
    N['amplitude_medias'] = float(max(g.mean() for g in grupos) - min(g.mean() for g in grupos))
    N['por_otimizador'] = {o: {'media': float(np.mean(v)), 'mediana': float(np.median(v)), 'melhor': float(max(v)),
                               'pior': float(min(v)), 'desvio': float(np.std(v, ddof=1))} for o, v in por.items()}
    linhas = ['\\textbf{Otimizador} & \\textbf{Média} & \\textbf{Mediana} & \\textbf{Melhor} & \\textbf{Pior} & \\textbf{Desvio} \\\\ \\hline']
    for o, v in sorted(por.items(), key=lambda kv: -np.mean(kv[1])):
        s = N['por_otimizador'][o]
        linhas.append(' & '.join([ROTULO[o], br(s['media']), br(s['mediana']), br(s['melhor']), br(s['pior']), br(s['desvio'])]) + ' \\\\')
    escrever('tab_agregado.tex', linhas,
             'Agregado por otimizador sobre as 3 execuções independentes (recompensa bruta em teste). '
             f"ANOVA de um fator: $F = {br(F, 3)}$, $p = {br(p_anova, 3)}$; Kruskal--Wallis: $H = {br(H, 3)}$, "
             f"$p = {br(p_kw, 3)}$; $\\eta^2 = {br(eta2, 3)}$.", 'tab:agregado', 'l c c c c c')

    # ── 2×2 do controle ─────────────────────────────────────────────────────────────────────────
    zero = ler(os.path.join(R3, 'hpo_reavaliacao_zero', 'resumo.json'))
    melhor_zero = max((v for k, v in zero.items() if isinstance(v, dict) and 'teste' in v and not k.startswith('padrao')),
                      key=lambda v: media3(v['validacao_objetivo']))
    mimeplant = max(inc, key=lambda d: d['media_val'])
    pre_registrado = max([d for d in inc if d['otim'] in ('TPE', 'RS')], key=lambda d: d['media_val'])
    N['mimeplant'] = {'cand': mimeplant['cand'], 'otim': mimeplant['otim'], 'teste': mimeplant['teste'],
                      'media': mimeplant['media_teste'], 'val': mimeplant['media_val']}
    N['pre_registrado'] = {'cand': pre_registrado['cand'], 'media': pre_registrado['media_teste']}
    N['zero_hpo'] = {'media': media3(melhor_zero['teste']), 'teste': {g: melhor_zero['teste'][g]['iqm'] for g in REGIMES}}
    q = {'il_hpo': pre_registrado['media_teste'], 'zero_hpo': N['zero_hpo']['media'],
         'il_padrao': N['ppo_il_media'], 'zero_padrao': N['ppo_zero_media']}
    N['quadrantes'] = q
    N['ganho_imitacao_com_hpo'] = q['il_hpo'] - q['zero_hpo']
    N['ganho_imitacao_sem_hpo'] = q['il_padrao'] - q['zero_padrao']
    N['ganho_hpo_sem_imitacao'] = q['zero_hpo'] - q['zero_padrao']
    N['ganho_conjunto'] = q['il_hpo'] - q['zero_padrao']
    linhas = ['& \\textbf{Configuração padrão} & \\textbf{Com HPO} & \\textbf{Ganho da HPO} \\\\ \\hline',
              f"\\textbf{{Com imitação}} & {br(q['il_padrao'], 0)} & \\textbf{{{br(q['il_hpo'], 0)}}} & $+{br(q['il_hpo'] - q['il_padrao'], 0)}$ \\\\",
              f"\\textbf{{Do zero}} & {br(q['zero_padrao'], 0)} & {br(q['zero_hpo'], 0)} & $+{br(N['ganho_hpo_sem_imitacao'], 0)}$ \\\\ \\hline",
              f"\\textbf{{Ganho da imitação}} & $+{br(N['ganho_imitacao_sem_hpo'], 0)}$ & $+{br(N['ganho_imitacao_com_hpo'], 0)}$ & \\\\"]
    escrever('tab_quadrantes.tex', linhas,
             'Desenho 2 $\\times$ 2 do controle (D-32): com e sem imitação, com e sem HPO. Recompensa bruta (IQM) '
             'no conjunto de teste, média dos 3 regimes, 5 sementes de treino.', 'tab:quadrantes', 'l c c c')

    # ── ablações ────────────────────────────────────────────────────────────────────────────────
    abl = ler(os.path.join(R3, 'hpo_ablacoes', 'resumo.json'))
    NOMES_ABL = {'abl_sem_kl': 'Sem regularização KL ($\\beta_{KL}=0$)',
                 'abl_zero': 'Sem imitação (parte do zero)',
                 'abl_kl_forte': 'KL forte (valores padrão)',
                 'abl_sem_temperatura': 'Sem temperatura inicial ($\\tau=1$)'}
    N['ablacoes'] = {}
    linhas = ['\\textbf{Variante} & \\textbf{Normal} & \\textbf{Seco} & \\textbf{Úmido} & \\textbf{Média} & \\textbf{$\\Delta$} \\\\ \\hline',
              ' & '.join(['\\textbf{MimePlant (base)}'] + [br(pre_registrado['teste'][g], 0) for g in REGIMES]
                         + ['\\textbf{' + br(pre_registrado['media_teste'], 0) + '}', '---']) + ' \\\\']
    for k, rot in NOMES_ABL.items():
        v = abl.get(k)
        if not (isinstance(v, dict) and 'teste' in v):
            continue
        vals = {g: v['teste'][g]['iqm'] for g in REGIMES}
        m = media3(v['teste'])
        N['ablacoes'][k] = m
        linhas.append(' & '.join([rot] + [br(vals[g], 0) for g in REGIMES] + [br(m, 0), '$' + br(m - pre_registrado['media_teste'], 0) + '$']) + ' \\\\')
    escrever('tab_ablacoes.tex', linhas,
             'Ablações sobre o MimePlant: cada variante remove um ingrediente e é retreinada com as 5 sementes. '
             'Recompensa bruta (IQM) no conjunto de teste.', 'tab:ablacoes', 'l c c c c c')

    # ── custo computacional ─────────────────────────────────────────────────────────────────────
    linhas = ['\\textbf{Camada} & \\textbf{Passos de ambiente} & \\textbf{FRE} & \\textbf{Ganho} & \\textbf{Ganho por FRE} \\\\ \\hline',
              f"Imitação (demonstrações + clone) & 200.750 & 0,20 & $+{br(N['clone_media'] - N['aleatorio_media'], 0)}$ & {br((N['clone_media'] - N['aleatorio_media']) / 0.2, 0)} \\\\",
              f"RL sem HPO (5 sementes) & 5.000.000 & 5,00 & $+{br(N['ppo_il_media'] - N['clone_media'], 0)}$ & {br((N['ppo_il_media'] - N['clone_media']) / 5, 0)} \\\\",
              f"HPO (1 execução de 32 FRE + reavaliação) & 38.000.000 & 38,00 & $+{br(q['il_hpo'] - N['ppo_il_media'], 0)}$ & {br((q['il_hpo'] - N['ppo_il_media']) / 38, 1)} \\\\"]
    escrever('tab_custos.tex', linhas,
             'Custo computacional de cada camada do \\textit{MimePlant} e o retorno que ela acrescenta. '
             '1 FRE $=$ $10^6$ passos de ambiente $=$ uma execução completa de RL.', 'tab:custos', 'l r r r r')

    # ── figura: curvas de validação do RL sem HPO ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for braco, cor, rot in (('ppo_il', '#0072B2', 'IL + RL (parte do clone)'), ('ppo_zero', '#D55E00', 'RL do zero')):
        curvas = []
        for f in sorted(glob.glob(os.path.join(R3, 'rl_padrao', braco, 'seed_*', 'metricas.json'))):
            d = ler(f)
            curvas.append([(p['passos'], p['media']) for p in d['curva_validacao_rollback']])
        passos = sorted({p for c in curvas for p, _ in c})
        mat = np.array([[dict(c).get(p, np.nan) for p in passos] for c in curvas], dtype=float)
        med = np.nanmedian(mat, axis=0)
        lo, hi = np.nanmin(mat, axis=0), np.nanmax(mat, axis=0)
        x = np.array(passos) / 1e6
        ax.plot(x, med, color=cor, lw=2, marker='o', ms=3, label=rot)
        ax.fill_between(x, lo, hi, color=cor, alpha=0.15)
    ax.axhline(ref['oraculo_perfeito']['validacao_rollback']['in_dist']['media'], color='k', ls='--', lw=1, label='Professor')
    ax.set_xlabel('Passos de ambiente (milhões)')
    ax.set_ylabel('Recompensa bruta (validação)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(SAIDA, 'fig_curvas_rl.pdf'))
    plt.close(fig)

    # ── figura: dispersão por otimizador ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    ordem = sorted(por, key=lambda o: -np.mean(por[o]))
    for i, o in enumerate(ordem):
        v = por[o]
        ax.scatter([i] * len(v), v, color=CORES[o], s=45, zorder=3, label=None)
        ax.hlines(np.mean(v), i - 0.22, i + 0.22, color=CORES[o], lw=2.5, zorder=4)
    ax.axhline(N['ppo_il_media'], color='gray', ls=':', lw=1.2, label='IL + RL sem HPO')
    ax.axhline(N['professor_media'], color='k', ls='--', lw=1.2, label='Professor')
    ax.set_xticks(range(len(ordem)))
    ax.set_xticklabels([ROTULO[o] for o in ordem], fontsize=8)
    ax.set_ylabel('Recompensa bruta em teste')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=0.3, axis='y')
    fig.savefig(os.path.join(SAIDA, 'fig_otimizadores.pdf'))
    plt.close(fig)

    # ── figura: melhor-até-agora × FRE (escala bruta) ───────────────────────────────────────────
    def curva_estudo(estudo):
        pad = 'trabalho_*' if estudo.startswith('dehb') else 'trial_*'
        pontos = []
        for f in glob.glob(os.path.join(DIR_HPO, estudo, pad, 'metricas.json')):
            try:
                d = ler(f)
            except Exception:
                continue
            s = (d.get('por_seed') or [{}])[0].get('validacao_objetivo')
            if not s or not d.get('fre'):
                continue
            pontos.append((os.path.getmtime(f), d['fre'], sum(s[g]['media'] for g in REGIMES) / 3))
        pontos.sort()
        acum, melhor, xs, ys = 0.0, None, [], []
        for _, fre, val in pontos:
            acum += fre
            melhor = val if melhor is None else max(melhor, val)
            xs.append(acum)
            ys.append(melhor)
        return xs, ys

    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    grade = np.arange(1, 33)
    for o, estudos in (('TPE', ['tpe_r3_a', 'tpe_r3_b', 'tpe_r3_c']), ('RS', ['rs_r3_a', 'rs_r3_b', 'rs_r3_c']),
                       ('DEHB', ['dehb_r3_a', 'dehb_r3_b', 'dehb_r3_c']), ('GP', ['gp_r3_a', 'gp_r3_b', 'gp_r3_c']),
                       ('QMC', ['qmc_r3_a', 'qmc_r3_b', 'qmc_r3_c'])):
        interps = []
        for e in estudos:
            xs, ys = curva_estudo(e)
            if xs:
                interps.append(np.interp(grade, xs, ys, left=np.nan))
        if interps:
            m = np.nanmean(np.array(interps), axis=0)
            ax.plot(grade, m, color=CORES[o], lw=1.8, label=ROTULO[o])
    ax.set_xlabel('Orçamento consumido (FRE)')
    ax.set_ylabel('Melhor configuração até o momento\n(recompensa bruta, validação)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(SAIDA, 'fig_melhor_ate_agora.pdf'))
    plt.close(fig)

    # ── figura: quadrantes ──────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    rot = ['RL do zero\npadrão', 'IL + RL\npadrão', 'RL do zero\ncom HPO', 'IL + RL\ncom HPO']
    val = [q['zero_padrao'], q['il_padrao'], q['zero_hpo'], q['il_hpo']]
    cor = ['#BBBBBB', '#BBBBBB', '#56B4E9', '#0072B2']
    b = ax.bar(rot, val, color=cor)
    ax.bar_label(b, labels=[br(v, 0) for v in val], fontsize=9, padding=2)
    ax.axhline(N['professor_media'], color='k', ls='--', lw=1.2, label='Professor')
    ax.set_ylim(min(val) - 150, max(max(val), N['professor_media']) + 120)
    ax.set_ylabel('Recompensa bruta em teste')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    fig.savefig(os.path.join(SAIDA, 'fig_quadrantes.pdf'))
    plt.close(fig)

    with open(os.path.join(SAIDA, 'numeros.json'), 'w', encoding='utf-8') as f:
        json.dump(N, f, ensure_ascii=False, indent=1)
    print('tabelas em', TAB)
    print('figuras e numeros.json em', SAIDA)
    for k in ('professor_media', 'clone_media', 'ppo_il_media', 'ppo_zero_media'):
        print(f'  {k}: {br(N[k])}')
    print('  MimePlant (conjunto ampliado):', N['mimeplant']['cand'], br(N['mimeplant']['media']))
    print('  MimePlant (pré-registrado):', N['pre_registrado']['cand'], br(N['pre_registrado']['media']))
    print('  quadrantes:', {k: br(v) for k, v in q.items()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
