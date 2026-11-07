"""Tabelas e figuras do Capítulo 5 da versão final da dissertação (Texto/final), em RECOMPENSA BRUTA.

Organizado pelas quatro perguntas que o capítulo responde:
  1. quanto pontua o professor heurístico (o bot);
  2. quanto esforço o RL do zero gasta para alcançar o ponto de partida do IL + RL (o nível do clone), com e sem HPO;
  3. o quanto o HPO importa no IL + RL;
  4. a comparação das cinco condições: professor, RL puro, RL + HPO, IL + RL puro, IL + RL + HPO.

Lê só arquivos de resultado da rodada 3 (nada é digitado à mão) e grava:
  <final>/tabelas/tab_referencias.tex        professor, clone e agente aleatório
  <final>/tabelas/tab_cinco_condicoes.tex    as cinco condições
  <final>/tabelas/tab_quadrantes.tex         ganhos do HPO e da imitação (2 x 2)
  <final>/tabelas/tab_otimizadores.tex       os 15 incumbentes do IL + RL
  <final>/tabelas/tab_agregado.tex           por otimizador, com o ganho sobre o IL + RL puro
  <final>/tabelas/tab_otimizadores_zero.tex  TPE, busca aleatória e GP-BO no RL do zero
  <final>/tabelas/tab_hp_escolhidos.tex      hiperparâmetros: manual, MimePlant e RL + HPO
  <final>/tabelas/tab_esforco.tex            esforço até o ponto de partida do IL + RL
  <final>/tabelas/tab_ablacoes.tex           ablações
  <final>/fig_cinco_condicoes.pdf, fig_otimizadores.pdf, fig_esforco.pdf
  ../analise_final/numeros_final.json        todos os números citados no texto

Agregação: IQM dos retornos por episódio, agrupados entre as sementes de treino (professor e agente aleatório: 50
episódios por regime; clone e agentes treinados: 250), e depois a média simples dos três regimes. Os testes pareados usam
a recompensa média de cada semente de treino (média dos três regimes).

Escolhas (todas em validação-objetivo; teste só reportado), pela regra do código (`reavaliar_topk.escolher_melhor_candidato`:
maior média, entre as sementes, da recompensa NORMALIZADA, 0 = aleatório e 1 = professor, nos três regimes):
  IL + RL + HPO (MimePlant) = melhor entre os estudos pré-registrados (TPE e busca aleatória);
  RL + HPO                  = melhor entre os três estudos do zero (TPE, busca aleatória e GP-BO).

Uso:  python -X utf8 execucao/gerar_artefatos_final.py [pasta do LaTeX]
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats          # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import gerar_artefatos_dissertacao as G  # noqa: E402

SAIDA = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(os.path.dirname(G.REPO), 'final')
G.SAIDA = SAIDA
G.TAB = os.path.join(SAIDA, 'tabelas')
R3 = G.R3
REGIMES = G.REGIMES
br, ler, media3, escrever = G.br, G.ler, G.media3, G.escrever
SEMENTES = [42, 123, 2024, 7, 1337]
DIR_NUM = os.path.join(G.EXPERIMENTOS, 'analise_final')
PASSOS_DEMOS = 200 * 365          # 200 episódios do professor, gerados uma vez e usados pelas 5 sementes
ROT_ZERO = {'tpe_zero': 'TPE', 'rs_zero': 'Busca aleatória', 'gp_zero': 'GP-BO'}
ROT_IL = {'TPE': 'TPE', 'RS': 'Busca aleatória', 'DEHB': 'DEHB', 'GP': 'GP-BO', 'QMC': 'Sobol'}
COR = {'prof': '#333333', 'rl': '#E69F00', 'rl_hpo': '#D55E00', 'il': '#56B4E9', 'il_hpo': '#0072B2'}
FRE_NOMINAL = 32.0


def sinal(x, casas=0):
    return ('+' if x >= 0 else '−') + br(abs(x), casas)


def tex_sinal(x, casas=0):
    return '$' + ('+' if x >= 0 else '-') + br(abs(x), casas) + '$'


def ganho_tab(x, tol=5.0):
    """Ganho na tabela dos 15 incumbentes: uma casa decimal no empate (|x| <= tol, critério do texto), para não parecer ganho."""
    if abs(x) <= tol:
        return '$' + ('+' if x >= 0 else '-') + br(abs(x), 1).replace(',', '{,}') + '$'
    return tex_sinal(x)


def iqm_agrupado(listas):
    """IQM dos retornos por episódio de todas as sementes juntas (a mesma regra de `avaliacao.iqm`, usada por
    `reavaliar_topk.bootstrap_iqm` nos agentes treinados): ordena e tira a média entre os percentis 25 e 75."""
    v = np.sort(np.concatenate([np.asarray(x, float) for x in listas]))
    return float(v[int(np.floor(0.25 * len(v))):int(np.ceil(0.75 * len(v)))].mean())


def fre_estudo(nome):
    """FRE consumido por um estudo de HPO (soma dos passos de todas as tentativas, concluídas e podadas)."""
    return float(ler(os.path.join(G.DIR_HPO, nome, 'PROGRESSO.json'))['fre_consumidos'])


def cand(pasta, nome=None):
    d = {k: v for k, v in ler(os.path.join(R3, pasta, 'resumo.json')).items() if isinstance(v, dict) and 'teste' in v}
    return d if nome is None else d[nome]


def resumo(rotulo, nome, pasta, v):
    return {'rotulo': rotulo, 'cand': nome, 'pasta': pasta, 'teste': {g: v['teste'][g]['iqm'] for g in REGIMES},
            'media': media3(v['teste']), 'val': media3(v['validacao_objetivo']), 'busca': v.get('valor_tuning'),
            'val_norm': v['validacao_objetivo'].get('normalizada_media_3_regimes', {}).get('media'),
            'por_semente': [float(x) for x in np.mean([v['teste'][g]['por_semente'] for g in REGIMES], axis=0)],
            'por_semente_regime': {g: [float(x) for x in v['teste'][g]['por_semente']] for g in REGIMES},
            'ic95': {g: v['teste'][g].get('ic95') for g in REGIMES},
            'val_norm_por_semente': v['validacao_objetivo'].get('normalizada_media_3_regimes', {}).get('por_semente')}


def pareado(a, b):
    """a − b por semente de treino (recompensa bruta em teste, média dos 3 regimes)."""
    d = np.asarray(a['por_semente']) - np.asarray(b['por_semente'])
    t, p = stats.ttest_rel(a['por_semente'], b['por_semente'])
    return {'diferenca': a['media'] - b['media'], 'vence': int((d > 0).sum()), 'n': int(len(d)), 't': float(t), 'p': float(p)}


def welch(a, b):
    """a − b entre agentes que NÃO partem dos mesmos pesos (com e sem imitação): teste t de Welch sobre as médias por semente."""
    t, p = stats.ttest_ind(a['por_semente'], b['por_semente'], equal_var=False)
    return {'diferenca': a['media'] - b['media'], 'n': int(len(a['por_semente'])), 't': float(t), 'p': float(p),
            'min_a': float(min(a['por_semente'])), 'max_b': float(max(b['por_semente']))}


def holm(ps):
    """Valores p ajustados por Holm (família de testes)."""
    ps = np.asarray(ps, float)
    ordem = np.argsort(ps)
    adj = np.empty(len(ps))
    acum = 0.0
    for r, i in enumerate(ordem):
        acum = max(acum, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, acum)
    return [float(x) for x in adj]


def metricas(pasta, nome, s):
    return ler(os.path.join(R3, pasta, nome, f'seed_{s}', 'metricas.json'))


def cruzar(curva, alvo):
    """Passos (interpolados entre blocos) do primeiro ponto da curva de validação que alcança `alvo`; None se não alcança."""
    ant = None
    for p in curva:
        if p['media'] >= alvo - 1e-6:      # tolerância de ponto flutuante: o primeiro ponto do IL + RL puro é o próprio clone
            if ant is None:
                return float(p['passos'])
            return float(ant['passos'] + (alvo - ant['media']) / (p['media'] - ant['media']) * (p['passos'] - ant['passos']))
        ant = p
    return None


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.makedirs(G.TAB, exist_ok=True)
    os.makedirs(DIR_NUM, exist_ok=True)
    N = {}

    # ── referências ─────────────────────────────────────────────────────────────────────────────
    ref = ler(os.path.join(R3, 'referencias', 'referencias.json'))
    clone_t = ler(os.path.join(R3, 'il_warmstart', 'resumo.json'))['tabela']['teste']
    il_ws = [ler(os.path.join(R3, 'il_warmstart', f'seed_{s}', 'metricas.json')) for s in SEMENTES]
    prof = {g: ref['oraculo_perfeito']['teste'][g]['iqm'] for g in REGIMES}
    alea = {g: ref['aleatorio']['teste'][g]['iqm'] for g in REGIMES}
    pausas = {g: ref['oraculo_subotimo']['teste'][g]['iqm'] for g in REGIMES}
    prof_media_ep = {g: ref['oraculo_perfeito']['teste'][g]['media'] for g in REGIMES}   # média dos 50 episódios (teste pareado)
    # clone: IQM dos 250 episódios de teste (5 sementes x 50), a mesma estatística dos agentes treinados e do professor
    clone = {g: iqm_agrupado([w['avaliacao']['teste'][g]['recompensas'] for w in il_ws]) for g in REGIMES}
    clone_media_sementes = {g: clone_t[g]['media_entre_sementes'] for g in REGIMES}      # estatística antiga, só para registro
    m = lambda d: sum(d[g] for g in REGIMES) / 3  # noqa: E731
    N['professor'] = {**prof, 'media': m(prof), 'media_episodios': {**prof_media_ep, 'media': m(prof_media_ep)}}
    N['aleatorio'] = {**alea, 'media': m(alea)}
    N['professor_pausas'] = {**pausas, 'media': m(pausas)}
    N['clone'] = {**clone, 'media': m(clone), 'pct_professor': 100 * m(clone) / m(prof),
                  'media_entre_sementes': {**clone_media_sementes, 'media': m(clone_media_sementes),
                                           'pct_professor': 100 * m(clone_media_sementes) / m(prof)}}
    linhas = ['\\textbf{Política} & \\textbf{Normal} & \\textbf{Seco} & \\textbf{Úmido} & \\textbf{Média} & \\textbf{\\% prof.} \\\\ \\hline']
    for rot, d in (('\\textbf{Professor heurístico}', prof),
                   ('Clone por imitação (BC + DAgger)', clone), ('Agente aleatório', alea)):
        linhas.append(' & '.join([rot] + [br(d[g], 0) for g in REGIMES] + [br(m(d), 0), br(100 * m(d) / m(prof), 0) + '\\%']) + ' \\\\')
    escrever('tab_referencias.tex', linhas,
             'Recompensa bruta (IQM) das políticas de referência no conjunto de teste: 50 episódios por regime no professor, '
             'no professor e no agente aleatório; 250 no clone (cinco sementes de treino $\\times$ 50). '
             'A última coluna é a média dos três regimes em porcentagem da média do professor.',
             'tab:referencias', 'l c c c c c', tam='\\small', curta='Professor, clone e agente aleatório (teste)')

    # ── as cinco condições ──────────────────────────────────────────────────────────────────────
    base = cand('hpo_reavaliacao')
    # confere que o IQM agrupado dos episódios reproduz o do resumo.json (mesma regra do código)
    for pasta_b, nome_b in (('ppo_il', 'padrao_ppo_il'), ('ppo_zero', 'padrao_ppo_zero')):
        for g in REGIMES:
            v_chk = iqm_agrupado([metricas('rl_padrao', pasta_b, s)['avaliacao']['teste'][g]['recompensas'] for s in SEMENTES])
            assert abs(v_chk - base[nome_b]['teste'][g]['iqm']) < 1e-6, ('IQM agrupado difere do resumo.json', nome_b, g)
    rl = resumo('RL puro', 'padrao_ppo_zero', 'rl_padrao/ppo_zero', base['padrao_ppo_zero'])
    il = resumo('IL + RL puro', 'padrao_ppo_il', 'rl_padrao/ppo_il', base['padrao_ppo_il'])
    inc = G.coletar_incumbentes()
    # regra do código (`escolher_melhor_candidato`): maior média, entre as sementes, da recompensa normalizada em validação-objetivo
    pre = max([d for d in inc if d['otim'] in ('TPE', 'RS')], key=lambda d: d['val_norm'])
    rank15 = sorted(inc, key=lambda d: -d['val_norm'])
    il_hpo = resumo('IL + RL com HPO (\\textit{MimePlant})', pre['cand'], 'hpo_reavaliacao', base[pre['cand']])
    zeros = {}
    for pasta in ('hpo_reavaliacao_zero', 'hpo_reavaliacao_zero_gp'):
        for k, v in cand(pasta).items():
            pref = next(p for p in ROT_ZERO if k.startswith(p))
            zeros[pref] = resumo(ROT_ZERO[pref], k, pasta, v)
    escolhido = max(zeros, key=lambda k: zeros[k]['val_norm'])
    rl_hpo = dict(zeros[escolhido], rotulo='RL com HPO')
    N['escolha_rl_hpo'] = {'otimizador': ROT_ZERO[escolhido], 'cand': rl_hpo['cand'],
                           'validacao': {k: z['val'] for k, z in zeros.items()},
                           'validacao_normalizada': {k: z['val_norm'] for k, z in zeros.items()}}
    N['mimeplant'] = {'cand': il_hpo['cand'], 'estudo': pre['cand'].rsplit('_trial', 1)[0], 'validacao': il_hpo['val'],
                      'validacao_normalizada': il_hpo['val_norm'], 'primeiro_entre_os_15': rank15[0]['cand'] == il_hpo['cand'],
                      'ranking_15_validacao_normalizada': [{'cand': d['cand'], 'val_norm': d['val_norm'], 'val_bruta': d['media_val'],
                                                            'teste': d['media_teste']} for d in rank15[:5]],
                      'primeiro_pela_media_bruta_entre_15': max(inc, key=lambda d: d['media_val'])['cand']}
    cinco = [('rl', rl), ('rl_hpo', rl_hpo), ('il', il), ('il_hpo', il_hpo)]
    # MimePlant contra o professor, regime a regime: média de cada semente de treino contra a média do professor nos mesmos
    # 50 episódios de teste (teste t de uma amostra); "acima em k de 5" conta as sementes cuja média passa a do professor
    vs_prof = {}
    for g in REGIMES:
        ps_g = np.asarray(il_hpo['por_semente_regime'][g])
        t_g, p_g = stats.ttest_1samp(ps_g, prof_media_ep[g])
        vs_prof[g] = {'dif_media': float(ps_g.mean() - prof_media_ep[g]), 'acima_em': int((ps_g > prof_media_ep[g]).sum()),
                      'n': int(len(ps_g)), 't': float(t_g), 'p': float(p_g)}
    ps_m = np.asarray(il_hpo['por_semente'])
    t_m, p_m = stats.ttest_1samp(ps_m, m(prof_media_ep))
    vs_prof['media'] = {'dif_media': float(ps_m.mean() - m(prof_media_ep)), 'acima_em': int((ps_m > m(prof_media_ep)).sum()),
                        'n': int(len(ps_m)), 't': float(t_m), 'p': float(p_m)}
    N['mimeplant_vs_professor'] = vs_prof

    # Friedman entre as cinco condições (professor, RL puro, RL com HPO, IL + RL puro e MimePlant), só como descrição: os blocos são
    # os 150 episódios de teste (3 regimes x 50 sementes de episódio, as mesmas para todos) e, nos agentes treinados, o valor do
    # bloco é a média das 5 sementes de treino nesse episódio. Os blocos dos três regimes não são independentes entre si.
    def ep_agente(c, g):
        loc = c['pasta'].split('/') if c['pasta'].startswith('rl_padrao/') else (c['pasta'], c['cand'])
        return np.mean([metricas(loc[0], loc[1], s)['avaliacao']['teste'][g]['recompensas'] for s in SEMENTES], axis=0)
    ordem5 = [('professor', None), ('rl', rl), ('rl_hpo', rl_hpo), ('il', il), ('il_hpo', il_hpo)]
    blocos5 = []
    for g in REGIMES:
        cols = [np.asarray(ref['oraculo_perfeito']['teste'][g]['recompensas'], float) if c is None else ep_agente(c, g) for _, c in ordem5]
        assert all(len(x) == 50 for x in cols), 'esperados 50 episódios de teste por regime'
        blocos5.append(np.column_stack(cols))
    X5 = np.vstack(blocos5)
    chi2_f, p_f = stats.friedmanchisquare(*X5.T)
    postos = np.apply_along_axis(stats.rankdata, 1, X5).mean(axis=0)
    # variante com blocos = sementes de treino (5), só os 4 agentes treinados (média dos 3 regimes por semente)
    Y4 = np.column_stack([np.asarray(c['por_semente'], float) for _, c in ordem5[1:]])
    chi2_s, p_s = stats.friedmanchisquare(*Y4.T)
    N['friedman'] = {'condicoes': [k for k, _ in ordem5], 'blocos': int(X5.shape[0]), 'k': int(X5.shape[1]),
                     'chi2': float(chi2_f), 'p': float(p_f), 'w_kendall': float(chi2_f / (X5.shape[0] * (X5.shape[1] - 1))),
                     'posto_medio': {k: float(v) for (k, _), v in zip(ordem5, postos)},
                     'quatro_agentes_blocos_sementes': {'condicoes': [k for k, _ in ordem5[1:]], 'blocos': int(Y4.shape[0]),
                                                        'chi2': float(chi2_s), 'p': float(p_s)}}
    pm = m(prof)
    linhas = ['\\textbf{Agente} & \\textbf{Normal} & \\textbf{Seco} & \\textbf{Úmido} & \\textbf{Média} & \\textbf{Dif. p/ prof.} \\\\ \\hline',
              ' & '.join(['Professor heurístico'] + [br(prof[g], 0) for g in REGIMES] + [br(pm, 0), '---']) + ' \\\\ \\hline']
    for chave, c in cinco:
        forte = chave == 'il_hpo'
        f = (lambda s: '\\textbf{' + s + '}') if forte else (lambda s: s)
        linhas.append(' & '.join([f(c['rotulo'])] + [f(br(c['teste'][g], 0)) for g in REGIMES]
                                 + [f(br(c['media'], 0)), tex_sinal(c['media'] - pm)]) + ' \\\\')
        N[chave] = {**c['teste'], 'media': c['media'], 'dif_professor': c['media'] - pm, 'cand': c['cand'],
                    'dif_regime': {g: c['teste'][g] - prof[g] for g in REGIMES}}
    escrever('tab_cinco_condicoes.tex', linhas,
             'As cinco condições comparadas. Recompensa bruta (IQM) no conjunto de teste: 50 episódios por regime no professor '
             'e 250 em cada agente treinado (cinco sementes de treino). '
             'RL puro e IL~+~RL puro usam a configuração manual; as versões com HPO usam a configuração escolhida em validação.',
             'tab:cinco_condicoes', 'l c c c c c', tam='\\small', tabcol='2.5pt', curta='As cinco condições (teste)')

    # ── ganhos: 2 x 2 ───────────────────────────────────────────────────────────────────────────
    g_hpo_il, g_hpo_rl = il_hpo['media'] - il['media'], rl_hpo['media'] - rl['media']
    g_il_puro, g_il_hpo = il['media'] - rl['media'], il_hpo['media'] - rl_hpo['media']
    N['ganhos'] = {'hpo_no_il_rl': g_hpo_il, 'hpo_no_rl': g_hpo_rl, 'imitacao_sem_hpo': g_il_puro, 'imitacao_com_hpo': g_il_hpo,
                   'conjunto': il_hpo['media'] - rl['media'], 'soma_isolados': g_il_puro + g_hpo_rl}
    # família F4: pareado onde as duas execuções partem dos mesmos pesos (efeito da HPO); Welch entre agentes com e sem imitação
    N['pareados'] = {'hpo_no_il_rl': pareado(il_hpo, il), 'hpo_no_rl': pareado(rl_hpo, rl)}
    N['welch'] = {'imitacao_sem_hpo': welch(il, rl), 'imitacao_com_hpo': welch(il_hpo, rl_hpo)}
    N['pareados_imitacao_so_registro'] = {'imitacao_sem_hpo': pareado(il, rl), 'imitacao_com_hpo': pareado(il_hpo, rl_hpo)}
    # interação imitação x HPO: ganho da HPO por semente no IL + RL contra o ganho da HPO por semente no RL do zero (Welch, F4);
    # o pareamento por semente entre os dois braços é só registro (os braços não partem dos mesmos pesos)
    g_il_sem = np.asarray(il_hpo['por_semente']) - np.asarray(il['por_semente'])
    g_rl_sem = np.asarray(rl_hpo['por_semente']) - np.asarray(rl['por_semente'])
    inter = g_il_sem - g_rl_sem
    t_w, p_w = stats.ttest_ind(g_il_sem, g_rl_sem, equal_var=False)
    t_i, p_i = stats.ttest_1samp(inter, 0)
    N['interacao'] = {'estimativa_iqm': N['ganhos']['conjunto'] - N['ganhos']['soma_isolados'],
                      'ganho_hpo_il_por_semente': [float(x) for x in g_il_sem], 'ganho_hpo_rl_por_semente': [float(x) for x in g_rl_sem],
                      'welch_t': float(t_w), 'welch_p': float(p_w), 'teste': 'Welch entre os ganhos da HPO por semente',
                      'por_semente': [float(x) for x in inter], 'media_por_semente': float(inter.mean()),
                      'positivas': int((inter > 0).sum()), 'n': int(len(inter)), 'pareado_t_so_registro': float(t_i), 'pareado_p_so_registro': float(p_i)}
    linhas = ['& \\textbf{Sem HPO} & \\textbf{Com HPO} & \\textbf{Ganho da HPO} \\\\ \\hline',
              f"\\textbf{{IL + RL}} & {br(il['media'], 0)} & \\textbf{{{br(il_hpo['media'], 0)}}} & {tex_sinal(g_hpo_il)} \\\\",
              f"\\textbf{{RL do zero}} & {br(rl['media'], 0)} & {br(rl_hpo['media'], 0)} & {tex_sinal(g_hpo_rl)} \\\\ \\hline",
              f"\\textbf{{Ganho da imitação}} & {tex_sinal(g_il_puro)} & {tex_sinal(g_il_hpo)} & \\\\"]
    escrever('tab_quadrantes.tex', linhas,
             'Ganho de cada ingrediente, a partir da Tabela~\\ref{tab:cinco_condicoes}: com e sem imitação, com e sem HPO. '
             'Recompensa bruta (IQM) em teste, média dos três regimes.', 'tab:quadrantes', 'l c c c', tam='\\small',
             curta='Ganho da imitação e da HPO (2 $\\times$ 2)')

    # ── HPO no IL + RL: 15 incumbentes e agregado por otimizador ───────────────────────────────
    linhas = ['\\textbf{Otimizador} & \\textbf{Execução} & \\textbf{Normal} & \\textbf{Seco} & \\textbf{Úmido} & \\textbf{Média} & \\textbf{Ganho} \\\\ \\hline']
    for d in sorted(inc, key=lambda x: -x['media_teste']):
        mp = d['cand'] == il_hpo['cand']
        f = (lambda s: s)                       # sem destaque em negrito (pedido do autor); só o asterisco marca o adotado
        linhas.append(' & '.join([f(ROT_IL[d['otim']] + ('$^{*}$' if mp else '')), f('\\textit{' + d['cand'].split('_')[2] + '}')]
                                 + [f(br(d['teste'][g], 0)) for g in REGIMES]
                                 + [f(br(d['media_teste'], 0)), ganho_tab(d['media_teste'] - il['media'])]) + ' \\\\')
    escrever('tab_otimizadores.tex', linhas,
             'HPO no IL~+~RL: a melhor configuração de cada uma das quinze buscas (cinco otimizadores, três execuções independentes '
             f"\\textit{{a}}, \\textit{{b}} e \\textit{{c}}), retreinada com cinco sementes. Recompensa bruta (IQM) em teste; o ganho é medido "
             f"sobre o IL~+~RL puro (${br(il['media'], 0)}$), calculado antes do arredondamento. $^{{*}}$ Agente adotado como "
             f"\\textit{{MimePlant}}: o de maior média normalizada em validação-objetivo ($0$~=~aleatório, $1$~=~professor) entre os seis "
             f"incumbentes de TPE e de busca aleatória (e também entre os quinze).",
             'tab:otimizadores', 'l c c c c c c', tam='\\small', tabcol='3pt', curta='Os quinze agentes do IL~+~RL com HPO (teste)')
    por = {}
    for d in inc:
        por.setdefault(d['otim'], []).append(d['media_teste'])
    grupos = [np.array(v) for v in por.values()]
    F, p_anova = stats.f_oneway(*grupos)
    H, p_kw = stats.kruskal(*grupos)
    todos = np.concatenate(grupos)
    eta2 = sum(len(g) * (g.mean() - todos.mean()) ** 2 for g in grupos) / ((todos - todos.mean()) ** 2).sum()
    N['otimizadores_il'] = {'anova_F': float(F), 'anova_p': float(p_anova), 'kruskal_H': float(H), 'kruskal_p': float(p_kw),
                            'eta2': float(eta2), 'media_15': float(todos.mean()), 'ganho_medio_15': float(todos.mean() - il['media']),
                            'melhor_15': float(todos.max()), 'pior_15': float(todos.min()),
                            'acima_do_puro': int((todos > il['media'] + 5).sum()), 'empate': int((abs(todos - il['media']) <= 5).sum()),
                            'abaixo': int((todos < il['media'] - 5).sum()), 'no_nivel_do_professor': int((todos >= pm).sum()),
                            'amplitude_medias': float(max(g.mean() for g in grupos) - min(g.mean() for g in grupos)),
                            'amplitude_15': float(todos.max() - todos.min()), 'desvio_15': float(todos.std(ddof=1)),
                            'por': {o: {'media': float(np.mean(v)), 'ganho': float(np.mean(v) - il['media']), 'melhor': float(max(v)),
                                        'pior': float(min(v)), 'desvio': float(np.std(v, ddof=1))} for o, v in por.items()}}
    # as 15 buscas contra o IL + RL puro, pelo critério dos testes pareados (média de cada semente de treino, 3 regimes)
    ps_inc = {}
    for pasta in ('hpo_reavaliacao', 'hpo_reavaliacao_dehb', 'hpo_reavaliacao_gp', 'hpo_reavaliacao_qmc'):
        for k, v in cand(pasta).items():
            if not k.startswith('padrao'):
                ps_inc[k] = {'por_semente': [float(x) for x in np.mean([v['teste'][g]['por_semente'] for g in REGIMES], axis=0)],
                             'media': media3(v['teste'])}
    pare15 = {k: pareado(v, il) for k, v in ps_inc.items()}
    sig15 = sorted(k for k, r in pare15.items() if r['vence'] == r['n'] and r['p'] < 0.05)
    bf_stat, bf_p = stats.levene(*grupos, center='median')    # Brown-Forsythe: a dispersão entre execuções difere entre otimizadores?
    N['otimizadores_il'].update({'vencem_5_de_5_p05': len(sig15), 'lista_vencem_5_de_5_p05': sig15,
                                 'pareados_15': {k: {'diferenca': r['diferenca'], 'vence': r['vence'], 'p': r['p']} for k, r in pare15.items()},
                                 'dispersao_brown_forsythe_p': float(bf_p), 'tolerancia_empate': 5.0})
    linhas = ['\\textbf{Otimizador} & \\textbf{Média} & \\textbf{Ganho médio} & \\textbf{Melhor} & \\textbf{Pior} & \\textbf{Desvio-padrão} \\\\ \\hline']
    ordem = sorted(por, key=lambda o: -np.mean(por[o]))
    for o in ordem:
        s = N['otimizadores_il']['por'][o]
        linhas.append(' & '.join([ROT_IL[o], br(s['media'], 0), tex_sinal(s['ganho']), br(s['melhor'], 0), br(s['pior'], 0), br(s['desvio'], 0)]) + ' \\\\')
    escrever('tab_agregado.tex', linhas,
             'HPO no IL~+~RL, por otimizador: média, melhor e pior das três execuções independentes e desvio-padrão amostral '
             f"entre elas. Recompensa bruta (IQM) em teste; ganho sobre o IL~+~RL puro (${br(il['media'], 0)}$), calculado antes do "
             'arredondamento.', 'tab:agregado', 'l c c c c c',
             tam='\\small', tabcol='2pt', curta='HPO no IL~+~RL, por otimizador (teste)')

    # ── viés da busca: nota de busca (1 semente) contra a validação com 5 sementes, na mesma escala ──
    reav = [resumo('', k, p, v) for p in ('hpo_reavaliacao', 'hpo_reavaliacao_dehb', 'hpo_reavaliacao_gp', 'hpo_reavaliacao_qmc',
                                           'hpo_reavaliacao_zero', 'hpo_reavaliacao_zero_gp')
            for k, v in cand(p).items() if not k.startswith('padrao')]
    com = [r for r in reav if r['busca'] is not None and r['val_norm'] is not None]
    N['vies_busca'] = {'n': len(com), 'busca_maior_que_reavaliacao': sum(r['busca'] > r['val_norm'] for r in com),
                       'queda_media_pct': float(100 * np.mean([r['busca'] - r['val_norm'] for r in com]))}

    # ── HPO no RL do zero ───────────────────────────────────────────────────────────────────────
    linhas = ['\\textbf{Otimizador} & \\textbf{Normal} & \\textbf{Seco} & \\textbf{Úmido} & \\textbf{Média} & \\textbf{Ganho} & \\textbf{Vence em} \\\\ \\hline',
              ' & '.join(['RL puro (sem HPO)'] + [br(rl['teste'][g], 0) for g in REGIMES] + [br(rl['media'], 0), '---', '---']) + ' \\\\ \\hline']
    N['otimizadores_zero'] = {}
    for k in sorted(zeros, key=lambda k: -zeros[k]['media']):
        z, pr = zeros[k], pareado(zeros[k], rl)
        N['otimizadores_zero'][k] = {**z['teste'], 'media': z['media'], 'ganho': z['media'] - rl['media'], 'busca': z['busca'],
                                     'cand': z['cand'], **{'pareado_' + a: b for a, b in pr.items()}}
        f = (lambda s: '\\textbf{' + s + '}') if k == escolhido else (lambda s: s)
        linhas.append(' & '.join([f(z['rotulo'])] + [br(z['teste'][g], 0) for g in REGIMES]
                                 + [f(br(z['media'], 0)), tex_sinal(z['media'] - rl['media']), f"{pr['vence']} de {pr['n']}"]) + ' \\\\')
    N['gp_zero_vs_tpe_zero'] = pareado(zeros['gp_zero'], zeros['tpe_zero'])
    escrever('tab_otimizadores_zero.tex', linhas,
             'HPO no RL do zero: uma busca de 32 FRE nominais por otimizador, melhor configuração retreinada com cinco sementes. '
             'Recompensa bruta (IQM) em teste; o ganho é medido sobre o RL puro, e a última coluna conta em quantas das cinco '
             'sementes de treino a configuração otimizada supera o RL puro. Em negrito, o agente escolhido em validação '
             '(o RL com HPO das cinco condições).', 'tab:otimizadores_zero', 'l c c c c c c', tam='\\small', tabcol='3pt',
             curta='HPO no RL do zero, por otimizador (teste)')

    # ── hiperparâmetros escolhidos ──────────────────────────────────────────────────────────────
    hp_pad = metricas('rl_padrao', 'ppo_il', 42)['hp']
    hp_mp = metricas('hpo_reavaliacao', il_hpo['cand'], 42)['hp']
    hp_z = metricas(rl_hpo['pasta'], rl_hpo['cand'], 42)['hp']
    sci = lambda x: '$' + f'{x:.1e}'.replace('.', '{,}').replace('e-0', ' \\times 10^{-').replace('e-', ' \\times 10^{-') + '}$'  # noqa: E731
    dec = lambda x, c=2: '$' + br(x, c).replace(',', '{,}') + '$'  # noqa: E731
    campos = [('Taxa de aprendizado do ator', 'lr', sci), ('Recorte do PPO ($\\epsilon_{\\text{clip}}$)', 'clip', dec),
              ('Coeficiente de entropia', 'c_ent', sci), ('Fator de desconto ($\\gamma$)', 'gamma', lambda x: dec(x, 3)),
              ('Parâmetro do GAE ($\\lambda_{\\text{GAE}}$)', 'lambda_gae', lambda x: dec(x, 3)), ('Épocas por atualização', 'n_epocas', lambda x: str(int(x))),
              ('Sequências por minilote', 'minilote_seqs', lambda x: str(int(x))), ('Escala da recompensa', 'escala', lambda x: str(int(x))),
              ('Peso inicial do KL ($\\beta_{\\text{KL}}^{(0)}$)', 'beta_kl0', lambda x: dec(x, 3)), ('Decaimento do KL', 'beta_kl_decay', lambda x: dec(x, 3)),
              ('Temperatura inicial ($\\tau$)', 'temperatura_inicial', dec), ('Iterações de aquecimento do crítico', 'k_aquec', lambda x: str(int(x)))]
    linhas = ['\\textbf{Hiperparâmetro} & \\textbf{Manual} & \\textbf{IL + RL com HPO} & \\textbf{RL com HPO} \\\\ \\hline']
    N['hp'] = {'manual': {}, 'il_hpo': {}, 'rl_hpo': {}}
    for i, (rot, k, fmt) in enumerate(campos):
        acopl = i >= 8
        linhas.append(' & '.join([rot, fmt(hp_pad[k]), fmt(hp_mp[k]), '---' if acopl else fmt(hp_z[k])]) + ' \\\\' + (' \\hline' if i == 7 else ''))
        N['hp']['manual'][k], N['hp']['il_hpo'][k], N['hp']['rl_hpo'][k] = hp_pad[k], hp_mp[k], (None if acopl else hp_z[k])
    escrever('tab_hp_escolhidos.tex', linhas,
             'Hiperparâmetros da configuração manual e das configurações escolhidas pela HPO (o \\textit{MimePlant} e o RL com '
             'HPO). Os quatro últimos regulam o acoplamento com o clone e não existem no RL do zero.', 'tab:hp_escolhidos',
             'l c c c', tam='\\small', tabcol='3pt', curta='Hiperparâmetros: manual e escolhidos pela HPO')

    # ── esforço até o ponto de partida do IL + RL (curvas de validação, regime normal) ──────────
    curvas = {}
    agentes = [('rl', 'RL puro', 'rl_padrao', 'ppo_zero'), ('rl_tpe', 'RL com HPO (TPE)', zeros['tpe_zero']['pasta'], zeros['tpe_zero']['cand']),
               ('rl_gp', 'RL com HPO (GP-BO)', zeros['gp_zero']['pasta'], zeros['gp_zero']['cand']),
               ('il', 'IL + RL puro', 'rl_padrao', 'ppo_il'), ('il_hpo', 'IL + RL com HPO', 'hpo_reavaliacao', il_hpo['cand'])]
    for chave, rot, pasta, nome in agentes:
        curvas[chave] = {'rotulo': rot, 'runs': [metricas(pasta, nome, s) for s in SEMENTES]}
    nivel_clone_sem = [float(r['nivel_il_rollback']) for r in curvas['il']['runs']]   # nível do clone de cada semente (validação-rollback)
    nivel_clone = float(np.mean(nivel_clone_sem))
    prof_val = float(ref['oraculo_perfeito']['validacao_rollback']['in_dist']['media'])
    passos_dagger = ler(os.path.join(R3, 'il_warmstart', 'resumo.json'))['receita']
    passos_il = PASSOS_DEMOS + passos_dagger['n_rounds_dagger'] * passos_dagger['passos_por_round']
    min_il = float(np.mean([(w['tempos_s']['bc'] + w['tempos_s']['dagger']) / 60 for w in il_ws]))
    min_demos = float(ler(os.path.join(R3, '..', 'demos_v5', 'config.json'))['tempo_s']) / 60
    N['esforco'] = {'nivel_clone': nivel_clone, 'nivel_clone_por_semente': nivel_clone_sem, 'professor_validacao': prof_val,
                    'imitacao': {'passos_por_semente': passos_il, 'fre': passos_il / 1e6, 'minutos_treino': min_il,
                                 'minutos_demonstracoes': min_demos, 'passos_demos': PASSOS_DEMOS}, 'agentes': {}}

    def linha_alvo(runs, alvos):
        """Passos até alcançar o alvo (um por execução; interpolação linear entre blocos). Os agentes que partem do clone são
        medidos contra o clone da própria semente; os que partem do zero, contra o nível médio dos cinco clones."""
        ps, mins = [], []
        for r, alvo in zip(runs, alvos):
            x = cruzar(r['curva_validacao_rollback'], alvo)
            if x is not None:
                ps.append(x)
                mins.append(x / r['passos'] * r['tempo_treino_s'] / 60)
        return {'n': len(ps), 'passos': ps, 'passos_mediana': float(np.median(ps)) if ps else None,
                'passos_min': min(ps) if ps else None, 'passos_max': max(ps) if ps else None,
                'minutos_mediana': float(np.median(mins)) if mins else None}

    def cel(a):
        if a['n'] == 0:
            return 'nenhuma', '---'
        faixa = (br(a['passos_min'] / 1e3, 0) if round(a['passos_min'] / 1e3) == round(a['passos_max'] / 1e3)
                 else f"{br(a['passos_min'] / 1e3, 0)} a {br(a['passos_max'] / 1e3, 0)}")
        return f"{a['n']} de 5", faixa

    fre_busca = {'rl_tpe': fre_estudo(zeros['tpe_zero']['cand'].rsplit('_trial', 1)[0]),
                 'rl_gp': fre_estudo(zeros['gp_zero']['cand'].rsplit('_trial', 1)[0]),
                 'il_hpo': fre_estudo(N['mimeplant']['estudo'])}
    N['esforco']['fre_busca'] = fre_busca
    linhas = ['\\textbf{Agente} & \\textbf{Custo prévio (FRE)} & \\textbf{Sementes} & \\textbf{Passos (mil)} & \\textbf{Nível final} \\\\ \\hline']
    def m1(x):   # decimal em modo matemático, como nas demais tabelas
        return '$' + br(x, 1).replace(',', '{,}') + '$'
    previo = {'rl': '---', 'rl_tpe': f"{m1(fre_busca['rl_tpe'])} (busca)", 'rl_gp': f"{m1(fre_busca['rl_gp'])} (busca)",
              'il': f"{m1(passos_il / 1e6)} (imitação)", 'il_hpo': f"{m1(passos_il / 1e6)} + {m1(fre_busca['il_hpo'])} (busca)"}
    for chave, rot, _, _ in agentes:
        runs = curvas[chave]['runs']
        alvos = nivel_clone_sem if chave in ('il', 'il_hpo') else [nivel_clone] * len(runs)
        a = linha_alvo(runs, alvos)
        final = float(np.mean([r['curva_validacao_rollback'][-1]['media'] for r in runs]))
        prim = runs[0]['curva_validacao_rollback'][0]
        N['esforco']['agentes'][chave] = {'partida': a, 'final_media': final,
                                          'primeiro_ponto': {'passos': float(prim['passos']), 'media': float(np.mean([r['curva_validacao_rollback'][0]['media'] for r in runs]))},
                                          'minutos_treino': float(np.mean([r['tempo_treino_s'] for r in runs]) / 60)}
        ca = cel(a)
        linhas.append(' & '.join([rot, previo[chave], ca[0], ca[1], br(final, 0)]) + ' \\\\' + (' \\hline' if chave == 'rl_gp' else ''))
    escrever('tab_esforco.tex', linhas,
             'Esforço para alcançar o ponto de partida do IL~+~RL, isto é, o nível do clone '
             f"(${br(nivel_clone, 0)}$ em média nos cinco clones), medido na curva de validação do treinamento por reforço "
             '(validação-\\textit{rollback}: média de 20 episódios no regime normal, em recompensa bruta, a cada bloco de cerca de $111$ mil passos). '
             '\\emph{Sementes}: quantas das cinco alcançam o alvo dentro de $10^6$ passos. \\emph{Passos}: gastos pelas que alcançam, '
             'por interpolação linear entre blocos. \\emph{Nível final}: média do último bloco, na mesma medida (não é teste). '
             'Os agentes que partem do clone são medidos contra o clone da própria semente: o IL~+~RL puro começa nele, e o '
             'IL~+~RL com HPO começa abaixo dele no primeiro bloco medido (temperatura inicial alta) e volta. '
             'O custo prévio é o que se gasta antes de o treinamento final começar (FRE efetivos).',
             'tab:esforco', 'l c c c c', tam='\\small', tabcol='2.5pt', curta='Esforço até o ponto de partida do IL~+~RL')

    # ── ablações (sobre o MimePlant) ────────────────────────────────────────────────────────────
    abl = ler(os.path.join(R3, 'hpo_ablacoes', 'resumo.json'))
    nomes = [('abl_sem_kl', 'Sem penalidade KL ($\\beta_{\\text{KL}}^{(0)}=0$)'),
             ('abl_kl_forte', 'Acoplamento manual ($\\beta_{\\text{KL}}^{(0)}=0{,}1$; decaimento $0{,}98$; $\\tau=1$)'),
             ('abl_sem_temperatura', 'Sem temperatura inicial ($\\tau=1$)'), ('abl_zero', 'Sem imitação (parte do zero)')]
    linhas = ['\\textbf{Variante} & \\textbf{Normal} & \\textbf{Seco} & \\textbf{Úmido} & \\textbf{Média} & \\textbf{Diferença} \\\\ \\hline',
              ' & '.join(['\\textbf{\\textit{MimePlant} completo}'] + [br(il_hpo['teste'][g], 0) for g in REGIMES]
                         + ['\\textbf{' + br(il_hpo['media'], 0) + '}', '---']) + ' \\\\ \\hline']
    N['ablacoes'] = {}
    for k, rot in nomes:
        v = abl[k]
        mm = media3(v['teste'])
        ps_k = np.mean([v['teste'][g]['por_semente'] for g in REGIMES], axis=0)
        var_k = {'por_semente': [float(x) for x in ps_k], 'media': mm}
        # variante − MimePlant: pareado por semente quando as duas partem do mesmo clone; Welch na que parte do zero (regra F4)
        pr_k = welch(var_k, il_hpo) if k == 'abl_zero' else pareado(var_k, il_hpo)
        N['ablacoes'][k] = {'media': mm, 'dif': mm - il_hpo['media'], 'por_semente': [float(x) for x in ps_k],
                            'abaixo_em': int((np.asarray(ps_k) < np.asarray(il_hpo['por_semente'])).sum()), 'n': pr_k['n'],
                            't': pr_k['t'], 'p': pr_k['p'], 'teste': 'welch' if k == 'abl_zero' else 'pareado'}
        linhas.append(' & '.join([rot] + [br(v['teste'][g]['iqm'], 0) for g in REGIMES] + [br(mm, 0), tex_sinal(mm - il_hpo['media'])]) + ' \\\\')
    escrever('tab_ablacoes.tex', linhas,
             'Ablações sobre o \\textit{MimePlant}, cada variante retreinada com as cinco sementes. A variante de acoplamento '
             'manual troca três hiperparâmetros pelos valores manuais ($\\beta_{\\text{KL}}^{(0)}$, decaimento do KL e $\\tau$) e mantém o '
             'aquecimento do crítico em 19; a de partida do zero troca a inicialização (e os quatro hiperparâmetros de acoplamento '
             'deixam de ter efeito); as outras duas alteram um só ingrediente. Recompensa bruta (IQM) em teste.',
             'tab:ablacoes', '>{\\raggedright\\arraybackslash}p{6.4cm} c c c c c', tam='\\small', tabcol='2.5pt', curta='Ablações sobre o \\textit{MimePlant} (teste)')

    # ── custo por camada ────────────────────────────────────────────────────────────────────────
    fre_rl = sum(r['passos'] for r in curvas['il']['runs']) / 1e6          # cinco execuções de reforço de 1.003.020 passos
    fre_reav = sum(metricas('hpo_reavaliacao', il_hpo['cand'], s)['passos'] for s in SEMENTES) / 1e6
    fre_hpo = fre_busca['il_hpo'] + fre_reav                                # a busca do MimePlant (efetiva) + a reavaliação com 5 sementes
    seis = [d for d in inc if d['otim'] in ('TPE', 'RS')]                   # as seis buscas entre as quais o MimePlant foi escolhido
    fre_seis = sum(fre_estudo(d['cand'].rsplit('_trial', 1)[0]) for d in seis) + len(seis) * fre_reav
    ganho_uma_busca = float(np.mean([d['media_teste'] for d in seis]) - il['media'])   # o que uma busca típica acrescenta
    N['custo'] = {'imitacao_fre_5_sementes': (PASSOS_DEMOS + 5 * (passos_il - PASSOS_DEMOS)) / 1e6,
                  'ganho_imitacao': m(clone) - m(alea), 'ganho_rl': il['media'] - m(clone), 'ganho_hpo': g_hpo_il,
                  'fre_rl': fre_rl, 'fre_busca': fre_busca['il_hpo'], 'fre_reavaliacao': fre_reav, 'fre_hpo': fre_hpo,
                  'fre_hpo_seis_buscas': fre_seis, 'ganho_medio_uma_busca': ganho_uma_busca}
    N['custo']['pontos_por_fre'] = {'imitacao': N['custo']['ganho_imitacao'] / N['custo']['imitacao_fre_5_sementes'],
                                    'rl': N['custo']['ganho_rl'] / fre_rl, 'hpo': g_hpo_il / fre_hpo,
                                    'hpo_seis_buscas': g_hpo_il / fre_seis, 'hpo_uma_busca_tipica': ganho_uma_busca / fre_hpo}

    plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.bbox': 'tight', 'figure.dpi': 150, 'pdf.fonttype': 42})   # fonte TrueType: texto pesquisável no PDF
    milhar = FuncFormatter(lambda v, _: br(v, 0))                                            # eixos com ponto de milhar, como no texto

    # ── figura: diferença de cada agente para o professor, por regime ───────────────────────────
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    series = [('RL puro', rl, COR['rl']), ('RL com HPO', rl_hpo, COR['rl_hpo']),
              ('IL + RL puro', il, COR['il']), ('IL + RL com HPO', il_hpo, COR['il_hpo'])]
    grupos_x = [('in_dist', 'Normal'), ('dry', 'Seco'), ('humid', 'Úmido'), ('m', 'Média dos três')]
    larg = 0.2
    for i, (rot, c, cor) in enumerate(series):
        dif = {**{g: c['teste'][g] - prof[g] for g in REGIMES}, 'm': c['media'] - pm}
        xs = [j + (i - 1.5) * larg for j in range(len(grupos_x))]
        ax.bar(xs, [dif[k] for k, _ in grupos_x], larg * 0.92, color=cor, label=rot, zorder=3)
    ax.axhline(0, color='k', lw=1.2, zorder=4)
    ax.text(-0.46, 6, 'professor', fontsize=8, ha='left', va='bottom')
    ax.set_xticks(range(len(grupos_x)))
    ax.set_xticklabels([r for _, r in grupos_x])
    ax.set_ylabel('Diferença para o professor\n(recompensa bruta em teste)')
    ax.yaxis.set_major_formatter(milhar)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.14), frameon=False)
    fig.savefig(os.path.join(SAIDA, 'fig_cinco_condicoes.pdf'))
    plt.close(fig)

    # ── figura: otimizadores no IL + RL e no RL do zero ─────────────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.3), sharey=True, gridspec_kw={'width_ratios': [5, 3.6]})
    for i, o in enumerate(ordem):
        v = por[o]
        a1.scatter([i] * len(v), v, color=COR['il_hpo'], s=38, zorder=3)
        a1.hlines(np.mean(v), i - 0.25, i + 0.25, color=COR['il_hpo'], lw=2.5, zorder=4)
    a1.axhline(il['media'], color=COR['il'], ls=':', lw=1.5, label='IL + RL puro')
    a1.axhline(pm, color='k', ls='--', lw=1.1, label='Professor')
    mp_i = next(i for i, o in enumerate(ordem) if o == next(d['otim'] for d in inc if d['cand'] == il_hpo['cand']))
    a1.scatter([mp_i], [il_hpo['media']], s=140, facecolors='none', edgecolors='k', linewidths=1.3, zorder=5, label='MimePlant (adotado)')
    a1.set_xticks(range(len(ordem)))
    a1.set_xticklabels([ROT_IL[o].replace('Busca ', 'Busca\n') for o in ordem], fontsize=7.5)
    a1.set_title('IL + RL (3 buscas por otimizador)', fontsize=9)
    a1.set_ylabel('Recompensa bruta em teste')
    a1.yaxis.set_major_formatter(milhar)
    a1.legend(fontsize=7.5, loc='lower left')
    a1.grid(alpha=0.3, axis='y')
    oz = sorted(zeros, key=lambda k: -zeros[k]['media'])
    for i, k in enumerate(oz):
        a2.scatter([i], [zeros[k]['media']], color=COR['rl_hpo'], s=38, zorder=3)   # um ponto por otimizador: sem traço de média
    a2.scatter([oz.index(escolhido)], [rl_hpo['media']], s=140, facecolors='none', edgecolors='k', linewidths=1.3, zorder=5, label='escolhido (RL com HPO)')
    a2.axhline(rl['media'], color=COR['rl'], ls=':', lw=1.5, label='RL puro')
    a2.axhline(pm, color='k', ls='--', lw=1.1)
    a2.set_xticks(range(len(oz)))
    a2.set_xticklabels([ROT_ZERO[k].replace('Busca ', 'Busca\n') for k in oz], fontsize=7)
    a2.set_xlim(-0.6, len(oz) - 0.4)
    a2.set_title('RL do zero (1 busca por otimizador)', fontsize=9)
    a2.legend(fontsize=7.5, loc='lower left')
    a2.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, 'fig_otimizadores.pdf'))
    plt.close(fig)

    # ── figura: curvas de treinamento e os dois alvos ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    estilo = {'rl': (COR['rl'], '-'), 'rl_gp': (COR['rl_hpo'], '-'), 'il': (COR['il'], '-'), 'il_hpo': (COR['il_hpo'], '-')}
    for chave in ('rl', 'rl_gp', 'il', 'il_hpo'):
        runs = curvas[chave]['runs']
        xs = [p['passos'] / 1e3 for p in runs[0]['curva_validacao_rollback']]
        ys = np.array([[p['media'] for p in r['curva_validacao_rollback']] for r in runs])     # só pontos medidos (sem ponto inicial artificial)
        cor, ls = estilo[chave]
        ax.plot(xs, ys.mean(axis=0), color=cor, ls=ls, lw=2, marker='o', ms=3, label=curvas[chave]['rotulo'].replace(' (GP-BO)', ''))
        ax.fill_between(xs, ys.min(axis=0), ys.max(axis=0), color=cor, alpha=0.10, lw=0)
    ax.axhline(prof_val, color='k', ls='--', lw=1.1, label='Professor')
    ax.axhline(nivel_clone, color='gray', ls=':', lw=1.3, label='Ponto de partida do IL + RL (clone)')
    ax.set_xlabel('Passos de ambiente no treinamento por reforço (milhares)')
    ax.set_ylabel('Recompensa bruta em validação\n(regime normal)')
    ax.yaxis.set_major_formatter(milhar)
    ax.xaxis.set_major_formatter(milhar)
    ax.set_ylim(3000, 5700)
    ax.set_xlim(-15, 1020)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2, loc='lower right')
    fig.savefig(os.path.join(SAIDA, 'fig_esforco.pdf'))
    plt.close(fig)

    # ── números citados no texto que não saem das tabelas ────────────────────────────────────
    ps_mp = np.asarray(il_hpo['por_semente'])
    sem42 = {}
    for chave, c in cinco:
        por_reg = []
        for g in REGIMES:
            # as configurações manuais ficam em rl_padrao/<braço>/seed_<s>; as da HPO, em <pasta>/<candidato>/seed_<s>
            loc = c['pasta'].split('/') if c['pasta'].startswith('rl_padrao/') else (c['pasta'], c['cand'])
            por_reg.append(iqm_agrupado([metricas(loc[0], loc[1], s)['avaliacao']['teste'][g]['recompensas'] for s in SEMENTES if s != 42]))
        sem42[chave] = {'media_sem_42': float(np.mean(por_reg)), 'dif': float(np.mean(por_reg) - c['media'])}
    manuais = {}
    for nome_m, c in (('padrao_ppo_il', il), ('padrao_ppo_zero', rl)):
        vn = c['val_norm_por_semente']
        manuais[nome_m] = {'por_semente': [float(x) for x in vn], 'melhor_indice': int(np.argmax(vn)), 'melhor_semente': SEMENTES[int(np.argmax(vn))],
                           'semente_42_menos_media_pp': float(100 * (vn[0] - np.mean(vn)))}
    ultimo = {chave: int(sum(r['curva_validacao_rollback'][-1]['media'] >= nivel_clone for r in curvas[chave]['runs'])) for chave in ('rl', 'rl_tpe', 'rl_gp')}
    ent = []
    for r in curvas['il']['runs']:
        ppo_pts = [c_ for c_ in r.get('curva_treino', []) if c_.get('fase') == 'ppo']
        k_ent = next((k for k in ppo_pts[0] if 'entrop' in k.lower()), None) if ppo_pts else None
        if k_ent:
            ent.append(float(ppo_pts[0][k_ent]))
    desvios = [N['otimizadores_il']['por'][o]['desvio'] for o in N['otimizadores_il']['por']]
    qm_dentro = float(np.mean(np.square(desvios)))
    k_g, n_g = len(desvios), 3
    fcrit = stats.f.ppf(0.95, k_g - 1, k_g * (n_g - 1))
    poder = {}
    for delta in (100, 150, 200, 250, 300):
        medias = np.array([0] * (k_g - 1) + [delta], float)
        lam = n_g * np.sum((medias - medias.mean()) ** 2) / qm_dentro
        poder[str(delta)] = float(1 - stats.ncf.cdf(fcrit, k_g - 1, k_g * (n_g - 1), lam))
    familia = {'hpo_no_il_rl': N['pareados']['hpo_no_il_rl']['p'], 'hpo_no_rl': N['pareados']['hpo_no_rl']['p'],
               'imitacao_sem_hpo': N['welch']['imitacao_sem_hpo']['p'], 'imitacao_com_hpo': N['welch']['imitacao_com_hpo']['p'],
               'tpe_zero': N['otimizadores_zero']['tpe_zero']['pareado_p'], 'gp_zero_vs_tpe_zero': N['gp_zero_vs_tpe_zero']['p']}
    N['extras'] = {'desvio_sementes_mimeplant': float(ps_mp.std(ddof=1)), 'sem_semente_42': sem42, 'semente_42_nas_manuais': manuais,
                   'ic95_teste': {chave: c['ic95'] for chave, c in cinco},
                   'ic95_contem_professor': {chave: {g: bool(c['ic95'][g][0] <= prof[g] <= c['ic95'][g][1]) for g in REGIMES} for chave, c in cinco},
                   'ultimo_bloco_acima_do_clone': ultimo, 'entropia_1a_atualizacao_il_rl_puro': {'min': min(ent), 'max': max(ent)} if ent else None,
                   'desvio_combinado_otimizadores': float(np.sqrt(qm_dentro)), 'poder_anova_um_grupo_deslocado': poder,
                   'media_seis_buscas_tpe_rs': float(np.mean([d['media_teste'] for d in seis])),
                   'holm': {'familia': familia, 'ajustado': dict(zip(familia, holm(list(familia.values()))))}}

    with open(os.path.join(DIR_NUM, 'numeros_final.json'), 'w', encoding='utf-8') as f:
        json.dump(N, f, ensure_ascii=False, indent=1)
    print('professor', br(pm, 0), '| RL puro', br(rl['media'], 0), '| RL + HPO', br(rl_hpo['media'], 0), f"({rl_hpo['cand']})",
          '| IL + RL puro', br(il['media'], 0), '| IL + RL + HPO', br(il_hpo['media'], 0), f"({il_hpo['cand']})")
    print('ganhos', {k: round(v) for k, v in N['ganhos'].items()})
    print('pareados', {k: (round(v['diferenca']), f"{v['vence']}/{v['n']}", round(v['p'], 3)) for k, v in N['pareados'].items()})
    print('validação dos estudos do zero', {k: round(v) for k, v in N['escolha_rl_hpo']['validacao'].items()})
    print('GP zero − TPE zero', N['gp_zero_vs_tpe_zero'])
    print('viés', N['vies_busca'])
    e = N['esforco']
    print('esforço: clone', round(e['nivel_clone']), 'imitação', e['imitacao'], 'FRE das buscas', {k: round(v, 3) for k, v in e['fre_busca'].items()})
    for k, v in e['agentes'].items():
        print(' ', k, 'partida', v['partida']['n'], [round(x / 1e3) for x in v['partida']['passos']], v['partida']['minutos_mediana'],
              '| 1º ponto', round(v['primeiro_ponto']['media']), '| final', round(v['final_media']), 'min', round(v['minutos_treino']))
    print('MimePlant:', N['mimeplant']['cand'], 'primeiro entre os 15:', N['mimeplant']['primeiro_entre_os_15'] if 'primeiro_entre_os_15' in N['mimeplant'] else N['mimeplant']['primeiro_entre_os_15'])
    print('MimePlant × professor:', {k: (round(v['dif_media']), f"{v['acima_em']}/{v['n']}", round(v['p'], 3)) for k, v in N['mimeplant_vs_professor'].items()})
    print('welch', {k: (round(v['diferenca']), round(v['p'], 4)) for k, v in N['welch'].items()})
    print('interação:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in N['interacao'].items()})
    print('extras:', {k: v for k, v in N['extras'].items() if k not in ('ic95_teste',)})
    print('15 buscas: vencem em 5 de 5 com p<0,05:', N['otimizadores_il']['vencem_5_de_5_p05'], '| Brown-Forsythe p', round(N['otimizadores_il']['dispersao_brown_forsythe_p'], 3))
    print('ablações:', {k: (round(v['dif']), f"abaixo em {v['abaixo_em']}/{v['n']}", round(v['p'], 4)) for k, v in N['ablacoes'].items()})
    print('custo', N['custo'])
    print('otimizadores IL', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in N['otimizadores_il'].items() if k != 'por'})
    return 0


if __name__ == '__main__':
    sys.exit(main())
