"""diag_regime_seco.py — diagnóstico com dados da queda do professor no regime seco (pergunta da banca de qualificação: "por que piorou tanto?").

Roda o professor heurístico (e, de referência, o agente aleatório) nos três regimes, nas sementes de VALIDAÇÃO (500–549, 50 episódios; nunca as de
teste), com um laço idêntico ao de `avaliacao.avaliar` que, além da recompensa, registra a cada passo a temperatura, a chuva e a umidade do solo
das células com planta. Para conferir o laço, a recompensa média do professor é comparada com a de `results/rodada3/referencias/referencias.json`.

Mede: componentes da recompensa, colheita, regas, umidade do solo nos dias com planta, dias com umidade abaixo do limiar de rega do professor
(0,58), variação diária da umidade com e sem rega, chuva e dias de penalidade por calor (> 35 °C com plantas).
Uso:  python -X utf8 analise/diag_regime_seco.py [pasta de saída]   ->  Resultados/diag_regime_seco/*.json e RESUMO_SECO.txt
"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))        # este script fica em <repositório>/analise
sys.path.insert(0, REPO)
import protocolo as P                                  # noqa: E402

P.preparar_sys_path()
import torch                                           # noqa: E402
torch.set_num_threads(2)
import avaliacao as A                                  # noqa: E402
from treinar_warmstart_il import novo_env              # noqa: E402

SAIDA = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(REPO, 'Resultados', 'diag_regime_seco')
_REFS = [os.path.join(REPO, 'results', 'rodada3', 'referencias', 'referencias.json'),
         os.path.join(REPO, 'Resultados', 'rodada3', 'referencias', 'referencias.json')]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
REG = ['in_dist', 'dry', 'humid']
NOME = {'in_dist': 'Normal', 'dry': 'Seco', 'humid': 'Úmido'}
INI, TAM, NCEL = P.INICIO_GRADE, P.TAM_CELULA, P.N_CELULAS
LIMIAR_REGA = 0.58
SEMENTES = P.sementes_validacao_rollback() + P.sementes_validacao_objetivo()      # 500–549


def episodio(politica, env, bruto, s, regime):
    obs, info = env.reset(seed=int(s))
    P.aplicar_regime(env, regime)
    obs = bruto._obter_observacao()
    info['mascara_acoes'] = env._gerar_mascara(obs) if hasattr(env, '_gerar_mascara') else info.get('mascara_acoes')
    politica.iniciar_episodio(int(s), bruto)
    contagem = np.zeros(P.N_ACOES, dtype=np.int64)
    comp, total, fim = {}, 0.0, False
    temp, chuva, umid_ar, umid, plantado, acao_t, qual = [], [], [], [], [], [], []
    while not fim:
        g = np.asarray(obs[INI:INI + NCEL * TAM], dtype=float).reshape(NCEL, TAM)
        com_planta = g[:, 0] > 0
        temp.append(float(obs[0]) * 40.0)
        umid_ar.append(float(obs[1]))
        chuva.append(float(obs[2]) * 50.0)
        plantado.append(bool(com_planta.any()))
        umid.append(float(g[com_planta, 3].mean()) if com_planta.any() else np.nan)
        qual.append(float(g[com_planta, 4].mean()) if com_planta.any() else np.nan)
        mascara = info.get('mascara_acoes', np.ones(P.N_ACOES, dtype=np.float32))
        acao = politica.agir(obs, mascara)
        acao_t.append(int(acao))
        obs, r, terminado, truncado, info = env.step(acao)
        total += float(r)
        contagem[acao] += 1
        for k, v in info.get('recompensa_comp', {}).items():
            comp[k] = comp.get(k, 0.0) + float(v)
        fim = terminado or truncado
    kp = info.get('kpis', {})
    temp, chuva, umid_ar, umid, qual = map(np.array, (temp, chuva, umid_ar, umid, qual))
    plantado, acao_t = np.array(plantado), np.array(acao_t)
    # penalidades do fator de crescimento da planta (environment/plant/plant.py): ×0,7 se T<10 ou T>35; ×0,8 se umidade do ar <0,3 ou >0,9; ×0,9 se chuva <2 mm
    pen_t, pen_u, pen_c = (temp < 10) | (temp > 35), (umid_ar < 0.3) | (umid_ar > 0.9), chuva < 2.0
    mult = np.where(pen_t, 0.7, 1.0) * np.where(pen_u, 0.8, 1.0) * np.where(pen_c, 0.9, 1.0)
    regou = acao_t == 2                                  # `regar`
    d = np.diff(umid)                                    # variação da umidade média das plantas de um dia para o seguinte
    valido = plantado[:-1] & plantado[1:] & ~np.isnan(d)
    sem_agua = valido & ~regou[:-1] & (chuva[:-1] <= 0)
    com_rega = valido & regou[:-1]
    com_planta = plantado & ~np.isnan(umid)
    return {
        'recompensa': total, 'comp': comp,
        'produtividade': float(kp.get('produtividade_acumulada', np.nan)),
        'regas': int(contagem[2]), 'colheitas': int(contagem[1]), 'plantios': int(contagem[0]), 'esperar': int(contagem[13]),
        'dias_com_planta': int(com_planta.sum()),
        'umidade_media_plantas': float(np.nanmean(umid[com_planta])) if com_planta.any() else np.nan,
        'frac_dias_abaixo_limiar': float((umid[com_planta] < LIMIAR_REGA).mean()) if com_planta.any() else np.nan,
        'delta_umidade_dia_sem_agua': float(d[sem_agua].mean()) if sem_agua.any() else np.nan,
        'delta_umidade_dia_com_rega': float(d[com_rega].mean()) if com_rega.any() else np.nan,
        'frac_dias_chuva': float((chuva > 0).mean()), 'chuva_media_mm': float(chuva.mean()),
        'temperatura_media': float(temp.mean()), 'umidade_ar_media': float(umid_ar.mean()),
        'dias_calor_com_planta': int((plantado & (temp > 35)).sum()),
        'frac_dias_umidade_ar_fora': float(pen_u[com_planta].mean()) if com_planta.any() else np.nan,
        'frac_dias_chuva_baixa': float(pen_c[com_planta].mean()) if com_planta.any() else np.nan,
        'frac_dias_temperatura_fora': float(pen_t[com_planta].mean()) if com_planta.any() else np.nan,
        'multiplicador_clima_medio': float(mult[com_planta].mean()) if com_planta.any() else np.nan,
        'qualidade_solo_plantas': float(np.nanmean(qual[com_planta])) if com_planta.any() else np.nan,
        'multiplicador_qualidade_medio': float((1.0 + qual[com_planta]).mean()) if com_planta.any() else np.nan,
    }


def rodar(nome, politica, env):
    bruto = P.env_bruto(env)
    out = {}
    with P.rng_isolado():
        for reg in REG:
            out[reg] = [episodio(politica, env, bruto, s, reg) for s in SEMENTES]
            P.restaurar_regime(env)
            log(f'  {nome} {reg}: {len(out[reg])} episódios | recompensa média {np.mean([e["recompensa"] for e in out[reg]]):.1f}')
    return out


def medias(eps, chave, sub=None):
    v = [(e[chave][sub] if sub else e[chave]) for e in eps]
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1)), v


def main():
    env = novo_env()
    resultados = {'professor': rodar('professor', A.PoliticaOraculo(P.novo_professor()), env),
                  'aleatorio': rodar('aleatório', A.PoliticaAleatoria(0), env)}
    os.makedirs(SAIDA, exist_ok=True)
    json.dump({'sementes': SEMENTES, 'limiar_rega': LIMIAR_REGA, 'resultados': resultados},
              open(os.path.join(SAIDA, 'episodios.json'), 'w', encoding='utf-8'), ensure_ascii=False)

    ref = json.load(open(next(p for p in _REFS if os.path.exists(p)), encoding='utf-8'))['oraculo_perfeito']
    L = ['# Diagnóstico do regime seco: por que o professor piora tanto?\n',
         'Professor heurístico nos três regimes, nas sementes de **validação** 500–549 (50 episódios; teste nunca usado). '
         'Valores: média ± desvio-padrão entre episódios.\n']
    L.append('## 0. Conferência do laço contra a avaliação oficial (professor, recompensa média)\n')
    L.append('| Regime | Este diagnóstico (500–549) | `referencias.json` (500–519 e 520–549, ponderado) |')
    L.append('|---|---|---|')
    for r in REG:
        mine = np.mean([e['recompensa'] for e in resultados['professor'][r]])
        oficial = (20 * ref['validacao_rollback'][r]['media'] + 30 * ref['validacao_objetivo'][r]['media']) / 50
        L.append(f'| {NOME[r]} | {mine:.1f} | {oficial:.1f} |')

    prof = resultados['professor']
    linhas = [('Recompensa total', 'recompensa', None, '{:.0f}'),
              ('Produtividade acumulada (peso colhido)', 'produtividade', None, '{:.1f}'),
              ('Colheitas (ações `colher`)', 'colheitas', None, '{:.1f}'),
              ('Regas (ações `regar`)', 'regas', None, '{:.0f}'),
              ('Dias com planta', 'dias_com_planta', None, '{:.0f}'),
              ('Umidade média do solo nas plantas', 'umidade_media_plantas', None, '{:.3f}'),
              ('Fração dos dias com planta abaixo de 0,58', 'frac_dias_abaixo_limiar', None, '{:.2f}'),
              ('Variação diária da umidade em dia sem rega e sem chuva', 'delta_umidade_dia_sem_agua', None, '{:+.3f}'),
              ('Variação diária da umidade em dia de rega', 'delta_umidade_dia_com_rega', None, '{:+.3f}'),
              ('Fração dos dias com chuva', 'frac_dias_chuva', None, '{:.2f}'),
              ('Chuva média (mm/dia)', 'chuva_media_mm', None, '{:.1f}'),
              ('Temperatura média (°C)', 'temperatura_media', None, '{:.1f}'),
              ('Dias com planta e calor acima de 35 °C', 'dias_calor_com_planta', None, '{:.0f}'),
              ('Fração dos dias com planta e umidade do ar fora de [0,3; 0,9] (×0,8 no crescimento)', 'frac_dias_umidade_ar_fora', None, '{:.2f}'),
              ('Fração dos dias com planta e chuva abaixo de 2 mm (×0,9 no crescimento)', 'frac_dias_chuva_baixa', None, '{:.2f}'),
              ('Fração dos dias com planta e temperatura fora de [10; 35] °C (×0,7 no crescimento)', 'frac_dias_temperatura_fora', None, '{:.2f}'),
              ('Multiplicador médio do clima sobre o crescimento (produto das três penalidades)', 'multiplicador_clima_medio', None, '{:.2f}'),
              ('Qualidade média do solo nas plantas', 'qualidade_solo_plantas', None, '{:.3f}'),
              ('Multiplicador médio da qualidade do solo, 1 + qualidade', 'multiplicador_qualidade_medio', None, '{:.3f}')]
    L.append('\n## 1. Professor, por regime\n')
    L.append('| Medida | Normal | Seco | Úmido |')
    L.append('|---|---|---|---|')
    tab = {}
    for rot, ch, sub, fmt in linhas:
        cel = []
        for r in REG:
            m, sd, _ = medias(prof[r], ch, sub)
            cel.append((fmt + ' ± ' + fmt.replace('+', '')).format(m, sd))
            tab.setdefault(ch, {})[r] = m
        L.append(f'| {rot} | ' + ' | '.join(cel) + ' |')

    comps = ['colheita', 'custo_acao', 'clima', 'ervas', 'doenca', 'eventos', 'frutos_pendentes', 'solo_final', 'polinizadores_final', 'quimicos_final']
    L.append('\n## 2. Decomposição da recompensa do professor (média por episódio) e diferença do seco para o normal\n')
    L.append('| Componente | Normal | Seco | Seco − Normal | Parcela da queda total |')
    L.append('|---|---|---|---|---|')
    tot = {r: np.mean([e['recompensa'] for e in prof[r]]) for r in REG}
    queda = tot['dry'] - tot['in_dist']
    decomp = {}
    for c in comps:
        vals = {r: np.mean([e['comp'].get(c, 0.0) for e in prof[r]]) for r in REG}
        d = vals['dry'] - vals['in_dist']
        decomp[c] = {'normal': vals['in_dist'], 'seco': vals['dry'], 'dif': d}
        L.append(f"| {c} | {vals['in_dist']:.1f} | {vals['dry']:.1f} | {d:+.1f} | {100 * d / queda:.0f}% |")
    L.append(f"| **Total** | {tot['in_dist']:.1f} | {tot['dry']:.1f} | {queda:+.1f} | 100% |")

    L.append('\n## 3. Testes pareados por episódio (mesmas sementes), seco contra normal\n')
    L.append('| Medida | Diferença média (seco − normal) | t pareado p |')
    L.append('|---|---|---|')
    testes = {}
    for rot, ch, sub, fmt in linhas:
        a = np.array(medias(prof['in_dist'], ch)[2], float)
        b = np.array(medias(prof['dry'], ch)[2], float)
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() > 2:
            t = stats.ttest_rel(b[ok], a[ok])
            testes[ch] = {'dif': float((b[ok] - a[ok]).mean()), 'p': float(t.pvalue)}
            L.append(f"| {rot} | {(b[ok] - a[ok]).mean():+.3f} | {t.pvalue:.3g} |")

    L.append('\n## 4. Agente aleatório (referência), recompensa média por regime\n')
    L.append('| Regime | Recompensa | Regas |')
    L.append('|---|---|---|')
    for r in REG:
        L.append(f"| {NOME[r]} | {np.mean([e['recompensa'] for e in resultados['aleatorio'][r]]):.0f} | {np.mean([e['regas'] for e in resultados['aleatorio'][r]]):.0f} |")
    open(os.path.join(SAIDA, 'RESUMO_SECO.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    json.dump({'medias': tab, 'decomposicao': decomp, 'testes_pareados': testes, 'recompensa_total': {r: float(v) for r, v in tot.items()}},
              open(os.path.join(SAIDA, 'numeros_seco.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    sys.exit(main())
