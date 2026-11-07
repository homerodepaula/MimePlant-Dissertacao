"""Testes de mecanismo do FarmGym (v3 e v4, 06/09/2026): cada subsistema que existe no codigo tem de agir.
Os testes marcados v4 falhariam no v3, e os marcados v3 falhariam no v2."""
import numpy as np
import pytest

import protocolo as P
import avaliacao as A
from conftest import rodar_episodio

PEST, ESP_BASICO, ESP_AV, REMOVER, CERCA = (P.NOMES_ACOES.index(n) for n in
    ('pesticida', 'colocar_espantalho_basico', 'colocar_espantalho_avancado', 'remover_espantalho', 'colocar_cerca'))


def _episodios_oraculo(env, bot, sementes, regime='in_dist'):
    return A.avaliar(A.PoliticaOraculo(bot), env, sementes, regime)


# ── doenca ──────────────────────────────────────────────────────────────────────────────────────
def test_doenca_ocorre_e_o_termo_da_recompensa_nao_e_sempre_zero(env, bot_perfeito):
    res = _episodios_oraculo(env, bot_perfeito, P.sementes_validacao_rollback()[:8])
    assert 'doenca' in res['comp']
    assert any(d < 0 for d in res['comp']['doenca'])                       # em algum episodio a doenca custou
    assert any(k['impacto_doenca_medio'] > 0 for k in res['kpis_brutos']) or any(d < 0 for d in res['comp']['doenca'])


def test_pesticida_e_valida_com_plantas_e_trata_a_doenca(env):
    import sys, random
    obs, info = env.reset(seed=3)
    assert info['mascara_acoes'][PEST] == 0.0                # sem plantas
    obs, r, _, _, info = env.step(P.ACAO_PLANTAR)
    assert info['mascara_acoes'][PEST] == 1.0                # com planta
    bruto = P.env_bruto(env)
    # o Enum tem de vir do MESMO modulo que o ambiente usa (`pest.disease_system`); `from disease_system import ...`
    # carregaria uma segunda copia do modulo pelo sys.path e os estagios nao seriam iguais (criticas.md, secao 8.3)
    EstagioDoenca = sys.modules[type(bruto.sistema_doencas).__module__].EstagioDoenca
    pos = next(iter(bruto.planta.plantas))
    bruto.sistema_doencas.doencas_plantas[pos] = {'infeccao_fungica': EstagioDoenca.SEVERO}
    assert bruto.sistema_doencas.calcular_impacto_doenca(*pos) > 0.5
    # o pesticida chama aplicar_tratamento('amplo_espectro') em toda celula (60 % por aplicacao); com 20 aplicacoes o
    # estagio passa a RECUPERANDO com probabilidade 1 - 0,4**20
    random.seed(1)
    for _ in range(20):
        env.step(PEST)
    assert bruto.sistema_doencas.doencas_plantas.get(pos, {}).get('infeccao_fungica') in (EstagioDoenca.RECUPERANDO, None)
    assert bruto.sistema_doencas.calcular_impacto_doenca(*pos) < 0.5


# ── aves e espantalho ───────────────────────────────────────────────────────────────────────────
def test_aves_comem_fruto_e_o_espantalho_reduz_a_perda(env):
    """Mesmo episodio, mesma semente: com espantalhos em todas as celulas a perda de fruto para aves e menor."""
    def politica(colocar_espantalhos):
        def f(obs, mascara, bruto, t):
            if mascara[P.ACAO_COLHER] > 0.5:
                return P.ACAO_COLHER
            if mascara[P.ACAO_PLANTAR] > 0.5 and t < 250:
                return P.ACAO_PLANTAR
            if colocar_espantalhos and t < 60 and mascara[ESP_AV] > 0.5 and bruto.instalacoes.n_espantalhos() < 16:
                return ESP_AV
            if mascara[P.ACAO_REGAR] > 0.5 and t % 4 == 0:
                return P.ACAO_REGAR
            return P.ACAO_ESPERAR
        return f
    perdas = {}
    for com in (False, True):
        with P.rng_isolado():
            _, recs, infos = rodar_episodio(env, 11, politica(com))
        perdas[com] = infos[-1]['kpis']['fruto_perdido_aves']
    assert perdas[False] > 0.0                                   # sem espantalho, aves comem fruto
    assert perdas[True] < perdas[False]                          # com espantalhos, comem menos


def test_observacao_das_aves_e_a_populacao_que_o_espantalho_afasta(env):
    obs, info = env.reset(seed=5)
    bruto = P.env_bruto(env)
    for i in range(16):
        x, y = divmod(i, 4)
        assert abs(obs[P.INICIO_GRADE + P.TAM_CELULA * i + 7] - min(1.0, bruto.passaros.obter_populacao_em(x, y) / 50.0)) < 1e-6


# ── eventos extremos ────────────────────────────────────────────────────────────────────────────
def test_eventos_extremos_custam_quando_o_agente_age_durante_o_evento(env):
    obs, info = env.reset(seed=8)
    bruto = P.env_bruto(env)
    bruto.eventos_extremos.eventos_ativos.append({'tipo': 'tempestade', 'passo_inicio': 0, 'duracao': 10 ** 6, 'severidade': 1.0,
                                                  'impactos': dict(bruto.eventos_extremos.tipos_evento['tempestade']['impactos'])})
    obs, r, _, _, info = env.step(P.ACAO_PLANTAR)
    assert info['recompensa_comp']['eventos'] == pytest.approx(-7.0)     # plantar na tempestade
    obs, r, _, _, info = env.step(P.ACAO_ESPERAR)
    assert info['recompensa_comp']['eventos'] == 0.0                     # esperar nao tem categoria


def test_eventos_extremos_expiram_ao_fim_da_duracao(env):
    """Ate 06/09 o modulo lia a chave 'passo' (o ambiente escreve 'step'): passo_inicio era 0 e nada expirava."""
    obs, info = env.reset(seed=8)
    bruto = P.env_bruto(env)
    ev = bruto.eventos_extremos
    injetado = {'tipo': 'tempestade', 'passo_inicio': bruto.passo_atual, 'duracao': 2, 'severidade': 1.0,
                'impactos': dict(ev.tipos_evento['tempestade']['impactos'])}
    ev.eventos_ativos.append(injetado)
    # a dinamica roda antes de `passo_atual += 1`: o evento ve os passos 0, 1 (0 - 0 < 2 e 1 - 0 < 2: ativo) e expira no 2
    env.step(P.ACAO_ESPERAR); env.step(P.ACAO_ESPERAR)
    assert injetado in ev.eventos_ativos
    env.step(P.ACAO_ESPERAR)
    assert injetado not in ev.eventos_ativos


def test_eventos_extremos_acontecem_em_algum_regime(env, bot_perfeito):
    vistos = 0
    for regime in P.NOMES_REGIMES:
        res = _episodios_oraculo(env, bot_perfeito, P.sementes_validacao_rollback()[:6], regime)
        vistos += sum(1 for e in res['comp'].get('eventos', []) if e < 0)
    assert vistos > 0


# ── fertilizante, pragas e polinizadores ────────────────────────────────────────────────────────
def test_fertilizante_afeta_o_peso_do_fruto_pela_lei_do_minimo():
    from plant import EstadoPlanta, EstagioPlanta
    p1, p2 = EstadoPlanta(), EstadoPlanta()
    for p in (p1, p2):
        p.estagio = EstagioPlanta.FRUTIFICACAO
    for _ in range(10):
        p1.atualizar(1.0, fator_nutriente=1.0)
        p2.atualizar(1.0, fator_nutriente=0.6)
    assert p1.peso_fruto > p2.peso_fruto
    assert p2.peso_fruto == pytest.approx(0.6 * p1.peso_fruto)


def test_polinizacao_no_florescimento_define_o_peso_do_fruto():
    """Duas plantas iguais em florescimento: com polinizadores 1,0 vs 0,0 o fruto sai com 100 % vs 60 % do peso (POLINIZACAO_PISO)."""
    from plant import EstadoPlanta, EstagioPlanta, POLINIZACAO_PISO
    p1, p2 = EstadoPlanta(), EstadoPlanta()
    for p in (p1, p2):
        p.estagio = EstagioPlanta.FLORESCIMENTO
    for _ in range(60):                       # 30 passos de florescimento (limiar 15 a 0,5/passo) + 30 de frutificacao
        p1.atualizar(1.0, 1.0, densidade_polinizadores=1.0)
        p2.atualizar(1.0, 1.0, densidade_polinizadores=0.0)
    assert p1.estagio == p2.estagio == EstagioPlanta.FRUTIFICACAO
    assert p1.peso_fruto > 0 and p2.peso_fruto == pytest.approx(POLINIZACAO_PISO * p1.peso_fruto)


def test_pragas_e_doenca_retardam_o_crescimento():
    from plant import Planta
    pl = Planta((4, 4)); pl.adicionar_planta(0, 0)
    clima = {'temperatura': 20, 'umidade_ar': 0.5, 'chuva': 5.0}
    solo = {(0, 0): {'qualidade': 0.5, 'nutrientes': {'N': 0.3, 'P': 0.3, 'K': 0.3, 'C': 0.3}}}
    f0 = pl._calcular_fator_crescimento(clima, solo, {}, {}, (0, 0), 0.0, 0.0)
    f_pragas = pl._calcular_fator_crescimento(clima, solo, {}, {}, (0, 0), 50.0, 0.0)
    f_doenca = pl._calcular_fator_crescimento(clima, solo, {}, {}, (0, 0), 0.0, 0.5)
    assert f_pragas < f0 and f_doenca < f0


def test_polinizadores_nao_colapsam_sob_o_oraculo(env, bot_perfeito):
    res = _episodios_oraculo(env, bot_perfeito, P.sementes_validacao_rollback()[:5])
    assert np.mean(res['kpis']['biodiversidade']) > 0.1     # v2: 0,00 em todas as referencias


def test_espantalho_e_cerca_coexistem_na_mesma_celula(env):
    obs, info = env.reset(seed=2)
    bruto = P.env_bruto(env)
    env.step(P.ACAO_PLANTAR)
    env.step(ESP_BASICO)
    assert bruto.instalacoes.n_espantalhos() == 1
    env.step(CERCA)
    assert bruto.instalacoes.n_espantalhos() == 1 and bruto.instalacoes.n_cercas() == 16   # a cerca nao apagou o espantalho


# ── observacao ──────────────────────────────────────────────────────────────────────────────────
def test_nenhuma_dimensao_da_observacao_e_constante_nos_episodios_do_oraculo(env, bot_perfeito):
    """v2: 24 dimensoes constantes por construcao (5 + 16 de doenca, 3 de eventos). Aqui: nenhuma, nos 3 regimes."""
    obs_all = []
    with P.rng_isolado():
        for regime in P.NOMES_REGIMES:
            for s in P.sementes_validacao_rollback()[:6]:
                obs, info = env.reset(seed=s)
                P.aplicar_regime(env, regime)
                bruto = P.env_bruto(env)
                obs = bruto._obter_observacao()
                bot_perfeito.resetar_episodio(seed_episodio=s)
                for t in range(P.PASSOS_MAXIMOS):
                    obs_all.append(obs.copy())
                    obs, r, te, tr, info = env.step(bot_perfeito.selecionar_acao(bruto))
            P.restaurar_regime(env)
    O = np.array(obs_all)
    constantes = np.where(O.std(0) < 1e-9)[0].tolist()
    assert constantes == [], f'dimensoes constantes: {constantes}'


# ── FarmGym v4 (06/09/2026): os defeitos que a auditoria de 12 frentes encontrou ────────────────
def test_layout_da_observacao_tem_uma_fonte_so(env):
    """O layout estava redigitado em quatro arquivos e dois falhavam EM SILÊNCIO: o wrapper do RL fazia reshape de um
    vetor de tamanho errado lendo campos trocados, e este arquivo de teste assertava no campo errado."""
    from rede_lstm import OBS_DIM as OBS_REDE, N_ACOES as ACOES_REDE
    from agents.rl import wrapper
    obs, info = env.reset(seed=3)
    assert P.OBS_DIM == OBS_REDE == env.observation_space.shape[0] == len(obs) == 176
    assert P.OBS_DIM == P.INICIO_GRADE + P.TAM_CELULA * P.N_CELULAS
    assert wrapper.INICIO_GRADE == P.INICIO_GRADE and wrapper.TAM_CELULA == P.TAM_CELULA
    assert P.N_ACOES == ACOES_REDE == env.action_space.n == len(P.NOMES_ACOES) == 14
    assert P.NOMES_ACOES[P.ACAO_ESPERAR] == 'esperar' and 'observar' not in P.NOMES_ACOES


def test_praga_esta_na_observacao_por_celula_e_no_agregado(env):
    """v3: a praga não aparecia em nenhuma das 148 dimensões — o agente era punido por algo que não via."""
    obs, info = env.reset(seed=5)
    bruto = P.env_bruto(env)
    for t in range(40):
        obs, r, te, tr, info = env.step(P.ACAO_PLANTAR if info['mascara_acoes'][P.ACAO_PLANTAR] > 0.5 else P.ACAO_ESPERAR)
    for i in range(16):
        x, y = divmod(i, 4)
        esperado = min(1.0, bruto.pragas.obter_populacao_em(x, y) / 100.0)
        assert abs(obs[P.INICIO_GRADE + P.TAM_CELULA * i + 8] - esperado) < 1e-6
    plantadas = list(bruto.planta.plantas)
    esperado = min(1.0, max((bruto.pragas.obter_populacao_em(*p) for p in plantadas), default=0.0) / 100.0)
    assert abs(obs[14] - esperado) < 1e-6


def test_populacao_de_pragas_cresce_e_depende_do_hospedeiro(env):
    """v3: `int(mudanca)` truncava a variação para zero em toda população alcançável, e a célula vazia crescia como a
    célula com planta. Prova do defeito: 'só esperar' terminava com a mesma praga de quem cultivava o campo inteiro."""
    def total_apos(politica, n=90, seed=7):
        with P.rng_isolado():
            obs, info = env.reset(seed=seed)
            bruto = P.env_bruto(env)
            for t in range(n):
                obs, r, te, tr, info = env.step(politica(info['mascara_acoes']))
            return bruto.pragas.obter_populacao_total() / 16.0
    com_lavoura = total_apos(lambda m: P.ACAO_PLANTAR if m[P.ACAO_PLANTAR] > 0.5 else P.ACAO_ESPERAR)
    sem_lavoura = total_apos(lambda m: P.ACAO_ESPERAR)
    assert com_lavoura > 25.0                    # a população cresce de verdade
    assert com_lavoura >= 3.0 * sem_lavoura      # e segue a lavoura, não o calendário


def test_doenca_e_apagada_quando_a_planta_sai(env):
    """v3: nada limpava `doencas_plantas`; a observação mostrava doença em célula vazia e o replantio herdava a
    infecção no mesmo passo."""
    import sys
    obs, info = env.reset(seed=11)
    bruto = P.env_bruto(env)
    env.step(P.ACAO_PLANTAR)
    pos = next(iter(bruto.planta.plantas))
    EstagioDoenca = sys.modules[type(bruto.sistema_doencas).__module__].EstagioDoenca
    bruto.sistema_doencas.doencas_plantas[pos] = {'infeccao_fungica': EstagioDoenca.SEVERO}
    assert bruto.sistema_doencas.calcular_impacto_doenca(*pos) > 0.5
    del bruto.planta.plantas[pos]                       # colheita/morte
    env.step(P.ACAO_ESPERAR)
    assert bruto.sistema_doencas.calcular_impacto_doenca(*pos) == 0.0
    i = P.INICIO_GRADE + P.TAM_CELULA * (pos[0] * 4 + pos[1])
    obs = bruto._obter_observacao()
    assert obs[i + 0] == 0.0 and obs[i + 6] == 0.0      # célula vazia e sem doença na observação


def test_defensivo_nao_cura_virose(env):
    """v4: a eficácia passa a ser por patógeno. Na v3 o defensivo curava 60 % de qualquer doença, inclusive o mosaico
    viral — que é a que mais ocorre no regime de treino —, o que fazia do pesticida uma bala de prata."""
    import sys, random
    obs, info = env.reset(seed=13)
    bruto = P.env_bruto(env)
    env.step(P.ACAO_PLANTAR)
    D = sys.modules[type(bruto.sistema_doencas).__module__]
    assert D.SistemaDoencas((4, 4)).tipos_doenca['mosaico_viral'].controle_quimico == 0.0
    pos = next(iter(bruto.planta.plantas))
    bruto.sistema_doencas.doencas_plantas[pos] = {'mosaico_viral': D.EstagioDoenca.SEVERO}
    random.seed(0)
    for _ in range(20):
        bruto.sistema_doencas.aplicar_tratamento(*pos, 'amplo_espectro')
    assert bruto.sistema_doencas.doencas_plantas[pos]['mosaico_viral'] == D.EstagioDoenca.SEVERO


def test_nenhuma_acao_paga_recompensa_por_passo_fora_da_colheita(env):
    """v3: `regar` durante tempestade pagava +2,0 × severidade — o único termo por passo estritamente positivo fora da
    colheita, e portanto um atrator para o RL."""
    for evento in ('tempestade', 'seca', 'estresse_calor', 'estresse_frio'):
        obs, info = env.reset(seed=17)
        bruto = P.env_bruto(env)
        ev = bruto.eventos_extremos
        ev.eventos_ativos.append({'tipo': evento, 'passo_inicio': bruto.passo_atual, 'duracao': 10 ** 6,
                                  'severidade': 1.0, 'impactos': dict(ev.tipos_evento[evento]['impactos'])})
        for a in range(P.N_ACOES):
            if info['mascara_acoes'][a] < 0.5:
                continue
            obs, r, te, tr, info = env.step(a)
            comp = info['recompensa_comp']
            assert comp['custo_acao'] + comp['eventos'] <= 1e-9, (evento, P.NOMES_ACOES[a], comp)
            if comp['colheita'] == 0:
                assert r <= 1e-6 + comp['colheita'], (evento, P.NOMES_ACOES[a], r)


def test_estresse_reduz_o_rendimento_e_nao_so_a_duracao():
    """v3: o peso do fruto integrava sempre para ~25 × nutriente × polinização, então clima, solo, ervas, doença e
    pragas mudavam só o relógio do ciclo. Agora a média do fator na janela de enchimento multiplica o peso."""
    from plant import EstadoPlanta, EstagioPlanta
    def peso(fator):
        p = EstadoPlanta(); p.estagio = EstagioPlanta.FRUTIFICACAO
        for _ in range(400):
            p.atualizar(fator, 1.0, densidade_polinizadores=1.0)
            if p.estagio != EstagioPlanta.FRUTIFICACAO:
                break
        return p.peso_fruto
    assert peso(1.4) >= 1.8 * peso(0.7)


def test_regar_e_invalida_sem_planta(env):
    """v3: `regar` custava 0,5 e não fazia nada quando não havia planta — contra a regra declarada na própria máscara."""
    obs, info = env.reset(seed=19)
    assert info['mascara_acoes'][P.ACAO_REGAR] == 0.0
    obs, r, te, tr, info = env.step(P.ACAO_PLANTAR)
    assert info['mascara_acoes'][P.ACAO_REGAR] == 1.0
