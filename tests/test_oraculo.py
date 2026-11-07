"""Testes do oráculo (Bot Ajustável): o professor é o manejo intensivo (v5, D-33) e não tem pausas (D-27); o rótulo dele
não depende de memória própria; o professor da rodada 2 (v4) continua construível; pausas reproduzíveis por episódio."""
import numpy as np
import pytest

import protocolo as P


def _acoes_do_bot(env, bot, seed_env, seed_episodio, n_passos=150):
    obs, info = env.reset(seed=seed_env)
    bruto = P.env_bruto(env)
    bot.resetar_episodio(seed_episodio=seed_episodio)
    acoes = []
    for _ in range(n_passos):
        a = bot.selecionar_acao(bruto)
        acoes.append(a)
        env.step(a)
    return acoes


def test_professor_e_o_manejo_intensivo_e_o_v4_continua_construivel():
    """D-33: trocar o professor é trocar `PROFESSOR_VERSAO`; a rodada 2 continua reproduzível por `novo_professor_v4()`."""
    assert P.PROFESSOR_VERSAO == 'v5'
    assert type(P.novo_professor()).__name__ == 'BotIntensivo'
    assert type(P.novo_professor_v4()).__name__ == 'BotPerfeito' and P.novo_professor_v4().imperfeicao == 0.0
    assert type(P.novo_oraculo_variante()) is type(P.novo_professor())   # a variante é o professor COM pausas
    with pytest.raises(ValueError):
        P.classe_do_professor('v9')


def test_professor_nunca_escolhe_acao_mascarada_nem_deixa_planta_pronta_morrer(env, bot_perfeito):
    """O v5 colhe em LOTE, então pode haver planta pronta sem que a ação seja `colher` — mas nenhuma morre esperando."""
    from plant import EstagioPlanta
    obs, info = env.reset(seed=3)
    bruto = P.env_bruto(env)
    bot_perfeito.resetar_episodio(seed_episodio=3)
    plantios = colhidas = esperas_com_pronta = 0
    for _ in range(P.PASSOS_MAXIMOS):
        prontas = sum(1 for p in bruto.planta.plantas.values() if p.estagio == EstagioPlanta.COLHEITA)
        a = bot_perfeito.selecionar_acao(bruto)
        assert info['mascara_acoes'][a] > 0.5, P.NOMES_ACOES[a]
        assert not (a == P.ACAO_ESPERAR and prontas)             # com planta pronta e nada a fazer, colhe
        plantios += int(a == P.ACAO_PLANTAR)
        colhidas += prontas if a == P.ACAO_COLHER else 0
        esperas_com_pronta += int(prontas > 0 and a != P.ACAO_COLHER)
        obs, r, te, tr, info = env.step(a)
    assert esperas_com_pronta > 0                                 # a colheita em lote aconteceu
    assert colhidas >= 30                                         # ≈ 3 ciclos de 16 células (o v4 faz ≈ 32 plantios)
    assert plantios == colhidas + len(bruto.planta.plantas)       # toda planta foi colhida ou está viva no fim: nenhuma morreu


def test_rotulo_do_professor_nao_depende_da_memoria_do_bot(env):
    """No DAgger quem executa a ação pode ser o aluno. Um rotulador que lembra 'apliquei herbicida no passo t' rotula
    errado a partir da primeira divergência; o v5 lê a carência do ESTADO do ambiente. Um bot recém-criado, sem história
    nenhuma, tem de dar o mesmo rótulo que o bot que conduziu o episódio, em todo passo."""
    obs, info = env.reset(seed=11)
    bruto = P.env_bruto(env)
    condutor = P.novo_professor(); condutor.resetar_episodio(seed_episodio=11)
    for t in range(200):
        a = condutor.selecionar_acao(bruto)
        novato = P.novo_professor(); novato.resetar_episodio(seed_episodio=11)
        assert novato.selecionar_acao(bruto) == a, f'passo {t}'
        env.step(a)


@pytest.mark.lento
def test_professor_v5_supera_o_v4_nos_tres_regimes(env):
    """D-33, em sementes de CALIBRAGEM (20000+): nunca validação nem teste."""
    import avaliacao as A
    for regime in P.NOMES_REGIMES:
        v5 = np.mean(A.avaliar(A.PoliticaOraculo(P.novo_professor()), env, [20000, 20001], regime)['recompensas'])
        v4 = np.mean(A.avaliar(A.PoliticaOraculo(P.novo_professor_v4()), env, [20000, 20001], regime)['recompensas'])
        assert v5 > 1.15 * v4, (regime, v5, v4)


def test_bot_v4_colhe_sempre_que_ha_planta_colhivel(env):
    """Propriedade do professor da rodada 2 (manejo reativo): colher tem prioridade máxima, sem lote."""
    from plant import EstagioPlanta
    bot_perfeito = P.novo_professor_v4()
    obs, info = env.reset(seed=3)
    bruto = P.env_bruto(env)
    bot_perfeito.resetar_episodio(seed_episodio=3)
    colheitas = 0
    for _ in range(P.PASSOS_MAXIMOS):
        ha_colhivel = any(p.estagio == EstagioPlanta.COLHEITA for p in bruto.planta.plantas.values())
        a = bot_perfeito.selecionar_acao(bruto)
        if ha_colhivel:
            assert a == P.ACAO_COLHER
            colheitas += 1
        assert a != P.ACAO_ESPERAR or not ha_colhivel
        env.step(a)
    assert colheitas >= 10


def test_professor_dos_experimentos_nao_tem_pausas(env):
    """D-27: o professor (demonstrações, DAgger, âncora) é o oráculo sem pausas. Guarda contra regressão silenciosa."""
    assert P.PROFESSOR_IMPERFEICAO == 0.0
    prof = P.novo_professor()
    assert prof.imperfeicao == 0.0 and prof.lazy_steps == P.ORACULO_LAZY_STEPS
    assert P.novo_oraculo_variante().imperfeicao == P.ORACULO_IMPERFEICAO_VARIANTE > 0
    # O professor nunca entra em pausa: comparado com a variante nas MESMAS sementes, ele age em todo passo em que a
    # hierarquia manda agir. A asserção é a diferença entre os dois, não uma propriedade que qualquer bot satisfaria.
    var = P.novo_oraculo_variante()
    for s in (3, 11, 500):
        a_prof = _acoes_do_bot(env, prof, seed_env=s, seed_episodio=s, n_passos=200)
        a_var = _acoes_do_bot(env, var, seed_env=s, seed_episodio=s, n_passos=200)
        assert a_prof != a_var                                  # a variante pausa; o professor não
        esperas_prof = sum(1 for x in a_prof if x == P.ACAO_ESPERAR)
        esperas_var = sum(1 for x in a_var if x == P.ACAO_ESPERAR)
        assert esperas_prof < esperas_var                       # e por isso espera menos
    import inspect
    import gerar_demos
    import treinar_warmstart_il
    # o gerador de demos usa o professor por padrão e o warm-start constrói o rotulador do DAgger pelo protocolo
    assert 'P.PROFESSOR_IMPERFEICAO' in inspect.getsource(gerar_demos.main)
    assert 'P.novo_professor()' in inspect.getsource(treinar_warmstart_il.treinar_semente)


def test_pausas_reproduziveis_por_episodio(env, bot_subotimo):
    a1 = _acoes_do_bot(env, bot_subotimo, seed_env=7, seed_episodio=7)
    a2 = _acoes_do_bot(env, bot_subotimo, seed_env=7, seed_episodio=7)
    assert a1 == a2
    a3 = _acoes_do_bot(env, bot_subotimo, seed_env=7, seed_episodio=8)
    assert a1 != a3            # outra semente de episódio ⇒ outro padrão de pausas
    assert P.ACAO_ESPERAR in a1


def test_semente_de_pausas_segue_o_protocolo(bot_subotimo):
    bot_subotimo.resetar_episodio(seed_episodio=5)
    esperado = np.random.RandomState(P.seed_pausas_oraculo(5)).random()
    assert bot_subotimo._rand() == esperado


def test_bot_subotimo_espera_mais_que_o_perfeito(env, bot_perfeito, bot_subotimo):
    ap = _acoes_do_bot(env, bot_perfeito, seed_env=9, seed_episodio=9, n_passos=365)
    asub = _acoes_do_bot(env, bot_subotimo, seed_env=9, seed_episodio=9, n_passos=365)
    assert asub.count(P.ACAO_ESPERAR) > ap.count(P.ACAO_ESPERAR)
