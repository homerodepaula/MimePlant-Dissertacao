"""Testes do núcleo do DEHB (`hpo/dehb_nucleo.py`): cada regra reproduzida da referência 0.1.2 e cada desvio deliberado.
Os números dourados das escadas foram conferidos contra o código do pacote (dehb.py:151-176, 436-444), lido e não instalado."""
import os
import sys
import json

import numpy as np
import pytest

import protocolo as P

sys.path.insert(0, os.path.join(P.RAIZ, 'hpo'))
import dehb_nucleo as D  # noqa: E402

P.preparar_sys_path()
from hp import ESPACO_MIMEPLANT  # noqa: E402


def esfera(v):
    return float(np.sum((np.asarray(v) - 0.3) ** 2))


def novo(L=2, semente=7, dim=12):
    return D.Dehb(dim, D.escada_de_passos(1_000_000, L), semente=semente)


def rodar(d, n, k=1, semente_ordem=0, f=esfera):
    """n avaliações com k em voo; a ordem de chegada é sorteada (semeada). Devolve os eventos, na ordem em que ocorreram."""
    rng, voo, eventos = np.random.default_rng(semente_ordem), [], []
    while len(d.historico) < n:
        while len(voo) < k and d._proximo_id < n:
            t = d.pedir()
            voo.append(t)
            eventos.append(('pedir', t.registro(), d.estado_do_gerador()))
        t = voo.pop(int(rng.integers(len(voo))))
        d.informar(t.id, f(t.vetor))
        eventos.append(('informar', t.id, t.aptidao))
    return eventos


# ── escada ───────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('L, brackets, pops, custo', [
    (2, [[3, 1], [2]], [3, 2], 4.0),
    (3, [[9, 3, 1], [3, 1], [3]], [9, 3, 3], 8.0),
    (4, [[27, 9, 3, 1], [9, 3, 1], [6, 2], [4]], [27, 9, 6, 4], 15.0),
])
def test_escada_dourada_da_referencia(L, brackets, pops, custo):
    assert [D.plano_do_bracket(i, L)[0] for i in range(L)] == brackets
    assert D.plano_do_bracket(L, L) == D.plano_do_bracket(0, L)                     # o ciclo tem período L
    assert [D.plano_do_bracket(i, L)[1] for i in range(L)] == [list(range(i, L)) for i in range(L)]   # as s+1 fidelidades mais ALTAS
    assert D.tamanhos_das_populacoes(L) == pops
    assert D.custo_de_uma_iteracao(D.escada_de_passos(3 ** (L + 5), L)) == pytest.approx(custo)


def test_escada_em_inteiros_e_a_do_podador_dos_outros_estudos():
    assert D.escada_de_passos(1_000_000, 2) == [333_333, 1_000_000]                 # = N // 3 e N (hpo_rl: min_resource = N // 3)
    assert D.escada_de_passos(1_000_000, 3) == [111_111, 333_333, 1_000_000]
    assert D.escada_de_passos(1460, 2) == [486, 1460]                                # smoke
    for ruim in ((1_000_000, 1, 3), (10, 5, 3), (1_000_000, 2, 1)):
        with pytest.raises(ValueError):
            D.escada_de_passos(*ruim)
    with pytest.raises(ValueError):
        D.Dehb(3, [1000, 1000])
    with pytest.raises(ValueError):
        D.Dehb(3, [3000, 1000])


# ── agendamento ──────────────────────────────────────────────────────────────────────────────────
def test_degrau_so_abre_com_todos_os_resultados_do_anterior_e_um_bracket_novo_e_aberto_na_espera():
    d = novo(L=2)
    t = [d.pedir() for _ in range(3)]
    assert [(x.bracket, x.degrau, x.nivel) for x in t] == [(0, 0, 0)] * 3
    quarto = d.pedir()                                                               # bracket 0 espera os 3 resultados: abre o bracket 1
    assert (quarto.bracket, quarto.degrau, quarto.nivel) == (1, 0, 1)
    d.informar(t[0].id, 0.5); d.informar(t[1].id, 0.4)
    assert d.brackets[0].degrau_agendavel() is None                                  # ainda falta 1 resultado do degrau 0
    d.informar(t[2].id, 0.6)
    assert d.brackets[0].degrau_agendavel() == 1
    prox = d.pedir()                                                                 # brackets ativos têm prioridade, na ordem de criação
    assert (prox.bracket, prox.degrau, prox.nivel) == (0, 1, 1)


@pytest.mark.parametrize('L, k', [(2, 1), (2, 3), (3, 4), (2, 6)])
def test_nunca_agenda_mais_que_o_plano_e_o_rodizio_do_pai_cobre_a_subpopulacao(L, k):
    d = novo(L=L)
    rodar(d, 60, k=k)
    for b in d.brackets:
        assert all(a <= n for a, n in zip(b.agendados, b.ns)) and all(c <= a for c, a in zip(b.concluidos, b.agendados))
    for nivel, sub in enumerate(d.subpops):
        pais = [t.id_pai for t in sorted(d.historico, key=lambda t: t.id) if t.nivel == nivel]
        assert pais == [i % len(sub.vetores) for i in range(len(pais))]              # rodízio, também nas promoções


# ── população inicial, promoção e evolução ───────────────────────────────────────────────────────
def test_populacao_inicial_nunca_e_avaliada_so_filhos():
    d = novo()
    iniciais = np.concatenate([s.vetores.copy() for s in d.subpops])
    rodar(d, 40, k=3)
    for t in d.historico:
        if t.origem == 'evolucao':
            assert not np.any(np.all(iniciais == t.vetor, axis=1))
    assert d.historico[0].origem == 'evolucao'                                       # não há fase de partida aleatória


def test_promocao_sobe_o_melhor_de_baixo_sem_mutar_e_sem_consumir_sorteio():
    d = novo(L=3)
    baixos = [d.pedir() for _ in range(9)]
    for i, t in enumerate(baixos):
        d.informar(t.id, float(10 - i) if i != 4 else 0.01)                          # o melhor é o 5º pedido
    antes = json.dumps(d.estado_do_gerador(), sort_keys=True)
    promovidos = [d.pedir() for _ in range(3)]
    assert json.dumps(d.estado_do_gerador(), sort_keys=True) == antes               # promoção não sorteia nada
    assert [t.origem for t in promovidos] == ['promocao'] * 3 and all(t.nivel == 1 for t in promovidos)
    assert np.array_equal(promovidos[0].vetor, baixos[4].vetor)                      # o melhor primeiro
    assert np.array_equal(promovidos[1].vetor, baixos[8].vetor) and np.array_equal(promovidos[2].vetor, baixos[7].vetor)
    assert [t.id_pai for t in promovidos] == [0, 1, 2]                               # a promoção também consome o rodízio


def test_promocao_nao_some_quando_um_resultado_atrasa_com_tres_em_voo():
    """Desvio 1. No pacote a promoção depende do contador GLOBAL de brackets: com 3 em voo, um resultado atrasado do bracket 0
    deixa o contador passar de L antes de o degrau alto abrir, e a promoção some conforme a ordem de chegada."""
    d = novo(L=2)
    b0 = [d.pedir() for _ in range(3)]
    d.informar(b0[0].id, 0.3); d.informar(b0[1].id, 0.2)
    outros = [d.pedir() for _ in range(6)]                                           # o terceiro do bracket 0 atrasa: abrem-se brackets novos
    assert max(t.bracket for t in outros) >= 2
    for t in outros:
        if t.id in d.pendentes and t.bracket != 0:
            d.informar(t.id, 0.9)
    d.informar(b0[2].id, 0.25)
    while True:
        t = d.pedir()
        if t.bracket == 0:
            break
        d.informar(t.id, 0.9)
    assert t.origem == 'promocao' and np.array_equal(t.vetor, b0[1].vetor)


def test_sem_nenhum_avaliado_com_sucesso_em_baixo_a_promocao_vira_evolucao():
    d = novo(L=2)
    for t in [d.pedir() for _ in range(3)]:
        d.informar(t.id, None)                                                       # as três falharam
    t = d.pedir()
    assert (t.bracket, t.degrau, t.origem) == (0, 1, 'evolucao')                     # o pacote levanta IndexError aqui (dehb.py:559)


def test_pool_de_pais_no_regime_de_dois_niveis():
    """Populações 3 e 2: em cima o pool vindo de baixo tem 1 indivíduo e recebe 3 mutantes da população global; em baixo o pool
    são os 3 da própria subpopulação, o alvo sai, e entra 1 vetor uniforme. É o caminho de borda do artigo virando o principal."""
    d = novo(L=2)
    rodar(d, 80, k=3)
    evol = [t for t in d.historico if t.origem == 'evolucao']
    altos = [t.pais for t in evol if t.nivel == 1 and t.degrau == 1]                 # degrau alto do bracket par: pool = 1 melhor de baixo
    assert altos and all(p['pool'] == 1 and p['global'] == 3 for p in altos)
    solo = [t.pais for t in evol if t.nivel == 1 and t.degrau == 0]                  # bracket ímpar: pool = os 2 da própria subpopulação
    assert solo and all(p['pool'] == 2 and p['global'] == 2 for p in solo)
    baixos = [t.pais for t in evol if t.nivel == 0]
    assert baixos and all(p == {'pool': 3, 'global': 0, 'uniforme': 1} for p in baixos)


def test_filho_tem_ao_menos_uma_dimensao_do_mutante_e_fica_no_hipercubo():
    d = D.Dehb(12, D.escada_de_passos(1_000_000, 2), semente=3, prob_cruzamento=0.0)  # CR = 0: só a dimensão forçada vem do mutante
    alvo = d.subpops[0].vetores[0].copy()
    filho, _ = d._evoluir(alvo, nivel_pool=0, n=3)
    assert int(np.sum(filho != alvo)) == 1
    d = D.Dehb(12, D.escada_de_passos(1_000_000, 2), semente=3, fator_mutacao=5.0)    # F grande: mutante quase sempre fora de [0, 1]
    rodar(d, 60, k=2)
    assert all(np.all((t.vetor >= 0) & (t.vetor <= 1)) for t in d.historico)
    assert any(t.origem == 'evolucao' and not np.any((t.vetor == 0) | (t.vetor == 1)) for t in d.historico)   # re-sorteio, não recorte


# ── seleção e falhas ─────────────────────────────────────────────────────────────────────────────
def test_selecao_menor_ou_igual_contra_o_ocupante_atual_e_falha_nunca_substitui_avaliado():
    d = novo(L=2)
    a, b, c = [d.pedir() for _ in range(3)]
    assert d.informar(a.id, 0.5).substituiu                                          # 0,5 <= inf
    sub = d.subpops[0]
    assert sub.aptidao[a.id_pai] == 0.5 and np.array_equal(sub.vetores[a.id_pai], a.vetor)
    d.informar(b.id, float('nan')); d.informar(c.id, None)
    assert np.isinf(sub.aptidao[b.id_pai]) and np.isinf(sub.aptidao[c.id_pai])
    assert d.brackets[0].concluidos[0] == 3                                          # a falha FECHA a vaga: o bracket não trava
    rodar(d, 12, k=1, f=lambda v: 0.5)                                               # empates: <= substitui
    mesmo_pai = [t for t in d.historico if t.nivel == 0 and t.id_pai == a.id_pai and t.id != a.id]
    assert mesmo_pai and all(t.substituiu for t in mesmo_pai)
    d2 = novo(L=2)
    x = d2.pedir(); d2.informar(x.id, 0.1)
    for _ in range(40):
        t = d2.pedir()
        r = d2.informar(t.id, None)
        if t.nivel == 0 and t.id_pai == x.id_pai:
            assert not r.substituiu and d2.subpops[0].aptidao[x.id_pai] == 0.1
    with pytest.raises(KeyError):
        d2.informar(x.id, 0.0)                                                       # informar duas vezes


def test_melhor_so_conta_a_fidelidade_maxima():
    d = novo(L=2)
    rodar(d, 30, k=3, f=lambda v: esfera(v))
    m = d.melhor_na_fidelidade_maxima()
    assert m.nivel == 1 and m.aptidao == min(t.aptidao for t in d.historico if t.nivel == 1)


# ── determinismo, replay e adoção ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('L, k', [(2, 1), (2, 3), (3, 6)])
def test_replay_dos_eventos_reconstroi_o_estado_bit_a_bit(L, k):
    d = novo(L=L, semente=11)
    eventos = rodar(d, 70, k=k, semente_ordem=5)
    r = novo(L=L, semente=11)
    for e in eventos:
        if e[0] == 'pedir':
            assert r.pedir().registro() == json.loads(json.dumps(e[1]))              # inclusive depois de ida e volta pelo JSON
        else:
            r.informar(e[1], json.loads(json.dumps(e[2])) if np.isfinite(e[2]) else None)
    assert json.dumps(r.assinatura(), sort_keys=True) == json.dumps(d.assinatura(), sort_keys=True)
    assert json.dumps(novo(L=L, semente=12).pedir().registro()) != json.dumps(novo(L=L, semente=11).pedir().registro())


def test_adotar_reconstroi_sem_sortear_e_valida_o_pedido():
    d = novo(L=2, semente=11)
    eventos = rodar(d, 50, k=3, semente_ordem=2)
    a = novo(L=2, semente=999)                                                        # semente ERRADA: a adoção não pode depender dela
    antes = json.dumps(a.estado_do_gerador(), sort_keys=True)
    for e in eventos:
        if e[0] == 'pedir':
            a.adotar_pedido(e[1])
        else:
            a.informar(e[1], e[2])
    assert json.dumps(a.estado_do_gerador(), sort_keys=True) == antes               # adotar não consome sorteio
    a.restaurar_gerador([e for e in eventos if e[0] == 'pedir'][-1][2])
    for chave in ('pedidos', 'informados', 'pendentes', 'brackets', 'aptidao', 'proximo_pai', 'gerador'):
        assert json.dumps(a.assinatura()[chave], sort_keys=True) == json.dumps(d.assinatura()[chave], sort_keys=True), chave
    assert a.pedir().registro()['id'] == d.pedir().registro()['id']
    ruim = dict(eventos[0][1], id=5)
    with pytest.raises(ValueError):
        novo().adotar_pedido(ruim)                                                   # fora de ordem
    with pytest.raises(ValueError):
        novo().adotar_pedido(dict(eventos[0][1], nivel=1))                           # nível que não é o do degrau


# ── espaço de busca ──────────────────────────────────────────────────────────────────────────────
def test_decodificacao_cabe_nas_distribuicoes_do_optuna_inclusive_nos_cantos():
    import optuna
    esp = D.EspacoVetorial(ESPACO_MIMEPLANT)
    assert esp.nomes == list(ESPACO_MIMEPLANT) and esp.dimensoes == 12
    dist = {}
    for nome, spec in ESPACO_MIMEPLANT.items():
        dist[nome] = (optuna.distributions.FloatDistribution(spec[1], spec[2], log=(spec[0] == 'log')) if spec[0] in ('log', 'float')
                      else optuna.distributions.IntDistribution(spec[1], spec[2]) if spec[0] == 'int'
                      else optuna.distributions.CategoricalDistribution(list(spec[1])))
    for v in (0.0, float(np.nextafter(0, 1)), 0.5, float(np.nextafter(1, 0)), 1.0):
        hp = esp.para_hp(np.full(12, v))
        for nome, x in hp.items():
            assert type(x) in (int, float), (nome, type(x))                          # tipos nativos: JSON e Optuna
            assert dist[nome]._contains(dist[nome].to_internal_repr(x)), (nome, v, x)   # exp(log lo + 1·log(hi/lo)) passa de hi sem o recorte
        optuna.trial.create_trial(value=0.0, params=hp, distributions=dist)
        json.dumps(hp, allow_nan=False)
    with pytest.raises(ValueError):
        esp.para_hp(np.full(12, 1.0000001))


def test_inteiros_e_categoricos_por_faixas_iguais_na_ordem_da_lista():
    esp = D.EspacoVetorial({'n': ('int', 2, 6), 'c': ('cat', [10, 20, 30]), 'x': ('log', 1e-5, 3e-4)})
    vs = np.linspace(0, 1, 10_001)
    ns = np.array([esp.para_hp([v, 0.5, 0.5])['n'] for v in vs])
    contagem = [int((ns == i).sum()) for i in range(2, 7)]
    assert max(contagem) - min(contagem) <= 2                                        # massa uniforme, extremos incluídos (o pacote dá meia faixa a 2 e a 6)
    cs = [esp.para_hp([0.5, v, 0.5])['c'] for v in (0.0, 0.33, 0.34, 0.66, 0.67, 1.0)]
    assert cs == [10, 10, 20, 20, 30, 30]
    hp = {'n': 4, 'c': 20, 'x': 1e-4}
    assert esp.para_hp(esp.para_vetor(hp)) == pytest.approx(hp)
    for ruim in ({}, {'a': ('cat', [])}, {'a': ('float', 1.0, 1.0)}, {'a': ('log', 0.0, 1.0)}, {'a': ('nada', 0, 1)}):
        with pytest.raises(ValueError):
            D.EspacoVetorial(ruim)


# ── sanidade de otimização ───────────────────────────────────────────────────────────────────────
def test_ganha_da_busca_aleatoria_com_o_mesmo_numero_de_avaliacoes_cheias():
    vitorias = 0
    for s in range(10):
        d = novo(L=2, semente=s)
        rodar(d, 96, k=3, semente_ordem=s)
        n_cheias = sum(1 for t in d.historico if t.nivel == 1)
        aleatoria = min(esfera(v) for v in np.random.default_rng(1000 + s).uniform(size=(n_cheias, 12)))
        vitorias += int(d.melhor_na_fidelidade_maxima().aptidao < aleatoria)
    assert vitorias >= 8
