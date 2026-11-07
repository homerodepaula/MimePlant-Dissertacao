"""Regras finas do núcleo do DEHB (`hpo/dehb_nucleo.py`) que `test_dehb_nucleo.py` não trava — escrito depois de uma revisão por
MUTAÇÃO: cada teste daqui mata pelo menos uma troca plausível de uma linha que passava na suíte. Onde dá, o operador de variação
é provado com F = 0 e CR = 1 (o filho vira exatamente um vetor do pool: dá para dizer de que subpopulação ele saiu) e com o
gerador CLONADO (dá para saber r1, r2, r3 e contar os sorteios). Populações e aptidões são montadas à mão de propósito.
Uma auditoria adversarial (2ª rodada de mutantes) acrescentou as validações de entrada, eta != 3, a aptidão 0,0, o melhor com falhas e os
dourados com a ordem de chegada real; o dourado é rede de segurança — cada regra tem aqui um teste direto com mensagem própria."""
import os
import sys
import json
import hashlib

import numpy as np
import pytest

import protocolo as P

sys.path.insert(0, os.path.join(P.RAIZ, 'hpo'))
import dehb_nucleo as D  # noqa: E402

P.preparar_sys_path()
from hp import ESPACO_MIMEPLANT  # noqa: E402


def novo(L=2, semente=7, dim=12, **kw):
    return D.Dehb(dim, D.escada_de_passos(1_000_000, L), semente=semente, **kw)


def estado(d):
    return json.dumps(d.estado_do_gerador(), sort_keys=True)


def clone_do_gerador(d):
    """Um gerador independente no MESMO estado do de `d`: o teste sorteia nele o que o núcleo vai sortear."""
    g = np.random.default_rng(0)
    g.bit_generator.state = d.estado_do_gerador()
    return g


def linha_de(v, matriz):
    """Índice da linha de `matriz` idêntica (bit a bit) a `v`, ou None."""
    iguais = np.flatnonzero(np.all(np.asarray(matriz) == np.asarray(v), axis=1))
    return int(iguais[0]) if len(iguais) else None


def modelo_do_operador(g, pool, alvo, F, CR):
    """Modelo de referência de rand1 + cruzamento binomial + conserto de fronteira para um pool com 3 ou mais vetores e SEM o
    alvo (de.py:481-488, 152-175), sorteando em `g`. Devolve (filho, mutante, máscara do cruzamento, máscara de fora)."""
    r1, r2, r3 = pool[g.choice(len(pool), size=3, replace=False)]
    mutante = r1 + F * (r2 - r3)
    cruza = g.random(len(alvo)) < CR
    if not cruza.any():
        cruza[g.integers(0, len(alvo))] = True
    filho = np.where(cruza, mutante, alvo)
    fora = (filho < 0.0) | (filho > 1.0)
    if fora.any():
        filho[fora] = g.uniform(0.0, 1.0, size=int(fora.sum()))
    return filho, mutante, cruza, fora


def primeira_iteracao_L3(d, baixos, medios, topo=None, filhos_b1=None):
    """Conduz, com 1 em voo, o começo da 1ª iteração de L = 3: 9 de baixo, 3 promoções ao nível 1, 1 promoção ao nível 2 e
    (se `filhos_b1` vier) as 3 evoluções do bracket 1 no nível 1. Devolve os trabalhos de cada etapa."""
    etapas = []
    for aptidoes in (baixos, medios, [topo] if topo is not None else None, filhos_b1):
        if aptidoes is None:
            break
        feitos = []
        for a in aptidoes:
            t = d.pedir()
            d.informar(t.id, a)
            feitos.append(t)
        etapas.append(feitos)
    return etapas


# ── a) sequência dourada ─────────────────────────────────────────────────────────────────────────
MSG_DOURADO = ('a sequência de sorteios mudou: suba VERSAO_NUCLEO (o replay do jornal de estudos em andamento depende dela) '
               'e atualize o dourado')


def aptidao_dourada(t):
    """Determinística e em Python puro (sem soma vetorizada). Arredondada a 1 casa para haver EMPATES (o <= da seleção e o
    argsort estável entram no dourado) e com falhas em ids fixos (o +inf também entra)."""
    if t.id % 11 == 5:
        return None
    return round(sum((x - 0.3) ** 2 for x in t.registro()['vetor']), 1)


CHEGADAS = {                                                                         # (trabalhos em voo, nº do passo) -> índice do que CHEGA agora
    'alternada': lambda voo, passo: (2 * passo + 1) % len(voo),                      # fixa e fora de ordem: 2º, 1º, 3º, 2º, ...
    'fila': lambda voo, passo: 0,                                                    # na ordem dos pedidos
    'longa_por_ultimo': lambda voo, passo: min(range(len(voo)), key=lambda i: (voo[i].nivel, voo[i].id)),   # a ordem REAL: a fidelidade máxima demora 9x
}


def sequencia_dourada(L, k, n=40, chegada='alternada', **kw):
    d = novo(L=L, semente=20260919, dim=5, **kw)
    voo, registros, passo = [], [], 0
    while len(registros) < n:
        while len(voo) < k and len(registros) < n:
            voo.append(d.pedir())
            registros.append(voo[-1].registro())
        t = voo.pop(CHEGADAS[chegada](voo, passo))
        d.informar(t.id, aptidao_dourada(t))
        passo += 1
    return hashlib.sha256(json.dumps({'pedidos': registros, 'gerador': d.estado_do_gerador()}, sort_keys=True).encode('utf-8')).hexdigest()


# (L, em voo, regra de chegada, nº de pedidos, F, CR); F = CR = None é o PADRÃO do núcleo, sem passar nada (mudar F_PADRAO muda o dourado).
# Os de 'longa_por_ultimo' passam de propósito pelo cenário da PROMOÇÃO REPETIDA (defeito corrigido no núcleo 1.1, mais abaixo): na correção
# (19/09/2026) o sha256 dos 3 mudou e o dos 4 primeiros NÃO — a regra nova só muda pedidos de estudos com --niveis 3 ou mais e mais de 1 em voo;
# estudos de 2 níveis (a escada do projeto) e execuções sequenciais (a validação contra o pacote) saem bit a bit iguais aos do núcleo 1.0.
DOURADOS = {
    (2, 1, 'alternada', 40, None, None): '35fb1d01a16da5239e817b91898a5dcc6b8663eb3f4f199fbecadb3f957f31d2',
    (2, 3, 'alternada', 40, None, None): '8b74521d3034ae7c3aad2d498f718e50426c4bac555356dd3061877de1926d9f',
    (3, 1, 'alternada', 40, None, None): 'd5fe4e27b81835f3b67b8ef3d8a79b6b6e8caa60bcf7b3f00f04b5176c92a45a',
    (3, 3, 'alternada', 40, None, None): '7b02389eb27b125c72e43aad7fe1abab87de69232154d37335457ab887e3e43f',
    (3, 3, 'longa_por_ultimo', 100, None, None): '576362892bc79243fa15eaff3d68c01c68b0bb32398526125622c233051da68b',
    (4, 3, 'longa_por_ultimo', 120, None, None): '75bb50bc0de9e4e2424d357f227a0491b6bcb386759cdd62661d3ca4359d63a5',
    (4, 3, 'longa_por_ultimo', 120, 0.8, 0.9): 'c3169da964d049a2d49891d79524b6978cc7994451556eee5d3e603c8b1bb035',
}


@pytest.mark.parametrize('L, k, chegada, n, F, CR', list(DOURADOS))
def test_sequencia_dourada_dos_primeiros_pedidos_e_do_gerador(L, k, chegada, n, F, CR):
    assert D.VERSAO_NUCLEO == '1.1', 'VERSAO_NUCLEO mudou: atualize esta trava JUNTO com os dourados acima (as duas coisas andam juntas)'
    obtido = sequencia_dourada(L, k, n=n, chegada=chegada, **({} if F is None else {'fator_mutacao': F, 'prob_cruzamento': CR}))
    assert obtido == DOURADOS[(L, k, chegada, n, F, CR)], f'{MSG_DOURADO} (L={L}, {k} em voo, chegada {chegada}, {n} pedidos, F={F}, CR={CR}: sha256 obtido {obtido})'


def test_padroes_sao_os_do_artigo_e_chegam_ao_objeto():
    assert (D.F_PADRAO, D.CR_PADRAO, D.ETA_PADRAO, D.TAM_MIN_POOL) == (0.5, 0.5, 3, 3), 'F = CR = 0,5 e eta = 3 (artigo, Ap. A, e pacote); rand1 precisa de 3'
    d = novo()
    assert (d.F, d.CR, d.eta) == (0.5, 0.5, 3)
    d = novo(fator_mutacao=0.8, prob_cruzamento=0.9)
    assert (d.F, d.CR) == (0.8, 0.9)


def test_o_dourado_e_sensivel_a_ordem_de_chegada_e_nao_ao_relogio():
    assert sequencia_dourada(2, 3) == sequencia_dourada(2, 3)                        # função só de (semente, sequência de pedir/informar)
    com_3_em_voo = {c: sequencia_dourada(3, 3, chegada=c) for c in CHEGADAS}
    assert len(set(com_3_em_voo.values())) == len(CHEGADAS), f'com 3 em voo, cada ordem de chegada dos resultados dá outra sequência de pedidos: {com_3_em_voo}'
    com_1_em_voo = {c: sequencia_dourada(3, 1, chegada=c) for c in CHEGADAS}
    assert len(set(com_1_em_voo.values())) == 1, 'com 1 em voo não há ordem de chegada a variar: a regra de chegada não pode entrar na sequência'


# ── agendamento e população inicial ──────────────────────────────────────────────────────────────
def test_degrau_de_cima_espera_o_ultimo_RESULTADO_de_baixo_e_nao_o_ultimo_pedido():
    d = novo(L=3)
    baixos = [d.pedir() for _ in range(9)]                                           # degrau 0 do bracket 0 todo PEDIDO, nada informado
    assert d.brackets[0].degrau_agendavel() is None
    t = d.pedir()
    assert (t.bracket, t.degrau, t.origem) == (1, 0, 'evolucao'), 'sem resultado de baixo não há o que promover: abre-se o bracket seguinte'
    for x in baixos[:-1]:
        d.informar(x.id, 0.5)
    assert d.brackets[0].degrau_agendavel() is None                                  # falta 1 resultado
    d.informar(baixos[-1].id, 0.4)
    t = d.pedir()
    assert (t.bracket, t.degrau, t.origem) == (0, 1, 'promocao') and np.array_equal(t.vetor, baixos[-1].vetor)


def test_populacao_inicial_e_uniforme_no_cubo_inteiro_e_nunca_avaliada():
    d = novo(L=4, semente=1)
    assert [len(s.vetores) for s in d.subpops] == [27, 9, 6, 4] and all(np.all(np.isinf(s.aptidao)) for s in d.subpops)
    v = np.concatenate([s.vetores for s in d.subpops]).ravel()                       # 46 x 12 = 552 sorteios
    assert v.min() < 0.02 and v.max() > 0.98 and abs(v.mean() - 0.5) < 0.05 and abs(np.mean(v < 0.25) - 0.25) < 0.06
    assert d.pendentes == {} and d.historico == [] and d.brackets == []              # nada é pedido nem aberto na construção


def test_escada_aceita_exatamente_eta_elevado_a_niveis_menos_um_e_recusa_um_passo_a_menos():
    for (passos_max, niveis, eta), esperada in {(9, 3, 3): [1, 3, 9], (27, 4, 3): [1, 3, 9, 27], (4, 3, 2): [1, 2, 4], (10, 3, 3): [1, 3, 10]}.items():
        try:
            escada = D.escada_de_passos(passos_max, niveis, eta=eta)
        except ValueError as e:
            pytest.fail(f'passos_max >= eta^(niveis-1) é escada VÁLIDA (na igualdade a menor fidelidade é 1 passo) e foi recusada: {e}')
        assert escada == esperada, f'escada_de_passos({passos_max}, {niveis}, eta={eta}) = {escada}; esperado {esperada} (N // eta^k, do maior k ao menor)'
    recusas = {'um passo a menos que 3^2': (8, 3, 3), 'um passo a menos que 3^3': (26, 4, 3), 'um passo a menos que 2^2': (3, 3, 2),
               'um nível só': (1000, 1, 3), 'eta = 1': (1000, 3, 1)}
    for caso, (passos_max, niveis, eta) in recusas.items():
        with pytest.raises(ValueError, match='escada inválida'):
            escada = D.escada_de_passos(passos_max, niveis, eta=eta)
            pytest.fail(f'escada_de_passos aceitou {caso} e devolveu {escada} (fidelidade 0 ou repetida)')


def test_construcao_recusa_menos_de_duas_fidelidades_escada_fora_de_ordem_e_dimensao_zero():
    assert D.Dehb(1, [100, 300]).pedir().vetor.shape == (1,)                         # o mínimo válido: 1 dimensão e 2 fidelidades
    recusas = {'uma fidelidade só (DE pura, sem Hyperband, sem avisar)': (3, [1000]), 'nenhuma fidelidade': (3, []), 'fidelidades repetidas': (3, [100, 100, 300]),
               'fidelidades decrescentes': (3, [300, 100]), 'zero dimensões': (0, [100, 300]), 'dimensões negativas': (-2, [100, 300])}
    for caso, (dimensoes, fidelidades) in recusas.items():
        with pytest.raises(ValueError, match='fidelidades devem ser|dimensoes < 1'):
            D.Dehb(dimensoes, fidelidades)
            pytest.fail(f'Dehb aceitou {caso}: dimensoes={dimensoes}, fidelidades={fidelidades}')


def test_eta_diferente_de_3_chega_a_escada_ao_plano_as_populacoes_ao_custo_e_aos_brackets():
    """Nenhum outro teste usa eta != 3, e o coordenador hoje o recusa (conferir_escada); mas a API do núcleo aceita eta e o cabeçalho do jornal o
    grava: um `eta` esquecido numa chamada interna dimensionaria metade do algoritmo com 3 e a outra metade com o eta pedido."""
    assert D.escada_de_passos(1000, 3, eta=2) == [250, 500, 1000], 'a escada divide por ETA (com 3 seria [111, 333, 1000])'
    assert D.escada_de_passos(1024, 3, eta=4) == [64, 256, 1024]
    assert [D.plano_do_bracket(i, 3, 2) for i in range(3)] == [([4, 2, 1], [0, 1, 2]), ([2, 1], [1, 2]), ([3], [2])]
    assert [D.plano_do_bracket(i, 3, 4) for i in range(3)] == [([16, 4, 1], [0, 1, 2]), ([4, 1], [1, 2]), ([3], [2])]
    assert D.tamanhos_das_populacoes(3, 2) == [4, 2, 3] and D.tamanhos_das_populacoes(3, 4) == [16, 4, 3]
    custo = D.custo_de_uma_iteracao([250, 500, 1000], eta=2)
    assert custo == 8.0, f'custo de uma iteração com eta=2: (4·¼ + 2·½ + 1) + (2·½ + 1) + 3 = 8 FRE; veio {custo} (10,25 é o do plano de eta=3)'
    for eta, pops, planos in ((2, [4, 2, 3], [[4, 2, 1], [2, 1], [3]]), (4, [16, 4, 3], [[16, 4, 1], [4, 1], [3]])):
        d = D.Dehb(4, D.escada_de_passos(1024, 3, eta=eta), eta=eta, semente=1)
        assert [len(s.vetores) for s in d.subpops] == pops, f'Dehb(eta={eta}) dimensionou as subpopulações com outro eta'
        n_pedidos = sum(sum(ns) for ns in planos)
        for _ in range(n_pedidos):                                                   # a 1ª iteração inteira, 1 em voo
            t = d.pedir()
            d.informar(t.id, float(np.sum(t.vetor)))
        assert [b.ns for b in d.brackets] == planos, f'Dehb(eta={eta}) abriu os brackets com o plano de outro eta'
        assert all(b.terminou() for b in d.brackets) and d.pedir().bracket == 3      # e a iteração fecha exatamente com esses pedidos
        assert sum(t.origem == 'promocao' for t in d.historico) == sum(sum(ns[1:]) for ns in planos)


# ── b) F e CR são honrados ───────────────────────────────────────────────────────────────────────
def test_com_F_zero_e_CR_um_o_filho_e_exatamente_um_vetor_do_pool():
    d = novo(L=3, fator_mutacao=0.0, prob_cruzamento=1.0)
    alvo = d.subpops[1].vetores[0].copy()                                            # alvo de outra subpopulação: não está no pool
    origens = set()
    for _ in range(40):
        filho, pais = d._evoluir(alvo, nivel_pool=0, n=9)
        assert pais == {'pool': 9, 'global': 0, 'uniforme': 0}
        i = linha_de(filho, d.subpops[0].vetores)
        assert i is not None, f'com F=0 e CR=1 o filho tem de ser r1, um vetor do pool, sem tirar nem pôr: {filho}'
        origens.add(i)
    assert len(origens) >= 6                                                         # r1 é SORTEADO no pool, não é sempre o mesmo


def test_F_multiplica_a_diferenca_r2_menos_r3_na_proporcao_certa():
    d = novo(L=3)
    pool, alvo = d.subpops[0].vetores.copy(), d.subpops[1].vetores[0].copy()
    partida = d.estado_do_gerador()
    mutantes = {}
    for F in (0.0, 0.5, 0.9, 1.0, 2.0):
        d.restaurar_gerador(partida)                                                 # mesmos r1, r2, r3 para todo F
        d.F = F
        mutantes[F], uniformes = d._rand1(pool, alvo)
        assert uniformes == 0                                                        # pool de 9 sem o alvo: nada a completar
    i1 = linha_de(mutantes[0.0], pool)
    assert i1 is not None                                                            # F = 0: o mutante é r1
    g = clone_do_gerador(d)
    g.bit_generator.state = partida
    r1, r2, r3 = pool[g.choice(len(pool), size=3, replace=False)]
    assert np.array_equal(r1, pool[i1])
    for F, m in mutantes.items():
        assert np.array_equal(m, r1 + F * (r2 - r3)), f'F={F} não foi honrado: o mutante não é r1 + F·(r2 - r3)'
    assert np.allclose(mutantes[2.0] - r1, 4 * (mutantes[0.5] - r1), atol=1e-12)     # a proporção, sem depender de quem são r2 e r3
    assert not np.allclose(mutantes[0.9], mutantes[0.5])


def test_os_tres_do_rand1_sao_distintos_e_o_alvo_sai_do_sorteio():
    d = novo(L=3, fator_mutacao=1.0)
    pool, alvo = d.subpops[0].vetores[:4].copy(), d.subpops[0].vetores[0].copy()     # o alvo É a linha 0 do pool
    for _ in range(60):
        m, uniformes = d._rand1(pool, alvo)
        assert uniformes == 0                                                        # sobraram 3 depois de tirar o alvo
        achados = [(a, b, c) for a in (1, 2, 3) for b in (1, 2, 3) for c in (1, 2, 3)
                   if len({a, b, c}) == 3 and np.array_equal(m, pool[a] + 1.0 * (pool[b] - pool[c]))]
        assert achados, 'o mutante não é r1 + (r2 - r3) com r1, r2, r3 DISTINTOS e diferentes do alvo'
    d.F = 0.0
    assert all(linha_de(d._rand1(pool, alvo)[0], pool) in (1, 2, 3) for _ in range(60))   # com F=0, r1 nunca é o alvo
    sozinho = d.subpops[0].vetores[:1].copy()                                        # pool de 1 que é o próprio alvo: não se tira o único
    antes = clone_do_gerador(d)
    m, uniformes = d._rand1(sozinho, sozinho[0].copy())
    assert uniformes == 2, f'o único vetor do pool fica, mesmo sendo o alvo: faltam 2 uniformes para o sorteio de 3, entraram {uniformes}'
    completo = np.concatenate([sozinho, antes.uniform(0.0, 1.0, size=(2, d.dimensoes))])
    assert linha_de(m, completo) is not None                                         # os 2 que faltavam são uniformes NOVOS, do mesmo gerador


def test_com_CR_um_o_filho_e_o_mutante_inteiro():
    d = novo(L=3, prob_cruzamento=1.0)
    alvo = d.subpops[1].vetores[0].copy()
    dentro_total = 0
    for _ in range(25):
        partida = d.estado_do_gerador()
        mutante, _ = d._rand1(d.subpops[0].vetores, alvo)                            # nunca avaliados: o pool dos 9 melhores é a subpopulação, na ordem
        d.restaurar_gerador(partida)
        filho, _ = d._evoluir(alvo, nivel_pool=0, n=9)
        dentro = (mutante >= 0.0) & (mutante <= 1.0)
        assert np.array_equal(filho[dentro], mutante[dentro]), 'CR=1: toda dimensão vem do mutante'
        assert not np.any(filho == alvo)
        dentro_total += int(dentro.sum())
    assert dentro_total >= 200                                                       # a comparação acima não é vazia


def test_com_CR_zero_so_a_dimensao_forcada_vem_do_mutante_e_ela_e_sorteada():
    d = novo(L=3, prob_cruzamento=0.0, fator_mutacao=0.0)                            # F=0: o mutante é um vetor do pool, dentro do cubo
    alvo = d.subpops[1].vetores[0].copy()
    forcadas = set()
    for _ in range(40):
        filho, _ = d._evoluir(alvo, nivel_pool=0, n=9)
        mudou = np.flatnonzero(filho != alvo)
        assert len(mudou) == 1, f'CR=0: exatamente 1 dimensão do mutante, vieram {len(mudou)}'
        assert filho[mudou[0]] in d.subpops[0].vetores[:, mudou[0]]
        forcadas.add(int(mudou[0]))
    assert len(forcadas) >= 6                                                        # a dimensão forçada é sorteada, não é sempre a primeira


@pytest.mark.parametrize('CR', [0.2, 0.5, 0.8])
def test_fracao_de_dimensoes_vindas_do_mutante_acompanha_CR(CR):
    d = novo(L=3, prob_cruzamento=CR, fator_mutacao=0.0)
    alvo = d.subpops[1].vetores[0].copy()
    do_mutante = sum(int(np.sum(d._evoluir(alvo, nivel_pool=0, n=9)[0] != alvo)) for _ in range(400))
    assert abs(do_mutante / (400 * 12) - CR) < 0.04, f'CR={CR}: {do_mutante / (400 * 12):.3f} das dimensões vieram do mutante'   # ~6 desvios-padrão


# ── c) pool de pais ──────────────────────────────────────────────────────────────────────────────
def test_pool_sao_os_n_melhores_em_ordem_estavel_e_nao_os_piores():
    d = novo(L=3, fator_mutacao=0.0, prob_cruzamento=1.0)
    baixo, alvo = d.subpops[0], d.subpops[1].vetores[0].copy()
    baixo.aptidao[:] = [5, 1, 7, 2, 9, 3, 8, 4, 6]
    vistos = {linha_de(d._evoluir(alvo, nivel_pool=0, n=3)[0], baixo.vetores) for _ in range(60)}
    assert vistos == {1, 3, 5}, f'o pool tem de ser os 3 MELHORES de baixo (linhas 1, 3, 5), veio {sorted(vistos, key=str)}'
    baixo.aptidao[:] = [2, 1, 1, 2, 1, 1, 2, 1, 1]                                   # empate: vale a ordem das posições (argsort estável)
    vistos = {linha_de(d._evoluir(alvo, nivel_pool=0, n=4)[0], baixo.vetores) for _ in range(80)}
    assert vistos == {1, 2, 4, 5}


def test_pool_inclui_nunca_avaliados_quando_faltam_avaliados():
    d = novo(L=4, fator_mutacao=0.0, prob_cruzamento=1.0)                            # nível 0 com 27: grande o bastante para uma ordenação instável embaralhar os +inf
    baixo, alvo = d.subpops[0], d.subpops[1].vetores[0].copy()
    baixo.aptidao[20] = 0.5                                                          # 1 avaliado, 26 nunca avaliados (+inf)
    vistos = set()
    for _ in range(120):
        filho, pais = d._evoluir(alvo, nivel_pool=0, n=9)
        assert pais == {'pool': 9, 'global': 0, 'uniforme': 0}                       # os +inf COMPLETAM o pool: nada de população global aqui
        vistos.add(linha_de(filho, baixo.vetores))
    assert vistos == {20, 0, 1, 2, 3, 4, 5, 6, 7}                                    # o avaliado primeiro; depois os +inf na ordem das posições


def test_pool_vem_da_fidelidade_de_baixo_quando_o_degrau_nao_e_o_primeiro():
    """Bracket 3 de L = 3 (2ª iteração: não promove mais). Os 3 pedidos do degrau 1 são EVOLUÇÃO no nível 1 com o pool = os 3
    melhores do nível 0. Com F = 0 e CR = 1 ligados só nessa hora, cada filho é um vetor do pool: dá para ver de onde ele saiu."""
    d = novo(L=3, semente=5)
    for i in range(29):                                                              # 20 da 1ª iteração + os 9 do degrau 0 do bracket 3
        t = d.pedir()
        # os 9 do bracket 3 ganham de todos: o nível 0 fica só com vetores NOVOS (um promovido da 1ª iteração mora nos dois níveis,
        # e se o alvo estivesse no pool ele sairia do sorteio e entraria um uniforme — aqui não se quer esse ruído)
        d.informar(t.id, float(np.sum((t.vetor - 0.3) ** 2)) if i < 20 else -float(i))
    d.F, d.CR = 0.0, 1.0
    baixo, proprio = d.subpops[0], d.subpops[1]
    melhores = set(np.argsort(baixo.aptidao, kind='stable')[:3].tolist())
    for _ in range(3):
        t = d.pedir()
        assert (t.bracket, t.degrau, t.nivel, t.origem) == (3, 1, 1, 'evolucao'), 'na 2ª iteração o degrau de cima é evolução, não promoção'
        assert linha_de(t.vetor, baixo.vetores) in melhores, 'o filho tem de sair dos 3 melhores da fidelidade de BAIXO'
        assert linha_de(t.vetor, proprio.vetores) is None                           # e não da própria subpopulação
        assert t.pais == {'pool': 3, 'global': 0, 'uniforme': 0}                     # n = vagas do degrau de DESTINO (3), não as do degrau de baixo (9)
        d.informar(t.id, 9.0)


def test_no_primeiro_degrau_o_pool_e_a_propria_subpopulacao_sem_o_alvo():
    d = novo(L=3, fator_mutacao=0.0, prob_cruzamento=1.0)
    t = d.pedir()                                                                    # bracket 0, degrau 0: alvo = posição 0 do nível 0
    assert (t.nivel, t.id_pai, t.pais) == (0, 0, {'pool': 9, 'global': 0, 'uniforme': 0})
    assert linha_de(t.vetor, d.subpops[0].vetores) in range(1, 9)                    # um dos OUTROS 8: o alvo saiu do sorteio


def test_o_alvo_do_cruzamento_e_o_pai_da_vez_no_rodizio():
    d = novo(L=2, prob_cruzamento=0.0)                                               # CR=0: o filho é o ALVO com 1 dimensão trocada
    iniciais = [s.vetores.copy() for s in d.subpops]
    for t in [d.pedir() for _ in range(5)]:                                          # 3 em baixo (pais 0, 1, 2) e 2 em cima (pais 0, 1)
        assert int(np.sum(t.vetor != iniciais[t.nivel][t.id_pai])) == 1, f'o filho {t.id} não descende da posição {t.id_pai} do nível {t.nivel}'


def test_pool_pequeno_recebe_mutantes_de_TODAS_as_subpopulacoes():
    """Degrau alto de L = 2: pool = 1 melhor de baixo + 3 mutantes rand1 da concatenação de todas as subpopulações (de.py:655-661).
    Com F = 0 esses mutantes são vetores da população global: o filho tem de poder sair também da subpopulação de cima."""
    d = novo(L=2, fator_mutacao=0.0, prob_cruzamento=1.0)
    d.subpops[0].aptidao[:] = [3.0, 1.0, 2.0]
    alvo = d.subpops[1].vetores[0].copy()
    de_onde = set()
    for _ in range(80):
        filho, pais = d._evoluir(alvo, nivel_pool=0, n=1)
        assert pais == {'pool': 1, 'global': 3, 'uniforme': 0}
        de_onde.add(('baixo', linha_de(filho, d.subpops[0].vetores)) if linha_de(filho, d.subpops[0].vetores) is not None
                    else ('cima', linha_de(filho, d.subpops[1].vetores)))
    assert ('baixo', 1) in de_onde                                                   # o melhor de baixo está no pool
    assert ('cima', 1) in de_onde                                                    # a global inclui a subpopulação de cima ...
    assert ('cima', 0) not in de_onde and ('cima', None) not in de_onde              # ... menos o alvo; e ninguém de fora das populações


def test_mutantes_da_populacao_global_entram_no_pool_como_sairam_sem_conserto_de_fronteira():
    """Pool pequeno (degrau alto de L = 2): os 3 extras são rand1 da população global e entram no pool COMO SAÍRAM, mesmo fora de [0, 1]
    (de.py:655-661); só o FILHO é consertado. Modelo de referência no gerador clonado, com F = 1,5 para os extras saírem bastante do cubo."""
    d = novo(L=2, semente=13, fator_mutacao=1.5, prob_cruzamento=1.0)
    d.subpops[0].aptidao[:] = [3.0, 1.0, 2.0]
    alvo = d.subpops[1].vetores[0].copy()
    todos = np.concatenate([s.vetores for s in d.subpops])
    sem_o_alvo = np.delete(todos, linha_de(alvo, todos), axis=0)
    extras_fora = 0
    for _ in range(30):
        g = clone_do_gerador(d)
        extras = []
        for _ in range(3):
            r1, r2, r3 = sem_o_alvo[g.choice(len(sem_o_alvo), size=3, replace=False)]
            extras.append(r1 + 1.5 * (r2 - r3))
        pool = np.concatenate([d.subpops[0].vetores[[1]], np.array(extras)])         # o melhor de baixo (linha 1) + os 3 extras, nesta ordem
        esperado = modelo_do_operador(g, pool, alvo, F=1.5, CR=1.0)[0]
        filho, pais = d._evoluir(alvo, nivel_pool=0, n=1)
        assert pais == {'pool': 1, 'global': 3, 'uniforme': 0}
        assert np.array_equal(filho, esperado), ('o filho não é o do modelo de referência: pool = melhor de baixo + 3 rand1 da população global COMO SAÍRAM '
                                                 '(sem recorte nem re-sorteio); só o FILHO passa pelo conserto de fronteira, que é re-sorteio')
        assert estado(d) == json.dumps(g.bit_generator.state, sort_keys=True)
        extras_fora += sum(bool(np.any((e < 0.0) | (e > 1.0))) for e in extras)
    assert extras_fora >= 45                                                         # o cenário exercita a regra: quase todo extra tem componente fora do cubo


def test_so_a_primeira_copia_do_alvo_sai_do_sorteio_quando_ele_aparece_duas_vezes():
    """Um promovido mora em DUAS subpopulações: na população global o alvo aparece duas vezes. A referência tira só a primeira cópia
    (de.py:663-696, `break` no primeiro igual); a outra concorre como qualquer vetor, e não entra uniforme nenhum para completar o sorteio."""
    d = novo(L=3, fator_mutacao=0.0)                                                 # F=0: o mutante É r1
    v = d.subpops[0].vetores
    pool, alvo = np.array([v[0], v[1], v[0], v[2]]), v[0].copy()
    resto = pool[1:]                                                                 # o que sobra depois de tirar a PRIMEIRA cópia
    copia_sorteada = 0
    for _ in range(60):
        g = clone_do_gerador(d)
        m, uniformes = d._rand1(pool, alvo)
        assert uniformes == 0, 'tirando UMA cópia do alvo sobram 3 no pool; tirar as duas faz entrar um vetor uniforme que a referência não sorteia'
        assert np.array_equal(m, resto[g.choice(3, size=3, replace=False)][0]) and estado(d) == json.dumps(g.bit_generator.state, sort_keys=True)
        copia_sorteada += int(np.array_equal(m, alvo))
    assert copia_sorteada >= 8, 'a segunda cópia do alvo continua no sorteio (esperado ~1/3 das vezes)'


# ── d) promoção ──────────────────────────────────────────────────────────────────────────────────
def test_promover_da_o_melhor_primeiro_pula_quem_ja_esta_em_cima_trunca_em_n_e_nao_sorteia():
    d = novo(L=3)
    baixo, alto = d.subpops[0], d.subpops[1]
    baixo.aptidao[:] = [5, 1, 7, 2, 9, 3, 8, 4, 6]
    alto.ja_pedidos.add(baixo.vetores[1].tobytes())                                  # o melhor de baixo JÁ foi pedido em cima (no pedir, é o _registrar que anota)
    antes = estado(d)
    a = d._promover(nivel_baixo=0, nivel_alto=1, n=2)
    assert linha_de(a, baixo.vetores) == 3, 'quem já foi pedido na fidelidade de cima não sobe de novo: o próximo melhor é a linha 3'
    assert len(alto.fila_promocao) == 1, f'fila com {len(alto.fila_promocao)} depois do 1º promovido; esperado 1 (n=2 NOVOS: filtra-se quem já está em cima, depois trunca-se)'
    b = d._promover(nivel_baixo=0, nivel_alto=1, n=2)
    assert linha_de(b, baixo.vetores) == 5 and alto.fila_promocao == []
    alto.ja_pedidos.update(v.tobytes() for v in (a, b))                              # os dois promovidos foram pedidos em cima
    baixo.aptidao[8] = 0.5                                                           # e apareceu um melhor em baixo
    c = d._promover(nivel_baixo=0, nivel_alto=1, n=2)
    assert linha_de(c, baixo.vetores) == 8, 'fila vazia se refaz com o estado ATUAL de baixo (fila não truncada serviria um candidato velho)'
    assert estado(d) == antes, 'promover não consome sorteio'
    a[:] = -1.0
    assert linha_de(a, baixo.vetores) is None and np.all(baixo.vetores >= 0)         # o promovido é CÓPIA: mexer nele não mexe na população


def test_empate_na_promocao_sobe_primeiro_a_posicao_mais_baixa():
    d = novo(L=3)
    d.subpops[0].aptidao[:] = [2, 1, 1, 2, 1, 1, 2, 1, 1]                            # argsort ESTÁVEL: o replay não pode depender do desempate da plataforma
    subiram = [linha_de(d._promover(0, 1, n=3), d.subpops[0].vetores) for _ in range(3)]
    assert subiram == [1, 2, 4]


def test_promover_nunca_devolve_falhado_e_sem_candidato_novo_devolve_None_sem_sortear():
    d = novo(L=3)
    baixo, alto = d.subpops[0], d.subpops[1]
    antes = estado(d)
    assert d._promover(0, 1, n=3) is None                                            # ninguém avaliado
    baixo.aptidao[:] = [np.inf, 0.7, np.inf, np.inf, 0.2, np.inf, np.inf, np.inf, np.inf]
    subiram = [d._promover(0, 1, n=3) for _ in range(2)]
    assert [linha_de(v, baixo.vetores) for v in subiram] == [4, 1]                   # só os 2 com sucesso, o melhor primeiro
    alto.ja_pedidos.update(v.tobytes() for v in subiram)
    assert d._promover(0, 1, n=3) is None, 'todos os bons de baixo já foram pedidos em cima: None (o pedido vira evolução), nunca um +inf nem um repetido'
    assert alto.fila_promocao == [] and estado(d) == antes, '_promover não sorteia nem quando devolve None'


def test_falhado_nunca_e_promovido_e_a_vaga_que_sobra_vira_evolucao():
    d = novo(L=3)
    baixos = [d.pedir() for _ in range(9)]
    for i, t in enumerate(baixos):
        d.informar(t.id, {2: 0.8, 6: 0.4}.get(i))                                    # 7 das 9 falharam
    pedidos = []
    for _ in range(3):
        pedidos.append(d.pedir())
        d.informar(pedidos[-1].id, 0.3)
    assert [t.origem for t in pedidos] == ['promocao', 'promocao', 'evolucao'], 'só 2 tiveram sucesso em baixo: a 3ª vaga do degrau é evolução'
    assert np.array_equal(pedidos[0].vetor, baixos[6].vetor) and np.array_equal(pedidos[1].vetor, baixos[2].vetor)
    assert [t.id_pai for t in pedidos] == [0, 1, 2] and all((t.bracket, t.degrau, t.nivel) == (0, 1, 1) for t in pedidos)
    falhados = np.array([t.vetor for i, t in enumerate(baixos) if i not in (2, 6)])
    assert linha_de(pedidos[2].vetor, falhados) is None


def test_promocao_do_bracket_seguinte_pula_quem_ja_subiu_e_usa_o_estado_atual_de_baixo():
    d = novo(L=3)
    _, medios, (topo,), _ = primeira_iteracao_L3(d, baixos=[1.0 + i for i in range(9)], medios=[0.5, 0.6, 0.7], topo=0.4, filhos_b1=[9.0, 9.0, 9.0])
    assert topo.origem == 'promocao' and np.array_equal(topo.vetor, medios[0].vetor)
    t = d.pedir()                                                                    # bracket 1, degrau 1: promoção do nível 1 para o 2
    assert (t.bracket, t.degrau, t.nivel, t.origem, t.id_pai) == (1, 1, 2, 'promocao', 1)
    assert np.array_equal(t.vetor, medios[1].vetor), 'o melhor do nível 1 já está no nível 2: sobe o SEGUNDO melhor, não o mesmo de novo'

    d = novo(L=3)
    _, medios, _, filhos = primeira_iteracao_L3(d, baixos=[1.0 + i for i in range(9)], medios=[0.5, 0.6, 0.7], topo=0.4, filhos_b1=[9.0, 0.1, 9.0])
    t = d.pedir()
    assert t.origem == 'promocao' and np.array_equal(t.vetor, filhos[1].vetor), 'sobe o melhor ATUAL de baixo (o filho de 0,1), não o resto de uma fila velha'


def test_sem_candidato_novo_para_subir_o_pedido_vira_evolucao_e_nada_e_reavaliado():
    """Desvio 1, segundo caso. Só 1 do nível 1 teve sucesso e ele já subiu ao nível 2; o bracket 1 não produz ninguém melhor. O
    pacote promoveria (reavaliaria) o mesmo vetor na fidelidade máxima (dehb.py:556-560); aqui o pedido é uma EVOLUÇÃO."""
    d = novo(L=3)
    primeira_iteracao_L3(d, baixos=[1.0 + i for i in range(9)], medios=[0.5, None, None], topo=0.4, filhos_b1=[None, None, None])
    antes = estado(d)
    t = d.pedir()
    assert (t.bracket, t.degrau, t.nivel, t.id_pai) == (1, 1, 2, 1)
    assert t.origem == 'evolucao', 'todos os bons de baixo já estão em cima: evoluir, e não reavaliar o mesmo vetor'
    assert t.pais == {'pool': 1, 'global': 3, 'uniforme': 0} and estado(d) != antes   # evolução de verdade: pool de baixo e sorteios
    assert linha_de(t.vetor, np.array([x.vetor for x in d.historico])) is None        # nenhum vetor já avaliado volta


@pytest.mark.parametrize('L', [2, 3, 4])
def test_com_um_em_voo_nenhum_vetor_e_avaliado_duas_vezes_na_mesma_fidelidade(L):
    d = novo(L=L, semente=3)
    for _ in range(90):
        t = d.pedir()
        d.informar(t.id, float(np.sum((t.vetor - 0.3) ** 2)))
    chaves = [(t.nivel, t.vetor.tobytes()) for t in d.historico]
    assert len(set(chaves)) == len(chaves)
    assert sum(t.origem == 'promocao' for t in d.historico) == sum(n for i in range(L) for n in D.plano_do_bracket(i, L)[0][1:])   # toda vaga de promoção da 1ª iteração promoveu


# DEFEITO CORRIGIDO em 19/09/2026 (núcleo 1.1; os 3 testes abaixo nasceram como xfail estrito). O filtro de "já promovido" de _promover olhava só a
# POPULAÇÃO ATUAL de cima. Escapam dele (1) o promovido que ainda está EM VOO em cima e (2) o promovido que já foi avaliado em cima e depois
# SUBSTITUÍDO lá por um filho melhor. Nos dois casos o mesmo vetor é pedido de novo na mesma fidelidade — com treino determinístico, uma execução
# inteira jogada fora, que é o que o desvio 1 diz evitar. Corrigir só (1) deixa (2). Numa revisão por mutação, o XPASS(strict) destes testes não
# conta como morte do mutante (acontece sob "fila não truncada" e "seleção sem comparar com o ocupante", por outro caminho).
DEFEITO_PROMOCAO_REPETIDA = ('defeito relatado: _promover só exclui quem está na POPULAÇÃO ATUAL de cima; não vê os pedidos EM VOO em cima nem quem já foi avaliado '
                             'em cima e depois substituído lá — com 3 níveis ou mais e mais de 1 em voo o mesmo vetor é promovido duas vezes (execução inteira repetida)')


def test_promocao_nao_repete_um_vetor_que_ainda_esta_em_voo_na_fidelidade_de_cima():
    d = novo(L=3)
    primeira_iteracao_L3(d, baixos=[1.0 + i for i in range(9)], medios=[0.5, 0.6, 0.7])
    x = d.pedir()                                                                    # promoção ao nível 2 (a execução mais longa): fica EM VOO
    assert (x.bracket, x.degrau, x.nivel, x.origem) == (0, 2, 2, 'promocao')
    for _ in range(3):
        t = d.pedir()
        d.informar(t.id, 9.0)                                                        # os filhos do bracket 1 não ganham de ninguém no nível 1
    y = d.pedir()                                                                    # bracket 1 promove ao nível 2 com x ainda pendente
    assert (y.bracket, y.degrau, y.nivel) == (1, 1, 2) and x.id in d.pendentes
    assert not np.array_equal(y.vetor, x.vetor), 'o mesmo vetor foi pedido duas vezes na fidelidade máxima'


def test_promocao_nao_repete_um_vetor_que_ja_foi_avaliado_em_cima_e_depois_substituido_la():
    """O segundo caminho do defeito, com no máximo 2 em voo: x sobe ao nível 2, é avaliado e entra; um filho do bracket 2 ganha dele e toma a
    posição; quando o degrau alto do bracket 1 abre, x continua o melhor do nível 1 e não está mais na população de cima: sobe DE NOVO."""
    d = novo(L=3)
    _, medios, (x,) = primeira_iteracao_L3(d, baixos=[1.0 + i for i in range(9)], medios=[0.5, 0.6, 0.7], topo=0.4)
    assert x.origem == 'promocao' and x.substituiu and np.array_equal(x.vetor, medios[0].vetor) and x.id not in d.pendentes   # x JÁ foi avaliado em cima
    filhos = [d.pedir() for _ in range(3)]                                           # degrau 0 do bracket 1 (nível 1)
    d.informar(filhos[0].id, 9.0); d.informar(filhos[1].id, 9.0)                     # o 3º fica em voo: o degrau alto do bracket 1 não abre, abre-se o bracket 2
    for aptidao in (0.9, 0.9, 0.1):                                                  # pais 1, 2 e 0 do nível 2: o último ganha de x (0,4) e toma a posição dele
        t = d.pedir()
        assert (t.bracket, t.nivel, t.origem) == (2, 2, 'evolucao') and len(d.pendentes) == 2
        d.informar(t.id, aptidao)
    assert (t.id_pai, t.substituiu) == (x.id_pai, True) and linha_de(x.vetor, d.subpops[2].vetores) is None   # x saiu da população de cima
    d.informar(filhos[2].id, 9.0)                                                    # nenhum filho ganha de ninguém no nível 1: x ainda é o melhor de lá
    y = d.pedir()
    assert (y.bracket, y.degrau, y.nivel) == (1, 1, 2)
    assert not np.array_equal(y.vetor, x.vetor), 'x já foi AVALIADO na fidelidade máxima (trabalho informado) e foi pedido lá de novo'
    assert y.origem == 'promocao' and np.array_equal(y.vetor, medios[1].vetor)       # o que se espera: sobe o segundo melhor do nível 1


def pedidos_repetidos(L, k, semente, n=200):
    """Pedidos de um (fidelidade, vetor) que já tinha sido pedido, com k em voo e a ordem de chegada real (a avaliação mais longa chega por último)."""
    d = novo(L=L, semente=semente)
    voo, vistos, repetidos = [], {}, []
    while len(d.historico) < n:
        while len(voo) < k:
            voo.append(d.pedir())
            chave = (voo[-1].nivel, voo[-1].vetor.tobytes())
            if chave in vistos:
                repetidos.append((voo[-1].id, vistos[chave]))
            vistos[chave] = voo[-1].id
        t = voo.pop(CHEGADAS['longa_por_ultimo'](voo, 0))
        d.informar(t.id, float(np.sum((t.vetor - 0.3) ** 2)))
    return repetidos


@pytest.mark.parametrize('L', [3, 4])
def test_com_tres_em_voo_nenhum_vetor_e_pedido_duas_vezes_na_mesma_fidelidade(L):
    repetidos = {s: pedidos_repetidos(L, k=3, semente=s) for s in range(5)}
    assert not any(repetidos.values()), f'L={L}, 3 em voo: (id do pedido repetido, id do 1º pedido do mesmo vetor na mesma fidelidade) por semente: {repetidos}'


@pytest.mark.parametrize('k', [2, 3, 6])
def test_com_dois_niveis_nenhum_vetor_e_pedido_duas_vezes_mesmo_com_varios_em_voo(k):
    """Com L = 2 só há UMA promoção por estudo (bracket 0), então o defeito acima não alcança os estudos de 2 níveis: fica travado."""
    assert [pedidos_repetidos(2, k=k, semente=s) for s in range(5)] == [[]] * 5


def test_brackets_depois_da_primeira_iteracao_nao_promovem_mesmo_com_candidato_novo():
    d = novo(L=2)
    for _ in range(6):                                                               # 1ª iteração inteira: brackets 0 e 1
        t = d.pedir()
        d.informar(t.id, 0.5)
    for _ in range(3):
        t = d.pedir()
        assert (t.bracket, t.degrau) == (2, 0)
        d.informar(t.id, -1.0)                                                       # três candidatos NOVOS e ótimos em baixo, nenhum deles em cima
    t = d.pedir()
    assert (t.bracket, t.degrau, t.nivel) == (2, 1, 1)
    assert t.origem == 'evolucao' and t.pais == {'pool': 1, 'global': 3, 'uniforme': 0}, 'a promoção é só dos brackets de índice < L'


def test_promocao_e_decidida_pelo_indice_do_bracket_e_nao_por_quantos_brackets_ja_abriram():
    """Desvio 1, aqui sem depender do dourado: com os resultados de baixo atrasados já há 3 brackets abertos (> L) quando o degrau alto do bracket 0
    abre. Pelo contador global do pacote a promoção sumiria; pelo índice do bracket (0 < L) ela acontece."""
    d = novo(L=2)
    b0 = [d.pedir() for _ in range(3)]
    espera = [d.pedir() for _ in range(3)]                                           # 2 do bracket 1 e 1 do bracket 2, todos em voo
    assert [t.bracket for t in espera] == [1, 1, 2] and len(d.brackets) == 3
    for t, aptidao in zip(b0, (0.3, 0.2, 0.25)):
        d.informar(t.id, aptidao)
    p = d.pedir()
    assert (p.bracket, p.degrau, p.nivel) == (0, 1, 1)
    assert p.origem == 'promocao' and np.array_equal(p.vetor, b0[1].vetor), 'bracket 0 é da 1ª iteração: promove o melhor de baixo, não importa quantos brackets já abriram'


def test_promovido_em_voo_nao_muda_quando_a_posicao_de_origem_e_substituida():
    d = novo(L=2)
    b0 = [d.pedir() for _ in range(3)]
    for t, a in zip(b0, (0.3, 0.2, 0.25)):
        d.informar(t.id, a)
    promovido = d.pedir()
    foto = promovido.vetor.copy()
    assert promovido.origem == 'promocao' and np.array_equal(foto, b0[1].vetor)
    outros = [d.pedir() for _ in range(5)]                                           # bracket 1 (2 em cima) e bracket 2 (3 em baixo), com o promovido em voo
    for t in outros:
        if t.nivel == 0:
            assert d.informar(t.id, -1.0).substituiu                                 # toda posição de baixo troca de ocupante
    assert np.array_equal(promovido.vetor, foto) and promovido.registro()['vetor'] == foto.tolist()
    d.informar(promovido.id, 0.1)
    assert np.array_equal(d.subpops[1].vetores[promovido.id_pai], foto)             # entrou em cima o vetor que foi AVALIADO


# ── e) seleção ───────────────────────────────────────────────────────────────────────────────────
def test_dois_filhos_em_voo_na_mesma_posicao_o_segundo_compete_com_o_primeiro():
    d = novo(L=2)
    pedidos = [d.pedir() for _ in range(8)]                                          # bracket 0 (3 em baixo), bracket 1 (2 em cima), bracket 2 (3 em baixo)
    a, b = [t for t in pedidos if t.nivel == 0 and t.id_pai == 0]
    assert (a.bracket, b.bracket) == (0, 2) and {a.id, b.id} <= set(d.pendentes)
    assert d.informar(b.id, 0.3).substituiu                                          # o mais novo chega primeiro: 0,3 <= +inf
    r = d.informar(a.id, 0.5)
    assert not r.substituiu, 'a seleção é contra o ocupante ATUAL (0,3, do outro filho), não contra o pai que havia na hora do pedido'
    sub = d.subpops[0]
    assert sub.aptidao[0] == 0.3 and np.array_equal(sub.vetores[0], b.vetor)
    c, e = [t for t in pedidos if t.nivel == 0 and t.id_pai == 1]
    d.informar(e.id, 0.3)
    assert d.informar(c.id, 0.3).substituiu and np.array_equal(sub.vetores[1], c.vetor), 'empate com o ocupante atual: a seleção é <=, o filho entra'


def test_filho_pior_que_o_ocupante_avaliado_nao_entra():
    d = novo(L=2)
    t = d.pedir()
    d.informar(t.id, 0.1)
    while True:
        u = d.pedir()
        if (u.nivel, u.id_pai) == (t.nivel, t.id_pai):
            break
        d.informar(u.id, 0.5)
    r = d.informar(u.id, 0.9)
    assert not r.substituiu and r.aptidao == 0.9, 'filho pior (0,9 > 0,1) não substitui'
    assert d.subpops[0].aptidao[t.id_pai] == 0.1 and np.array_equal(d.subpops[0].vetores[t.id_pai], t.vetor)


@pytest.mark.parametrize('ruim', [float('-inf'), float('inf'), float('nan'), None, np.float64('nan'), -np.inf])
def test_aptidao_nao_finita_e_falha_nao_substitui_avaliado_e_fecha_a_vaga(ruim):
    d = novo(L=2)
    a, b, c = [d.pedir() for _ in range(3)]
    d.informar(a.id, 0.5); d.informar(b.id, 0.6)
    r = d.informar(c.id, ruim)
    assert r.aptidao == float('inf'), f'{ruim!r} tem de virar +inf (falha); -inf como aptidão ganharia de todo mundo para sempre'
    # inf <= inf: o vetor do falhado ocupa a posição de um NUNCA avaliado, e a posição continua +inf. É o <= da referência; muda quem
    # entra nos próximos sorteios, então o replay depende disso (mudar = subir VERSAO_NUCLEO).
    assert r.substituiu and np.array_equal(d.subpops[0].vetores[c.id_pai], c.vetor) and d.subpops[0].aptidao[c.id_pai] == float('inf')
    assert d.brackets[0].concluidos == [3, 0] and d.brackets[0].degrau_agendavel() == 1   # a falha fecha a vaga e o degrau de cima abre
    p = d.pedir()
    assert p.origem == 'promocao' and np.array_equal(p.vetor, a.vetor)               # o falhado não é o "melhor" de baixo
    cheios = [p]
    for t in [d.pedir() for _ in range(5)]:                                          # bracket 1 e, em baixo de novo, o bracket 2
        if t.nivel == 0 and t.id_pai == a.id_pai:
            assert not d.informar(t.id, ruim).substituiu, 'falha nunca substitui um ocupante avaliado'
        elif t.nivel == 1:
            cheios.append(t)
    assert d.subpops[0].aptidao[a.id_pai] == 0.5 and np.array_equal(d.subpops[0].vetores[a.id_pai], a.vetor)
    for t in cheios:
        d.informar(t.id, ruim)
    assert len(cheios) == 3 and sum(t.nivel == 1 for t in d.historico) == 3          # sem isto o assert de baixo seria vazio: nada cheio teria sido informado
    assert d.melhor_na_fidelidade_maxima() is None, f'as 3 avaliações cheias falharam ({ruim!r}): não há "melhor" — um Trabalho com +inf não é resultado'
    assert all(np.isfinite(x) or x == float('inf') for s in d.subpops for x in s.aptidao)


def test_melhor_na_fidelidade_maxima_ignora_as_cheias_que_falharam_e_nao_se_perde_com_falhas_depois():
    d = novo(L=2)
    for _ in range(12):
        t = d.pedir()
        d.informar(t.id, None if t.nivel == 1 else 0.5)
    assert sum(t.nivel == 1 for t in d.historico) >= 4                               # há avaliações cheias no histórico, todas falhadas
    assert d.melhor_na_fidelidade_maxima() is None, 'toda avaliação cheia falhou: devolver um Trabalho com aptidão +inf faria dele o "melhor trial" do estudo'
    cheias = []
    for aptidao in (0.7, None, 0.9, float('nan')):
        while True:
            t = d.pedir()
            if t.nivel == 1:
                break
            d.informar(t.id, 0.1)                                                    # as baratas são melhores, e não contam
        d.informar(t.id, aptidao)
        cheias.append(t)
        assert d.melhor_na_fidelidade_maxima() is cheias[0], 'o melhor é a cheia de 0,7: nem as falhas nem a de 0,9 que vieram depois mudam isso'


@pytest.mark.parametrize('otima', [0.0, -0.0, 0, np.float64(0.0), -3.5])
def test_aptidao_zero_ou_negativa_e_resultado_legitimo_e_nao_falha(otima):
    """`if aptidao and ...` no lugar de `if aptidao is not None and ...` faria do 0,0 — o ótimo de muita função objetivo — uma falha (+inf)."""
    d = novo(L=2)
    a, b, c = [d.pedir() for _ in range(3)]
    r = d.informar(a.id, otima)
    assert r.aptidao == otima and np.isfinite(r.aptidao), f'{otima!r} é aptidão legítima (minimiza-se; o objetivo pode ser <= 0), e foi registrada como {r.aptidao}'
    assert r.substituiu and d.subpops[0].aptidao[a.id_pai] == otima and np.array_equal(d.subpops[0].vetores[a.id_pai], a.vetor)
    d.informar(b.id, 0.5); d.informar(c.id, 0.7)
    p = d.pedir()
    assert p.origem == 'promocao' and np.array_equal(p.vetor, a.vetor), 'o de aptidão 0 (ou negativa) é o MELHOR de baixo: é ele que sobe'
    d.informar(p.id, otima)
    assert d.melhor_na_fidelidade_maxima() is p and p.aptidao == otima


# ── f) fronteira ─────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('CR', [1.0, 0.5])
def test_componente_fora_do_cubo_e_ressorteada_so_ela_e_o_numero_de_sorteios_confere(CR):
    d = novo(L=3, semente=13, fator_mutacao=1.5, prob_cruzamento=CR)
    alvo = d.subpops[1].vetores[0].copy()
    n_fora = n_dentro = 0
    for _ in range(30):
        g = clone_do_gerador(d)
        esperado, mutante, cruza, fora = modelo_do_operador(g, d.subpops[0].vetores, alvo, F=1.5, CR=CR)
        filho, _ = d._evoluir(alvo, nivel_pool=0, n=9)
        assert np.all((filho > 0.0) & (filho < 1.0)), f'0.0 ou 1.0 exato é sinal de RECORTE; a regra é re-sortear: {filho}'
        intactas = cruza & ~fora
        assert np.array_equal(filho[intactas], mutante[intactas]), 'só a componente que saiu do cubo é re-sorteada; as outras ficam'
        assert np.array_equal(filho, esperado)
        assert estado(d) == json.dumps(g.bit_generator.state, sort_keys=True), 'nº de sorteios consumidos: 1 uniforme por componente fora, nem mais nem menos'
        n_fora += int(fora.sum()); n_dentro += int(intactas.sum())
    assert n_fora >= 30 and n_dentro >= 30                                           # o cenário exercita os dois lados


# ── g) adotar_pedido ─────────────────────────────────────────────────────────────────────────────
def pedido(id, bracket, degrau, nivel, id_pai, dim=12, **kw):
    return dict({'id': id, 'vetor': [0.25] * dim, 'fidelidade': 0, 'nivel': nivel, 'id_pai': id_pai, 'bracket': bracket, 'degrau': degrau,
                 'origem': 'evolucao', 'pais': {'pool': 3, 'global': 0, 'uniforme': 1}}, **kw)


def test_adotar_recusa_degrau_lotado_nivel_errado_id_fora_de_ordem_e_pai_inexistente():
    d = novo(L=2)
    for i in range(3):
        t = d.adotar_pedido(pedido(i, bracket=0, degrau=0, nivel=0, id_pai=i))
        assert t.fidelidade == 333_333 and t.id in d.pendentes                       # a fidelidade vem da ESCADA do estudo, não do registro
    antes = json.dumps(d.assinatura(), sort_keys=True)
    recusas = {
        'degrau lotado': pedido(3, bracket=0, degrau=0, nivel=0, id_pai=0),
        'nível que não é o do degrau': pedido(3, bracket=0, degrau=1, nivel=0, id_pai=0),
        'degrau que a escada não tem': pedido(3, bracket=0, degrau=2, nivel=1, id_pai=0),
        'id_pai além da subpopulação': pedido(3, bracket=1, degrau=0, nivel=1, id_pai=2),
        'id_pai negativo': pedido(3, bracket=1, degrau=0, nivel=1, id_pai=-1),
    }
    for caso, reg in recusas.items():
        with pytest.raises(ValueError, match='incompatível'):
            d.adotar_pedido(reg)
            pytest.fail(f'adotar aceitou um pedido inválido: {caso}')
    for id_ruim in (2, 4):                                                           # repetido e pulado
        with pytest.raises(ValueError, match='fora de ordem'):
            d.adotar_pedido(pedido(id_ruim, bracket=1, degrau=0, nivel=1, id_pai=0))
            pytest.fail(f'adotar aceitou o id {id_ruim} quando o próximo é o 3 (id repetido ou pulado)')
    antes, depois = json.loads(antes), d.assinatura()
    # Comportamento ATUAL, afirmado de propósito: adotar_pedido abre os brackets que faltam ANTES de validar, então as recusas com bracket=1
    # deixam o bracket 1 aberto e vazio. É inofensivo (um bracket vazio aberto antes da hora agenda igual; o coordenador não captura o ValueError:
    # o estudo para), mas a recusa NÃO deixa o estado intocado. Se um dia a validação vier antes de abrir, este assert avisa.
    assert len(antes['brackets']) == 1 and depois['brackets'] == [[0, [3, 0], [0, 0]], [1, [0], [0]]]
    for chave in ('pedidos', 'informados', 'pendentes', 'aptidao', 'vetores', 'proximo_pai', 'filas', 'gerador'):   # fora isso, recusa não agenda, não sorteia, não mexe
        assert json.dumps(depois[chave], sort_keys=True) == json.dumps(antes[chave], sort_keys=True), chave
    bom = d.adotar_pedido(pedido(3, bracket=1, degrau=0, nivel=1, id_pai=1))
    assert d.brackets[1].agendados == [1]                                            # adotar consome a vaga, como pedir
    assert d.subpops[1].proximo_pai == 0, 'o rodízio segue do id_pai do pedido ADOTADO (1 + 1, módulo 2), não do contador que a subpopulação tinha'
    assert d.pedir().id == bom.id + 1


def test_adotar_abre_os_brackets_que_faltam_e_preserva_o_registro():
    d = novo(L=2)
    reg = pedido(0, bracket=2, degrau=0, nivel=0, id_pai=0, origem='promocao', pais={})
    t = d.adotar_pedido(reg)
    assert [(b.indice, b.ns, b.niveis, b.agendados) for b in d.brackets] == [(0, [3, 1], [0, 1], [0, 0]), (1, [2], [1], [0]), (2, [3, 1], [0, 1], [1, 0])]
    assert (t.origem, t.pais) == ('promocao', {}), 'origem e pais do pedido adotado são os do REGISTRO (o espelho e a análise leem daí), não um padrão'
    assert t.registro() == dict(reg, fidelidade=333_333)
    original = novo(L=3, semente=4)
    regs = []
    for _ in range(25):
        p = original.pedir()
        regs.append(json.loads(json.dumps(p.registro())))
        original.informar(p.id, float(np.sum(p.vetor)))
    adotante = novo(L=3, semente=99)
    for r in regs:
        assert adotante.adotar_pedido(r).registro() == r                             # vetor, origem e pais voltam como foram gravados
        adotante.informar(r['id'], float(np.sum(r['vetor'])))
    for chave in ('brackets', 'aptidao', 'vetores', 'proximo_pai'):
        assert json.dumps(adotante.assinatura()[chave]) == json.dumps(original.assinatura()[chave]), chave


# ── h) sanidade de otimização, sem viés de centro ────────────────────────────────────────────────
OTIMO = np.array([0.11, 0.88, 0.94, 0.87, 0.91, 0.84, 0.06, 0.93])                   # sorteado UMA vez (default_rng(20260919)), longe de 0,5 em toda dimensão


def perda(v):
    return float(np.sum((np.asarray(v) - OTIMO) ** 2))


def corrida_a_custo_igual(L, semente, fre_alvo=64.0, sigma=0.2, **kw):
    """(melhor do DEHB na fidelidade máxima, melhor da busca aleatória) com o MESMO custo em FRE (execuções completas
    equivalentes): a busca aleatória faz int(FRE gasto pelo DEHB) avaliações cheias. A fidelidade baixa é a alta + ruído."""
    d = D.Dehb(len(OTIMO), D.escada_de_passos(1_000_000, L), semente=semente, **kw)
    fre = 0.0
    while fre < fre_alvo:
        t = d.pedir()
        fracao = t.fidelidade / d.fidelidades[-1]
        ruido = float(np.random.default_rng([semente, t.id]).normal(0.0, sigma * (1.0 - fracao))) if fracao < 1 else 0.0
        d.informar(t.id, perda(t.vetor) + ruido)
        fre += fracao
    aleatoria = min(perda(v) for v in np.random.default_rng(1000 + semente).uniform(size=(int(fre), len(OTIMO))))
    return d.melhor_na_fidelidade_maxima().aptidao, aleatoria


@pytest.mark.parametrize('L, min_vitorias, max_razao', [(2, 15, 0.6), (3, 15, 0.85)])
def test_ganha_da_busca_aleatoria_a_custo_igual_em_FRE_com_otimo_longe_do_centro(L, min_vitorias, max_razao):
    pares = [corrida_a_custo_igual(L, s) for s in range(20)]
    vitorias = sum(a < b for a, b in pares)
    razao = float(np.median([a / b for a, b in pares]))
    assert vitorias >= min_vitorias and razao <= max_razao, f'L={L}: {vitorias}/20 vitórias, razão mediana DEHB/aleatória {razao:.2f}'


# ── i) espaço vetorial ───────────────────────────────────────────────────────────────────────────
def test_descricao_e_ordenada_como_o_dicionario_e_faz_ida_e_volta_pelo_json():
    esp = D.EspacoVetorial(ESPACO_MIMEPLANT)
    desc = esp.descricao()
    assert [nome for nome, _ in desc] == list(ESPACO_MIMEPLANT) == esp.nomes
    assert json.loads(json.dumps(desc, allow_nan=False)) == desc                      # só listas e tipos nativos: o que se grava é o que se lê
    assert dict((nome, tuple(spec)) for nome, spec in desc) == {k: tuple(v) for k, v in ESPACO_MIMEPLANT.items()}
    espaco = {'z': ('cat', ('b', 'a')), 'a': ('int', 2, 6), 'm': ('log', 1e-5, 3e-4)}  # fora da ordem alfabética, e opções em TUPLA
    esp = D.EspacoVetorial(espaco)
    assert esp.descricao() == [['z', ['cat', ['b', 'a']]], ['a', ['int', 2, 6]], ['m', ['log', 1e-5, 3e-4]]]
    assert json.loads(json.dumps(esp.descricao())) == esp.descricao()
    invertido = D.EspacoVetorial(dict(reversed(list(espaco.items()))))
    assert invertido.descricao() == esp.descricao()[::-1] != esp.descricao()         # reordenar o espaço MUDA a descrição (e invalida o replay)


def test_a_ordem_das_dimensoes_e_a_do_dicionario():
    espaco = {'z': ('float', 0.0, 1.0), 'a': ('float', 10.0, 20.0)}
    assert D.EspacoVetorial(espaco).para_hp([0.25, 0.5]) == {'z': 0.25, 'a': 15.0}
    assert list(D.EspacoVetorial(espaco).para_hp([0.25, 0.5])) == ['z', 'a']
    assert D.EspacoVetorial(dict(reversed(list(espaco.items())))).para_hp([0.25, 0.5]) == {'a': 12.5, 'z': 0.5}
    assert D.EspacoVetorial(espaco).para_vetor({'a': 15.0, 'z': 0.25}).tolist() == [0.25, 0.5]


def test_para_vetor_e_para_hp_sao_estaveis_nos_12_hp_reais():
    esp = D.EspacoVetorial(ESPACO_MIMEPLANT)
    rng = np.random.default_rng(0)
    vetores = list(rng.uniform(size=(200, 12))) + [np.zeros(12), np.ones(12), np.full(12, np.nextafter(1, 0))]
    for v in vetores:
        hp = esp.para_hp(v)
        v2 = esp.para_vetor(hp)
        assert v2.shape == (12,) and np.all((v2 >= 0) & (v2 <= 1))
        hp2 = esp.para_hp(v2)
        for nome, spec in ESPACO_MIMEPLANT.items():
            if spec[0] in ('int', 'cat'):
                assert hp2[nome] == hp[nome] and type(hp2[nome]) is type(hp[nome]), (nome, hp[nome], hp2[nome])   # discretos: EXATO
            else:
                assert hp2[nome] == pytest.approx(hp[nome], rel=1e-12), (nome, hp[nome], hp2[nome])
                assert v2[esp.nomes.index(nome)] == pytest.approx(v[esp.nomes.index(nome)], abs=1e-9)             # contínuos: a inversa acha o ponto
        assert np.allclose(esp.para_vetor(hp2), v2, atol=1e-12)                      # ponto fixo: mais uma volta não anda


def test_para_hp_recusa_vetor_fora_do_cubo_nao_finito_ou_de_forma_errada():
    esp = D.EspacoVetorial({'x': ('float', 0.1, 0.3), 'n': ('int', 2, 6)})
    assert esp.para_hp([0.0, 1.0]) == {'x': 0.1, 'n': 6} and esp.para_hp(np.array([1.0, 0.0])) == {'x': 0.3, 'n': 2}   # as bordas do cubo são válidas
    recusas = {'componente negativa (sairia recortada em lo, em silêncio)': [-1e-9, 0.5], 'componente acima de 1': [0.5, 1.0000001],
               'NaN (nan < 0 e nan > 1 são falsos: decodificaria no limite inferior)': [float('nan'), 0.5], '+inf': [float('inf'), 0.5], '-inf': [0.5, float('-inf')],
               'vetor curto (o zip truncaria: dicionário PARCIAL de HP)': [0.5], 'vetor longo': [0.5, 0.5, 0.5], 'matriz 1 x d': [[0.5, 0.5]], 'escalar': 0.5}
    for caso, vetor in recusas.items():
        with pytest.raises(ValueError, match=r'vetor fora de \[0, 1\]\^2'):
            hp = esp.para_hp(vetor)
            pytest.fail(f'para_hp aceitou {caso}: {vetor} -> {hp}')


def test_float_decodificado_e_recortado_em_hi_quando_a_conta_em_ponto_flutuante_estoura():
    """Desvio 3 no tipo 'float'. Nos 3 floats do espaço real lo + 1,0·(hi - lo) não passa de hi, então o teste dos cantos de `test_dehb_nucleo.py`
    só exercita o recorte do 'log'. Nestes pares a conta ESTOURA (0,06 + 0,82 = 0,8800000000000001) e o Optuna recusaria o trial."""
    for lo, hi in ((0.06, 0.88), (0.07, 0.93), (0.07, 0.84)):
        assert lo + 1.0 * (hi - lo) > hi                                             # a pré-condição do cenário: sem recorte, passa de hi
        x = D.EspacoVetorial({'x': ('float', lo, hi)}).para_hp([1.0])['x']
        assert x == hi and type(x) is float, f"'float' em ({lo}, {hi}) com v = 1,0 decodificou {x!r}: fora de [lo, hi], tem de sair recortado em hi"
        assert D.EspacoVetorial({'x': ('float', lo, hi)}).para_hp([0.0])['x'] == lo


def test_para_vetor_recorta_no_cubo_o_hp_que_esta_fora_da_faixa():
    esp = D.EspacoVetorial({'x': ('float', 0.1, 0.3), 'n': ('int', 2, 6), 'lr': ('log', 1e-5, 3e-4)})
    acima, abaixo = esp.para_vetor({'x': 0.6, 'n': 9, 'lr': 1e-2}), esp.para_vetor({'x': 0.0, 'n': 2, 'lr': 1e-7})
    assert acima.tolist() == [1.0, 1.0, 1.0] and abaixo.tolist() == [0.0, 0.1, 0.0], f'HP fora da faixa vai para a BORDA do cubo: veio {acima.tolist()} e {abaixo.tolist()}'
    bordas = (esp.para_hp(acima), esp.para_hp(abaixo))                               # o que para_vetor devolve, para_hp aceita — e decodifica EXATAMENTE nos limites
    assert bordas == ({'x': 0.3, 'n': 6, 'lr': 3e-4}, {'x': 0.1, 'n': 2, 'lr': 1e-5}), f'a borda do cubo decodifica em hi e em lo EXATOS (desvio 3, dos dois lados): {bordas}'


def test_para_vetor_poe_inteiros_e_categoricos_no_centro_da_faixa():
    esp = D.EspacoVetorial({'n': ('int', 2, 6), 'c': ('cat', [10, 20, 30, 40])})
    assert esp.para_vetor({'n': 2, 'c': 10}).tolist() == [0.1, 0.125]
    assert esp.para_vetor({'n': 6, 'c': 40}).tolist() == [0.9, 0.875]
    for n in range(2, 7):
        for c in (10, 20, 30, 40):
            v = esp.para_vetor({'n': n, 'c': c})
            for empurrao in (-0.09, 0.0, 0.09):                                      # centro da faixa: um empurrão menor que meia faixa não troca o valor
                assert esp.para_hp(v + empurrao) == {'n': n, 'c': c}


# ── j) auditoria das correções do núcleo 1.1 (19/09/2026): ja_pedidos (N1) e a conversão da aptidão (N2) ──────────────────────────
# Revisão por mutação SOBRE AS LINHAS NOVAS. O que os testes de cima já matavam: não anotar em _registrar, filtrar pela população (a regra do 1.0),
# anotar só no informar, tobytes de float32, aceitar -inf. O que passava: anotar só as promoções, não anotar na ADOÇÃO, esquecer o promovido que
# falhou em cima, converter a aptidão DEPOIS de tirar de pendentes, capturar só TypeError (texto levanta ValueError). Cada um ganhou um teste aqui.
from decimal import Decimal  # noqa: E402
from fractions import Fraction  # noqa: E402


def corrida_com_eventos(L, k, n, semente=4, chegada='longa_por_ultimo', dim=5):
    """(núcleo, eventos) de n pedidos com k em voo, na forma do jornal do coordenador: ('pedir', registro DEPOIS de ida e volta pelo JSON, gerador)
    e ('informar', id, aptidão). Os últimos pedidos ficam EM VOO de propósito (é o estado de uma queda)."""
    d = novo(L=L, semente=semente, dim=dim)
    voo, eventos, passo, pedidos = [], [], 0, 0
    while pedidos < n:
        while len(voo) < k and pedidos < n:
            voo.append(d.pedir())
            pedidos += 1
            eventos.append(('pedir', json.loads(json.dumps(voo[-1].registro())), d.estado_do_gerador()))
        t = voo.pop(CHEGADAS[chegada](voo, passo))
        eventos.append(('informar', t.id, aptidao_dourada(t)))
        d.informar(t.id, eventos[-1][2])
        passo += 1
    return d, eventos


def reconstruir(eventos, L, modo, semente=4, dim=5):
    """O que o _replay do coordenador faz: 'verificar' repete os pedidos; 'adotar' reinstala os gravados e restaura o gerador do último."""
    r = novo(L=L, semente=semente, dim=dim)
    for e in eventos:
        if e[0] == 'informar':
            r.informar(e[1], e[2])
        elif modo == 'verificar':
            assert r.pedir().registro() == e[1]
        else:
            r.adotar_pedido(e[1])
    if modo == 'adotar':
        r.restaurar_gerador([e for e in eventos if e[0] == 'pedir'][-1][2])
    return r


@pytest.mark.parametrize('L, k', [(2, 3), (3, 3), (4, 6)])
def test_ja_pedidos_e_o_conjunto_de_TODOS_os_pedidos_de_cada_fidelidade_tambem_depois_do_replay_e_da_adocao(L, k):
    """ja_pedidos não entra em assinatura() (é função só da sequência de pedidos, que o replay já confere um a um), então nenhum teste de replay o
    olhava. Aqui: (1) todo pedido é anotado na fidelidade DELE — promoção, evolução, em voo, avaliado, falhado; (2) a chave é o float64 do vetor, e o
    vetor que volta do JSON do jornal tem os MESMOS bytes do que o pedir sorteou; (3) replay e adoção chegam ao mesmo conjunto do original."""
    d, eventos = corrida_com_eventos(L, k, n=90)
    pedidos = [e[1] for e in eventos if e[0] == 'pedir']
    assert {p['origem'] for p in pedidos} == {'promocao', 'evolucao'} and len(d.pendentes) == k - 1 and any(t.aptidao == float('inf') for t in d.historico)
    esperado = [{np.array(p['vetor'], dtype=np.float64).tobytes() for p in pedidos if p['nivel'] == nivel} for nivel in range(L)]
    for nome, nucleo in (('original', d), ('replay por pedir/informar', reconstruir(eventos, L, 'verificar')), ('adoção por adotar_pedido', reconstruir(eventos, L, 'adotar'))):
        obtido = [s.ja_pedidos for s in nucleo.subpops]
        assert [len(x) for x in obtido] == [len(x) for x in esperado], f'{nome}: nº de vetores anotados por fidelidade {[len(x) for x in obtido]}; pedidos por fidelidade {[len(x) for x in esperado]}'
        assert obtido == esperado, f'{nome}: ja_pedidos não é o conjunto dos bytes (float64) dos vetores pedidos em cada fidelidade'
    assert not any(s.fila_promocao for s in d.subpops)                               # sem promoção em curso (a fila é o único estado que a adoção não refaz) ...
    a, b = d.pedir(), reconstruir(eventos, L, 'adotar').pedir()
    assert a.registro() == b.registro(), '... o pedido seguinte à adoção é o do original'


def test_adocao_no_meio_de_uma_promocao_nao_repete_nem_pula_candidato():
    """O modo ADOTAR do coordenador recomeça com a fila de promoção VAZIA (é estado derivado). No núcleo 1.0 isso repetia o melhor de baixo, que já
    tinha subido e ainda estava em voo. No 1.1 a fila se refaz filtrada por ja_pedidos — que adotar_pedido também anota: sobe o que faltava subir."""
    d, eventos = novo(L=3), []

    def pedir_e_gravar():
        t = d.pedir()
        eventos.append(('pedir', json.loads(json.dumps(t.registro())), d.estado_do_gerador()))
        return t

    baixos = [pedir_e_gravar() for _ in range(9)]
    for i, t in enumerate(baixos):
        eventos.append(('informar', t.id, 1.0 + i))
        d.informar(t.id, 1.0 + i)
    p0, p1 = pedir_e_gravar(), pedir_e_gravar()                                      # 2 das 3 promoções ao nível 1, as duas EM VOO: a queda é aqui
    assert [p.origem for p in (p0, p1)] == ['promocao'] * 2 and len(d.subpops[1].fila_promocao) == 1
    adotado = reconstruir(eventos, 3, 'adotar', semente=7, dim=12)
    assert adotado.subpops[1].fila_promocao == [] and sorted(adotado.pendentes) == [p0.id, p1.id]   # a pré-condição: a fila em curso se perdeu
    a, b = d.pedir(), adotado.pedir()
    assert b.origem == 'promocao' and not any(np.array_equal(b.vetor, p.vetor) for p in (p0, p1)), 'depois da adoção, um vetor que já está EM VOO em cima subiu de novo'
    assert np.array_equal(b.vetor, baixos[2].vetor) and a.registro() == b.registro(), 'sobe o 3º melhor de baixo, como no original: a adoção não repete nem pula'
    assert d.pedir().registro() == adotado.pedir().registro()                        # e o pedido seguinte (evolução do bracket 1, com sorteio) também


def test_vetor_ja_pedido_em_cima_por_EVOLUCAO_tambem_nao_sobe_de_novo():
    """"Já pedido em cima" vale para qualquer origem. O filho de uma evolução PODE sair idêntico a um vetor de baixo (com pool pequeno os extras vêm
    da população global, onde o promovido aparece duas vezes: r2 = r3 dá uma cópia exata — acontece, ver a varredura da auditoria). Aqui o pedido de
    cima entra pelo jornal (adotar_pedido, origem 'evolucao') com o vetor do melhor de baixo."""
    d = novo(L=3)
    baixos = [d.pedir() for _ in range(9)]
    for i, t in enumerate(baixos):
        d.informar(t.id, 1.0 + i)
    d.adotar_pedido(pedido(9, bracket=0, degrau=1, nivel=1, id_pai=0, vetor=json.loads(json.dumps(baixos[0].registro()['vetor']))))
    t = d.pedir()
    assert t.origem == 'promocao' and not np.array_equal(t.vetor, baixos[0].vetor), 'o melhor de baixo já foi pedido em cima (por evolução): pedi-lo de novo repete a execução'
    assert np.array_equal(t.vetor, baixos[1].vetor)


def test_promovido_que_FALHOU_em_cima_continua_ja_pedido_e_nao_sobe_de_novo():
    """Treino determinístico dada a semente: repetir em cima o vetor que falhou lá falharia de novo (e a falha transitória já teve as tentativas do
    coordenador). Com 2 em voo o falhado ainda é tirado da população de cima por um filho — só ja_pedidos lembra dele."""
    d = novo(L=3)
    _, medios = primeira_iteracao_L3(d, baixos=[1.0 + i for i in range(9)], medios=[0.5, 0.6, 0.7])
    x = d.pedir()
    assert (x.nivel, x.origem) == (2, 'promocao') and np.array_equal(x.vetor, medios[0].vetor)
    d.informar(x.id, None)                                                           # falhou na fidelidade máxima
    filhos = [d.pedir() for _ in range(3)]                                           # degrau 0 do bracket 1
    d.informar(filhos[0].id, 9.0); d.informar(filhos[1].id, 9.0)                     # o 3º fica em voo: abre-se o bracket 2, no nível 2
    for _ in range(3):
        t = d.pedir()
        assert (t.bracket, t.nivel) == (2, 2)
        d.informar(t.id, 0.9)
    assert linha_de(x.vetor, d.subpops[2].vetores) is None                           # o falhado saiu da população de cima (pré-condição do cenário)
    d.informar(filhos[2].id, 9.0)
    y = d.pedir()
    assert (y.bracket, y.degrau, y.nivel, y.origem) == (1, 1, 2, 'promocao')
    assert not np.array_equal(y.vetor, x.vetor), 'o promovido que FALHOU na fidelidade máxima foi pedido lá de novo'
    assert np.array_equal(y.vetor, medios[1].vetor)


def test_duas_linhas_iguais_em_baixo_nao_poem_o_mesmo_vetor_duas_vezes_na_fila_de_promocao():
    """ja_pedidos só sabe do que JÁ foi pedido em cima, e os candidatos de uma mesma fila ainda não foram. A população de baixo PODE ter duas linhas
    iguais: a mesma tripla (r1, r2, r3) sorteada duas vezes, com o pool parado e o cruzamento inteiro, dá o mesmo filho; as duas avaliações
    (determinísticas) empatam e cada uma entra na posição do seu pai. Com 1 dimensão isso é comum (32 promoções repetidas em 1920 corridas da auditoria);
    abaixo, só pedir/informar com 1 em voo. Primeiro a regra, com a população montada à mão; depois o caso real."""
    d = novo(L=3)
    baixo = d.subpops[0]
    baixo.vetores[3] = baixo.vetores[1]                                              # o mesmo vetor em duas posições, com a mesma aptidão
    baixo.aptidao[:] = [5, 1, 7, 1, 9, 3, 8, 4, 6]
    subiram = [linha_de(d._promover(0, 1, n=3), baixo.vetores) for _ in range(3)]
    assert subiram == [1, 5, 7], f'linhas de baixo que subiram: {subiram}; a linha 3 é o MESMO vetor da 1 — sobe uma vez só, e a vaga vai para o próximo melhor'
    assert d.subpops[1].fila_promocao == []

    d = D.Dehb(1, D.escada_de_passos(1_000_000, 3), semente=11)
    baixos = []
    for _ in range(9):
        baixos.append(d.pedir())
        d.informar(baixos[-1].id, float((baixos[-1].vetor[0] - 0.3) ** 2))
    assert np.array_equal(baixos[8].vetor, baixos[6].vetor) and np.array_equal(d.subpops[0].vetores[8], d.subpops[0].vetores[6]), 'pré-condição do cenário: o filho 8 saiu idêntico ao 6'
    promovidos = [d.pedir() for _ in range(3)]
    assert [t.origem for t in promovidos] == ['promocao'] * 3
    assert len({t.vetor.tobytes() for t in promovidos}) == 3, f'o mesmo vetor foi promovido duas vezes à mesma fidelidade: {[t.vetor.tolist() for t in promovidos]}'
    assert [linha_de(t.vetor, d.subpops[0].vetores) for t in promovidos] == [3, 6, 1]    # o melhor, o segundo (uma vez só) e o PRÓXIMO melhor


# COMPORTAMENTO DO PACOTE, MANTIDO DE PROPÓSITO (auditoria de 19/09/2026; docstring do núcleo). ja_pedidos só é consultado na PROMOÇÃO; a EVOLUÇÃO
# pode devolver um vetor idêntico a um já pedido na mesma fidelidade. Dois caminhos: (1) pool pequeno (degrau de 1 ou 2 vagas): os extras vêm da
# população global, onde o promovido mora em dois níveis — r2 = r3 dá uma CÓPIA exata de r1, e o filho herda o resto do alvo; (2) a mesma tripla
# (r1, r2, r3) sorteada duas vezes com o pool parado. Frequência em corridas de 300 pedidos: 19% das corridas com 5 dimensões; com as 12 do espaço
# real, 24 de 1800 (2 de 600 com 2 níveis), ~1e-5 por pedido. Evitar exigiria re-sortear (outra sequência de sorteios, mais um desvio do pacote):
# não compensa. O xfail estrito TRAVA o comportamento: se um dia alguém mudar isso, o XPASS avisa — e VERSAO_NUCLEO e os dourados têm de mudar junto.
DEFEITO_EVOLUCAO_REPETIDA = ('comportamento do pacote mantido de propósito: _evoluir pode devolver um vetor já pedido na mesma fidelidade '
                             '(ja_pedidos só filtra a promoção); com 12 dimensões ~1e-5 por pedido')


@pytest.mark.xfail(strict=True, reason=DEFEITO_EVOLUCAO_REPETIDA)
def test_evolucao_nao_pede_de_novo_um_vetor_ja_pedido_na_mesma_fidelidade():
    d = novo(L=2, semente=7, dim=5)                                                  # 1 em voo, 2 níveis: o pedido 11 (evolução, no topo) sai igual ao 3 (o promovido)
    vistos = {}
    for _ in range(12):
        t = d.pedir()
        chave = (t.nivel, t.vetor.tobytes())
        assert chave not in vistos, f'o pedido {t.id} ({t.origem}, pais {t.pais}) repete o vetor do pedido {vistos[chave]} na fidelidade {t.fidelidade}'
        vistos[chave] = t.id
        d.informar(t.id, sum((x - 0.3) ** 2 for x in t.registro()['vetor']))


class _FloatQueLevanta:
    def __init__(self, erro):
        self.erro = erro

    def __float__(self):
        raise self.erro


def conferir_trabalho_fechado(d, t, r, concluidos_antes):
    """Os invariantes de um informar que deu certo, seja qual for o valor: saiu de pendentes, fechou a vaga, entrou no histórico com float nativo."""
    assert r is t and t.id not in d.pendentes and d.historico[-1] is t and sum(h.id == t.id for h in d.historico) == 1
    assert d.brackets[t.bracket].concluidos[t.degrau] == concluidos_antes + 1
    assert type(t.aptidao) is float and (np.isfinite(t.aptidao) or t.aptidao == float('inf')), f'aptidão registrada: {t.aptidao!r}'
    assert json.dumps(t.aptidao) and isinstance(t.substituiu, bool)                  # o coordenador grava isto no jornal
    assert all(np.isfinite(a) or a == float('inf') for s in d.subpops for a in s.aptidao), 'NaN ou -inf na população: a seleção (<=) e o argsort se perdem'
    with pytest.raises(KeyError, match='não está pendente'):
        d.informar(t.id, 0.0)                                                        # informar duas vezes continua recusado


def dois_nucleos_com_um_avaliado_e_um_em_voo():
    nucleos = []
    for _ in range(2):
        d = novo(L=2)
        a, b, c = [d.pedir() for _ in range(3)]
        d.informar(a.id, 0.5)
        nucleos.append((d, b))
    return nucleos


NAO_NUMEROS = {'texto': 'abc', 'texto vazio': '', 'bytes': b'\xff\xfe', 'lista': [0.5], 'tupla': (0.5,), 'dicionario': {'aptidao': 0.5}, 'objeto': object(),
               'complexo': 1j, 'Decimal sNaN': Decimal('sNaN'), 'Decimal NaN': Decimal('NaN'), 'Decimal -Infinity': Decimal('-Infinity'), "texto 'nan'": 'nan',
               "texto '-inf'": '-inf', 'float32 nan': np.float32('nan'), 'float32 -inf': np.float32('-inf')}


@pytest.mark.parametrize('caso', list(NAO_NUMEROS))
def test_aptidao_que_nao_e_numero_finito_de_qualquer_tipo_e_falha_igual_a_None_sem_excecao(caso):
    """N2. Antes, np.isfinite('abc') levantava TypeError DEPOIS do pop: o trabalho sumia de pendentes sem fechar a vaga e o bracket travava para
    sempre. float() de texto levanta ValueError e de lista TypeError: as duas têm de virar falha. O estado final é o de um gêmeo informado com None."""
    (d, t), (gemeo, tg) = dois_nucleos_com_um_avaliado_e_um_em_voo()
    antes = d.brackets[t.bracket].concluidos[t.degrau]
    try:
        r = d.informar(t.id, NAO_NUMEROS[caso])
    except (TypeError, ValueError) as e:
        pytest.fail(f'informar levantou {type(e).__name__} para {caso} ({NAO_NUMEROS[caso]!r}): o que não é número finito é FALHA (+inf), não exceção: {e}')
    assert r.aptidao == float('inf'), f'{caso} virou aptidão {r.aptidao!r}'
    conferir_trabalho_fechado(d, t, r, antes)
    gemeo.informar(tg.id, None)
    assert json.dumps(d.assinatura(), sort_keys=True) == json.dumps(gemeo.assinatura(), sort_keys=True), f'{caso}: o estado não é o de uma falha comum (None)'


@pytest.mark.parametrize('numero', [np.float32(0.25), np.float64(0.25), np.int64(3), Decimal('0.25'), Fraction(1, 4), np.array(0.25)], ids=lambda x: type(x).__name__)
def test_numero_finito_de_outro_tipo_e_aceito_e_guardado_como_float_nativo(numero):
    (d, t), (gemeo, tg) = dois_nucleos_com_um_avaliado_e_um_em_voo()
    antes = d.brackets[t.bracket].concluidos[t.degrau]
    r = d.informar(t.id, numero)
    assert r.aptidao == float(numero), f'{numero!r} é número finito: aptidão {float(numero)}, e não {r.aptidao!r}'
    conferir_trabalho_fechado(d, t, r, antes)                                        # inclusive type(...) is float: np.float32 não passa no json.dumps do jornal
    gemeo.informar(tg.id, float(numero))
    assert json.dumps(d.assinatura(), sort_keys=True) == json.dumps(gemeo.assinatura(), sort_keys=True)


@pytest.mark.parametrize('ambiguo', ['0.5', b'0.5', ' 1e3 ', True], ids=repr)
def test_texto_numerico_e_bool_deixam_o_estado_integro_seja_qual_for_a_leitura(ambiguo):
    """Comportamento ATUAL, anotado e NÃO travado: float() entende texto numérico, bytes numéricos e bool, então '0.5' vale 0,5 e True vale 1,0 (o
    coordenador nunca manda isso: numero_finito filtra antes). O que se trava é a integridade: ou o número, ou falha — nunca meio caminho."""
    (d, t), _ = dois_nucleos_com_um_avaliado_e_um_em_voo()
    antes = d.brackets[t.bracket].concluidos[t.degrau]
    r = d.informar(t.id, ambiguo)
    assert r.aptidao in (float(ambiguo), float('inf'))
    conferir_trabalho_fechado(d, t, r, antes)


@pytest.mark.parametrize('ruim', [_FloatQueLevanta(RuntimeError('sensor')), _FloatQueLevanta(ZeroDivisionError()), 10 ** 400], ids=['RuntimeError', 'ZeroDivisionError', 'int_de_400_digitos_OverflowError'])
def test_conversao_que_levanta_outra_excecao_nao_toca_no_estado_e_o_trabalho_continua_pendente(ruim):
    """A conversão vem ANTES do pop. Se float() levanta algo que não é TypeError/ValueError (um __float__ defeituoso; OverflowError de um inteiro
    gigante), ou a exceção sai com o estado INTACTO e o trabalho ainda pendente, ou o valor vira falha — o que não pode é o trabalho sair de
    pendentes sem fechar a vaga do degrau (bracket travado para sempre, que era o defeito)."""
    (d, t), _ = dois_nucleos_com_um_avaliado_e_um_em_voo()
    antes, concluidos = json.dumps(d.assinatura(), sort_keys=True), d.brackets[t.bracket].concluidos[t.degrau]
    try:
        r = d.informar(t.id, ruim)
    except (RuntimeError, ArithmeticError):
        assert t.id in d.pendentes, 'a exceção saiu DEPOIS de o trabalho sair de pendentes: a vaga do degrau nunca mais fecha'
        assert json.dumps(d.assinatura(), sort_keys=True) == antes and t.aptidao is None and all(h is not t for h in d.historico)
        r = d.informar(t.id, 0.4)                                                    # e o trabalho ainda pode ser informado direito
        assert r.aptidao == 0.4
    else:
        assert r.aptidao == float('inf')
    conferir_trabalho_fechado(d, t, r, concluidos)
