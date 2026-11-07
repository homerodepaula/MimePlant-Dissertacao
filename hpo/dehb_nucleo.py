"""hpo/dehb_nucleo.py — DEHB (Awad, Mallik e Hutter, IJCAI 2021) em implementação nativa: só numpy, interface pedir/informar.

Por que nativa (18/09/2026, D-34). O pacote de referência (`dehb` 0.1.2, de automl/DEHB) exige `numpy < 2`: instalá-lo rebaixaria
o numpy 2.2.6 fixado em `requirements-lock.txt` — o numpy com que toda referência do projeto reproduz bit a bit — e traria
Dask, distributed e ConfigSpace para rodar numa GPU só. Este módulo reimplementa o algoritmo a partir do artigo e do código
de referência (lido, não instalado); `tests/test_dehb_nucleo.py` trava cada regra abaixo.

O que é reproduzido da referência 0.1.2, modo assíncrono `ask`/`tell` (arquivo:linha do pacote):
  escada        bracket i usa as s+1 fidelidades mais altas, s = L-1-(i mod L), com n0 = floor(L/(s+1))·eta^s configurações no
                primeiro degrau e max(int(n0·eta^-j), 1) nos seguintes (dehb.py:151-176). É `floor` ANTES de multiplicar, como
                no pacote; o Alg. 2 do artigo usa `ceil` e só coincide com ele até 2 níveis.
  populações    uma subpopulação por fidelidade, do tamanho do maior n daquela fidelidade entre os brackets (dehb.py:436-444),
                sorteada uniforme em [0,1]^d, aptidão inicial +inf = "nunca avaliado" (dehb.py:446-460). A população inicial
                NUNCA é avaliada: já o primeiro pedido é um filho de pais aleatórios, e `filho <= inf` o põe no lugar do pai
                (dehb.py:578-613, 1024-1027). O DEHB não tem fase de partida aleatória como a do TPE.
  agendamento   Successive Halving SÍNCRONO dentro do bracket (um degrau só abre com todos os resultados do anterior) e brackets
                ASSÍNCRONOS entre si: se nenhum bracket ativo pode agendar, abre-se outro (dehb.py:615-649; bracket_manager.py).
  alvo          o pai é percorrido em rodízio dentro da subpopulação da fidelidade, também nas promoções (dehb.py:563-573).
  promoção      nos brackets da PRIMEIRA iteração e acima do primeiro degrau: o melhor já avaliado da fidelidade de baixo que
                ainda não esteja na de cima sobe SEM mutação e sem consumir sorteio (dehb.py:503-561, 578-586).
  evolução      POOL DE PAIS = os n melhores da fidelidade de baixo (no primeiro degrau, da própria), n = configurações do degrau
                de destino, incluídos os nunca avaliados; pool com menos de 3 recebe (3 - len + 1) MUTANTES rand1 tirados da
                concatenação de todas as subpopulações, sem conserto de fronteira (dehb.py:588-602; de.py:655-661); o alvo sai do
                sorteio e, se sobrarem menos de 3, entram vetores uniformes novos (de.py:663-696); rand1 com F = 0,5; cruzamento
                binomial com CR = 0,5, forçando uma dimensão do mutante só quando nenhuma cruzou (de.py:481-488); componente
                fora de [0,1] é re-sorteada, nunca recortada (de.py:152-175).
  seleção       no informar, o filho substitui o OCUPANTE ATUAL da posição do pai se aptidão_filho <= aptidão_ocupante
                (dehb.py:1023-1027). Minimiza-se. Com mais avaliações em voo do que indivíduos numa subpopulação, dois filhos
                disputam a mesma posição e o segundo compete com o primeiro — é da referência.

Desvios deliberados em relação ao pacote (cada um com teste):
  1. Promoção decidida pelo ÍNDICE DO BRACKET (< L), e não pelo contador global de brackets: com avaliações em paralelo um
     resultado atrasado faz o contador passar de L antes de o degrau alto do bracket 0 abrir, e a promoção — parte
     definidora do Alg. 3 do artigo — sumiria conforme a ordem de chegada. "Já subiu" quer dizer já foi PEDIDO na fidelidade de
     cima, em qualquer estado: em voo, avaliado, ou avaliado e depois substituído lá por um filho melhor. O pacote olha só a
     população ATUAL de cima, e com mais de um em voo e 3 níveis ou mais o mesmo vetor sobe duas vezes (medido: 11 de 30 sementes
     com L = 3 e 3 em voo; com L = 2 só há uma promoção por estudo e o caso não existe). Sem candidato NOVO para subir (ninguém
     avaliado com sucesso em baixo, ou todos já subiram), cai na evolução: o pacote levanta IndexError no primeiro caso
     (dehb.py:559) e, no segundo, reavalia o melhor de baixo de novo (dehb.py:556-560) — aqui, com treino determinístico dada a
     semente, isso seria uma execução inteira jogada fora. Com falhas (desvio 5) a fila pode ficar menor que n.
  2. Inteiros por faixas de largura igual, como os categóricos. O pacote arredonda `lo + v·(hi-lo)`, o que dá meia faixa aos
     extremos — um prior diferente do `suggest_int` do TPE e da busca aleatória com que o DEHB é comparado aqui.
  3. Valor decodificado recortado em [lo, hi]: exp(log lo + 1,0·log(hi/lo)) dá 0,0003000000000000001 para hi = 3e-4, e o
     Optuna recusa o trial. Só o valor decodificado é recortado; o vetor da população nunca.
  4. Um gerador só (`numpy.random.Generator`) e `argsort` estável: o estado é função de (semente, sequência de pedir/informar
     com os valores informados). É o que permite a retomada por REPLAY do jornal em `hpo_dehb.py`. O pacote usa um gerador por
     fidelidade e `argsort` instável, e o checkpoint nativo dele não guarda gerador, populações nem trabalhos em voo.
  5. Avaliação que falhou vale +inf: nunca substitui um ocupante avaliado e nunca é promovida (o pacote não trata NaN/inf).
  Mantido do pacote de propósito: a EVOLUÇÃO pode produzir um vetor idêntico a um já pedido na mesma fidelidade (pool pequeno com
     r2 = r3; ou a mesma tripla sorteada duas vezes). Medido em 19/09/2026 com as 12 dimensões do espaço real: 24 de 1800 corridas de
     300 pedidos (2 de 600 com 2 níveis), ~1e-5 por pedido — da ordem de 0,1 % de chance num estudo de 64 FRE. Evitar exigiria
     re-sortear (outra sequência de sorteios, mais um desvio): não compensa. Um teste xfail estrito trava o comportamento.
  6. Não há incumbente global. O do pacote mistura fidelidades (dehb.py:1029-1036); com rand1 ele não entra na evolução, e
     aqui as fidelidades medem quantidades diferentes. `melhor_na_fidelidade_maxima()` é a única noção de "melhor".
Categóricos são codificados por faixas iguais de [0,1] NA ORDEM DA LISTA (ordinal de fato, como no pacote e no Ap. C.2 do artigo).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

VERSAO_NUCLEO = '1.1'     # muda quando a sequência de sorteios ou as regras mudam: o replay do jornal confere
#                           1.1 (19/09/2026): a promoção deixa de repetir vetor já pedido em cima (muda pedidos com L >= 3 e > 1 em voo)
#                           e não põe duas vezes na fila o mesmo vetor (duas linhas iguais em baixo; só visto com 1 dimensão)
F_PADRAO = 0.5            # fator de mutação (artigo, Ap. A, e pacote)
CR_PADRAO = 0.5           # probabilidade de cruzamento (idem)
ETA_PADRAO = 3
TAM_MIN_POOL = 3          # rand1 precisa de três indivíduos distintos


# ── espaço de busca: [0, 1]^d <-> hiperparâmetros ────────────────────────────────────────────────
class EspacoVetorial:
    """Tradução entre o hipercubo unitário, onde a DE opera, e o espaço de `agents/rl/hp.py`:
    nome -> ('log', lo, hi) | ('float', lo, hi) | ('int', lo, hi) | ('cat', [opções]). A ordem das dimensões é a do dicionário."""

    def __init__(self, espaco: Dict[str, tuple]):
        if not espaco:
            raise ValueError('espaço de busca vazio')
        for nome, spec in espaco.items():
            if spec[0] not in ('log', 'float', 'int', 'cat'):
                raise ValueError(f'tipo desconhecido em {nome!r}: {spec[0]!r}')
            if spec[0] == 'cat':
                if not len(spec[1]):
                    raise ValueError(f'categórico sem opções: {nome!r}')
            elif not spec[1] < spec[2]:
                raise ValueError(f'limites inválidos em {nome!r}: {spec[1:]}')
            if spec[0] == 'log' and spec[1] <= 0:
                raise ValueError(f'escala log exige limite inferior positivo: {nome!r}')
        self.espaco = {k: tuple(v) for k, v in espaco.items()}
        self.nomes = list(espaco)
        self.dimensoes = len(self.nomes)

    @staticmethod
    def _faixa(v: float, n: int) -> int:
        """Índice da faixa de largura 1/n que contém v; v = 1,0 cai na última."""
        return int(min(n - 1, max(0, np.floor(float(v) * n))))

    def para_hp(self, vetor: Sequence[float]) -> Dict:
        """Hiperparâmetros (tipos nativos, prontos para JSON e para o Optuna) do ponto `vetor` de [0, 1]^d."""
        vetor = np.asarray(vetor, dtype=float)
        if vetor.shape != (self.dimensoes,) or not np.all(np.isfinite(vetor)) or np.any(vetor < 0) or np.any(vetor > 1):
            raise ValueError(f'vetor fora de [0, 1]^{self.dimensoes}: {vetor}')
        hp = {}
        for v, nome in zip(vetor, self.nomes):
            spec = self.espaco[nome]
            if spec[0] == 'log':
                x = np.exp(np.log(spec[1]) + v * (np.log(spec[2]) - np.log(spec[1])))
                hp[nome] = float(min(spec[2], max(spec[1], x)))
            elif spec[0] == 'float':
                hp[nome] = float(min(spec[2], max(spec[1], spec[1] + v * (spec[2] - spec[1]))))
            elif spec[0] == 'int':
                hp[nome] = int(spec[1]) + self._faixa(v, int(spec[2]) - int(spec[1]) + 1)
            else:
                hp[nome] = spec[1][self._faixa(v, len(spec[1]))]
        return hp

    def para_vetor(self, hp: Dict) -> np.ndarray:
        """Inversa de `para_hp` (inteiros e categóricos vão para o CENTRO da faixa). Serve a testes e a registros."""
        vetor = np.zeros(self.dimensoes)
        for i, nome in enumerate(self.nomes):
            spec, valor = self.espaco[nome], hp[nome]
            if spec[0] == 'log':
                vetor[i] = (np.log(valor) - np.log(spec[1])) / (np.log(spec[2]) - np.log(spec[1]))
            elif spec[0] == 'float':
                vetor[i] = (valor - spec[1]) / (spec[2] - spec[1])
            elif spec[0] == 'int':
                vetor[i] = (int(valor) - int(spec[1]) + 0.5) / (int(spec[2]) - int(spec[1]) + 1)
            else:
                vetor[i] = (list(spec[1]).index(valor) + 0.5) / len(spec[1])
        return np.clip(vetor, 0.0, 1.0)

    def descricao(self) -> List:
        """Forma serializável e ORDENADA do espaço (vai para o cabeçalho do jornal: reordenar o espaço invalida o replay)."""
        return [[nome, [self.espaco[nome][0]] + [list(x) if isinstance(x, (list, tuple)) else x for x in self.espaco[nome][1:]]]
                for nome in self.nomes]


# ── escada do Hyperband ──────────────────────────────────────────────────────────────────────────
def escada_de_passos(passos_max: int, niveis: int, eta: int = ETA_PADRAO) -> List[int]:
    """Fidelidades INTEIRAS em ordem crescente: N // eta^(L-1), ..., N // eta, N. Em inteiros e pelo NÚMERO DE NÍVEIS de
    propósito: o pacote conta os níveis com `-int(log(min/max)/log(eta)) + 1`, e o truncamento de um quociente em ponto
    flutuante que deveria ser inteiro perde um nível em alguns pares (min, max)."""
    if niveis < 2 or eta < 2 or passos_max < eta ** (niveis - 1):
        raise ValueError(f'escada inválida: passos_max={passos_max} niveis={niveis} eta={eta}')
    return [int(passos_max) // eta ** k for k in range(niveis - 1, -1, -1)]


def plano_do_bracket(iteracao: int, L: int, eta: int = ETA_PADRAO) -> Tuple[List[int], List[int]]:
    """(n de configurações por degrau, índice da fidelidade de cada degrau) do bracket de índice `iteracao`."""
    s = L - 1 - (iteracao % L)
    n0 = int(np.floor(L / (s + 1))) * eta ** s
    ns = [max(n0 // eta ** j, 1) for j in range(s + 1)]     # o pacote faz int(n0 * eta**(-j)), em ponto flutuante; aqui é exato
    return ns, list(range(L - s - 1, L))


def tamanhos_das_populacoes(L: int, eta: int = ETA_PADRAO) -> List[int]:
    tam = [0] * L
    for i in range(L):
        ns, niveis = plano_do_bracket(i, L, eta)
        for n, nivel in zip(ns, niveis):
            tam[nivel] = max(tam[nivel], n)
    return tam


def custo_de_uma_iteracao(fidelidades: Sequence[float], eta: int = ETA_PADRAO) -> float:
    """Custo nominal dos L brackets de uma iteração do Hyperband, em unidades da fidelidade máxima."""
    L = len(fidelidades)
    return float(sum(n * fidelidades[nivel] for i in range(L) for n, nivel in zip(*plano_do_bracket(i, L, eta))) / fidelidades[-1])


# ── estado de um bracket (Successive Halving síncrono) ───────────────────────────────────────────
class Bracket:
    def __init__(self, indice: int, ns: List[int], niveis: List[int]):
        self.indice, self.ns, self.niveis = indice, list(ns), list(niveis)
        self.agendados = [0] * len(ns)
        self.concluidos = [0] * len(ns)

    def degrau_agendavel(self) -> Optional[int]:
        """O degrau que pode receber um trabalho agora, ou None. Um degrau só abre depois de TODOS os resultados do anterior."""
        for r, n in enumerate(self.ns):
            if self.agendados[r] < n:
                return r if (r == 0 or self.concluidos[r - 1] == self.ns[r - 1]) else None
        return None

    def terminou(self) -> bool:
        return all(c == n for c, n in zip(self.concluidos, self.ns))


@dataclass
class Trabalho:
    """Uma avaliação pedida pelo DEHB. `origem`: 'promocao' (subiu sem mutar) ou 'evolucao'. `pais`: de onde vieram os
    indivíduos do sorteio do rand1 — 'pool' (fidelidade de baixo), 'global' (mutantes da população global), 'uniforme'."""
    id: int
    vetor: np.ndarray
    fidelidade: float
    nivel: int
    id_pai: int
    bracket: int
    degrau: int
    origem: str
    pais: Dict[str, int] = field(default_factory=dict)
    aptidao: Optional[float] = None
    substituiu: Optional[bool] = None

    def registro(self) -> Dict:
        """Forma serializável do PEDIDO (floats exatos: o repr do JSON faz ida e volta sem perda)."""
        return {'id': self.id, 'vetor': [float(x) for x in self.vetor], 'fidelidade': self.fidelidade, 'nivel': self.nivel,
                'id_pai': self.id_pai, 'bracket': self.bracket, 'degrau': self.degrau, 'origem': self.origem, 'pais': dict(self.pais)}


@dataclass
class _Subpopulacao:
    vetores: np.ndarray
    aptidao: np.ndarray
    proximo_pai: int = 0
    fila_promocao: List[np.ndarray] = field(default_factory=list)
    ja_pedidos: set = field(default_factory=set)      # bytes de todo vetor já PEDIDO nesta fidelidade (estado derivado dos pedidos)


class Dehb:
    """DEHB assíncrono. Uso: `t = d.pedir()` → avaliar `t.vetor` em `t.fidelidade` → `d.informar(t.id, aptidão)` (menor é
    melhor; NaN/inf = a avaliação falhou). Vários `pedir()` antes de um `informar()` são permitidos: é assim que se rodam
    avaliações em paralelo. `fidelidades` vem em ordem crescente (ver `escada_de_passos`)."""

    def __init__(self, dimensoes: int, fidelidades: Sequence[float], eta: int = ETA_PADRAO,
                 fator_mutacao: float = F_PADRAO, prob_cruzamento: float = CR_PADRAO, semente: int = 0):
        if dimensoes < 1:
            raise ValueError('dimensoes < 1')
        if len(fidelidades) < 2 or list(fidelidades) != sorted(set(fidelidades)):
            raise ValueError(f'fidelidades devem ser pelo menos duas, distintas e crescentes: {list(fidelidades)}')
        self.dimensoes, self.eta = int(dimensoes), int(eta)
        self.F, self.CR = float(fator_mutacao), float(prob_cruzamento)
        self.fidelidades = list(fidelidades)
        self.L = len(self.fidelidades)
        self.rng = np.random.default_rng(int(semente))
        self.subpops = [_Subpopulacao(vetores=self.rng.uniform(0.0, 1.0, size=(n, self.dimensoes)), aptidao=np.full(n, np.inf))
                        for n in tamanhos_das_populacoes(self.L, self.eta)]
        self.brackets: List[Bracket] = []
        self.pendentes: Dict[int, Trabalho] = {}
        self.historico: List[Trabalho] = []
        self._proximo_id = 0

    # ── pedir ────────────────────────────────────────────────────────────────────────────────────
    def _abrir_bracket(self) -> Bracket:
        ns, niveis = plano_do_bracket(len(self.brackets), self.L, self.eta)
        self.brackets.append(Bracket(len(self.brackets), ns, niveis))
        return self.brackets[-1]

    def _bracket_para_agendar(self) -> Tuple[Bracket, int]:
        for b in self.brackets:
            r = b.degrau_agendavel()
            if r is not None:
                return b, r
        return self._abrir_bracket(), 0

    def pedir(self) -> Trabalho:
        b, r = self._bracket_para_agendar()
        nivel = b.niveis[r]
        sub = self.subpops[nivel]
        id_pai = sub.proximo_pai
        vetor, origem, pais = None, 'promocao', {}
        if b.indice < self.L and r > 0:
            vetor = self._promover(nivel_baixo=b.niveis[r - 1], nivel_alto=nivel, n=b.ns[r])
        if vetor is None:
            vetor, pais = self._evoluir(sub.vetores[id_pai], nivel_pool=(b.niveis[r - 1] if r > 0 else nivel), n=b.ns[r])
            origem = 'evolucao'
        return self._registrar(Trabalho(id=self._proximo_id, vetor=vetor, fidelidade=self.fidelidades[nivel], nivel=nivel,
                                        id_pai=id_pai, bracket=b.indice, degrau=r, origem=origem, pais=pais))

    def _registrar(self, t: Trabalho) -> Trabalho:
        sub = self.subpops[t.nivel]
        sub.proximo_pai = (t.id_pai + 1) % len(sub.vetores)
        sub.ja_pedidos.add(np.asarray(t.vetor, dtype=float).tobytes())
        self.brackets[t.bracket].agendados[t.degrau] += 1
        self.pendentes[t.id] = t
        self._proximo_id = t.id + 1
        return t

    def _promover(self, nivel_baixo: int, nivel_alto: int, n: int) -> Optional[np.ndarray]:
        baixo, alto = self.subpops[nivel_baixo], self.subpops[nivel_alto]
        if not alto.fila_promocao:
            ordem = [i for i in np.argsort(baixo.aptidao, kind='stable') if np.isfinite(baixo.aptidao[i])]
            if not ordem:
                return None                    # ninguém avaliado com sucesso em baixo: evoluir em vez de promover (desvio 1)
            candidatos = {}                    # bytes -> melhor posição: duas linhas IGUAIS em baixo (o mesmo filho sorteado duas vezes) sobem uma vez só
            for i in ordem:
                candidatos.setdefault(baixo.vetores[i].tobytes(), i)
            alto.fila_promocao = [baixo.vetores[i].copy() for chave, i in candidatos.items() if chave not in alto.ja_pedidos][:n]
            if not alto.fila_promocao:
                return None                    # todos os bons de baixo já foram pedidos em cima: evoluir (desvio 1)
        return alto.fila_promocao.pop(0)

    def _rand1(self, pool: np.ndarray, alvo: np.ndarray) -> Tuple[np.ndarray, int]:
        """(mutante r1 + F·(r2 - r3), nº de vetores uniformes usados para completar o sorteio). O alvo sai do sorteio."""
        iguais = np.all(pool == alvo, axis=1)
        if len(pool) > 1 and iguais.any():
            pool = np.delete(pool, int(np.argmax(iguais)), axis=0)
        faltam = max(0, TAM_MIN_POOL - len(pool))
        if faltam:
            pool = np.concatenate([pool, self.rng.uniform(0.0, 1.0, size=(faltam, self.dimensoes))])
        r1, r2, r3 = pool[self.rng.choice(len(pool), size=3, replace=False)]
        return r1 + self.F * (r2 - r3), faltam

    def _evoluir(self, alvo: np.ndarray, nivel_pool: int, n: int) -> Tuple[np.ndarray, Dict[str, int]]:
        fonte = self.subpops[nivel_pool]
        pool = fonte.vetores[np.argsort(fonte.aptidao, kind='stable')[:n]]
        pais = {'pool': int(len(pool)), 'global': 0, 'uniforme': 0}
        if len(pool) < TAM_MIN_POOL:
            todos = np.concatenate([s.vetores for s in self.subpops])
            extras = [self._rand1(todos, alvo)[0] for _ in range(TAM_MIN_POOL - len(pool) + 1)]
            pool = np.concatenate([pool, np.array(extras)])
            pais['global'] = len(extras)
        mutante, pais['uniforme'] = self._rand1(pool, alvo)
        cruza = self.rng.random(self.dimensoes) < self.CR
        if not cruza.any():
            cruza[self.rng.integers(0, self.dimensoes)] = True
        filho = np.where(cruza, mutante, alvo)
        fora = (filho < 0.0) | (filho > 1.0)
        if fora.any():
            filho[fora] = self.rng.uniform(0.0, 1.0, size=int(fora.sum()))
        return filho, pais

    # ── informar ─────────────────────────────────────────────────────────────────────────────────
    def informar(self, id_trabalho: int, aptidao: Optional[float]) -> Trabalho:
        """Registra o resultado (menor é melhor; None/NaN/inf = falhou) e faz a seleção da DE contra o ocupante ATUAL da
        posição do pai. Fecha a vaga do degrau no bracket mesmo quando a avaliação falhou — senão o bracket travaria."""
        if id_trabalho not in self.pendentes:
            raise KeyError(f'trabalho {id_trabalho} não está pendente (já informado, ou nunca pedido)')
        try:                                   # ANTES de tocar no estado: o que não é número finito é falha, nunca exceção a meio caminho
            valor = float(aptidao)
        except (TypeError, ValueError, OverflowError):
            valor = float('inf')
        t = self.pendentes.pop(id_trabalho)
        t.aptidao = valor if np.isfinite(valor) else float('inf')
        self.brackets[t.bracket].concluidos[t.degrau] += 1
        sub = self.subpops[t.nivel]
        t.substituiu = bool(t.aptidao <= sub.aptidao[t.id_pai])
        if t.substituiu:
            sub.vetores[t.id_pai] = t.vetor
            sub.aptidao[t.id_pai] = t.aptidao
        self.historico.append(t)
        return t

    # ── reconstrução a partir de um jornal, sem sortear (modo ADOTAR de hpo_dehb.py) ───────────
    def adotar_pedido(self, registro: Dict) -> Trabalho:
        """Reinstala um pedido já feito (por outra versão do núcleo, por exemplo) sem planejar nem consumir sorteio."""
        if int(registro['id']) != self._proximo_id:
            raise ValueError(f"pedido fora de ordem: esperado {self._proximo_id}, veio {registro['id']}")
        while len(self.brackets) <= int(registro['bracket']):
            self._abrir_bracket()
        b = self.brackets[int(registro['bracket'])]
        r, nivel = int(registro['degrau']), int(registro['nivel'])
        if r >= len(b.ns) or b.niveis[r] != nivel or b.agendados[r] >= b.ns[r] or not 0 <= int(registro['id_pai']) < len(self.subpops[nivel].vetores):
            raise ValueError(f'pedido incompatível com a escada deste estudo: {registro}')
        return self._registrar(Trabalho(id=int(registro['id']), vetor=np.asarray(registro['vetor'], dtype=float),
                                        fidelidade=self.fidelidades[nivel], nivel=nivel, id_pai=int(registro['id_pai']),
                                        bracket=b.indice, degrau=r, origem=str(registro.get('origem', 'evolucao')),
                                        pais=dict(registro.get('pais') or {})))

    def estado_do_gerador(self) -> Dict:
        return self.rng.bit_generator.state

    def restaurar_gerador(self, estado: Dict) -> None:
        self.rng.bit_generator.state = estado

    # ── consultas ────────────────────────────────────────────────────────────────────────────────
    def melhor_na_fidelidade_maxima(self) -> Optional[Trabalho]:
        cheios = [t for t in self.historico if t.nivel == self.L - 1 and np.isfinite(t.aptidao)]
        return min(cheios, key=lambda t: t.aptidao) if cheios else None

    def assinatura(self) -> Dict:
        """Resumo exato do estado, para conferir que um replay chegou ao mesmo ponto (inclui o gerador)."""
        return {'pedidos': self._proximo_id, 'informados': len(self.historico), 'pendentes': sorted(self.pendentes),
                'brackets': [[b.indice, list(b.agendados), list(b.concluidos)] for b in self.brackets],
                'aptidao': [[float(a) for a in s.aptidao] for s in self.subpops],
                'vetores': [s.vetores.tolist() for s in self.subpops],
                'proximo_pai': [s.proximo_pai for s in self.subpops],
                'filas': [[v.tolist() for v in s.fila_promocao] for s in self.subpops],
                'gerador': self.rng.bit_generator.state}
