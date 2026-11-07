"""Pragas do FarmGym.

FarmGym v4 (06/09/2026): a população de pragas passa a ter dinâmica de verdade. Até a v3 ela era **inerte**, e não por
calibração ruim: `nova_pop = max(0, pop_atual + int(mudanca_populacao))` truncava a variação para zero. As populações
alcançáveis vinham do reset (`randint(0, 10)` → 0..9) e, para pop ≤ 9, a variação por passo ficava entre −0,74 e +0,63,
que `int()` zera nos dois sentidos; crescer exigiria pop ≥ 15, inalcançável. A prova está nos dados da v3: **todas** as
políticas que não pulverizam terminavam o episódio com exatamente a mesma população total, inclusive a política "só
esperar", que nunca planta.

Quatro defeitos corrigidos aqui:
  1. `int()` no laço e no abate: a dinâmica não existia e o pesticida não removia populações pequenas.
  2. `if pop_atual <= 0: continue`: uma célula zerada (pelo reset ou pelo pesticida) ficava fora da dinâmica para
     sempre — estado absorvente, sem dispersão nem imigração.
  3. O residual do pesticida multiplicava a mudança LÍQUIDA (`mudanca *= (1 - efeito)`), o que também enfraquecia a
     mortalidade e a predação: com a mudança já negativa (inverno), pulverizar fazia a praga cair MENOS. Agora o
     residual é mortalidade extra, com teto.
  4. `_calcular_fator_crescimento` não tinha `else` para a célula sem planta: uma célula VAZIA crescia como uma célula
     com planta saudável. Agora a praga depende do hospedeiro, e é isso que faz a pressão de praga seguir a lavoura.
A dispersão também mudou de forma: era uma soma com limiar inalcançável (`pop_vizinho > 20`) que criava população do
nada; virou difusão (relaxação para a média da vizinhança), que conserva e recoloniza.
"""
import numpy as np
from typing import Dict, Tuple, List
import random

class Pragas:
    def __init__(self, tamanho_grade: Tuple[int, int] = (10, 10)):
        self.tamanho_grade = tamanho_grade

        self.populacao_maxima = 100.0
        # FarmGym v4: taxas recalibradas. As da v3 (0,07 / 0,04) só existiam no papel — o `int()` as anulava — e, com a
        # dependência de hospedeiro ligada, dariam saturação só no passo ~134, depois do fim do primeiro ciclo da
        # cultura. Alvo: sem controle, a média por célula sai de ~5 e cruza 90 por volta do passo 30; com o professor
        # pulverizando no limiar, a média do episódio fica baixa e o máximo não satura.
        self.taxa_reproducao = 0.16          # por passo, × fator de crescimento (clima × hospedeiro)
        self.taxa_dispersao = 0.02           # relaxação difusiva para a média da vizinhança de Moore
        self.taxa_mortalidade = 0.06         # por passo
        self.predacao_por_ave = 0.0015       # por ave e por praga (o espantalho, ao afastar aves, tira este predador)
        self.imigracao = 0.15                # entrada de fora do campo, por célula e por passo: mata o estado absorvente
        self.fator_sem_hospedeiro = 0.35     # célula sem planta: a praga sobrevive em restos e ervas, não cresce pleno
        self.mortalidade_pesticida_max = 0.30  # teto da mortalidade extra do residual, por passo

        self.resetar()

    def resetar(self):
        self.populacao_pragas = {}
        self.passo_tempo = 0
        self._inicializar_populacao_pragas()

    def _inicializar_populacao_pragas(self):
        for x in range(self.tamanho_grade[0]):
            for y in range(self.tamanho_grade[1]):
                # v4: float e nunca exatamente 0. Com `randint(0, 10)` uma célula em 16 nascia com 0 e, pelo guarda
                # `if pop_atual <= 0: continue`, ficava fora da dinâmica pelo episódio inteiro.
                self.populacao_pragas[(x, y)] = float(np.random.uniform(2.0, 8.0))

    def atualizar(self, clima: Dict, plantas: Dict, defensivos: Dict, passaros: Dict = None):
        self.passo_tempo += 1

        if passaros is None:
            passaros = {}

        # A dispersão lê o estado do passo anterior em todas as células: sem o instantâneo, a ordem de iteração do
        # dicionário mudaria o resultado (a célula processada depois veria vizinhos já atualizados).
        anterior = dict(self.populacao_pragas)
        nova_populacao = {}

        for pos in list(self.populacao_pragas.keys()):
            pop_atual = float(anterior[pos])

            fator_crescimento = self._calcular_fator_crescimento(pos, clima, plantas, defensivos)

            mudanca_populacao = pop_atual * self.taxa_reproducao * fator_crescimento
            mudanca_populacao -= pop_atual * self.taxa_mortalidade

            pop_passaros = float(passaros.get(pos, 0))
            mudanca_populacao -= pop_passaros * self.predacao_por_ave * pop_atual

            # v4: o residual do pesticida é MORTALIDADE EXTRA, não um multiplicador da mudança líquida.
            efeito_pesticida = self._calcular_efeito_pesticida(pos, defensivos)
            mudanca_populacao -= pop_atual * min(self.mortalidade_pesticida_max, efeito_pesticida)

            mudanca_populacao += self._calcular_fator_dispersao(pos, anterior)
            mudanca_populacao += self.imigracao

            # v4: sem int(). Com pop <= 14 o truncamento zerava a variação nos DOIS sentidos.
            nova_populacao[pos] = min(self.populacao_maxima, max(0.0, pop_atual + mudanca_populacao))

        self.populacao_pragas = nova_populacao

    def _calcular_fator_crescimento(self, pos: Tuple[int, int], clima: Dict, plantas: Dict, defensivos: Dict) -> float:
        fator = 1.0

        temp = clima.get('temperatura', 20)
        umidade_ar = clima.get('umidade_ar', 0.5)

        if 20 <= temp <= 30:
            fator *= 1.3
        elif temp < 10 or temp > 35:
            fator *= 0.4

        if umidade_ar > 0.7:
            fator *= 1.2

        dados_planta = plantas.get(pos, {})
        if dados_planta:
            estagio_planta = dados_planta.get('estagio', '')
            saude_planta = dados_planta.get('saude', 0)

            # v4: 'colheita' entra na lista — é a fase em que o fruto está exposto e a praga mais importa.
            if estagio_planta in ['crescimento', 'florescimento', 'frutificacao', 'colheita']:
                fator *= (0.5 + 0.5 * saude_planta)
            else:
                fator *= 0.3
        else:
            # v4: sem hospedeiro a praga apenas sobrevive. Na v3 faltava este `else`, então a célula VAZIA crescia
            # exatamente como a célula com planta saudável — e "só esperar" terminava com a mesma praga de quem
            # cultivava o campo inteiro.
            fator *= self.fator_sem_hospedeiro

        return fator

    def _calcular_fator_dispersao(self, pos: Tuple[int, int], anterior: Dict = None) -> float:
        """v4: difusão (relaxação para a média da vizinhança de Moore), sem limiar. A v3 exigia `pop_vizinho > 20`,
        inalcançável porque nenhuma população se movia, e ainda CRIAVA praga do nada: somava a chegada sem debitar a
        saída da célula de origem. A forma nova nunca leva a célula acima da média local."""
        estado = self.populacao_pragas if anterior is None else anterior
        x, y = pos
        vizinhos = [estado[(x + dx, y + dy)]
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    if (dx or dy) and (x + dx, y + dy) in estado]
        if not vizinhos:
            return 0.0
        return self.taxa_dispersao * (float(np.mean(vizinhos)) - float(estado[pos]))

    def _calcular_efeito_pesticida(self, pos: Tuple[int, int], defensivos: Dict) -> float:
        dados_defensivo = defensivos.get(pos, {})

        # v4: SO o inseticida controla praga. Ate a v3 o residual generico de qualquer defensivo entrava aqui com peso
        # 0,4, entao o HERBICIDA tambem matava praga -- de graca e sem estar declarado. Com a praga viva isso virou
        # decisivo: a politica que aplicava herbicida todo dia terminava com 23 pragas por celula contra 93 da que nao
        # aplicava, e por isso vencia a politica que usa o herbicida no limiar (criterio 2 do checklist). Cada acao
        # controla o seu alvo: o herbicida, as ervas; o pesticida, as pragas e a doenca.
        return min(0.95, dados_defensivo.get('efeito_pesticida', 0))

    def obter_observacao(self) -> Dict:
        return self.populacao_pragas.copy()

    def obter_populacao_em(self, x: int, y: int) -> float:
        pos = (x, y)
        return float(self.populacao_pragas.get(pos, 0.0))

    def obter_populacao_total(self) -> float:
        return float(sum(self.populacao_pragas.values()))

    def obter_populacao_maxima_no_campo(self) -> float:
        """v4: maior população entre as células. É o agregado que a observação expõe: a ação `pesticida` é do campo
        inteiro, então a estatística que decide a ação é o máximo, não a média."""
        return float(max(self.populacao_pragas.values())) if self.populacao_pragas else 0.0

    def esta_nivel_infestacao(self, x: int, y: int, limiar: float = 50.0) -> bool:
        return self.obter_populacao_em(x, y) >= limiar

    def aplicar_tratamento(self, x: int, y: int, efetividade: float = 0.8):
        pos = (x, y)
        if pos in self.populacao_pragas:
            # v4: sem int(). Com população 1..4 o abate truncava para 0 e a aplicação não fazia nada.
            self.populacao_pragas[pos] = max(0.0, float(self.populacao_pragas[pos]) * (1.0 - efetividade))

    def definir_parametros_populacao(self, pop_max: float, taxa_repro: float, taxa_dispersao: float):
        self.populacao_maxima = float(pop_max)
        self.taxa_reproducao = taxa_repro
        self.taxa_dispersao = taxa_dispersao
