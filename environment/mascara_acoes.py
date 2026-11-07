"""Wrapper de mascaramento de acoes invalidas para o ambiente FarmGym.

Injeta em info['mascara_acoes'] um vetor 0/1 indicando quais das 15 acoes sao
validas no estado atual. Usado por todos os agentes (BC, DAgger) e pelos
baselines (oraculo, aleatorio).

Regra geral: uma acao e invalida quando nao tem efeito nenhum no estado. As regras
de plantar/colher/regar/herbicida/pesticida/espantalhos leem a OBSERVACAO; as de
`remover_espantalho` e `colocar_cerca` leem o estado das instalacoes do ambiente
(a observacao nao carrega instalacoes). Correcao de 06/09/2026: antes, `remover_espantalho`
era sempre valida e custava zero, e agentes treinados do zero a usavam como um segundo
"esperar" em ate 79 % dos passos (criticas.md, §2.3).
"""
import numpy as np
import gymnasium as gym


class InvolucroMascaraAcoes(gym.Wrapper):
    """Wrapper que injeta máscaras de ações válidas na info do ambiente."""

    def __init__(self, env):
        super().__init__(env)
        self.num_acoes = env.action_space.n

    def _ambiente_bruto(self):
        bruto = self.env
        while hasattr(bruto, 'env'):
            bruto = bruto.env
        return bruto

    def step(self, acao):
        obs, recompensa, terminado, truncado, info = self.env.step(acao)
        info['mascara_acoes'] = self._gerar_mascara(obs)
        return obs, recompensa, terminado, truncado, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['mascara_acoes'] = self._gerar_mascara(obs)
        return obs, info

    def _gerar_mascara(self, obs):
        # Índices de ação (FarmGym v4): 0=plantar 1=colher 2=regar 3=fertN 4=fertP 5=fertK 6=fertC
        #                  7=herbicida 8=pesticida 9=espant_basico 10=espant_avancado
        #                  11=remover_espant 12=colocar_cerca 13=esperar   ('observar' saiu: no-op com custo)
        mascara = np.ones(self.num_acoes, dtype=np.float32)

        # v4: layout por célula = [tem_planta, saude, estagio, umidade, qualidade, ervas, doenca, aves, pragas, espantalho]
        deslocamento_base = 16
        tamanho_celula = 10
        num_celulas = 16

        tem_espaco_vazio = False
        tem_colheitavel = False
        tem_ervas = False
        tem_planta = False
        umidade_media = 0.0
        n_celulas_com_planta = 0

        for i in range(num_celulas):
            idx = deslocamento_base + (i * tamanho_celula)
            planta_presente = obs[idx + 0]
            estagio_planta  = obs[idx + 2]
            umidade_solo    = obs[idx + 3]
            densidade_ervas = obs[idx + 5]

            if planta_presente == 0:
                tem_espaco_vazio = True
            else:
                tem_planta = True
                n_celulas_com_planta += 1
                umidade_media += umidade_solo

            # COLHEITA = índice 5/6 nos enums; normalizado por 7 estágios → ≥5/7≈0.71
            if planta_presente > 0 and estagio_planta >= 0.71:
                tem_colheitavel = True
            if densidade_ervas > 0.1:
                tem_ervas = True

        if n_celulas_com_planta > 0:
            umidade_media /= n_celulas_com_planta

        if not tem_espaco_vazio:
            mascara[0] = 0.0   # plantar — só faz sentido se há espaço vazio
        if not tem_colheitavel:
            mascara[1] = 0.0   # colher — só faz sentido se há planta pronta
        if not tem_planta or umidade_media > 0.8:
            # v4: `regar` também é inválida sem nenhuma planta. Custava 0,5 e não fazia nada (só planta consome água),
            # contra a regra declarada acima.
            mascara[2] = 0.0   # regar — sem planta, ou solo já saturado
        if not tem_ervas:
            mascara[7] = 0.0   # herbicida — sem ervas daninhas
        if not tem_planta:
            mascara[8] = 0.0   # pesticida — sem plantas (v3: pragas nao estao na observacao; a decisao de aplicar e do agente)
        if not tem_planta:
            mascara[9]  = 0.0  # espantalho básico — sem plantas para proteger
            mascara[10] = 0.0  # espantalho avançado — sem plantas para proteger

        # Instalações não estão na observação: ler o estado do ambiente (06/09/2026).
        instalacoes = getattr(self._ambiente_bruto(), 'instalacoes', None)
        if instalacoes is not None:
            if instalacoes.n_espantalhos() == 0:
                mascara[11] = 0.0  # remover espantalho — não há espantalho para remover
            if instalacoes.n_cercas() >= num_celulas:
                mascara[12] = 0.0  # cerca viva — todas as células já têm cerca

        mascara[13] = 1.0      # esperar — sempre válido (é a única ação sempre disponível)
        return mascara
