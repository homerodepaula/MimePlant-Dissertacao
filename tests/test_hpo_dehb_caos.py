"""Ensaio de CAOS do coordenador do DEHB (`hpo/hpo_dehb.py`): o estudo inteiro roda pelo caminho de PRODUÇÃO (`H.main` →
`rodar` → `_rodar_com_trava`, jornal de verdade, espelho de verdade no Optuna) sob falhas SORTEADAS por uma semente, em várias
sessões encadeadas com --retomar, e no fim conferem-se invariantes (I1–I8, em `conferir`). Não é teste de um caminho: é a
pergunta "existe ALGUMA ordem de azares que estraga o estudo?", feita a muitas sementes.

O que é de mentira: os avaliadores (`ProcCaotico` no lugar do Popen: o PLANO sorteado decide o que cada um deixa em disco, em
quantos ciclos e com que código de saída) e o relógio (`monotonic` anda 10 s por consulta e `sleep` não dorme). Tudo o mais é o
código de produção, com as falhas injetadas por BAIXO dele:
  jornal    `open` do hpo_dehb: o jornal.jsonl fica preso (PermissionError) ou a escrita sai pela METADE e levanta OSError, 1 a 11
            vezes seguidas — até 7 as novas tentativas internas do `Jornal.acrescentar` absorvem; de 8 em diante o OSError chega
            ao coordenador, por tipo de evento (cabecalho, sessao, pedir, informar, tentativa_falhou, relancado_apos_queda);
  queda     `Queda` (BaseException: atravessa todo `except Exception` da produção, como um kill) antes da escrita no jornal, com a
            linha pela metade, ou depois do fsync e antes de a memória saber; antes/depois do Popen; antes/depois da _DONE.flag.
            Na queda os avaliadores vivos morrem (Job Object) — alguns terminam o que faziam no mesmo instante;
  disco     `os.replace` do hpo_dehb preso 1, 2 ou 13 vezes (13 esgota as 12 tentativas do escritor atômico); resultado.json
            preso na leitura 2, 7 ou 14 vezes (6 é a insistência do `_ler_valido`); _STATUS.txt preso; Popen que levanta OSError;
            o close() do avaliacao.log levanta OSError; e (só com o espelho de mentira) o jornal ENCOLHE por fora entre dois ciclos,
            como quando o OneDrive restaura uma versão anterior — a sessão tem de parar sem tocar nele e a seguinte segue dali;
  avaliador falha da configuração, falha transitória, morte sem nada, saída 0 sem resultado, resultado com bytes que não são
            UTF-8, JSON cortado, aptidão null/NaN, objetivo/indicador/passos/fre que não são número, impressão digital de outro
            pedido, processo pendurado (o vigia de batimento o mata; às vezes o kill demora);
  espelho   `Espelho._trial_de` levanta 1, 2 ou 6 vezes seguidas (6 esgota as 5 tentativas do `reconciliar`).
Determinismo: todo sorteio sai de `random.Random(semente)` e o relógio é o do teste; nada depende de tempo real nem de GPU.

Invariantes (em `conferir`, mais os que o `Caos` confere durante a execução):
  I1 jornal: JSONL válido, sem NaN/Infinity nem NULs, índices contíguos, cabeçalho único;  I2 cada 'pedir' com exatamente um
  'informar', depois dele; ids contíguos e em ordem; tentativas 1..n <= max, e 'transitoria_esgotada' se e só se n = max;
  I3 orçamento atingido, nenhum 'pedir' com ele já gasto, excesso dentro do que os trabalhos abertos no último pedir podiam
  custar, fre = passos / N, PROGRESSO.json de acordo;  I4 o núcleo puro refaz cada pedido bit a bit (e o gerador), um Coordenador
  novo abre o jornal final sem 'divergencia' nem tocar nele e chega à mesma assinatura da última sessão viva;  I5 espelho = eventos
  terminais, na ordem, COMPLETE/PRUNED/FAIL certos, datas estritamente crescentes;  I6 nenhum (fidelidade, vetor) avaliado com
  sucesso duas vezes, nenhum lançamento sem 'pedir' gravado / de trabalho já informado / com K avaliadores vivos / que o jornal não
  explique, nenhum resultado bom refeito;  I7 nenhum .tmp* (nem logo depois de um os.replace esgotar), _DONE.flag sem _FAILED.flag;
  I8 conservação: o cobrado (e o cobrado + o 'perdido em queda') de um trabalho não passa do que os avaliadores dele publicaram;
  e nenhuma sessão derrubada por exceção que não seja de E/S ou parada prevista (`MOTIVOS_DE_PARADA_LEGITIMOS`).

Fora do escopo (e por quê): órfão vivo de outra sessão e Job Object (precisam de processo de verdade: test_hpo_dehb_retomada.py);
queda DENTRO do escritor atômico (deixa .tmp por definição: o invariante I7 só vale para erro, não para kill); `avaliar()` de
produção (test_hpo_dehb_avaliador.py). RESSALVAS que o arnês tolera e anota em `caos.ressalvas`, em vez de acusar: a queda
exatamente ENTRE gravar 'tentativa_falhou'/'relancado_apos_queda' e guardar o _BATIMENTO.json já contado (ele é dado como perdido
de novo: infla só o campo informativo), e o PROGRESSO.json do fim ficar velho quando a ÚLTIMA gravação dele fica presa."""
import os
import sys
import json
import time
import types
import random
import datetime
import subprocess
import collections

import pytest

import protocolo as P

sys.path.insert(0, os.path.join(P.RAIZ, 'hpo'))
import hpo_dehb as H        # noqa: E402
import dehb_nucleo as NUC   # noqa: E402

N = 900_000                 # fidelidades 100.000 (só com 3 níveis), 300.000 e 900.000: todas múltiplas do bloco N // 9
TERMINAIS = ('informar', 'tentativa_falhou')
MAX_TENTATIVAS = 2
MAX_SESSOES = 60
LIMITE_SEM_BATIMENTO_H = 0.05          # 180 s do relógio de mentira = 18 ciclos; o avaliador normal leva no máximo 4
TAXAS = {
    'jornal': {'cabecalho': 0.05, 'sessao': 0.06, 'pedir': 0.04, 'informar': 0.05, 'tentativa_falhou': 0.12, 'relancado_apos_queda': 0.12},
    'queda_no_jornal': 0.02, 'popen_oserror': 0.05, 'popen_queda': 0.015, 'replace_preso': 0.02, 'espelho': 0.03, 'status': 0.03,
    'queda_na_flag': 0.06, 'termina_na_queda': 0.3, 'quedas_max': 6, 'log_nao_fecha': 0.04, 'jornal_encolhe': 0.004,
    'planos': (('ok', 56), ('ok_preso', 6), ('falha_config', 5), ('transitoria', 8), ('morte', 6), ('sem_resultado', 3), ('lixo_utf8', 3),
               ('json_cortado', 3), ('aptidao_null', 2), ('aptidao_nan', 2), ('numero_null', 3), ('impressao_errada', 2), ('pendurado', 2)),
}


# O que pode parar uma sessão sem ser defeito: E/S que insiste (no laço, ou na partida, onde não há laço que insista), o espelho que
# não reconcilia no fim, o jornal que encolheu por fora. Qualquer outra coisa na _FAILED.flag (KeyError, TypeError, ...) é defeito de
# programa — a cadeia relança e o estudo até termina, mas cada sessão derrubada leva junto os treinos em voo.
MOTIVOS_DE_PARADA_LEGITIMOS = ('PermissionError', 'OSError', 'erro de E/S persistente', 'espelho', 'encolheu')


class Queda(BaseException):
    """O coordenador 'morre' aqui. BaseException de propósito: nenhum `except Exception`/`except OSError` da produção a segura."""


class LacoSemFim(BaseException):
    """O laço de produção passou do limite de sonecas: o estudo não termina (invariante de progresso)."""


# ── bancada ──────────────────────────────────────────────────────────────────────────────────────
def montar_area(raiz):
    """DIR_HPO + precondições mínimas, em caminho com acento e espaço (função, e não só fixture: a varredura larga monta uma área
    por estudo)."""
    raiz = os.path.join(str(raiz), 'Área de Trabalho', 'estudo dehb')
    os.makedirs(os.path.join(raiz, 'hpo'))
    os.makedirs(os.path.join(raiz, 'il', 'seed_42'))
    with open(os.path.join(raiz, 'il', 'seed_42', 'modelo_final.pt'), 'wb') as f:
        f.write(b'x')
    refs = {k: {c: {r: {'media': m} for r in P.NOMES_REGIMES} for c in ('validacao_rollback', 'validacao_objetivo', 'teste')}
            for k, m in (('oraculo_perfeito', 5000.0), ('aleatorio', 600.0))}
    with open(os.path.join(raiz, 'referencias.json'), 'w', encoding='utf-8') as f:
        json.dump(refs, f)
    return {'hpo': os.path.join(raiz, 'hpo'), 'il': os.path.join(raiz, 'il'), 'refs': os.path.join(raiz, 'referencias.json'), 'raiz': raiz}


@pytest.fixture
def area(tmp_path):
    return montar_area(tmp_path)


def no_arquivo(area, study='s'):
    """Os eventos como estão NO ARQUIVO do jornal (a verdade que sobrevive a uma queda)."""
    arq = os.path.join(area['hpo'], study, 'jornal.jsonl')
    if not os.path.exists(arq):
        return []
    eventos = []
    for ln in open(arq, 'rb').read().split(b'\n'):
        try:
            eventos.append(json.loads(ln.decode('utf-8')))
        except ValueError:
            pass                                            # cauda cortada por uma queda: a sessão seguinte a descarta
    return eventos


class ArquivoDoJornal:
    """O arquivo do jornal visto pelo `Jornal.acrescentar`, com o azar da vez: `rasga` = a escrita sai pela metade e levanta;
    `depois` = levanta ao fechar, com a linha inteira e o fsync já feitos."""

    def __init__(self, real, rasga=None, depois=None):
        self._real, self._rasga, self._depois = real, rasga, depois

    def __getattr__(self, nome):
        return getattr(self._real, nome)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._real.close()
        if self._depois is not None and exc[0] is None:
            raise self._depois

    def write(self, dados):
        if self._rasga is None:
            return self._real.write(dados)
        self._real.write(dados[:max(1, len(dados) // 2)])
        self._real.flush()
        raise self._rasga


class ArquivoDeLog:
    """O avaliacao.log cujo close() levanta OSError uma vez (disco cheio no flush final): o arquivo fecha, mas quem chamou vê o erro."""

    def __init__(self, real):
        self._real, self._ja = real, False

    def __getattr__(self, nome):
        return getattr(self._real, nome)

    def close(self):
        self._real.close()
        if not self._ja:
            self._ja = True
            raise OSError(28, 'sem espaço no flush final do log (de mentira)')


def fechar_sem_reclamar(arquivo):
    """Fecha um log na QUEDA (na de verdade quem fecha é o sistema). O log 'que não fecha' do caos levantaria OSError aqui, e a
    Queda viraria um OSError comum dentro da produção: foi assim que o arnês acusou 'dois avaliadores do mesmo trabalho' à toa."""
    try:
        arquivo.close()
    except OSError:
        pass


def sobras_de_escrita_atomica(area):
    return [os.path.relpath(os.path.join(p, x), area['hpo']) for p, _, arquivos in os.walk(area['hpo']) for x in arquivos if '.tmp' in x]


class OsCaotico:
    """O módulo `os` visto SÓ pelo hpo_dehb: tudo igual, menos `replace`, que às vezes encontra o destino preso."""

    def __init__(self, caos):
        self._caos = caos

    def __getattr__(self, nome):
        return getattr(os, nome)

    def replace(self, origem, destino):
        if self._caos.replace_preso(str(destino)):
            nome = os.path.basename(str(destino))
            if nome.startswith('_BATIMENTO.json.anterior'):   # a guarda do batimento é arrumação (o erro é engolido): ele pode ser contado de novo
                self._caos.janela_do_batimento.add(int(os.path.basename(os.path.dirname(str(destino))).split('_')[1]))
            if nome == 'PROGRESSO.json':
                self._caos.progresso_gravado = False
            raise PermissionError(13, 'WinError 5: acesso negado (de mentira)', str(destino))
        if os.path.basename(str(destino)) == 'PROGRESSO.json':
            self._caos.progresso_gravado = True
        return os.replace(origem, destino)


class ProcCaotico:
    """Avaliador de mentira. O PLANO, sorteado no lançamento, decide o que ele deixa em disco, depois de quantas consultas e com
    que código. `gasto` = o máximo de passos que ele chegou a PUBLICAR (batimento, falha ou resultado): teto do que pode ser
    cobrado ou dado como perdido por conta dele."""

    def __init__(self, caos, dir_trabalho, saida):
        self.caos, self.dir, self.saida = caos, dir_trabalho, saida
        self.id = int(os.path.basename(dir_trabalho).split('_')[1])
        self.pid, self.rc, self.vivo, self.mortes, self.gasto, self.resultado_bom = 50_000 + len(caos.procs), None, True, 0, 0, False
        with open(os.path.join(dir_trabalho, 'trabalho.json'), encoding='utf-8') as f:
            self.tr = json.load(f)
        self.plano = caos.escolher_plano()
        self.ciclos = caos.rng.randint(0, 4)
        self.esperas_que_estouram = 1 if self.plano == 'pendurado' and caos.rng.random() < 0.4 else 0
        if caos.rng.random() < 0.7:
            self.bater(0)

    def bater(self, passos):
        with open(os.path.join(self.dir, '_BATIMENTO.json'), 'w', encoding='utf-8') as f:
            json.dump({'data': H.agora_iso(), 'pid': self.pid, 'passos': int(passos)}, f)
        self.gasto = max(self.gasto, int(passos))

    def poll(self):
        if self.rc is not None or self.plano == 'pendurado':
            return self.rc
        if self.ciclos > 0:
            self.ciclos -= 1
            if self.caos.rng.random() < 0.5:
                self.bater(self.gasto + self.tr['fidelidade'] // 10)
            return None
        self.terminar()
        return self.rc

    def kill(self):
        self.mortes += 1

    def wait(self, timeout=None):
        if self.esperas_que_estouram:
            self.esperas_que_estouram -= 1
            raise subprocess.TimeoutExpired('avaliador-de-mentira', timeout)
        self.rc, self.vivo = (-9 if self.rc is None else self.rc), False
        return self.rc

    def _falha(self, motivo, passos):
        with open(os.path.join(self.dir, 'falha.json'), 'w', encoding='utf-8') as f:
            json.dump({'impressao': self.tr['impressao'], 'id': self.id, 'motivo': motivo, 'detalhe': 'de mentira', 'passos': int(passos),
                       'inicio': H.agora_iso(), 'fim': H.agora_iso(), 'tempo_s': 0.0, 'hash_git': 'falso'}, f)
        self.gasto = max(self.gasto, int(passos))

    def _resultado(self, **troca):
        """O resultado.json do avaliador falso (mesma função sintética). As datas são SORTEADAS — fora de ordem entre trabalhos e às
        vezes com o início depois do fim: o espelho tem de pôr os trials em ordem assim mesmo."""
        tr, rng = self.tr, self.caos.rng
        valor = -float(sum((v - 0.3) ** 2 for v in tr['vetor']))
        fim = datetime.datetime(2026, 1, 1) + datetime.timedelta(seconds=rng.randint(0, 3_000_000))
        inicio = fim + datetime.timedelta(seconds=rng.choice([-7200, -60, 0, 30]))
        curva = [{'passos': int(tr['fidelidade']), 'proxy_norm': valor}] + ([{'passos': 1, 'proxy_norm': None}] if rng.random() < 0.2 else [])
        return dict({'impressao': tr['impressao'], 'id': self.id, 'aptidao': -valor, 'objetivo': valor if tr['fidelidade_maxima'] else None,
                     'indicador': valor, 'passos': int(tr['fidelidade']), 'fre': tr['fidelidade'] / tr['passos_max'],
                     'inicio': inicio.isoformat(timespec='milliseconds'), 'fim': fim.isoformat(timespec='milliseconds'), 'tempo_s': 0.0,
                     'curva': curva, 'hash_git': 'falso'}, **troca)

    def terminar(self):
        plano, fid, rng, arq = self.plano, int(self.tr['fidelidade']), self.caos.rng, os.path.join(self.dir, 'resultado.json')
        self.vivo, self.rc = False, 0
        if plano in ('ok', 'ok_preso'):
            with open(arq, 'w', encoding='utf-8') as f:
                json.dump(self._resultado(), f)
            self.gasto, self.resultado_bom = fid, True
            if plano == 'ok_preso':
                self.caos.leitura_presa[os.path.normcase(arq)] = rng.choice([2, 7, 14])
        elif plano == 'falha_config':
            self._falha('excecao_no_treino', fid // 2)
            self.rc = H.SAIDA_FALHA_DA_CONFIGURACAO
        elif plano == 'transitoria':
            self._falha('gpu_ou_sistema', fid // 4)
            self.rc = H.SAIDA_FALHA_TRANSITORIA
        elif plano == 'morte':
            if rng.random() < 0.6:
                self.bater(max(self.gasto, fid // 3))
            self.rc = rng.choice([1, -9, 3221225477])
        elif plano == 'sem_resultado':
            pass
        else:                                               # resultado que foi escrito e não serve: o treino inteiro foi gasto
            self.gasto = fid
            if plano == 'lixo_utf8':
                dados = b'\xff\xfe\xfa{"aptidao": 1.0, "id": ' + str(self.id).encode() + b'}'
            elif plano == 'json_cortado':
                texto = json.dumps(self._resultado())
                dados = texto[:len(texto) // 2].encode('utf-8')
            else:
                troca = {'aptidao_null': {'aptidao': None}, 'aptidao_nan': {'aptidao': float('nan')}, 'impressao_errada': {'impressao': 'f' * 64},
                         'numero_null': rng.choice([{'objetivo' if self.tr['fidelidade_maxima'] else 'indicador': None}, {'passos': None},
                                                    {'fre': 'muito'}, {'objetivo' if self.tr['fidelidade_maxima'] else 'indicador': float('inf')}])}[plano]
                dados = json.dumps(self._resultado(**troca)).encode('utf-8')
            with open(arq, 'wb') as f:
                f.write(dados)


class Caos:
    """O sorteador de azares de UM estudo e o caderno do que aconteceu (para os invariantes e para provar que o azar existiu)."""

    def __init__(self, semente, area, k, taxas=None, study='s'):
        self.rng, self.area, self.k, self.study = random.Random(semente), area, k, study
        self.taxas = dict(TAXAS, **(taxas or {}))
        self.contagem, self.violacoes = collections.Counter(), []
        self.procs, self.coordenadores, self.quedas, self.sessoes = [], [], 0, 0
        self.evento_atual, self.jornal_preso, self.jornal_rasga, self.queda_jornal = None, 0, 0, None
        self.leitura_presa, self.replace_sequencia, self.espelho_sequencia = {}, {}, 0
        self.agora, self.sonecas, self.log = 1000.0, 0, []
        self.id_atual, self.janela_do_batimento, self.ressalvas, self.progresso_gravado = None, set(), [], True
        self.conferir_tmp, self.jornal_encolhido, self.espelho_real = False, False, True

    # ── sorteios ──
    def sorteio(self, nome, p):
        if self.rng.random() < p:
            self.contagem[nome] += 1
            return True
        return False

    def cai(self, onde, p):
        """Sorteia uma QUEDA (limitadas por estudo: senão ele não acaba)."""
        if self.quedas < self.taxas['quedas_max'] and self.rng.random() < p:
            self.quedas += 1
            self.contagem[f'queda:{onde}'] += 1
            return True
        return False

    def escolher_plano(self):
        nomes, pesos = zip(*self.taxas['planos'])
        plano = self.rng.choices(nomes, weights=pesos)[0]
        self.contagem[f'plano:{plano}'] += 1
        return plano

    def relogio(self):
        self.agora += 10.0
        return self.agora

    def dormir(self, s):
        self.sonecas += 1
        if self.sonecas > 100_000:
            raise LacoSemFim(f'{self.sonecas} sonecas: o laço do coordenador não termina')
        if s == 0:                                          # a soneca do FIM de um ciclo bom (as das novas tentativas de E/S nunca são 0)
            self.no_fim_de_um_ciclo_bom()

    def no_fim_de_um_ciclo_bom(self):
        if self.conferir_tmp:                               # um os.replace esgotou as tentativas neste ciclo: o temporário tem de ter sumido JÁ
            self.conferir_tmp = False
            sobras = sobras_de_escrita_atomica(self.area)
            if sobras:
                self.violacoes.append(f'I7: temporário deixado para trás depois de o os.replace esgotar as tentativas: {sobras}')
        if not self.espelho_real and not self.jornal_encolhido and self.rng.random() < self.taxas['jornal_encolhe']:
            self.encolher_jornal()

    def encolher_jornal(self):
        """O jornal ENCOLHE por fora (o OneDrive restaura uma versão anterior): perde de 1 a 3 eventos do fim. O coordenador tem de
        parar sem tocar no arquivo (RuntimeError), e a sessão seguinte segue do que ficou. Só com o espelho de mentira: o de verdade
        passa a ter trials que o jornal não tem, e isso é, por desenho, caso de --reconstruir-espelho à mão."""
        arq = os.path.join(self.area['hpo'], self.study, 'jornal.jsonl')
        bruto = open(arq, 'rb').read()
        linhas = bruto.split(b'\n')[:-1]
        fora = self.rng.randint(1, 3)
        if len(linhas) - fora < 2:
            return
        with open(arq, 'r+b') as f:
            f.truncate(sum(len(ln) + 1 for ln in linhas[:-fora]))
        self.jornal_encolhido = True
        self.contagem['jornal_encolhido'] += 1

    # ── jornal ──
    def antes_de_acrescentar(self, tipo):
        if self.cai(f'jornal:{tipo}', self.taxas['queda_no_jornal']):
            self.queda_jornal = self.rng.choice(['antes', 'rasgando', 'depois'])
            self.contagem[f'queda_jornal:{self.queda_jornal}'] += 1
        elif self.rng.random() < self.taxas['jornal'].get(tipo, 0.0):
            n = self.rng.choice([1, 3, 8, 8, 11])           # < 8: o jornal absorve; >= 8: o OSError chega ao coordenador
            if self.rng.random() < 0.5:
                self.jornal_preso += n
            else:
                self.jornal_rasga += n
            self.contagem[f"jornal_{'absorvido' if n < 8 else 'oserror'}:{tipo}"] += 1

    def abrir_jornal(self, caminho, modo):
        if self.queda_jornal is not None:
            como, self.queda_jornal = self.queda_jornal, None
            if como == 'antes':
                raise Queda(f'antes de gravar {self.evento_atual}')
            if como == 'depois' and self.evento_atual in ('tentativa_falhou', 'relancado_apos_queda'):
                self.janela_do_batimento.add(self.id_atual)   # a queda cai ENTRE o evento gravado e a guarda do _BATIMENTO.json já contado
            return ArquivoDoJornal(open(caminho, modo), **({'rasga': Queda(f'no meio de {self.evento_atual}')} if como == 'rasgando'
                                                           else {'depois': Queda(f'depois de gravar {self.evento_atual}')}))
        if self.jornal_preso > 0:
            self.jornal_preso -= 1
            raise PermissionError(13, 'WinError 32: jornal preso (de mentira)', caminho)
        if self.jornal_rasga > 0:
            self.jornal_rasga -= 1
            return ArquivoDoJornal(open(caminho, modo), rasga=OSError(28, 'escrita pela metade (de mentira)'))
        return open(caminho, modo)

    # ── disco ──
    def abrir(self, caminho, *a, **k):
        nome, modo = os.path.basename(str(caminho)), (a[0] if a else k.get('mode', 'r'))
        if nome == 'jornal.jsonl' and self.evento_atual is not None and modo in ('r+b', 'w+b'):
            return self.abrir_jornal(caminho, modo)
        chave = os.path.normcase(str(caminho))
        if nome == 'resultado.json' and self.leitura_presa.get(chave, 0) > 0:
            self.leitura_presa[chave] -= 1
            self.contagem['leitura_presa'] += 1
            raise PermissionError(13, 'WinError 32: resultado preso (de mentira)', str(caminho))
        if nome == 'avaliacao.log' and self.sorteio('log_nao_fecha', self.taxas['log_nao_fecha']):
            return ArquivoDeLog(open(caminho, *a, **k))
        return open(caminho, *a, **k)

    def replace_preso(self, destino):
        if self.replace_sequencia.get(destino, 0) > 0:
            self.replace_sequencia[destino] -= 1
            return True
        if self.rng.random() < self.taxas['replace_preso']:
            n = self.rng.choice([1, 2, 13])
            self.replace_sequencia[destino] = n - 1
            self.contagem[f"replace_{'esgotado' if n > 12 else 'preso'}:{os.path.basename(destino).split('.')[0]}"] += 1
            self.conferir_tmp = self.conferir_tmp or n > 12
            return True
        return False

    # ── avaliadores ──
    def popen(self, cmd, **kw):
        if self.sorteio('popen_oserror', self.taxas['popen_oserror']):
            raise OSError('[WinError 1455] arquivo de paginação muito pequeno (de mentira)')
        if self.cai('popen_antes', self.taxas['popen_queda']):
            fechar_sem_reclamar(kw['stdout'])
            raise Queda('antes do Popen')
        self.conferir_lancamento(cmd[-1])
        proc = ProcCaotico(self, cmd[-1], kw.get('stdout'))
        self.procs.append(proc)
        if self.cai('popen_depois', self.taxas['popen_queda']):
            fechar_sem_reclamar(kw['stdout'])
            raise Queda('depois do Popen')
        return proc

    def conferir_lancamento(self, dir_trabalho):
        """I6, conferido NA HORA de cada lançamento, contra o ARQUIVO do jornal."""
        id_t = int(os.path.basename(dir_trabalho).split('_')[1])
        vivos = [p for p in self.procs if p.vivo]
        if len(vivos) >= self.k:
            self.violacoes.append(f'I6: lançou o trabalho {id_t} com {len(vivos)} avaliadores vivos (K = {self.k}): {[p.id for p in vivos]}')
        if any(p.id == id_t for p in vivos):
            self.violacoes.append(f'I6: lançou o trabalho {id_t} com outro avaliador DELE ainda vivo')
        eventos = self.coordenadores[-1].jornal.eventos       # a memória do jornal só recebe o evento DEPOIS do fsync (reler o arquivo custa 10 ms por lançamento)
        if not any(e['ev'] == 'pedir' and e['trabalho']['id'] == id_t for e in eventos):
            self.violacoes.append(f"I6: lançou o trabalho {id_t} sem o 'pedir' dele no arquivo do jornal (write-ahead)")
        if any(e['ev'] == 'informar' and e['id'] == id_t for e in eventos):
            self.violacoes.append(f'I6: lançou o trabalho {id_t}, que já estava informado')

    def derrubar(self):
        """A queda vista de fora: os filhos morrem com o coordenador (Job Object); alguns tinham ACABADO de terminar."""
        for p in self.procs:
            if p.vivo:
                if p.plano != 'pendurado' and self.rng.random() < self.taxas['termina_na_queda']:
                    p.terminar()
                elif self.rng.random() < 0.5:
                    p.bater(p.gasto + p.tr['fidelidade'] // 20)
                p.vivo = False
        for c in self.coordenadores:
            for v in c.em_voo.values():
                fechar_sem_reclamar(v['saida'])


class EspelhoDeMentira:
    """Para a varredura larga: o espelho de verdade custa ~0,2 s por trial no SQLite (7 s por estudo contra 1 s), e o que ele
    projeta não realimenta o coordenador. Com ele o invariante I5 não é conferido."""

    def __init__(self, *a, **k):
        pass

    def atributos_do_estudo(self, *a, **k):
        pass

    def reconciliar(self, eventos):
        return True

    def indice(self):
        return {}

    def fechar(self):
        pass


def argumentos(area, k, niveis, n_fre, semente, study='s'):
    return ['--study', study, '--n-fre', str(n_fre), '--orcamento', str(N), '--niveis', str(niveis), '--paralelo', str(k), '--seed', str(semente),
            '--warmstart', area['il'], '--referencias', area['refs'], '--intervalo-s', '0', '--max-tentativas', str(MAX_TENTATIVAS),
            '--max-falhas-seguidas', '1000000', '--limite-sem-batimento-h', str(LIMITE_SEM_BATIMENTO_H), '--retomar']


def estudo_caotico(area, semente, k, niveis, n_fre, taxas=None, study='s', espelho_real=True):
    """Roda UM estudo até a _DONE.flag, sessão após sessão, sob o caos da semente. Devolve o `Caos` (caderno + violações vistas
    durante a execução). O disjuntor fica desligado (--max-falhas-seguidas enorme): com falhas sorteadas ele dispararia por acaso
    e o estudo pararia de propósito — ele tem os testes dele."""
    ks = tuple(k) if isinstance(k, (tuple, list)) else (k,)  # vários K: cada sessão sorteia o seu (a cadeia pode ser relançada com outro --paralelo)
    caos = Caos(semente, area, ks[0], taxas, study)
    caos.espelho_real = espelho_real
    real_acrescentar, real_trial_de, real_status, real_flag = H.Jornal.acrescentar, H.Espelho._trial_de, P.escrever_status, P.escrever_flag

    def acrescentar(self, evento, *a, **kw):
        caos.evento_atual, caos.id_atual = evento.get('ev'), evento.get('id')
        try:
            caos.antes_de_acrescentar(caos.evento_atual)
            return real_acrescentar(self, evento, *a, **kw)
        finally:
            caos.evento_atual = None

    def trial_de(self, ev):
        if caos.espelho_sequencia > 0 or caos.rng.random() < caos.taxas['espelho']:
            if caos.espelho_sequencia == 0:
                caos.espelho_sequencia = caos.rng.choice([1, 2, 6])
                caos.contagem[f"espelho_{'esgotado' if caos.espelho_sequencia > 5 else 'falhou'}"] += 1
            caos.espelho_sequencia -= 1
            raise RuntimeError('database is locked (de mentira)')
        return real_trial_de(self, ev)

    def status(dir_out, etapa, detalhe=''):
        if caos.sorteio('status_preso', caos.taxas['status']):
            raise PermissionError(13, 'WinError 32: _STATUS.txt preso (de mentira)')
        return real_status(dir_out, etapa, detalhe)

    def flag(dir_out, nome, mensagem=''):
        if nome == '_DONE.flag' and caos.cai('flag_antes', caos.taxas['queda_na_flag']):
            raise Queda('antes da _DONE.flag')
        caminho = real_flag(dir_out, nome, mensagem)
        if nome == '_DONE.flag' and caos.cai('flag_depois', caos.taxas['queda_na_flag']):
            raise Queda('depois da _DONE.flag')
        return caminho

    class CoordenadorVigiado(H.Coordenador):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            caos.coordenadores.append(self)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(H, 'DIR_HPO', area['hpo'])
        mp.setattr(P, 'hash_git', lambda: 'git-do-teste')     # entra na impressão digital; chamar o git a cada sessão custaria mais que o estudo
        mp.setattr(H, 'time', types.SimpleNamespace(monotonic=caos.relogio, sleep=caos.dormir, time=time.time, strftime=time.strftime))
        mp.setattr(H, 'subprocess', types.SimpleNamespace(Popen=caos.popen, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT,
                                                          TimeoutExpired=subprocess.TimeoutExpired))
        mp.setattr(H, 'prender_filhos_ao_coordenador', lambda: (lambda proc: None))
        mp.setattr(H, 'manter_a_maquina_acordada', lambda ligar: None)
        mp.setattr(H, 'open', caos.abrir, raising=False)
        mp.setattr(H, 'os', OsCaotico(caos))
        mp.setattr(H, 'Coordenador', CoordenadorVigiado)
        mp.setattr(H.Jornal, 'acrescentar', acrescentar)
        mp.setattr(H.Espelho, '_trial_de', trial_de)
        if not espelho_real:
            mp.setattr(H, 'Espelho', EspelhoDeMentira)
        mp.setattr(H, 'print', lambda *a, **kw: caos.log.append(' '.join(str(x) for x in a)), raising=False)   # o log da sessão fica no caderno
        mp.setattr(P, 'escrever_status', status)
        mp.setattr(P, 'escrever_flag', flag)
        caos.fim = None
        for _ in range(MAX_SESSOES):
            caos.sessoes += 1
            inicio_do_log = len(caos.log)
            caos.k = ks[0] if len(ks) == 1 else caos.rng.choice(ks)
            try:
                rc = H.main(argumentos(area, caos.k, niveis, n_fre, semente, study), comando_avaliador=lambda d: ['avaliador-de-mentira', d])
            except Queda:
                caos.derrubar()
                continue
            except LacoSemFim as e:
                caos.violacoes.append(f'progresso: {e}')
                caos.derrubar()
                break
            except SystemExit as e:
                caos.violacoes.append(f'a sessão {caos.sessoes} recusou o próprio estudo: {e}')
                break
            vivos = [p.id for p in caos.procs if p.vivo and not p.mortes]
            if rc != 0:
                motivo = ler_texto(os.path.join(area['hpo'], study, '_FAILED.flag'))
                caos.log.append('_FAILED.flag: ' + motivo)
                if not any(m in motivo for m in MOTIVOS_DE_PARADA_LEGITIMOS):
                    caos.violacoes.append(f'a sessão {caos.sessoes} foi derrubada por algo que não é E/S nem parada prevista: {motivo}')
            if vivos and (rc == 0 or any('ABORTADO' in m for m in caos.log[inicio_do_log:])):   # exceção que sobe mata os filhos pelo Job Object; o _abortar tem de matá-los ele mesmo
                caos.violacoes.append(f'a sessão {caos.sessoes} terminou (rc = {rc}) e deixou avaliadores vivos e não encerrados: {vivos}')
            caos.derrubar()
            if rc == 0:
                caos.fim = caos.coordenadores[-1]
                break
            caos.contagem['sessao_abortada'] += 1
        else:
            caos.violacoes.append(f'o estudo não chegou ao fim em {MAX_SESSOES} sessões')
    return caos


# ── invariantes ──────────────────────────────────────────────────────────────────────────────────
def _constante_proibida(nome):
    raise ValueError(f'{nome} no jornal')


def conferir(area, caos, k, niveis, n_fre, study='s'):
    """As violações dos invariantes I1–I8 no que ficou EM DISCO (mais as que o `Caos` anotou durante a execução)."""
    v = list(caos.violacoes)
    k = max(k) if isinstance(k, (tuple, list)) else k       # com K por sessão, o teto dos pedidos abertos é o maior deles
    d = os.path.join(area['hpo'], study)
    arq = os.path.join(d, 'jornal.jsonl')
    if not os.path.exists(arq):
        return v + ['I1: não há jornal']
    # I1 — o arquivo do jornal
    bruto = open(arq, 'rb').read()
    if not bruto.endswith(b'\n') or b'\x00' in bruto:
        v.append('I1: o jornal não termina em fim de linha, ou tem NULs')
    ev = []
    for n, ln in enumerate(bruto.split(b'\n')[:-1]):
        try:
            ev.append(json.loads(ln.decode('utf-8'), parse_constant=_constante_proibida))
        except ValueError as e:
            return v + [f'I1: linha {n + 1} do jornal inválida ({e})']
    if [e.get('i') for e in ev] != list(range(len(ev))):
        v.append(f"I1: índices não contíguos: {[e.get('i') for e in ev]}")
    if [e['i'] for e in ev if e['ev'] == 'cabecalho'] != [0]:
        v.append('I1: o cabeçalho não é único e o primeiro')
    if any(e['ev'] == 'divergencia' for e in ev):
        v.append(f"I4: o jornal tem 'divergencia' (evento {[e['i'] for e in ev if e['ev'] == 'divergencia']}): uma retomada não reproduziu os pedidos")
    # I2 — pedir / informar / tentativas
    pedidos = [e for e in ev if e['ev'] == 'pedir']
    ids = [e['trabalho']['id'] for e in pedidos]
    if ids != list(range(len(ids))):
        v.append(f'I2: ids dos pedidos fora de ordem ou com buraco: {ids}')
    onde_pedir = {e['trabalho']['id']: e['i'] for e in pedidos}
    por_id = collections.defaultdict(list)
    for e in ev:
        if e['ev'] in TERMINAIS + ('relancado_apos_queda',):
            por_id[e['id']].append(e)
    for id_t in sorted(set(ids) | set(por_id)):
        meus = por_id.get(id_t, [])
        informes = [e for e in meus if e['ev'] == 'informar']
        if id_t not in onde_pedir:
            v.append(f"I2: o trabalho {id_t} tem eventos sem 'pedir'")
            continue
        if len(informes) != 1:
            v.append(f"I2: o trabalho {id_t} tem {len(informes)} 'informar' (tinha de ser exatamente um)")
            continue
        if any(not onde_pedir[id_t] < e['i'] <= informes[0]['i'] for e in meus):
            v.append(f"I2: o trabalho {id_t} tem evento antes do 'pedir' ou depois do 'informar': {[(e['ev'], e['i']) for e in meus]}")
        tentativas = [e['tentativa'] for e in meus if e['ev'] == 'tentativa_falhou']
        if tentativas != list(range(1, len(tentativas) + 1)) or len(tentativas) > MAX_TENTATIVAS:
            v.append(f'I4: tentativas do trabalho {id_t} = {tentativas} (max_tentativas = {MAX_TENTATIVAS})')
        if (informes[0].get('motivo') == 'transitoria_esgotada') != (len(tentativas) == MAX_TENTATIVAS):
            v.append(f"I2: trabalho {id_t} com {len(tentativas)} tentativas falhadas e informar de motivo {informes[0].get('motivo')!r}")
    # I3 — orçamento
    fid_fre = {e['trabalho']['id']: e['trabalho']['fidelidade'] / N for e in pedidos}
    total = sum(e['fre'] for e in ev if e['ev'] in TERMINAIS)
    if total < n_fre:
        v.append(f'I3: o estudo acabou com {total:.3f} FRE, abaixo do orçamento {n_fre}')
    for e in ev:
        if e['ev'] in TERMINAIS and (not H.numero_finito(e['fre']) or e['fre'] < 0 or abs(e['fre'] - e['passos'] / N) > 1e-12 or e['fre'] > fid_fre.get(e['id'], 0) + 1e-12):
            v.append(f"I3: evento {e['i']} ({e['ev']} do trabalho {e['id']}) com fre = {e['fre']!r} e passos = {e['passos']!r}")
    gasto_ate, abertos, max_abertos = 0.0, set(), 0
    for e in ev:
        if e['ev'] == 'pedir':
            if gasto_ate >= n_fre:
                v.append(f"I3: 'pedir' do trabalho {e['trabalho']['id']} com o orçamento já gasto ({gasto_ate:.3f} >= {n_fre})")
            abertos.add(e['trabalho']['id'])
            max_abertos = max(max_abertos, len(abertos))
            folga = sum(MAX_TENTATIVAS * fid_fre[i] for i in abertos)     # o pior caso do que ainda pode ser cobrado depois do ÚLTIMO pedir
        elif e['ev'] in TERMINAIS:
            gasto_ate += e['fre']
            if e['ev'] == 'informar':
                abertos.discard(e['id'])
    if pedidos and total >= n_fre + folga:
        v.append(f'I3: excesso de {total - n_fre:.3f} FRE, acima do que os trabalhos abertos no último pedir podiam custar ({folga:.3f})')
    if max_abertos > k:
        v.append(f'I6: {max_abertos} trabalhos pedidos e não informados ao mesmo tempo (K = {k})')
    try:
        if not caos.progresso_gravado:
            # PROGRESSO.json é rastro de melhor esforço (4 tentativas e segue): se a ÚLTIMA gravação ficou presa, o arquivo do fim é o velho
            caos.ressalvas.append('I3: a última gravação do PROGRESSO.json ficou presa; o arquivo do fim é o da gravação anterior')
            raise StopIteration
        prog = json.load(open(os.path.join(d, 'PROGRESSO.json'), encoding='utf-8'))
        if abs(prog['fre_consumidos'] - total) > 1e-3 or prog['em_voo'] or prog['na_fila']:
            v.append(f"I3: PROGRESSO.json diz {prog['fre_consumidos']} FRE, em voo {prog['em_voo']}, fila {prog['na_fila']}; o jornal soma {total:.3f}")
    except StopIteration:
        pass
    except (OSError, ValueError, KeyError) as e:
        v.append(f'I3: PROGRESSO.json ilegível ({e})')
    # I4 — replay: o núcleo puro refaz cada pedido bit a bit, e um Coordenador novo chega ao mesmo estado sem 'divergencia'
    cab = ev[0]
    nucleo = NUC.Dehb(len(cab['espaco']), cab['fidelidades'], eta=cab['eta'], semente=cab['semente'])
    for e in ev:
        if e['ev'] == 'pedir':
            t = nucleo.pedir()
            if t.registro() != e['trabalho'] or json.loads(json.dumps(nucleo.estado_do_gerador())) != e['gerador']:
                v.append(f"I4: o replay do núcleo não reproduz o pedido {e['trabalho']['id']} (evento {e['i']})")
                break
        elif e['ev'] == 'informar':
            try:
                nucleo.informar(e['id'], e.get('aptidao'))
            except KeyError as erro:
                v.append(f'I4: replay: {erro}')
                break
    else:
        if nucleo.pendentes:
            v.append(f'I4: o replay termina com pendentes {sorted(nucleo.pendentes)}')
        ns = H.argparse.Namespace(study=study, orcamento=N, niveis=niveis, eta=cab['eta'], seed=cab['semente'], inicializacao='professor', seeds_trial=[42],
                                  smoke=False, retomar=True, n_fre=n_fre, paralelo=k, warmstart=area['il'], referencias=area['refs'], dispositivo=None,
                                  max_tentativas=MAX_TENTATIVAS, max_falhas_seguidas=10 ** 6, ignorar_disjuntor=False,
                                  limite_sem_batimento_h=LIMITE_SEM_BATIMENTO_H, intervalo_s=0.0)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(H, 'DIR_HPO', area['hpo'])
            mp.setattr(P, 'hash_git', lambda: 'git-do-teste')
            novo = H.Coordenador(ns, log=lambda m: None)
            novo.jornal = H.Jornal(arq)
            novo.cab = novo.jornal.eventos[0]
            novo._replay()
        if len(novo.jornal.eventos) != len(ev) or os.path.getsize(arq) != len(bruto):
            v.append("I4: abrir o jornal final num Coordenador novo mexeu nele ('divergencia' nova, ou cauda descartada)")
        if novo.dehb.assinatura() != nucleo.assinatura():
            v.append('I4: o Coordenador novo não chega à assinatura do replay do núcleo')
        if getattr(caos.fim, 'dehb', None) is not None and caos.fim.dehb.assinatura() != nucleo.assinatura():
            v.append('I4: a assinatura do núcleo da última sessão VIVA difere da do replay do jornal')
        if any(n > MAX_TENTATIVAS for n in novo.tentativas.values()):
            v.append(f'I4: contador de tentativas além do máximo: {novo.tentativas}')
    # I5 — espelho
    terminais = [e for e in ev if e['ev'] in TERMINAIS]
    try:
        if not caos.espelho_real:
            raise StopIteration
        import optuna
        st = optuna.load_study(study_name=study, storage=f"sqlite:///{os.path.join(area['hpo'], study + '.db').replace(os.sep, '/')}")
        ts = st.get_trials(deepcopy=True)
        H.liberar_banco(st)
        estados = optuna.trial.TrialState
        if len(ts) != len(terminais):
            v.append(f'I5: {len(ts)} trials no espelho para {len(terminais)} eventos terminais no jornal')
        for t, e in zip(ts, terminais):
            certo = estados.FAIL if (e['ev'] == 'tentativa_falhou' or e.get('falhou')) else (estados.COMPLETE if e['fidelidade_maxima'] else estados.PRUNED)
            if t.user_attrs.get('evento_i') != e['i'] or t.user_attrs.get('id_trabalho') != e['id'] or t.state != certo:
                v.append(f"I5: trial {t.number} = evento {t.user_attrs.get('evento_i')} em {t.state.name}; o jornal pede o evento {e['i']} em {certo.name}")
            elif certo == estados.COMPLETE and t.value != e['objetivo']:
                v.append(f"I5: trial {t.number} com value {t.value}, objetivo {e['objetivo']} no jornal")
        datas = [t.datetime_complete for t in ts]
        if any(b <= a for a, b in zip(datas, datas[1:])) or any(t.datetime_start > t.datetime_complete for t in ts):
            v.append('I5: datas do espelho fora de ordem')
    except StopIteration:
        pass                                                # estudo rodado com o espelho de mentira: I5 não se aplica
    except Exception as e:
        v.append(f'I5: espelho ilegível ({type(e).__name__}: {e})')
    # I6 — nada avaliado duas vezes; lançamentos explicados pelo jornal
    vistos = {}
    for e in ev:
        if e['ev'] == 'informar' and not e.get('falhou'):
            chave = (e['nivel'], tuple(pedidos[e['id']]['trabalho']['vetor'])) if e['id'] < len(pedidos) else None
            if chave in vistos:
                v.append(f"I6: os trabalhos {vistos[chave]} e {e['id']} avaliaram com sucesso o MESMO vetor na fidelidade {e['fidelidade']}")
            vistos[chave] = e['id']
    for id_t in ([] if caos.jornal_encolhido else ids):     # jornal encolhido por fora perdeu eventos: os lançamentos já não se explicam por ele
        meus, lancados = por_id.get(id_t, []), [p for p in caos.procs if p.id == id_t]
        explicados = 1 + sum(1 for e in meus if e['ev'] in ('tentativa_falhou', 'relancado_apos_queda'))
        if len(lancados) > explicados:
            v.append(f'I6: o trabalho {id_t} foi lançado {len(lancados)} vezes e o jornal só explica {explicados}')
        if sum(1 for p in lancados if p.resultado_bom) > 1:
            v.append(f'I6: o trabalho {id_t} produziu resultado BOM mais de uma vez (treino refeito à toa)')
        # I8 — conservação: o cobrado mais o dado como perdido não passa do que os avaliadores do trabalho publicaram
        cobrado = sum(e['passos'] for e in meus if e['ev'] in TERMINAIS)
        perdido = sum(e['passos_perdidos'] for e in meus if e['ev'] == 'relancado_apos_queda')
        gasto = sum(p.gasto for p in lancados)
        if cobrado > gasto:
            v.append(f'I8: trabalho {id_t}: {cobrado} passos COBRADOS do orçamento > {gasto} publicados pelos avaliadores dele')
        elif cobrado + perdido > gasto:
            texto = f'I8: trabalho {id_t}: {cobrado} passos cobrados + {perdido} dados como perdidos > {gasto} publicados pelos avaliadores dele'
            # Janela conhecida (e só ela): queda DEPOIS de gravar 'tentativa_falhou'/'relancado_apos_queda' e ANTES de guardar o
            # _BATIMENTO.json já contado — a sessão seguinte o conta outra vez como perdido. Só o campo informativo infla, não o orçamento.
            (caos.ressalvas if id_t in caos.janela_do_batimento else v).append(texto)
    # I7 — pastas limpas e flags
    sobras = sobras_de_escrita_atomica(area)
    if sobras:
        v.append(f'I7: sobras de escrita atômica: {sobras}')
    if not os.path.exists(os.path.join(d, '_DONE.flag')) or os.path.exists(os.path.join(d, '_FAILED.flag')):
        v.append(f"I7: flags no fim: _DONE {os.path.exists(os.path.join(d, '_DONE.flag'))}, _FAILED {os.path.exists(os.path.join(d, '_FAILED.flag'))}")
    return v


def ler_texto(caminho):
    try:
        with open(caminho, encoding='utf-8') as f:
            return f.read().strip()
    except OSError:
        return '(ausente)'


def relato(titulo, violacoes, caos):
    """A mensagem de um ensaio vermelho: as violações e o fim do log das sessões (é por ele que se acha a ordem dos azares)."""
    return os.linesep.join([titulo + ':'] + [f'  {x}' for x in violacoes[:10]] + ['--- fim do log das sessões ---'] + caos.log[-25:])[:6000]


def rodar_e_conferir(area, semente, k, niveis, n_fre, taxas=None, espelho_real=True):
    caos = estudo_caotico(area, semente, k, niveis, n_fre, taxas, espelho_real=espelho_real)
    return caos, conferir(area, caos, k, niveis, n_fre)


# ── os ensaios ───────────────────────────────────────────────────────────────────────────────────
# (semente, K, níveis, n_fre, espelho de verdade?). Poucos e curtos: o arquivo roda no validador horário. Nenhuma semente foi
# escolhida por PASSAR — a varredura larga (abaixo, lenta) é que caça. As 11 a 19 são as primeiras da fila; a 1002 e a 1013 entraram
# por MATAR o que as outras deixavam vivo: a 1002, os mutantes 'término ignora _esgotados' e 'retomada dá tentativa extra ao
# esgotado' (correção C7: tentativa nº 3 com max_tentativas = 2); a 1013, o 'batimento cobrado não é guardado' (C6: I8).
ESTUDOS_RAPIDOS = [(11, 3, 2, 3.0, True), (12, 3, 3, 6.0, False), (14, 1, 2, 3.0, False), (15, 1, 3, 4.0, False), (16, 2, 3, 5.0, False),
                   (17, 3, 2, 4.0, False), (18, 3, 3, 6.0, False), (19, 3, 2, 4.0, False), (1002, 3, 3, 7.0, False), (1013, 3, 3, 7.0, False)]
_CADERNOS = {}


@pytest.mark.parametrize('semente, k, niveis, n_fre, espelho_real', ESTUDOS_RAPIDOS)
def test_estudo_sob_caos_termina_com_todos_os_invariantes(area, semente, k, niveis, n_fre, espelho_real):
    """Sessões encadeadas sob falhas sorteadas: o que sobra em disco tem de ser um estudo íntegro (I1–I8). A mensagem traz a semente
    e a configuração: `rodar_e_conferir(montar_area(pasta), semente, k, niveis, n_fre)` reproduz o estudo fora do pytest."""
    caos, violacoes = rodar_e_conferir(area, semente, k, niveis, n_fre, espelho_real=espelho_real)
    _CADERNOS[semente] = caos
    assert not violacoes, relato(f'semente {semente}, K = {k}, {niveis} níveis, {n_fre} FRE, {caos.sessoes} sessões, {caos.quedas} quedas', violacoes, caos)


def test_o_caos_dos_ensaios_rapidos_aconteceu_de_fato():
    """Um ensaio de caos que não sorteia azar nenhum passa sempre. Soma o caderno dos estudos rápidos e exige que cada FAMÍLIA de
    falha tenha acontecido, com folga larga sobre o que as sementes de hoje dão (o caminho muda quando a produção muda)."""
    if len(_CADERNOS) < len(ESTUDOS_RAPIDOS):
        pytest.skip('os ensaios rápidos não rodaram todos nesta execução (seleção com -k?)')
    total = collections.Counter()
    for caos in _CADERNOS.values():
        total.update(caos.contagem)
    soma = lambda prefixo: sum(n for nome, n in total.items() if nome.startswith(prefixo))   # noqa: E731
    sessoes = sum(c.sessoes for c in _CADERNOS.values())
    assert soma('queda:') >= 4 and sessoes >= len(_CADERNOS) + 4, f'quase nenhuma queda/retomada: {dict(total)}'
    assert soma('jornal_oserror:') >= 3 and soma('jornal_absorvido:') >= 2, f'o jornal quase não falhou: {dict(total)}'
    assert total['popen_oserror'] >= 2 and soma('replace_') >= 3 and total['leitura_presa'] >= 2, f'o disco quase não falhou: {dict(total)}'
    ruins = sum(n for nome, n in total.items() if nome.startswith('plano:') and nome not in ('plano:ok', 'plano:ok_preso'))
    assert ruins >= 15 and total['plano:ok'] >= 40, f'avaliadores: {ruins} planos de falha e {total["plano:ok"]} bons: {dict(total)}'


# ── o defeito que a varredura achou (sementes 17 e 20 com K = 3, 3 níveis, 7 FRE, antes da correção) ─────────────────────────────
def test_duas_quedas_seguidas_antes_do_relancamento_nao_contam_o_mesmo_batimento_duas_vezes_como_perdido(area):
    """Sessão 1 cai com um avaliador em voo (batimento de 60.000 passos). A sessão 2 registra 'relancado_apos_queda' com 60.000
    perdidos e cai ANTES de relançar (reinício em laço; ou OSError no 'relancado' do pendente seguinte, que aborta a partida). A
    sessão 3 encontrava o MESMO _BATIMENTO.json e o somava de novo em `fre_perdido_em_quedas_nao_cobrado` — é o par da correção
    C6: quem conta um batimento tem de guardá-lo na hora."""
    caos = Caos(0, area, 1, taxas={'planos': (('pendurado', 1),)})
    ns = H.argparse.Namespace(study='s', orcamento=N, niveis=2, eta=3, seed=5, inicializacao='professor', seeds_trial=[42], smoke=False, retomar=True,
                              n_fre=3.0, paralelo=1, warmstart=area['il'], referencias=area['refs'], dispositivo=None, max_tentativas=2,
                              max_falhas_seguidas=4, ignorar_disjuntor=False, limite_sem_batimento_h=1.0, intervalo_s=0.0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(H, 'DIR_HPO', area['hpo'])
        mp.setattr(P, 'hash_git', lambda: 'git-do-teste')
        mp.setattr(H, 'subprocess', types.SimpleNamespace(Popen=lambda cmd, **kw: caos.procs.append(ProcCaotico(caos, cmd[-1], kw.get('stdout'))) or caos.procs[-1],
                                                          DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired))
        mp.setattr(H, 'prender_filhos_ao_coordenador', lambda: (lambda proc: None))

        def sessao():
            c = H.Coordenador(ns, comando_avaliador=lambda d: ['avaliador-de-mentira', d], log=lambda m: None)
            assert c.abrir()
            c.espelho = EspelhoDeMentira()
            return c
        c1 = sessao()
        assert c1._ciclo() is False and [p.id for p in caos.procs] == [0]
        caos.procs[0].bater(60_000)
        c1.em_voo[0]['saida'].close()                       # a sessão 1 cai
        c2 = sessao()
        c2._retomar_pendentes()                             # ... e a sessão 2 cai antes de relançar
        c3 = sessao()
        c3._retomar_pendentes()
    perdidos = [e['passos_perdidos'] for e in no_arquivo(area) if e['ev'] == 'relancado_apos_queda']
    assert perdidos == [60_000, 0], f'o mesmo batimento foi dado como perdido em cada retomada: {perdidos}'
    assert c3.progresso()['fre_perdido_em_quedas_nao_cobrado'] == round(60_000 / N, 3)


# ── o defeito que a varredura achou (semente 1049 com K = 3, 3 níveis, 7 FRE, antes da correção) ──────────────────────────────────
def test_jornal_que_encolhe_por_fora_logo_depois_de_uma_escrita_nossa_falhar_nao_ganha_um_buraco_de_nuls(tmp_path):
    """A escrita de um evento falha nas 8 tentativas (arquivo preso pelo OneDrive): fica resto NOSSO além do tamanho conhecido, e a
    conferência de 'alguém mexeu por fora' era desligada INTEIRA até a escrita seguinte dar certo. Se nesse intervalo o arquivo
    encolhe (o mesmo OneDrive restaura uma versão anterior), o `truncate` o esticava com NULs e o evento ia parar depois do buraco:
    'jornal corrompido na linha N' em toda retomada, até alguém consertar à mão. Resto nosso só explica arquivo MAIOR."""
    arq = str(tmp_path / 'jornal.jsonl')
    jornal = H.Jornal(arq)
    for k in range(3):
        jornal.acrescentar({'ev': 'de_teste', 'k': k})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(H, 'time', types.SimpleNamespace(sleep=lambda s: None, strftime=time.strftime, time=time.time, monotonic=time.monotonic))
        mp.setattr(H, 'open', lambda caminho, *a, **k: ArquivoDoJornal(open(caminho, *a, **k), rasga=OSError(28, 'escrita pela metade (de mentira)')), raising=False)
        with pytest.raises(OSError, match='pela metade'):
            jornal.acrescentar({'ev': 'de_teste', 'k': 3})
    assert os.path.getsize(arq) > jornal._tamanho, 'o teste não deixou resto de escrita nossa no arquivo'
    primeira = open(arq, 'rb').read().split(b'\n')[0] + b'\n'
    with open(arq, 'r+b') as f:                             # por fora: volta a versão que só tinha o primeiro evento
        f.truncate(len(primeira))
    with pytest.raises(RuntimeError, match='encolheu'):
        jornal.acrescentar({'ev': 'de_teste', 'k': 3})
    assert open(arq, 'rb').read() == primeira, 'o jornal foi tocado depois de encolher por fora (buraco de NULs?)'
    assert [e['k'] for e in H.Jornal(arq).eventos] == [0]


# ── a varredura larga ────────────────────────────────────────────────────────────────────────────
@pytest.mark.lento
@pytest.mark.parametrize('k, niveis, n_fre, espelho_real, sementes', [
    (3, 3, 7.0, False, range(1000, 1030)), (3, 2, 5.0, False, range(2000, 2030)), (1, 2, 4.0, False, range(3000, 3020)),
    (1, 3, 5.0, False, range(4000, 4015)), (2, 3, 6.0, False, range(5000, 5015)), ((1, 2, 3), 3, 6.0, False, range(8000, 8015)),
    (3, 3, 6.0, True, range(6000, 6008)), (3, 2, 5.0, True, range(7000, 7008))])
def test_varredura_larga_de_sementes_sob_caos(tmp_path, k, niveis, n_fre, espelho_real, sementes):
    """141 estudos (~5 min com a máquina livre; os 16 com o espelho de verdade levam mais da metade). K em tupla = cada sessão sorteia o seu. DEHB_CAOS_SEMENTES=<n> multiplica o nº de sementes por
    configuração, para caçar."""
    vezes, ruins = int(os.environ.get('DEHB_CAOS_SEMENTES', '1')), {}
    for semente in range(sementes[0], sementes[0] + vezes * len(sementes)):
        caos, violacoes = rodar_e_conferir(montar_area(tmp_path / f's{semente}'), semente, k, niveis, n_fre, espelho_real=espelho_real)
        if violacoes:
            ruins[semente] = violacoes[:4]
    assert not ruins, f'K = {k}, {niveis} níveis, {n_fre} FRE: {len(ruins)} sementes com violação: ' + json.dumps(ruins, ensure_ascii=False, indent=1)[:6000]
