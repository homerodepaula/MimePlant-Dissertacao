"""hpo/hpo_dehb.py — DEHB como terceiro método de HPO da fase de RL do MimePlant (D-16: estudo de COMPARAÇÃO; D-34).

O algoritmo está em `dehb_nucleo.py` (nativo: o pacote `dehb` exige numpy < 2). Este arquivo é o COORDENADOR: pede
configurações ao DEHB, avalia cada uma num SUBPROCESSO (`--avaliar`), devolve a aptidão, e mantém o estudo retomável.

Condições IDÊNTICAS às dos estudos TPE e busca aleatória (`hpo_rl.py`), para a comparação valer:
  substrato   `ppo.treinar_rl` partindo do clone do professor, mesmo espaço de 12 hiperparâmetros (`hp.ESPACO_MIMEPLANT`), mesma
              semente de treino por avaliação (42), mesmo warm-start e mesmas referências de normalização;
  escada      a do podador dos outros estudos: `--niveis 2`, eta 3 → fidelidades N//3 e N. Com 2 níveis as subpopulações têm 3 e 2
              indivíduos e o preenchimento do pool de pais (caminho de borda no artigo) vira o caminho principal: o artigo não
              avalia esse regime, e a conclusão vale para "DEHB nesta escada". `--niveis 3` (N//9) fica para sensibilidade: o
              projeto mediu que N/9 não separa hiperparâmetros (`k_aquec` consome o início do orçamento; hpo_rl.py:153-161);
  objetivo    na fidelidade máxima, a média nos 3 regimes do retorno normalizado em VALIDAÇÃO-OBJETIVO (520–549) do melhor bloco
              escolhido em validação-rollback; nas fidelidades menores, o retorno normalizado em validação-rollback in-dist
              (500–519) no ÚLTIMO bloco — o mesmo número, no mesmo passo, que o podador dos outros estudos vê. A DE só compara
              aptidões DENTRO da mesma fidelidade, então as duas quantidades nunca se cruzam. Teste (999+) não entra aqui;
  orçamento   em FRE pelos passos REAIS treinados (`passos / N`), como no hpo_rl, inclusive o parcial de avaliação que morreu.
O que NÃO tem equivalente, e se declara: o DEHB re-treina do clone em cada fidelidade (o Hyperband do Optuna continua o mesmo
trial), o que dá ~8 % menos avaliações completas por FRE; não há fase de partida aleatória como a do TPE; um estudo por método é
comparação descritiva (n = 1), e a inferência fica na reavaliação com 5 sementes (`reavaliar_topk.py`, em pasta PRÓPRIA:
`protocolo.DIR_REAVALIACAO_DEHB` — o DEHB não concorre a MimePlant nem é base das ablações).

Resiliência (a máquina reinicia sozinha e a cadeia relança o mesmo comando com --retomar):
  jornal      `jornal.jsonl`, só de acréscimo, uma linha por evento, flush + fsync, escrito ANTES do efeito (pedir → jornal →
              lançar; resultado → jornal → aplicar → espelho). Leitura tolerante: última linha cortada é descartada (com cópia);
  retomada    REPLAY do jornal num DEHB novo com a mesma semente, conferindo cada pedido bit a bit. Se divergir (núcleo de
              outra versão, por exemplo), o estado é ADOTADO dos próprios eventos e o gerador restaurado, com registro;
  trabalhos   cada avaliação é um subprocesso com pasta própria (`trabalho_<id>`), resultado gravado de forma atômica com
              impressão digital (estudo, id, vetor, fidelidade, semente, git): resultado velho ou de outro estudo é recusado;
  quedas      se o COORDENADOR cai, o parcial já treinado pelas avaliações em voo se perde e NÃO entra no orçamento — no hpo_rl
              o trial do processo morto fica RUNNING para sempre e também não conta —, mas fica registrado
              (`relancado_apos_queda`, e `fre_perdido_em_quedas_nao_cobrado` no PROGRESSO.json);
  falhas      determinística (exceção no treino, aptidão não finita) → informada como falha (+inf: nunca substitui, nunca é
              promovida), trial FAIL com o FRE gasto; transitória (OOM, morte do processo, sem batimento) → repetida até
              `--max-tentativas`, e o parcial gasto entra no orçamento; `--max-falhas-seguidas` falhas finais seguidas param o
              estudo com _FAILED.flag (defeito sistêmico não pode virar "todas as configurações são ruins");
  exclusão    trava do sistema operacional por estudo: um segundo coordenador espera; filhos presos a um Job Object do Windows
              (morrem com o coordenador) e, se isso falhar, o órfão é adotado pela trava `_EXECUTANDO.lock` em vez de duplicado;
  espelho     `<study>.db` do Optuna é PROJEÇÃO do jornal (COMPLETE = fidelidade máxima com value = objetivo; PRUNED = fidelidade
              menor; FAIL = falha), reconciliado na partida e antes da _DONE.flag; erro nele nunca derruba o estudo. É por ele que
              `reavaliar_topk.py`, `analise_hpo.py` e `pos_hpo.trials_em_voo` leem o DEHB sem mudar uma linha.

Uso (a partir da raiz do código):
  python -X utf8 hpo/hpo_dehb.py --study dehb_r3 --n-fre 64 [--orcamento N] [--paralelo 3] [--niveis 2] [--retomar]
  python -X utf8 hpo/hpo_dehb.py --study smoke_dehb --smoke
  python -X utf8 hpo/hpo_dehb.py --study dehb_r3 --reconstruir-espelho
Saída: ../experimentos/hpo/<study>/{jornal.jsonl, trabalho_<id>/{trabalho,resultado,metricas,config}.json, PROGRESSO.json,
INDICE.json, _STATUS.txt, _DONE.flag} e ../experimentos/hpo/<study>.db. As pastas são por id de TRABALHO (ordem do pedido); o
número do trial no Optuna segue a ordem de conclusão — `INDICE.json` liga os dois.
"""
import os
import sys
import json
import time
import uuid
import zlib
import hashlib
import argparse
import datetime
import traceback
import subprocess

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (RAIZ, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import protocolo as P                                          # noqa: E402  (só numpy no topo; torch fica nas funções)

P.preparar_sys_path()
from hp import ESPACO_MIMEPLANT, ESPACOS, INICIALIZACAO_DO_METODO  # noqa: E402  (só numpy)
import dehb_nucleo as NUC                                       # noqa: E402

VF = os.path.dirname(RAIZ)
DIR_HPO = os.path.join(VF, 'experimentos', 'hpo')
BLOCOS_POR_EXECUCAO = 9            # = ppo.HP_PADRAO['blocos'] (um teste trava): o bloco tem N // 9 passos em QUALQUER fidelidade
EPISODIOS_POR_COLETA = 12          # = ppo.HP_PADRAO['V'] (idem): uma coleta tem V x 365 passos
SAIDA_FALHA_DA_CONFIGURACAO = 3    # códigos de saída do avaliador
SAIDA_FALHA_TRANSITORIA = 4
HP_SMOKE = {'V': 2, 'k_aquec': 0, 'n_epocas': 1, 'minilote_seqs': 4}   # k_aquec 0: senão a fidelidade baixa acaba sem nenhum bloco


def espaco_de(inicializacao: str):
    """O mesmo espaço de `hpo_rl.espaco_de` (que não pode ser importado aqui: ele carrega torch). Um teste trava a igualdade."""
    return ESPACO_MIMEPLANT if inicializacao == 'professor' else ESPACOS['ppo_zero']


def agora_iso() -> str:
    return datetime.datetime.now().isoformat(timespec='milliseconds')


# ── arquivos: escrita atômica, trava do SO, jornal ───────────────────────────────────────────────
def escrever_json_atomico(caminho: str, obj, tentativas: int = 12) -> None:
    """tmp + flush + fsync + os.replace, com novas tentativas: OneDrive e antivírus seguram arquivos desta pasta (WinError 5/32)."""
    tmp = f'{caminho}.tmp{os.getpid()}'

    def sem_temporario():
        try:
            os.remove(tmp)
        except OSError:
            pass
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=1, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        sem_temporario()                                  # meio JSON (valor não finito recusado, disco cheio) não fica na pasta
        raise
    for k in range(tentativas):
        try:
            os.replace(tmp, caminho)
            return
        except PermissionError:
            if k == tentativas - 1:
                sem_temporario()
                raise
            time.sleep(0.25 * (k + 1))


class Trava:
    """Trava exclusiva do sistema operacional sobre um arquivo. O SO a solta quando o processo morre, então não sobra trava
    velha depois de um reinício — ao contrário de um arquivo de pid."""

    def __init__(self, caminho: str):
        self.caminho, self._f = caminho, None

    def tentar(self) -> bool:
        os.makedirs(os.path.dirname(self.caminho), exist_ok=True)
        f = open(self.caminho, 'a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self._f = f
        return True

    def soltar(self) -> None:
        if self._f is not None:
            try:
                if os.name == 'nt':
                    import msvcrt
                    self._f.seek(0)
                    msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self._f.close()
            self._f = None


class Jornal:
    """Jornal JSONL só de acréscimo. Cada evento tem o índice contíguo 'i'. Uma queda no meio da escrita deixa, no máximo, a
    ÚLTIMA linha cortada: ela é descartada (com cópia do arquivo) e o arquivo truncado no último '\\n' válido, para que o
    acréscimo seguinte não cole dois registros. Linha inválida no MEIO é corrupção de verdade e aborta."""

    def __init__(self, caminho: str):
        self.caminho = caminho
        self.eventos = []
        self._tamanho = 0            # bytes válidos do arquivo: todo acréscimo parte daqui
        self._resto_proprio = False  # True enquanto uma escrita NOSSA pode ter ficado pela metade (erro de E/S): não é escritor estranho
        if os.path.exists(caminho):
            self._ler()

    def _ler(self) -> None:
        bruto = open(self.caminho, 'rb').read()
        dados = bruto.rstrip(b'\x00')                     # o NTFS pode deixar NULs na cauda depois de um desligamento abrupto
        linhas = dados.split(b'\n')
        resto = linhas.pop()                              # o que vem depois do último '\n' ('' se o arquivo está íntegro)
        validos = 0
        for k, ln in enumerate(linhas):
            try:
                ev = json.loads(ln.decode('utf-8'))
                if ev.get('i') != len(self.eventos):
                    raise ValueError(f"índice {ev.get('i')} onde se esperava {len(self.eventos)}")
            except Exception as e:
                if k == len(linhas) - 1 and not resto:
                    resto = ln                            # a última linha completa está estragada: trata como cauda cortada
                    break
                raise RuntimeError(f'jornal corrompido na linha {k + 1} de {self.caminho}: {e}') from e
            self.eventos.append(ev)
            validos += len(ln) + 1
        self._tamanho = validos
        if resto or validos != len(bruto):
            copia = f"{self.caminho}.corrompido.{time.strftime('%Y%m%d_%H%M%S')}"
            with open(copia, 'wb') as f:
                f.write(bruto)
            with open(self.caminho, 'r+b') as f:
                f.truncate(validos)
                f.flush()
                os.fsync(f.fileno())

    def acrescentar(self, evento: dict, tentativas: int = 8) -> dict:
        """Grava o evento no fim do arquivo e só então o põe em memória. A escrita parte SEMPRE do tamanho conhecido (trunca
        o que houver além dele), então repetir depois de um erro de E/S nunca cola um registro num resto de escrita parcial.
        Arquivo de tamanho DIFERENTE do conhecido antes de escrever = alguém mexeu no jornal por fora: aborta sem tocar nele.
        Maior: outro processo escreveu (não intercalar). Menor ou sumido (o OneDrive restaurou uma versão anterior, alguém
        truncou): `truncate` o esticaria com NULs e o evento novo ficaria depois de um buraco, perdido na leitura seguinte.
        Abortar é seguro: a retomada relê o que ficou em disco, e todo resultado já treinado continua na pasta do trabalho."""
        ev = dict(evento, i=len(self.eventos))
        linha = (json.dumps(ev, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        for k in range(tentativas):
            try:
                em_disco = os.path.getsize(self.caminho) if os.path.exists(self.caminho) else 0
                if em_disco > self._tamanho and not self._resto_proprio:
                    raise RuntimeError(f'{self.caminho} cresceu fora deste processo: há outro coordenador escrevendo no jornal?')
                if em_disco < self._tamanho:              # vale também com resto NOSSO pendente: escrita nossa nunca deixa o arquivo MENOR
                    raise RuntimeError(f'{self.caminho} encolheu (ou sumiu) fora deste processo: {em_disco} bytes em disco, '
                                       f'{self._tamanho} conhecidos. Relance com --retomar: o estudo segue do que ficou gravado')
                self._resto_proprio = True                # daqui até o fsync, bytes além de _tamanho são resto de escrita NOSSA
                with open(self.caminho, 'r+b' if os.path.exists(self.caminho) else 'w+b') as f:
                    f.truncate(self._tamanho)
                    f.seek(self._tamanho)
                    f.write(linha)                        # uma única escrita por evento
                    f.flush()
                    os.fsync(f.fileno())
                self._resto_proprio = False
                break
            except OSError:
                if k == tentativas - 1:
                    raise
                time.sleep(0.5 * (k + 1))
        self._tamanho += len(linha)
        self.eventos.append(ev)
        return ev


def impressao_digital(id_execucao: str, registro: dict, semente_treino, passos_max: int, hash_git: str) -> str:
    base = {'id_execucao': id_execucao, 'id': registro['id'], 'vetor': registro['vetor'], 'fidelidade': registro['fidelidade'],
            'seeds': list(semente_treino), 'N': int(passos_max), 'hash_git': hash_git}
    return hashlib.sha256(json.dumps(base, sort_keys=True).encode('utf-8')).hexdigest()


def manter_a_maquina_acordada(ligar: bool) -> None:
    """Pede ao Windows que não suspenda por ociosidade enquanto o estudo roda (SetThreadExecutionState). Lido em 18/09/2026:
    o plano de energia desta máquina suspende após 30 min na tomada. A rodada 2 rodou dias sem dormir — a heurística de
    ociosidade do Windows leva em conta o uso de CPU —, mas isso é comportamento não garantido, e um estudo de dias não pode
    depender dele. Não impede suspensão pedida pelo usuário (tampa, botão). Fora do Windows, ou se a API falhar, não faz nada."""
    if os.name != 'nt':
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if ligar else 0))
    except Exception:
        pass


def prender_filhos_ao_coordenador():
    """Job Object do Windows com KILL_ON_JOB_CLOSE: os filhos morrem junto com o coordenador (sem isso, um coordenador morto
    deixa 3 treinos de horas na GPU e o relançamento dispara outros 3). Devolve uma função `prender(Popen)`; fora do Windows,
    ou se a API falhar, devolve uma função que não faz nada — a segunda defesa é a trava `_EXECUTANDO.lock` de cada trabalho."""
    if os.name != 'nt':
        return lambda proc: None
    try:
        import ctypes
        from ctypes import wintypes

        class _LimitesBasicos(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                        ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                        ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                        ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]

        class _ContadoresES(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                                                       'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

        class _LimitesEstendidos(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', _LimitesBasicos), ('IoInfo', _ContadoresES),
                        ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                        ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]

        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        info = _LimitesEstendidos()
        info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not job or not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            return lambda proc: None

        def prender(proc):
            try:
                k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle)))
            except Exception:
                pass
        prender._job = job                                      # mantém o identificador vivo enquanto o coordenador viver
        return prender
    except Exception:
        return lambda proc: None


# ── espelho no Optuna: projeção do jornal ────────────────────────────────────────────────────────
def liberar_banco(study) -> None:
    """Fecha as conexões do SQLAlchemy de um study do Optuna. No Windows um .db com conexão aberta não pode ser renomeado
    nem apagado, e o motor do Optuna mantém a conexão em pool enquanto o objeto viver."""
    armazenamento = getattr(study, '_storage', None)
    for alvo in (armazenamento, getattr(armazenamento, '_backend', None)):
        motor = getattr(alvo, 'engine', None)
        if motor is not None:
            try:
                motor.dispose()
            except Exception:
                pass


class Espelho:
    """Projeção do jornal num study do Optuna. Invariante: o trial de número k é o k-ésimo evento terminal do jornal
    ('informar' ou 'tentativa_falhou'), e guarda `evento_i`. Qualquer erro aqui é registrado e NÃO derruba o estudo: o jornal é
    a verdade, e `reconciliar()` (na partida e antes da _DONE.flag) ou `--reconstruir-espelho` refazem o que faltar."""

    def __init__(self, study: str, espaco: dict, log):
        self.study_nome, self.espaco, self.log = study, espaco, log
        self.caminho = os.path.join(DIR_HPO, study + '.db')
        self._study = None
        self._ultima_data = None

    def _abrir(self):
        if self._study is None:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self._study = optuna.create_study(study_name=self.study_nome, direction='maximize', load_if_exists=True,
                                              storage=f"sqlite:///{self.caminho.replace(os.sep, '/')}")
        return self._study

    def distribuicoes(self):
        import optuna
        d = {}
        for nome, spec in self.espaco.items():
            if spec[0] == 'log':
                d[nome] = optuna.distributions.FloatDistribution(spec[1], spec[2], log=True)
            elif spec[0] == 'float':
                d[nome] = optuna.distributions.FloatDistribution(spec[1], spec[2])
            elif spec[0] == 'int':
                d[nome] = optuna.distributions.IntDistribution(int(spec[1]), int(spec[2]))
            else:
                d[nome] = optuna.distributions.CategoricalDistribution(list(spec[1]))
        return d

    def atributos_do_estudo(self, atributos: dict, sessao: dict) -> None:
        try:
            st = self._abrir()
            for k, v in atributos.items():
                st.set_user_attr(k, json.loads(json.dumps(v, default=str)))
            st.set_user_attr('sessoes', list(st.user_attrs.get('sessoes', [])) + [sessao])   # acumula: versões diferentes do código ficam visíveis
        except Exception as e:
            self.log(f'[espelho] atributos do estudo não gravados ({type(e).__name__}: {e})')

    def _trial_de(self, ev: dict):
        import optuna
        estados = optuna.trial.TrialState
        comuns = {'evento_i': ev['i'], 'id_trabalho': ev['id'], 'fidelidade': ev.get('fidelidade'), 'nivel': ev.get('nivel'),
                  'bracket': ev.get('bracket'), 'degrau': ev.get('degrau'), 'origem': ev.get('origem'),
                  'passos': ev.get('passos', 0), 'tempo_s': ev.get('tempo_s'), 'hash_git': ev.get('hash_git'),
                  'pasta': f"trabalho_{ev['id']}"}
        intermediarios = {int(p['passos']): float(p['proxy_norm']) for p in (ev.get('curva') or [])
                          if p.get('proxy_norm') is not None and np.isfinite(p['proxy_norm'])}
        params = ev.get('hp') or {}
        if ev['ev'] == 'tentativa_falhou' or ev.get('falhou'):
            t = optuna.trial.create_trial(state=estados.FAIL, params=params, distributions=self.distribuicoes() if params else {},
                                          intermediate_values=intermediarios,
                                          user_attrs=dict(comuns, fre_parcial=float(ev.get('fre', 0.0)), motivo=ev.get('motivo'),
                                                          tentativa=ev.get('tentativa')))
        elif ev.get('fidelidade_maxima'):
            t = optuna.trial.create_trial(state=estados.COMPLETE, value=float(ev['objetivo']), params=params,
                                          distributions=self.distribuicoes(), intermediate_values=intermediarios,
                                          user_attrs=dict(comuns, fre=float(ev['fre'])))
        else:
            t = optuna.trial.create_trial(state=estados.PRUNED, params=params, distributions=self.distribuicoes(),
                                          intermediate_values=intermediarios,
                                          user_attrs=dict(comuns, fre=float(ev['fre']), indicador=ev.get('indicador')))
        # datas do PRÓPRIO evento, e monótonas: analise_hpo.curva_melhor_ate_agora ordena por datetime_complete
        fim = datetime.datetime.fromisoformat(ev.get('fim') or ev['data'])
        if self._ultima_data is not None and fim <= self._ultima_data:
            fim = self._ultima_data + datetime.timedelta(milliseconds=1)
        t.datetime_start = min(datetime.datetime.fromisoformat(ev.get('inicio') or ev['data']), fim)
        t.datetime_complete = fim
        self._ultima_data = fim
        return t

    def reconciliar(self, eventos: list) -> bool:
        """Acrescenta ao banco os eventos terminais que faltam, na ordem do jornal. Devolve True se o espelho está completo."""
        terminais = [e for e in eventos if e['ev'] in ('informar', 'tentativa_falhou')]
        for tentativa in range(5):
            try:
                st = self._abrir()
                existentes = st.get_trials(deepcopy=False)
                for k, t in enumerate(existentes):
                    if k >= len(terminais) or t.user_attrs.get('evento_i') != terminais[k]['i']:
                        self.log(f'[espelho] o banco não é projeção deste jornal (trial {k}); use --reconstruir-espelho')
                        return False
                if existentes:
                    self._ultima_data = max([t.datetime_complete for t in existentes if t.datetime_complete] + ([self._ultima_data] if self._ultima_data else []))
                for ev in terminais[len(existentes):]:
                    st.add_trial(self._trial_de(ev))
                return True
            except Exception as e:
                self.log(f'[espelho] tentativa {tentativa + 1}/5 falhou ({type(e).__name__}: {e})')
                self.fechar()
                time.sleep(2.0 * (tentativa + 1))
        return False

    def fechar(self) -> None:
        if self._study is not None:
            liberar_banco(self._study)
            self._study = None

    def indice(self) -> dict:
        try:
            return {str(t.number): {'id_trabalho': t.user_attrs.get('id_trabalho'), 'pasta': t.user_attrs.get('pasta'),
                                    'estado': t.state.name, 'valor': t.value} for t in self._abrir().get_trials(deepcopy=False)}
        except Exception:
            return {}


# ── o avaliador (subprocesso): treina uma configuração numa fidelidade ───────────────────────────
def avaliar(dir_trabalho: str) -> int:
    """Roda UMA avaliação. Só aqui entram torch, avaliacao, ppo e bakeoff. Códigos de saída: 0 (resultado.json gravado),
    3 (falha da configuração: falha.json), 4 (falha transitória: GPU sem memória, erro de CUDA)."""
    dir_trabalho = os.path.abspath(dir_trabalho)
    trava = Trava(os.path.join(dir_trabalho, '_EXECUTANDO.lock'))
    if not trava.tentar():
        print(f'[avaliar] outro processo já executa {dir_trabalho}; saindo', flush=True)
        return SAIDA_FALHA_TRANSITORIA
    tr = json.load(open(os.path.join(dir_trabalho, 'trabalho.json'), encoding='utf-8'))
    t0, inicio = time.time(), agora_iso()
    estado = {'passos': 0}

    def batimento(extra=None):
        try:
            escrever_json_atomico(os.path.join(dir_trabalho, '_BATIMENTO.json'),
                                  dict({'data': agora_iso(), 'pid': os.getpid(), 'passos': estado['passos']}, **(extra or {})), tentativas=3)
        except Exception:
            pass

    def falha(codigo: int, motivo: str, detalhe: str) -> int:
        escrever_json_atomico(os.path.join(dir_trabalho, 'falha.json'),
                              {'impressao': tr['impressao'], 'id': tr['id'], 'motivo': motivo, 'detalhe': detalhe[-4000:],
                               'passos': int(estado['passos']), 'inicio': inicio, 'fim': agora_iso(), 'tempo_s': time.time() - t0,
                               'hash_git': P.hash_git()})
        return codigo

    batimento()
    try:
        import copy
        import torch
        import avaliacao as A
        import ppo as PPO
        from envs_lote import fabrica_env_padrao
        from bakeoff import carregar_referencias, normalizado
    except Exception:
        return falha(SAIDA_FALHA_TRANSITORIA, 'importacao', traceback.format_exc())

    try:
        N, fidelidade = int(tr['passos_max']), int(tr['fidelidade'])
        refs = carregar_referencias(tr['referencias'])
        r_alea = refs['aleatorio']['validacao_rollback']['in_dist']['media']
        r_perf = refs['oraculo_perfeito']['validacao_rollback']['in_dist']['media']
        n_rb, n_obj = (3, 3) if tr['smoke'] else (P.N_VALIDACAO_ROLLBACK, P.N_VALIDACAO_OBJETIVO)
        sementes_rb, sementes_obj = P.sementes_validacao_rollback()[:n_rb], P.sementes_validacao_objetivo()[:n_obj]
        env_aval = fabrica_env_padrao()
        hp = dict(tr['hp'], inicializacao=tr['inicializacao'], blocos=int(tr['blocos']))
        if tr['smoke']:
            hp.update(HP_SMOKE)
        indicadores, objetivos, detalhes, passos_total = [], [], [], 0
        for seed in tr['seeds']:
            politica_il = os.path.join(tr['warmstart'], f'seed_{seed}', 'modelo_final.pt')
            curva, melhor = [], {'valor': -np.inf, 'passos': 0, 'estado': None}
            base = passos_total

            def medir(passos, agente):
                res = A.avaliar(A.PoliticaRede(agente), env_aval, sementes_rb, 'in_dist')
                m = float(np.mean(res['recompensas']))
                curva.append({'passos': int(passos), 'media': m, 'proxy_norm': A.normalizar(m, r_alea, r_perf),
                              'ociosidade_pct': A.ociosidade_pct(res)})
                if m > melhor['valor']:
                    melhor.update(valor=m, passos=int(passos), estado=copy.deepcopy(agente.rede.state_dict()))

            def callback(passos, agente):
                estado['passos'] = base + int(passos)
                medir(passos, agente)
                batimento({'bloco': len(curva)})
                return False

            def log_do_treino(msg):
                batimento()                                   # o treino fala a cada atualização: é o batimento fino

            res = PPO.treinar_rl(politica_il, hp, int(seed), fidelidade, callback=callback, log=log_do_treino,
                                 dispositivo=tr.get('dispositivo'))
            passos_total += int(res['passos'])
            estado['passos'] = passos_total
            agente = res['agente']
            pos_treino = not curva
            if pos_treino:                                    # orçamento menor que um bloco (smoke): mede uma vez, no estado final
                medir(res['passos'], agente)
            indicadores.append(curva[-1]['proxy_norm'])       # o ÚLTIMO bloco: o que o podador dos outros estudos vê neste passo
            detalhe = {'seed': seed, 'passos': res['passos'], 'curva_validacao_rollback': curva, 'indicador_pos_treino': pos_treino,
                       'melhor': {'valor': melhor['valor'], 'passos': melhor['passos']}, 'curva_treino': res['curva']}
            if tr['fidelidade_maxima']:
                if melhor['estado'] is not None and melhor['valor'] >= curva[-1]['media']:
                    agente.rede.load_state_dict(melhor['estado'])
                agente.rede.eval()
                pol, por_regime = A.PoliticaRede(agente), {}
                for regime in P.NOMES_REGIMES:
                    r = A.avaliar(pol, env_aval, sementes_obj, regime)
                    media = float(np.mean(r['recompensas']))
                    por_regime[regime] = {'media': media, 'iqm': A.iqm(r['recompensas']),
                                          'normalizada': normalizado(media, refs, 'validacao_objetivo', regime),
                                          'ociosidade_pct': A.ociosidade_pct(r), 'acoes_pct': A.acoes_pct(r),
                                          'comp_medio': {k: float(np.mean(v)) for k, v in r['comp'].items()}}
                    batimento({'regime': regime})
                detalhe['validacao_objetivo'] = por_regime
                detalhe['objetivo'] = float(np.mean([por_regime[reg]['normalizada'] for reg in P.NOMES_REGIMES]))
                objetivos.append(detalhe['objetivo'])
            detalhes.append(detalhe)

        indicador = float(np.mean(indicadores))
        objetivo = float(np.mean(objetivos)) if tr['fidelidade_maxima'] else None
        valor = objetivo if tr['fidelidade_maxima'] else indicador
        if not np.isfinite(valor) or not np.isfinite(indicador):
            # o indicador conta também na fidelidade máxima: é ele que escolhe o bloco, e sem ele o objetivo não é o do protocolo
            return falha(SAIDA_FALHA_DA_CONFIGURACAO, 'aptidao_nao_finita', f'objetivo={objetivo} indicador={indicador}')
        fre = passos_total / (N * max(1, len(tr['seeds'])))
        with open(os.path.join(dir_trabalho, 'metricas.json'), 'w', encoding='utf-8') as f:
            json.dump({'id_trabalho': tr['id'], 'fidelidade': fidelidade, 'fre': fre, 'tempo_s': time.time() - t0, 'objetivo': objetivo,
                       'indicador': indicador, 'por_seed': detalhes}, f, ensure_ascii=False, indent=1, default=float)
        P.registrar_config(dir_trabalho, experimento='hpo_dehb', study=tr['study'], id_trabalho=tr['id'], hp=hp,
                           fidelidade=fidelidade, orcamento_passos=N, seeds_trial=tr['seeds'], warmstart=tr['warmstart'],
                           referencias=tr['referencias'], dehb={k: tr[k] for k in ('nivel', 'bracket', 'degrau', 'origem', 'pais')})
        escrever_json_atomico(os.path.join(dir_trabalho, 'resultado.json'),
                              {'impressao': tr['impressao'], 'id': tr['id'], 'aptidao': -valor, 'objetivo': objetivo, 'indicador': indicador,
                               'passos': passos_total, 'fre': fre, 'inicio': inicio, 'fim': agora_iso(), 'tempo_s': time.time() - t0,
                               'curva': [{'passos': p['passos'], 'proxy_norm': p['proxy_norm']} for p in detalhes[0]['curva_validacao_rollback']],
                               'hash_git': P.hash_git(), 'versoes': P.versoes(), 'dispositivo': P.dispositivo()})
        return 0
    except Exception as e:
        texto = traceback.format_exc()
        transitoria = isinstance(e, torch.cuda.OutOfMemoryError) or 'CUDA' in texto or 'cuDNN' in texto or isinstance(e, (OSError, MemoryError))
        return falha(SAIDA_FALHA_TRANSITORIA if transitoria else SAIDA_FALHA_DA_CONFIGURACAO,
                     'gpu_ou_sistema' if transitoria else 'excecao_no_treino', texto)
    finally:
        trava.soltar()


# ── o coordenador ────────────────────────────────────────────────────────────────────────────────
def atributos_do_estudo(cab: dict, n_fre, paralelo) -> dict:
    """Os user_attrs do study no Optuna, derivados SÓ do cabeçalho do jornal (e do orçamento da sessão): assim
    `--reconstruir-espelho` os refaz iguais. `inicializacao` é lida por `reavaliar_topk.candidatos_do_study`."""
    return {'sampler': 'dehb', 'poda': f"dehb(eta={cab['eta']}): fidelidades {cab['fidelidades']}", 'inicializacao': cab['inicializacao'],
            'orcamento_passos': cab['N'], 'n_fre_alvo': n_fre, 'seeds_trial': cab['seeds_trial'], 'semente_dehb': cab['semente'],
            'versao_nucleo': cab['versao_nucleo'], 'espaco': cab['espaco'], 'paralelo': paralelo,
            'objetivo': 'fidelidade maxima: media dos 3 regimes da recompensa normalizada em validacao-objetivo 520-549 (melhor bloco em validacao-rollback)',
            'aptidao_nas_fidelidades_menores': 'recompensa normalizada em validacao-rollback 500-519 (in-dist) no ULTIMO bloco; viram trials PRUNED',
            'papel': 'estudo de COMPARACAO (D-16/D-34): nao concorre a MimePlant; reavaliacao em protocolo.DIR_REAVALIACAO_DEHB'}


class Coordenador:
    def __init__(self, args, comando_avaliador=None, log=None):
        self.args = args
        self.N = int(args.orcamento)
        self.dir = os.path.join(DIR_HPO, args.study)
        self.log = log or (lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True))
        self.comando_avaliador = comando_avaliador or (lambda d: [sys.executable, '-X', 'utf8', os.path.abspath(__file__), '--avaliar', d])
        self.espaco_dict = espaco_de(args.inicializacao)
        self.espaco = NUC.EspacoVetorial(self.espaco_dict)
        self.fidelidades = NUC.escada_de_passos(self.N, args.niveis, args.eta)
        self.semente = int(args.seed) if args.seed is not None else zlib.crc32(args.study.encode('utf-8')) % 100_000
        self.hash_git = P.hash_git()
        self.em_voo = {}            # id -> {'proc', 'registro', 'saida'}
        self.adotados = {}          # id -> Trava que um avaliador órfão ainda segura (esperamos o resultado dele)
        self.tentativas = {}        # id -> nº de tentativas transitórias já falhadas
        self._fila = []             # registros a lançar (novos pedidos, repetições, pendentes da retomada), em ordem
        self._pedido_por_gravar = None   # pedido que o núcleo JÁ sorteou e o jornal ainda não gravou (erro de E/S): regrava, não pede outro
        self._esgotados = set()     # ids cuja última tentativa já está no jornal e só falta o 'informar' (erro de E/S no meio)
        self._silencio = {}         # id -> {'mtime', 'acumulado'}: vigia de batimento em tempo ATIVO do coordenador
        self._tique = None
        self.prender = prender_filhos_ao_coordenador()
        self._ultimo_log = 0.0

    # ── partida: jornal novo ou replay ──
    def cabecalho(self) -> dict:
        return {'ev': 'cabecalho', 'versao_nucleo': NUC.VERSAO_NUCLEO, 'study': self.args.study, 'id_execucao': uuid.uuid4().hex,
                'semente': self.semente, 'espaco': self.espaco.descricao(), 'N': self.N, 'fidelidades': self.fidelidades,
                'eta': self.args.eta, 'F': NUC.F_PADRAO, 'CR': NUC.CR_PADRAO, 'inicializacao': self.args.inicializacao,
                'seeds_trial': list(self.args.seeds_trial), 'smoke': bool(self.args.smoke), 'numpy': np.__version__, 'data': agora_iso()}

    def _novo_dehb(self):
        return NUC.Dehb(self.espaco.dimensoes, self.fidelidades, eta=self.args.eta, semente=self.semente)

    def abrir(self) -> bool:
        """Abre (ou cria) o jornal, confere que é o MESMO estudo e refaz o estado. Devolve False se o estudo já está concluído
        para o orçamento pedido — nesse caso nada é escrito: relançar um estudo pronto tem de ser inofensivo."""
        os.makedirs(self.dir, exist_ok=True)
        arq = os.path.join(self.dir, 'jornal.jsonl')
        ha_jornal = os.path.exists(arq) and os.path.getsize(arq) > 0
        restos = [x for x in os.listdir(self.dir) if x.startswith('trabalho_')] + ([self.args.study + '.db'] if os.path.exists(os.path.join(DIR_HPO, self.args.study + '.db')) else [])
        if ha_jornal and not self.args.retomar:
            raise SystemExit(f'já existe um estudo em {self.dir}: use --retomar para continuá-lo ou outro --study')
        if not ha_jornal and restos:
            raise SystemExit(f'{self.dir} não tem jornal mas tem restos de outro estudo ({restos[:3]}...): use outro --study ou limpe a pasta')
        self.jornal = Jornal(arq)
        if not self.jornal.eventos:
            self.jornal.acrescentar(self.cabecalho())
        self.cab = self.jornal.eventos[0]
        for chave, atual in (('semente', self.semente), ('espaco', self.espaco.descricao()), ('N', self.N), ('fidelidades', self.fidelidades),
                             ('eta', self.args.eta), ('inicializacao', self.args.inicializacao), ('seeds_trial', list(self.args.seeds_trial)),
                             ('smoke', bool(self.args.smoke))):
            if json.loads(json.dumps(self.cab.get(chave))) != json.loads(json.dumps(atual)):
                raise SystemExit(f'--retomar com {chave} diferente do estudo gravado ({self.cab.get(chave)!r} x {atual!r}): é outro estudo; use outro --study')
        feito = os.path.join(self.dir, '_DONE.flag')
        if os.path.exists(feito):
            if self.fre_consumidos() >= self.args.n_fre:
                return False
            os.remove(feito)                                # --n-fre maior que o já gasto: o estudo continua
        self._replay()
        self.jornal.acrescentar({'ev': 'sessao', 'pid': os.getpid(), 'hash_git': self.hash_git, 'n_fre': self.args.n_fre,
                                 'paralelo': self.args.paralelo, 'versao_nucleo': NUC.VERSAO_NUCLEO, 'data': agora_iso()})
        return True

    def _replay(self) -> None:
        """Refaz o estado do DEHB pelos eventos 'pedir' e 'informar' (os demais são contabilidade e rastro). VERIFICAR: repete os
        pedidos e confere cada um bit a bit. ADOTAR (núcleo de outra versão, ou divergência): reinstala os pedidos gravados sem
        sortear e restaura o gerador do último; a fila de promoção, que é estado derivado, recomeça vazia e é refeita dos vetores
        ainda não pedidos em cima (o conjunto dos já pedidos sai dos próprios pedidos reinstalados): nenhuma promoção se repete.
        Toda adoção fica registrada no jornal ('divergencia')."""
        eventos = self.jornal.eventos
        modo = 'verificar' if self.cab.get('versao_nucleo') == NUC.VERSAO_NUCLEO and not any(e['ev'] == 'divergencia' for e in eventos) else 'adotar'
        for tentativa in (modo, 'adotar'):
            self.dehb, divergiu = self._novo_dehb(), None
            for e in eventos:
                if e['ev'] == 'pedir':
                    if tentativa == 'verificar':
                        t = self.dehb.pedir()
                        if t.registro() != e['trabalho']:
                            divergiu = e['i']
                            break
                    else:
                        self.dehb.adotar_pedido(e['trabalho'])
                elif e['ev'] == 'informar':
                    self.dehb.informar(e['id'], e.get('aptidao'))
            if divergiu is None:
                break
            self.log(f'replay divergiu no evento {divergiu}: o estado será ADOTADO dos eventos gravados (o núcleo mudou?)')
            self.jornal.acrescentar({'ev': 'divergencia', 'no_evento': divergiu, 'versao_nucleo': NUC.VERSAO_NUCLEO, 'data': agora_iso()})
            eventos = self.jornal.eventos
        if tentativa == 'adotar':
            pedidos = [e for e in eventos if e['ev'] == 'pedir']
            if pedidos:
                self.dehb.restaurar_gerador(pedidos[-1]['gerador'])
        self.tentativas = {}
        for e in eventos:
            if e['ev'] == 'tentativa_falhou':
                self.tentativas[e['id']] = max(self.tentativas.get(e['id'], 0), int(e['tentativa']))

    def _registro(self, id_trabalho: int) -> dict:
        return next(e['trabalho'] for e in self.jornal.eventos if e['ev'] == 'pedir' and e['trabalho']['id'] == id_trabalho)

    # ── contabilidade ──
    def fre_consumidos(self) -> float:
        return float(sum(e.get('fre', 0.0) for e in self.jornal.eventos if e['ev'] in ('informar', 'tentativa_falhou')))

    def falhas_finais_seguidas(self) -> int:
        n = 0
        for e in reversed([e for e in self.jornal.eventos if e['ev'] == 'informar']):
            if not e.get('falhou'):
                break
            n += 1
        return n

    def vagas(self) -> int:
        return self.args.paralelo - len(self.em_voo) - len(self.adotados)

    def progresso(self, tentativas: int = 4) -> dict:
        inf = [e for e in self.jornal.eventos if e['ev'] == 'informar']
        cheios = [e for e in inf if e.get('fidelidade_maxima') and not e.get('falhou')]
        prog = {'study': self.args.study, 'sampler': 'dehb', 'fre_alvo': self.args.n_fre, 'fre_consumidos': round(self.fre_consumidos(), 3),
                'concluidos': len(cheios), 'podados': len([e for e in inf if not e.get('fidelidade_maxima') and not e.get('falhou')]),
                'falhas': len([e for e in inf if e.get('falhou')]), 'tentativas_falhadas': len([e for e in self.jornal.eventos if e['ev'] == 'tentativa_falhou']),
                'fre_perdido_em_quedas_nao_cobrado': round(sum(e.get('fre_nao_cobrado', 0.0) for e in self.jornal.eventos if e['ev'] == 'relancado_apos_queda'), 3),
                'em_voo': sorted(list(self.em_voo) + list(self.adotados)), 'na_fila': [r['id'] for r in self._fila],
                'brackets': len(self.dehb.brackets), 'fidelidades': self.fidelidades,
                'pais_por_origem': {k: int(sum((e['trabalho'].get('pais') or {}).get(k, 0) for e in self.jornal.eventos if e['ev'] == 'pedir'))
                                    for k in ('pool', 'global', 'uniforme')},
                'promocoes': len([e for e in self.jornal.eventos if e['ev'] == 'pedir' and e['trabalho']['origem'] == 'promocao']),
                'atualizado': time.strftime('%Y-%m-%d %H:%M:%S')}
        if cheios:
            melhor = max(cheios, key=lambda e: e['objetivo'])
            prog['melhor'] = {'id_trabalho': melhor['id'], 'valor': melhor['objetivo'], 'params': melhor['hp']}
            prog['melhor_ate_agora'] = [max(e['objetivo'] for e in cheios[:k + 1]) for k in range(len(cheios))]
        try:
            escrever_json_atomico(os.path.join(self.dir, 'PROGRESSO.json'), prog, tentativas=tentativas)
        except Exception as e:
            self.log(f'PROGRESSO.json não gravado ({type(e).__name__}: {e})')
        return prog

    # ── trabalhos ──
    def _dir_trabalho(self, id_trabalho: int) -> str:
        return os.path.join(self.dir, f'trabalho_{id_trabalho}')

    def _lancar(self, registro: dict) -> None:
        """Dispara o avaliador de um pedido JÁ gravado no jornal. Pode levantar OSError (arquivo preso): quem chama repete."""
        d = self._dir_trabalho(registro['id'])
        os.makedirs(d, exist_ok=True)
        for velho in ('resultado.json', 'falha.json', '_BATIMENTO.json'):
            if os.path.exists(os.path.join(d, velho)):
                os.replace(os.path.join(d, velho), os.path.join(d, f"{velho}.anterior.{time.strftime('%Y%m%d_%H%M%S')}"))
        nivel = int(registro['nivel'])
        trabalho = dict(registro, study=self.args.study, hp=self.espaco.para_hp(registro['vetor']), inicializacao=self.args.inicializacao,
                        passos_max=self.N, fidelidade=int(self.fidelidades[nivel]), fidelidade_maxima=(nivel == len(self.fidelidades) - 1),
                        blocos=blocos_da_fidelidade(self.fidelidades[nivel], self.N), seeds=list(self.args.seeds_trial),
                        warmstart=os.path.abspath(self.args.warmstart), referencias=os.path.abspath(self.args.referencias),
                        smoke=bool(self.args.smoke), dispositivo=self.args.dispositivo,
                        impressao=impressao_digital(self.cab['id_execucao'], registro, self.args.seeds_trial, self.N, self.hash_git))
        escrever_json_atomico(os.path.join(d, 'trabalho.json'), trabalho)
        saida = open(os.path.join(d, 'avaliacao.log'), 'ab')
        try:
            proc = subprocess.Popen(self.comando_avaliador(d), cwd=RAIZ, stdin=subprocess.DEVNULL, stdout=saida, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONPATH=RAIZ),
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception:
            saida.close()
            raise
        self.prender(proc)
        self.em_voo[registro['id']] = {'proc': proc, 'registro': registro, 'saida': saida}
        self._silencio[registro['id']] = {'mtime': None, 'acumulado': 0.0}
        self.log(f"trabalho {registro['id']}: {registro['origem']} | fidelidade {trabalho['fidelidade']} | bracket {registro['bracket']} degrau {registro['degrau']} | pid {proc.pid}")

    def _lancar_da_fila(self) -> None:
        while self._fila and self.vagas() > 0:
            self._lancar(self._fila[0])                       # se levantar OSError, o registro continua na fila
            self._fila.pop(0)

    def _no_topo(self, registro: dict) -> bool:
        return int(registro['nivel']) == len(self.fidelidades) - 1

    def _trabalho_gravado(self, registro: dict) -> dict:
        nivel = int(registro['nivel'])
        return {'hp': self.espaco.para_hp(registro['vetor']), 'fidelidade': int(self.fidelidades[nivel]), 'nivel': nivel,
                'fidelidade_maxima': nivel == len(self.fidelidades) - 1, 'bracket': registro['bracket'], 'degrau': registro['degrau'],
                'origem': registro['origem']}

    def _ler_valido(self, id_trabalho: int, nome: str, registro: dict):
        """O JSON do avaliador, se existir, estiver íntegro e tiver a impressão digital DESTE pedido. Arquivo momentaneamente
        PRESO (antivírus, OneDrive) não é arquivo estragado: insiste-se na leitura e, se continuar preso, levanta OSError para
        quem chama tentar de novo depois — renomear um resultado válido custaria horas de GPU. Só o que foi LIDO e não serve
        (JSON cortado, impressão de outro pedido) é renomeado."""
        caminho = os.path.join(self._dir_trabalho(id_trabalho), nome)
        if not os.path.exists(caminho):
            return None
        esperada = impressao_digital(self.cab['id_execucao'], registro, self.args.seeds_trial, self.N, self.hash_git)
        for k in range(6):
            try:
                with open(caminho, mode='rb') as f:          # em bytes: lixo que não é UTF-8 tem de cair no ramo 'ilegível', não subir
                    bruto = f.read()
                break
            except OSError:
                if k == 5:
                    raise
                time.sleep(0.5)
        try:
            d = json.loads(bruto.decode('utf-8'))
            if isinstance(d, dict) and d.get('impressao') == esperada and d.get('id') == id_trabalho:
                return d
            motivo = 'impressão digital de outro pedido, estudo ou versão do código'
        except ValueError as e:                               # inclui UnicodeDecodeError
            motivo = f'JSON ilegível ({e})'
        try:
            os.replace(caminho, f"{caminho}.rejeitado.{time.strftime('%Y%m%d_%H%M%S')}")
        except OSError:
            pass
        self.log(f'trabalho {id_trabalho}: {nome} recusado — {motivo}')
        return None

    def _informar(self, registro: dict, resultado=None, falha=None, motivo=None) -> None:
        """Evento terminal de um trabalho: jornal (fsync) → núcleo → espelho → progresso, nesta ordem."""
        base = dict(self._trabalho_gravado(registro), ev='informar', id=registro['id'], data=agora_iso(), hash_git=self.hash_git)
        if resultado is not None:
            ev = dict(base, falhou=False, aptidao=float(resultado['aptidao']), objetivo=resultado.get('objetivo'), indicador=resultado.get('indicador'),
                      passos=int(resultado['passos']), fre=float(resultado['fre']), tempo_s=resultado.get('tempo_s'), inicio=resultado.get('inicio'),
                      fim=resultado.get('fim'), hash_git=resultado.get('hash_git', self.hash_git),
                      curva=[p for p in (resultado.get('curva') or []) if isinstance(p, dict) and numero_finito(p.get('proxy_norm'))])
        else:
            falha = falha or {}
            passos = int(falha.get('passos', 0))
            ev = dict(base, falhou=True, aptidao=None, motivo=motivo or falha.get('motivo'), detalhe=(falha.get('detalhe') or '')[-600:], passos=passos,
                      fre=passos / (self.N * max(1, len(self.args.seeds_trial))), tempo_s=falha.get('tempo_s'), inicio=falha.get('inicio'), fim=falha.get('fim'))
        ev = self.jornal.acrescentar(ev)
        t = self.dehb.informar(ev['id'], ev['aptidao'])
        self.log(f"trabalho {ev['id']}: " + (f"FALHOU ({ev['motivo']})" if ev['falhou'] else
                 (f"objetivo {ev['objetivo']:.3f}" if ev['fidelidade_maxima'] else f"indicador {ev['indicador']:.3f}") + (' | substituiu o pai' if t.substituiu else ''))
                 + f" | {ev['fre']:.2f} FRE | total {self.fre_consumidos():.2f}/{self.args.n_fre}")
        self.espelho.reconciliar(self.jornal.eventos)
        self.progresso()
        try:
            P.escrever_status(self.dir, f"trabalho_{ev['id']}", f"falhou={ev['falhou']} fre={self.fre_consumidos():.3f}")
        except OSError:
            pass

    def _tentativa_falhou(self, registro: dict, motivo: str, falha=None, rc=None) -> None:
        """Falha TRANSITÓRIA: o parcial gasto entra no orçamento (como o FAIL do hpo_rl) e o trabalho é repetido, até o limite."""
        id_t = registro['id']
        passos = int((falha or {}).get('passos', 0)) or self._passos_do_batimento(id_t)
        self.jornal.acrescentar(dict(self._trabalho_gravado(registro), ev='tentativa_falhou', id=id_t, tentativa=self.tentativas.get(id_t, 0) + 1,
                                     motivo=motivo, codigo_saida=rc, passos=passos, fre=passos / (self.N * max(1, len(self.args.seeds_trial))),
                                     detalhe=((falha or {}).get('detalhe') or '')[-600:], data=agora_iso(), hash_git=self.hash_git))
        self.tentativas[id_t] = self.tentativas.get(id_t, 0) + 1
        self.log(f'trabalho {id_t}: tentativa {self.tentativas[id_t]} falhou ({motivo}, saída {rc}); {passos} passos cobrados')
        self._guardar_batimento(id_t)                         # já cobrado: uma queda agora não pode contá-lo de novo como 'perdido'
        self.espelho.reconciliar(self.jornal.eventos)
        if self.tentativas[id_t] < self.args.max_tentativas:
            self._fila.insert(0, registro)
        else:
            self._informar_esgotado(registro)

    def _informar_esgotado(self, registro: dict) -> None:
        """'informar' da transitória esgotada. A marca em `_esgotados` cobre o erro de E/S entre os dois eventos: o trabalho continua
        em voo, e o ciclo seguinte completa o 'informar' em vez de gravar (e cobrar) a mesma tentativa outra vez."""
        self._esgotados.add(registro['id'])
        self._informar(registro, motivo='transitoria_esgotada')
        self._esgotados.discard(registro['id'])

    def _guardar_batimento(self, id_trabalho: int) -> None:
        arq = os.path.join(self._dir_trabalho(id_trabalho), '_BATIMENTO.json')
        try:
            if os.path.exists(arq):
                os.replace(arq, f"{arq}.anterior.{time.strftime('%Y%m%d_%H%M%S')}")
        except OSError:
            pass                                              # é arrumação: o _lancar guarda de novo antes de relançar

    def _passos_do_batimento(self, id_trabalho: int) -> int:
        try:
            with open(os.path.join(self._dir_trabalho(id_trabalho), '_BATIMENTO.json'), encoding='utf-8') as f:
                return int(json.load(f).get('passos', 0))
        except Exception:
            return 0

    def _sem_batimento(self, id_trabalho: int, dt: float) -> bool:
        """True se o avaliador está calado há mais que o limite, medido em tempo ATIVO do coordenador. Relógio de parede não
        serve: depois de a máquina dormir 8 h, todos os filhos pareceriam mortos e seriam encerrados ao acordar."""
        arq = os.path.join(self._dir_trabalho(id_trabalho), '_BATIMENTO.json')
        try:
            mtime = os.path.getmtime(arq)
        except OSError:
            mtime = None
        s = self._silencio.setdefault(id_trabalho, {'mtime': None, 'acumulado': 0.0})
        if mtime != s['mtime']:
            s.update(mtime=mtime, acumulado=0.0)
        else:
            s['acumulado'] += dt
        return s['acumulado'] > self.args.limite_sem_batimento_h * 3600

    def _colher(self) -> None:
        agora = time.monotonic()
        dt = 0.0 if self._tique is None else min(agora - self._tique, max(60.0, 3 * self.args.intervalo_s))   # teto: suspensão não conta
        self._tique = agora
        for id_t, v in list(self.em_voo.items()):
            rc = v['proc'].poll()
            if rc is None:
                if not self._sem_batimento(id_t, dt):
                    continue
                self.log(f'trabalho {id_t}: sem batimento há mais de {self.args.limite_sem_batimento_h} h de atividade; encerrando o processo')
                v['proc'].kill()
                try:
                    v['proc'].wait(timeout=60)
                except subprocess.TimeoutExpired:
                    continue
                rc = -9
            self._tratar_termino(v['registro'], rc)           # se levantar OSError, o trabalho continua em em_voo e volta no próximo ciclo
            try:
                v['saida'].close()
            except OSError:
                pass                                          # o término JÁ está no jornal: um erro aqui não pode deixar o trabalho em voo
            del self.em_voo[id_t]
            self._silencio.pop(id_t, None)
        for id_t, trava in list(self.adotados.items()):
            if trava.tentar():                                # o órfão terminou (ou morreu): a trava soltou
                trava.soltar()
                self._tratar_termino(self._registro(id_t), None)
                del self.adotados[id_t]

    def _tratar_termino(self, registro: dict, rc) -> None:
        id_t = registro['id']
        if id_t in self._esgotados:
            return self._informar_esgotado(registro)
        resultado = self._ler_valido(id_t, 'resultado.json', registro)
        if resultado_aproveitavel(resultado, self._no_topo(registro)):
            return self._informar(registro, resultado=resultado)
        falha = self._ler_valido(id_t, 'falha.json', registro)
        if rc == SAIDA_FALHA_DA_CONFIGURACAO and falha is not None:
            return self._informar(registro, falha=falha)
        self._tentativa_falhou(registro, motivo=(falha or {}).get('motivo') or ('sem_resultado' if rc == 0 else 'processo_morreu'), falha=falha, rc=rc)

    def _retomar_pendentes(self) -> None:
        """Pedidos do jornal sem evento terminal: resultado válido em disco → informar; avaliador órfão vivo → adotar (nunca
        duplicar); senão → fila de lançamento. O parcial treinado antes da queda do COORDENADOR não entra no orçamento — no
        hpo_rl o trial do processo morto fica RUNNING para sempre e também não conta —, mas fica registrado."""
        for id_t in sorted(self.dehb.pendentes):
            registro = self._registro(id_t)
            trava = Trava(os.path.join(self._dir_trabalho(id_t), '_EXECUTANDO.lock'))
            if not trava.tentar():
                self.log(f'trabalho {id_t}: um avaliador de outra sessão ainda está vivo; será adotado, não duplicado')
                self.adotados[id_t] = trava
                continue
            trava.soltar()
            resultado = self._ler_valido(id_t, 'resultado.json', registro)
            if resultado_aproveitavel(resultado, self._no_topo(registro)):
                self.log(f'trabalho {id_t}: resultado íntegro já em disco; informado sem retreinar')
                self._informar(registro, resultado=resultado)
                continue
            if self.tentativas.get(id_t, 0) >= self.args.max_tentativas:
                self.log(f'trabalho {id_t}: a última tentativa já estava no jornal quando a sessão caiu; falta só o informar')
                self._informar_esgotado(registro)             # a queda foi ENTRE os dois eventos: não ganha tentativa extra
                continue
            perdidos = self._passos_do_batimento(id_t)
            self.jornal.acrescentar({'ev': 'relancado_apos_queda', 'id': id_t, 'passos_perdidos': perdidos,
                                     'fre_nao_cobrado': perdidos / (self.N * max(1, len(self.args.seeds_trial))), 'data': agora_iso()})
            self._guardar_batimento(id_t)                     # já contado como perdido: outra queda ANTES do relançamento não pode contá-lo de novo
            self.log(f'trabalho {id_t}: estava em voo na queda ({perdidos} passos perdidos, não cobrados); volta para a fila')
            self._fila.append(registro)

    # ── laço principal ──
    def rodar(self) -> int:
        trava = Trava(os.path.join(self.dir, '_COORDENADOR.lock'))
        os.makedirs(self.dir, exist_ok=True)
        ultimo = 0.0
        while not trava.tentar():
            if os.path.exists(os.path.join(self.dir, '_DONE.flag')):
                self.log('outro coordenador concluiu o estudo; nada a fazer')
                return 0
            if time.time() - ultimo > 1800:
                self.log('aguardando: outro coordenador deste estudo está vivo')
                ultimo = time.time()
            time.sleep(self.args.intervalo_s)
        manter_a_maquina_acordada(True)
        try:
            return self._rodar_com_trava()
        finally:
            manter_a_maquina_acordada(False)
            if getattr(self, 'espelho', None) is not None:
                self.espelho.fechar()
            trava.soltar()

    def _ciclo(self) -> bool:
        """Um ciclo do laço: colhe, relança, pede. Devolve True quando o estudo terminou."""
        self._gravar_pedido()                                 # o que ficou por gravar entra ANTES de qualquer 'informar' deste ciclo
        self._colher()
        self._lancar_da_fila()
        while self.vagas() - len(self._fila) > 0 and self.fre_consumidos() < self.args.n_fre:
            t = self.dehb.pedir()                             # write-ahead: memória → jornal (fsync) → fila → processo
            self._pedido_por_gravar = {'ev': 'pedir', 'trabalho': t.registro(), 'gerador': self.dehb.estado_do_gerador()}
            self._gravar_pedido()
            self._lancar_da_fila()
        return not self.em_voo and not self.adotados and not self._fila and self.fre_consumidos() >= self.args.n_fre

    def _gravar_pedido(self) -> None:
        """Grava no jornal o pedido que o núcleo já sorteou. Se o jornal levantar OSError, o pedido FICA guardado e o ciclo seguinte
        o regrava antes de tudo: pedir outro deixaria no núcleo um pendente que o jornal não tem (o degrau do bracket nunca
        fecharia, e a retomada seguinte recusaria o jornal por 'pedido fora de ordem'); e gravá-lo depois de um 'informar' que o
        núcleo só viu DEPOIS dele faria o replay divergir."""
        if self._pedido_por_gravar is None:
            return
        ev = self.jornal.acrescentar(dict(self._pedido_por_gravar, data=agora_iso()))
        self._pedido_por_gravar = None
        self._fila.append(ev['trabalho'])

    def _rodar_com_trava(self) -> int:
        if os.path.exists(os.path.join(self.dir, '_FAILED.flag')):
            os.remove(os.path.join(self.dir, '_FAILED.flag'))   # flag de uma sessão anterior: o validador a reportaria para sempre
        if not self.abrir():
            self.log(f'estudo já concluído ({self.fre_consumidos():.2f} FRE de {self.args.n_fre}); nada a fazer')
            return 0
        self.espelho = Espelho(self.args.study, self.espaco_dict, self.log)
        self.espelho.atributos_do_estudo(atributos_do_estudo(self.cab, self.args.n_fre, self.args.paralelo),
                                         {'data': agora_iso(), 'hash_git': self.hash_git, 'pid': os.getpid(), 'n_eventos': len(self.jornal.eventos)})
        self.espelho.reconciliar(self.jornal.eventos)
        P.escrever_status(self.dir, 'inicio', f'dehb n_fre={self.args.n_fre} N={self.N} fidelidades={self.fidelidades} K={self.args.paralelo}')
        self.log(f"study {self.args.study} | DEHB {self.args.niveis} níveis {self.fidelidades} | alvo {self.args.n_fre} FRE | já consumidos "
                 f"{self.fre_consumidos():.2f} em {len(self.dehb.historico)} avaliações | {len(self.dehb.pendentes)} em voo na última sessão | K={self.args.paralelo}")
        if self.falhas_finais_seguidas() >= self.args.max_falhas_seguidas and not self.args.ignorar_disjuntor:
            return self._abortar(f'as últimas {self.falhas_finais_seguidas()} avaliações falharam: o defeito parece sistêmico. Veja trabalho_<id>/falha.json; '
                                 f'corrigido o problema, relance com --ignorar-disjuntor')
        self._retomar_pendentes()
        erros_es = 0
        while True:
            try:
                if self._ciclo():
                    break
                erros_es = 0
            except OSError as e:
                # Erro transitório de E/S (WinError 5/32: antivírus, OneDrive) NÃO pode derrubar a sessão: o coordenador morto leva
                # junto os treinos em voo (Job Object). O jornal é idempotente na repetição; insistir custa segundos, cair custa horas.
                erros_es += 1
                self.log(f'erro de E/S ({type(e).__name__}: {e}); nova tentativa no próximo ciclo ({erros_es}/40)')
                if erros_es >= 40:
                    return self._abortar(f'erro de E/S persistente: {e}')
                time.sleep(min(30.0, 1.0 + erros_es))
            if self.falhas_finais_seguidas() >= self.args.max_falhas_seguidas and not self.args.ignorar_disjuntor:
                return self._abortar(f'{self.falhas_finais_seguidas()} avaliações seguidas falharam: defeito sistêmico, não das configurações')
            if time.time() - self._ultimo_log > 1800:        # o validador trata log parado como aviso: dizer que está vivo
                orfaos = {i: round(self._silencio.get(i, {}).get('acumulado', 0.0) / 60) for i in self.adotados}
                self.log(f"vivo: {self.fre_consumidos():.2f}/{self.args.n_fre} FRE | em voo {sorted(self.em_voo)} | fila {[r['id'] for r in self._fila]}"
                         + (f" | ÓRFÃOS adotados {sorted(self.adotados)}: se travarem, só encerrando o processo à mão (pid em trabalho_<id>/_BATIMENTO.json) {orfaos}" if self.adotados else ''))
                self.progresso()
                self._ultimo_log = time.time()
            time.sleep(self.args.intervalo_s)
        completo = self.espelho.reconciliar(self.jornal.eventos)
        prog = self.progresso(tentativas=12)                  # o último fica para sempre ao lado da _DONE.flag: insistir mais
        escrever_json_atomico(os.path.join(self.dir, 'INDICE.json'), self.espelho.indice())
        if not completo:
            return self._abortar('estudo concluído, mas o espelho no Optuna não pôde ser reconciliado; rode --reconstruir-espelho')
        self.log(f"fim: {prog['concluidos']} avaliações em fidelidade máxima, {prog['podados']} em fidelidade menor, {prog['falhas']} falhas, "
                 f"{prog['fre_consumidos']:.2f} FRE | melhor: {prog.get('melhor')}")
        P.escrever_flag(self.dir, '_DONE.flag', f"fre={prog['fre_consumidos']}")
        return 0

    def _abortar(self, mensagem: str) -> int:
        self.log('ABORTADO: ' + mensagem)
        for v in self.em_voo.values():
            try:
                v['proc'].kill()
            except Exception:
                pass
        P.escrever_flag(self.dir, '_FAILED.flag', mensagem[:300])
        return 1


def numero_finito(x) -> bool:
    """True só para número real finito. O que vem de um JSON em disco pode ser null, texto ou lista: np.isfinite levantaria
    TypeError, que não é OSError — derrubaria a sessão, e de novo a cada retomada, porque o arquivo continua lá."""
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return False
    try:
        return bool(np.isfinite(float(x)))
    except OverflowError:                                     # inteiro que não cabe em float
        return False


def resultado_aproveitavel(resultado, fidelidade_maxima: bool) -> bool:
    """O resultado.json (já com a impressão digital conferida) tem os números de que `_informar` e o espelho precisam? Na
    fidelidade máxima, o objetivo; nas menores, o indicador. Se não tem, é tratado como avaliação sem resultado (tentativa
    falhada), nunca como exceção."""
    return (isinstance(resultado, dict) and all(numero_finito(resultado.get(k)) for k in ('aptidao', 'passos', 'fre'))
            and numero_finito(resultado.get('objetivo' if fidelidade_maxima else 'indicador')))


def blocos_da_fidelidade(fidelidade: int, passos_max: int) -> int:
    """Nº de blocos de validação de uma avaliação: o bloco tem SEMPRE N // 9 passos, como no trial completo dos outros estudos
    (N // 3 com 3 blocos tem as mesmas fronteiras dos 3 primeiros blocos de N com 9)."""
    return max(1, round(int(fidelidade) / max(1, int(passos_max) // BLOCOS_POR_EXECUCAO)))


def conferir_escada(fidelidades, passos_max: int) -> None:
    """Toda fidelidade tem de ser um número inteiro de blocos de N // 9 passos; senão o indicador deixa de ser o número que o
    podador dos outros estudos vê naquele passo. Vale para eta = 3 com 2 ou 3 níveis; outro eta, ou 4 níveis, é recusado."""
    bloco = max(1, int(passos_max) // BLOCOS_POR_EXECUCAO)
    for f in fidelidades:
        k = round(f / bloco)
        if k < 1 or abs(k * bloco - f) > 0.02 * bloco:
            raise SystemExit(f'fidelidade {f} não é múltiplo do bloco de validação (N // {BLOCOS_POR_EXECUCAO} = {bloco} passos): '
                             f'use --eta 3 com --niveis 2 ou 3')


def reconstruir_espelho(args, log) -> int:
    dir_study = os.path.join(DIR_HPO, args.study)
    if not os.path.exists(os.path.join(dir_study, 'jornal.jsonl')):
        raise SystemExit(f'sem jornal em {dir_study}')
    trava = Trava(os.path.join(dir_study, '_COORDENADOR.lock'))
    if not trava.tentar():
        raise SystemExit('há um coordenador vivo neste estudo: o espelho só pode ser refeito com ele parado')
    try:
        jornal = Jornal(os.path.join(dir_study, 'jornal.jsonl'))
        banco = os.path.join(DIR_HPO, args.study + '.db')
        if os.path.exists(banco):
            os.replace(banco, f"{banco}.anterior.{time.strftime('%Y%m%d_%H%M%S')}")
        cab = jornal.eventos[0]
        sessoes = [e for e in jornal.eventos if e['ev'] == 'sessao']
        esp = Espelho(args.study, {nome: tuple([spec[0]] + spec[1:]) for nome, spec in cab['espaco']}, log)
        esp.atributos_do_estudo(atributos_do_estudo(cab, sessoes[-1]['n_fre'] if sessoes else None, sessoes[-1]['paralelo'] if sessoes else None),
                                {'data': agora_iso(), 'hash_git': P.hash_git(), 'pid': os.getpid(), 'reconstruido_de': len(jornal.eventos)})
        ok = esp.reconciliar(jornal.eventos)
        escrever_json_atomico(os.path.join(dir_study, 'INDICE.json'), esp.indice())
        esp.fechar()
        log(f"espelho reconstruído de {len(jornal.eventos)} eventos: {'ok' if ok else 'FALHOU'}")
        return 0 if ok else 1
    finally:
        trava.soltar()


def main(argv=None, comando_avaliador=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default=P.STUDY_DEHB)
    ap.add_argument('--n-fre', type=float, default=64.0, help='orçamento total em execuções completas equivalentes (passos reais / N)')
    ap.add_argument('--orcamento', type=int, default=P.ORCAMENTO_RL_PASSOS, help='passos de ambiente da fidelidade máxima (N)')
    ap.add_argument('--niveis', type=int, default=2, help='fidelidades da escada: 2 = N//3 e N (a do podador dos outros estudos); 3 acrescenta N//9')
    ap.add_argument('--eta', type=int, default=NUC.ETA_PADRAO)
    ap.add_argument('--paralelo', type=int, default=3, help='avaliações simultâneas (subprocessos) na GPU')
    ap.add_argument('--seeds-trial', type=int, nargs='+', default=[42],
                    help='sementes de treino por avaliação (média). O protocolo é UMA (42), como nos outros estudos; com várias o hpo_rl '
                         'desliga a poda e o DEHB segue multi-fidelidade — deixa de ser a mesma comparação')
    ap.add_argument('--seed', type=int, default=None, help='semente do DEHB (padrão: crc32 do nome do study; constante entre retomadas)')
    ap.add_argument('--warmstart', default=P.DIR_WARMSTART)
    ap.add_argument('--referencias', default=P.REFERENCIAS_PADRAO)
    ap.add_argument('--inicializacao', choices=['professor', 'zero'], default=INICIALIZACAO_DO_METODO)
    ap.add_argument('--max-tentativas', type=int, default=2, help='tentativas por trabalho diante de falha TRANSITÓRIA (GPU sem memória, processo morto)')
    ap.add_argument('--max-falhas-seguidas', type=int, default=4, help='falhas finais seguidas que param o estudo (defeito sistêmico)')
    ap.add_argument('--ignorar-disjuntor', action='store_true')
    ap.add_argument('--aceitar-fidelidade-curta', action='store_true',
                    help='roda mesmo com a fidelidade mínima menor que o aquecimento do crítico mais uma atualização (ver a mensagem de erro)')
    ap.add_argument('--limite-sem-batimento-h', type=float, default=3.0, help='horas de ATIVIDADE do coordenador sem batimento do avaliador antes de encerrá-lo')
    ap.add_argument('--intervalo-s', type=float, default=5.0)
    ap.add_argument('--dispositivo', default=None)
    ap.add_argument('--retomar', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--reconstruir-espelho', action='store_true')
    ap.add_argument('--avaliar', default=None, metavar='DIR', help='(interno) roda a avaliação descrita em DIR/trabalho.json')
    args = ap.parse_args(argv)

    if args.avaliar:
        return avaliar(args.avaliar)

    def log(msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    if args.reconstruir_espelho:
        return reconstruir_espelho(args, log)
    if args.smoke:
        args.orcamento, args.n_fre, args.paralelo = 4 * P.PASSOS_MAXIMOS, 2.0, min(args.paralelo, 2)
    if args.paralelo < 1 or args.max_tentativas < 1:
        raise SystemExit('--paralelo e --max-tentativas devem ser >= 1')
    # precondições, conferidas com json puro (o coordenador não carrega torch): sem elas TODA avaliação falharia igual
    for seed in args.seeds_trial:
        ckpt = os.path.join(args.warmstart, f'seed_{seed}', 'modelo_final.pt')
        if args.inicializacao == 'professor' and not os.path.exists(ckpt):
            raise SystemExit(f'warm-start ausente: {ckpt}')
    try:
        with open(args.referencias, encoding='utf-8') as f:
            refs = json.load(f)
        assert 'validacao_rollback' in refs['oraculo_perfeito'] and 'validacao_objetivo' in refs['oraculo_perfeito']
    except Exception as e:
        raise SystemExit(f'referências inválidas ou ausentes ({args.referencias}): {type(e).__name__}: {e}')
    coord = Coordenador(args, comando_avaliador=comando_avaliador, log=log)
    conferir_escada(coord.fidelidades, coord.N)
    # Abaixo de (k_aquec máximo + 1) coletas o ator de uma configuração com k_aquec alto NÃO recebe nenhuma atualização na
    # fidelidade mínima: a aptidão ali mediria k_aquec, e o pool de pais seria puxado para k_aquec baixo. Com N = 250.000 a
    # escada N//3 cai nesse caso (o podador da rodada 2 também caía); com N >= 500.000, não.
    k_aquec_max = int(coord.espaco_dict['k_aquec'][2]) if 'k_aquec' in coord.espaco_dict else 0
    minimo = (k_aquec_max + 1) * EPISODIOS_POR_COLETA * P.PASSOS_MAXIMOS
    if coord.fidelidades[0] < minimo and not args.smoke and not args.aceitar_fidelidade_curta:
        raise SystemExit(f'a fidelidade mínima ({coord.fidelidades[0]} passos) não cobre o aquecimento do crítico mais uma atualização do ator '
                         f'({minimo} passos): ali a aptidão mediria k_aquec. Use menos níveis, um orçamento maior, ou declare a '
                         f'escolha com --aceitar-fidelidade-curta.')
    try:
        return coord.rodar()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        P.escrever_flag(os.path.join(DIR_HPO, args.study), '_FAILED.flag', traceback.format_exc().splitlines()[-1][:300])
        return 1


if __name__ == '__main__':
    sys.exit(main())
