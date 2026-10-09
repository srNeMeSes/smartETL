"""Identidade da janela na barra de tarefas do Windows (nome, ícone e o que "Fixar" abre).

A janela do app pertence ao cliente do Flet (`flet.exe`), não ao Python nem ao smartETL.exe: sem
isto, o Windows mostra a descrição do cliente ("Flet description") e, ao fixar, fixa o
`flet.exe`, que sozinho não abre o app. Gravamos na janela as propriedades AppUserModel — ID
próprio, nome, ícone e o comando de reabrir (o smartETL.exe, ou o `pythonw main.py` rodando do
código) —, o que vale também para janelas de outro processo. Em outros sistemas, ou sem o
pywin32, não faz nada. Falhas só vão para o log: nunca impedem o app de abrir.

Cuidado: o Windows guarda, para cada ID, o PRIMEIRO comando de reabrir que viu (num atalho em
"User Pinned/ImplicitAppShortcuts", que fica lá mesmo depois de desafixar) e ignora mudanças
posteriores. Em 2026-10-08 um teste manual gravou "python.exe teste" no ID antigo e o ícone
fixado passou a abrir só um console. Por isso o ID é derivado do próprio comando de reabrir
(prefixo + 10 dígitos do SHA-1 do caminho): executável e código, ou o app em outra pasta, têm IDs
diferentes, e um atalho guardado sempre corresponde ao comando atual.
"""

import hashlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

PREFIXO_EXECUTAVEL = "smartETL.Desktop"  # smartETL.exe
PREFIXO_CODIGO = "smartETL.Codigo"  # pythonw main.py (desenvolvimento)
NOME = "smartETL"
ESPERA_JANELA = 10.0  # segundos procurando a janela do cliente Flet


@dataclass(frozen=True)
class Identidade:
    id_aplicativo: str
    nome: str
    icone: str  # "caminho.ico,0"
    comando: str  # linha de comando para reabrir o app


def comando_reabrir(empacotado: bool, executavel: str, script: Path) -> str:
    """O executável empacotado, ou o pythonw (sem console) do mesmo ambiente com o main.py."""
    if empacotado:
        return f'"{executavel}"'
    pasta = Path(executavel).parent
    interpretador = pasta / "pythonw.exe"
    if not interpretador.exists():
        interpretador = Path(executavel)
    return f'"{interpretador}" "{script}"'


def id_aplicativo(empacotado: bool, comando: str) -> str:
    """ID do AppUserModel ligado ao comando de reabrir (sem espaços, bem abaixo de 128 caracteres).

    O mesmo comando dá sempre o mesmo ID (o ícone fixado continua agrupando a janela); outro
    comando — outra pasta, executável × código — dá outro ID, sem atalho antigo guardado.
    """
    assinatura = hashlib.sha1(comando.lower().encode("utf-8")).hexdigest()[:10]
    return f"{PREFIXO_EXECUTAVEL if empacotado else PREFIXO_CODIGO}.{assinatura}"


def identidade(empacotado: bool, executavel: str, script: Path, icone: Path) -> Identidade:
    comando = comando_reabrir(empacotado, executavel, script)
    return Identidade(id_aplicativo(empacotado, comando), NOME, f"{icone},0", comando)


def _procurar_janela(titulo: str, espera: float) -> int:
    import win32gui

    limite = time.monotonic() + espera
    while True:
        hwnd = win32gui.FindWindow(None, titulo)
        if hwnd or time.monotonic() >= limite:
            return hwnd
        time.sleep(0.2)


def _gravar(hwnd: int, dados: Identidade) -> None:
    import pythoncom
    from win32com.propsys import propsys, pscon

    loja = propsys.SHGetPropertyStoreForWindow(hwnd, propsys.IID_IPropertyStore)
    for chave, valor in (
        (pscon.PKEY_AppUserModel_ID, dados.id_aplicativo),
        (pscon.PKEY_AppUserModel_RelaunchCommand, dados.comando),
        (pscon.PKEY_AppUserModel_RelaunchDisplayNameResource, dados.nome),
        (pscon.PKEY_AppUserModel_RelaunchIconResource, dados.icone),
    ):
        loja.SetValue(chave, propsys.PROPVARIANTType(valor, pythoncom.VT_LPWSTR))
    loja.Commit()


def aplicar(titulo: str, dados: Identidade, espera: float = ESPERA_JANELA) -> bool:
    """Grava a identidade na janela de título `titulo`. Devolve se conseguiu."""
    if sys.platform != "win32":
        return False
    try:
        import pythoncom
    except ImportError:
        log.info("pywin32 ausente: identidade da janela na barra de tarefas não aplicada")
        return False
    pythoncom.CoInitialize()  # roda fora da thread principal
    try:
        hwnd = _procurar_janela(titulo, espera)
        if not hwnd:
            log.warning("Janela '%s' não encontrada para a barra de tarefas", titulo)
            return False
        _gravar(hwnd, dados)
        log.info(
            "Identidade da janela aplicada (%s, ID %s; reabrir: %s)",
            dados.nome,
            dados.id_aplicativo,
            dados.comando,
        )
        return True
    except Exception:
        log.exception("Falha ao aplicar a identidade da janela na barra de tarefas")
        return False
    finally:
        pythoncom.CoUninitialize()
