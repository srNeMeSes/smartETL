"""Identidade da janela na barra de tarefas do Windows (app/identidade_windows.py).

A gravação real (pywin32, SHGetPropertyStoreForWindow) foi conferida à mão no app: com ela, o
botão direito no ícone mostra "smartETL" com o ícone laranja em vez de "Flet description".
Aqui ficam as regras (comando de reabrir, dados gravados) e os caminhos de falha.
"""

import sys
from pathlib import Path

from app import identidade_windows as iw
from app.identidade_windows import aplicar as aplicar_original

ICONE = Path("C:/app/assets/icon2.ico")


def test_comando_reabrir_no_executavel():
    assert iw.comando_reabrir(True, r"C:\Programas\smartETL\smartETL.exe", Path("x")) == (
        r'"C:\Programas\smartETL\smartETL.exe"'
    )


def test_comando_reabrir_do_codigo_usa_pythonw(tmp_path):
    (tmp_path / "python.exe").write_text("")
    script = tmp_path / "main.py"
    sem_pythonw = iw.comando_reabrir(False, str(tmp_path / "python.exe"), script)
    assert sem_pythonw == f'"{tmp_path / "python.exe"}" "{script}"'
    (tmp_path / "pythonw.exe").write_text("")  # sem console, como um app
    com_pythonw = iw.comando_reabrir(False, str(tmp_path / "python.exe"), script)
    assert com_pythonw == f'"{tmp_path / "pythonw.exe"}" "{script}"'


def test_identidade():
    dados = iw.identidade(True, r"C:\smartETL.exe", Path("main.py"), ICONE)
    assert (dados.id_aplicativo, dados.nome) == ("smartETL.TestesDeHipotese", "smartETL")
    assert dados.icone == f"{ICONE},0" and dados.comando == r'"C:\smartETL.exe"'


def test_aplicar_fora_do_windows_nao_faz_nada(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert aplicar_original("t", iw.identidade(True, "x", Path("m"), ICONE)) is False


def test_aplicar_grava_na_janela_encontrada(monkeypatch):
    if sys.platform != "win32":
        return
    gravados = []
    monkeypatch.setattr(iw, "_procurar_janela", lambda titulo, espera: 4242)
    monkeypatch.setattr(iw, "_gravar", lambda hwnd, dados: gravados.append((hwnd, dados)))
    dados = iw.identidade(True, "x", Path("m"), ICONE)
    assert aplicar_original("smartETL — Processamento de dados", dados) is True
    assert gravados == [(4242, dados)]


def test_aplicar_sem_janela_ou_com_erro_nao_quebra(monkeypatch):
    if sys.platform != "win32":
        return
    dados = iw.identidade(True, "x", Path("m"), ICONE)
    monkeypatch.setattr(iw, "_procurar_janela", lambda titulo, espera: 0)
    assert aplicar_original("t", dados) is False

    def falha(hwnd, dados):
        raise OSError("acesso negado")

    monkeypatch.setattr(iw, "_procurar_janela", lambda titulo, espera: 1)
    monkeypatch.setattr(iw, "_gravar", falha)
    assert aplicar_original("t", dados) is False  # só registra no log
