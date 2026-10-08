# Gera o executável do smartETL para Windows (pasta dist\smartETL, com smartETL.exe) e o zip
# para distribuir (dist\smartETL-windows.zip). Rodar da raiz do projeto, com o venv instalado:
#
#   powershell -ExecutionPolicy Bypass -File scripts\gerar_executavel.ps1
#
# Usa `flet pack` (PyInstaller) em modo pasta: abre em ~2 s, contra ~15 s do arquivo único
# (que descompacta pandas/scipy/statsmodels a cada abertura). O ícone assets\icon2.ico vai no
# executável e na janela do Flet (canto superior esquerdo e barra de tarefas).

$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot
Set-Location $raiz
$flet = Join-Path $raiz "venv\Scripts\flet.exe"

$excluir = "tkinter", "pytest", "IPython", "PyQt5", "PySide6" |
    ForEach-Object { "--pyinstaller-build-args=--exclude-module=$_" }

& $flet pack main.py -y -D --name smartETL --icon assets/icon2.ico `
    --add-data "assets/icon2.ico:assets" `
    --product-name smartETL --file-description "smartETL - Testes de Hipotese" `
    --product-version 1.0.0 --file-version 1.0.0.0 `
    --company-name "Evaniel Arcanjo Da Silva" --copyright "Evaniel Arcanjo Da Silva" `
    @excluir
if ($LASTEXITCODE -ne 0) { throw "flet pack falhou ($LASTEXITCODE)" }

$zip = Join-Path $raiz "dist\smartETL-windows.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path (Join-Path $raiz "dist\smartETL") -DestinationPath $zip
Write-Output "Executavel: dist\smartETL\smartETL.exe"
Write-Output "Para distribuir: $zip"
