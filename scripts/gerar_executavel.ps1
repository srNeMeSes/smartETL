# Gera o executável do Sigra para Windows (pasta dist\Sigra, com Sigra.exe) e o zip
# para distribuir (dist\Sigra-windows.zip). Rodar da raiz do projeto, com o venv instalado:
#
#   powershell -ExecutionPolicy Bypass -File scripts\gerar_executavel.ps1
#
# Usa `flet pack` (PyInstaller) em modo pasta: abre em ~2 s, contra ~15 s do arquivo único
# (que descompacta pandas/scipy/statsmodels a cada abertura). O ícone assets\icon2.ico vai no
# executável e na janela do Flet (canto superior esquerdo e barra de tarefas); a fonte Inter
# (assets\fonts) vai junto, para a interface não depender das fontes instaladas na máquina.

$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot
Set-Location $raiz
$flet = Join-Path $raiz "venv\Scripts\flet.exe"

$excluir = "tkinter", "pytest", "IPython", "PyQt5", "PySide6" |
    ForEach-Object { "--pyinstaller-build-args=--exclude-module=$_" }

& $flet pack main.py -y -D --name Sigra --icon assets/icon2.ico `
    --add-data "assets/icon2.ico:assets" --add-data "assets/fonts:assets/fonts" `
    --product-name Sigra --file-description "Sigra - Testes de Hipotese" `
    --product-version 1.0.0 --file-version 1.0.0.0 `
    --company-name "Evaniel Arcanjo Da Silva" --copyright "Evaniel Arcanjo Da Silva" `
    @excluir
if ($LASTEXITCODE -ne 0) { throw "flet pack falhou ($LASTEXITCODE)" }

$zip = Join-Path $raiz "dist\Sigra-windows.zip"
if (Test-Path $zip) { Remove-Item $zip }
# Logo depois do build o antivírus ainda lê os arquivos novos e o zip falha com "arquivo em uso"
# (aconteceu em dois builds seguidos): espera e tenta de novo.
for ($tentativa = 1; ; $tentativa++) {
    try {
        Compress-Archive -Path (Join-Path $raiz "dist\Sigra") -DestinationPath $zip -Force -ErrorAction Stop
        break
    } catch {
        if ($tentativa -ge 6) { throw }
        Write-Output "Zip: arquivo em uso, nova tentativa em 10 s ($tentativa/5)"
        Start-Sleep -Seconds 10
    }
}
Write-Output "Executavel: dist\Sigra\Sigra.exe"
Write-Output "Para distribuir: $zip"
