# Gera o pacote MSIX do Sigra para a Microsoft Store (dist\Sigra-<versão>-x64.msix), a partir do
# executável já gerado por scripts\gerar_executavel.ps1. Rodar da raiz do projeto:
#
#   powershell -ExecutionPolicy Bypass -File scripts\gerar_msix.ps1 -Versao 1.0.0.0
#
# A identidade padrão é a do Sigra no Partner Center (app > Gerenciamento de produtos > Identidade
# do produto): Name, Publisher e PublisherDisplayName precisam ser exatamente iguais aos de lá,
# senão a Loja recusa o pacote. A cada submissão, aumente a versão (o último número fica 0).
# A Loja assina o pacote na publicação: o .msix enviado vai sem assinatura.
#
# Teste neste PC (exige o Modo de Desenvolvedor), como se o app tivesse vindo da Loja (Menu
# Iniciar, barra de tarefas, pasta somente leitura):
#
#   powershell -ExecutionPolicy Bypass -File scripts\gerar_msix.ps1 -Teste -Registrar
#
# -Teste usa uma identidade própria (Sigra.Teste) e monta tudo em build\msix-teste, sem tocar no
# pacote da Loja; a identidade da Loja nunca é registrada aqui (um registro de desenvolvedor com
# ela conflitaria com a instalação real pela Loja). Remover o teste:
#   Get-AppxPackage Sigra.Teste | Remove-AppxPackage
#
# Dentro do pacote: dist\Sigra (executável), o cliente Flet já desempacotado em
# Sigra\_internal\flet_client (main.usar_cliente_flet_embutido), com o plugin de localização
# trocado por um vazio (msix\plugin_vazio: sem ele o Windows pede permissão de localização na
# primeira abertura, e o app não usa localização), as imagens geradas por
# scripts\gerar_imagens_msix.py e o resources.pri que indexa as escalas dessas imagens.

param(
    [string]$NomePacote = "SysArc.Sigra",
    [string]$Editor = "CN=79E73AAC-3E0F-4BEA-932A-DAA5AC3826DF",
    [string]$NomeEditor = "SysArc",
    [string]$Versao = "1.0.0.0",
    [switch]$Teste,
    [switch]$Registrar
)

$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot
Set-Location $raiz

if ($Registrar -and -not $Teste) {
    throw "Use -Registrar junto com -Teste: a identidade da Loja não é registrada neste PC."
}
if ($Teste) {
    $NomePacote = "Sigra.Teste"
    $Editor = "CN=Sigra Teste"
}

if ($Versao -notmatch '^\d+\.\d+\.\d+\.0$') {
    throw "Versão no formato X.Y.Z.0 (a Loja exige o último número 0): $Versao"
}
if (-not (Test-Path "dist\Sigra\Sigra.exe")) {
    throw "Gere o executável antes: scripts\gerar_executavel.ps1"
}
$python = Join-Path $raiz "venv\Scripts\python.exe"

# 1. makeappx e makepri: pacote NuGet oficial da Microsoft com as ferramentas do Windows SDK
#    (evita instalar o SDK inteiro). Baixado uma vez e guardado em .ferramentas\ (fora de
#    build\, que o flet pack apaga a cada executável; ignorado pelo git).
$versaoSdk = "10.0.26100.9169"
$ferramentas = Join-Path $raiz ".ferramentas"
$pastaSdk = Join-Path $ferramentas "windows-sdk-buildtools\$versaoSdk"
function Achar-Ferramenta($nome) {
    Get-ChildItem (Join-Path $pastaSdk "bin") -Recurse -Filter $nome -ErrorAction SilentlyContinue |
        Where-Object { $_.Directory.Name -eq "x64" } | Select-Object -First 1 -ExpandProperty FullName
}
if (-not (Achar-Ferramenta "makeappx.exe")) {
    $pacote = Join-Path $env:TEMP "sdk-buildtools-$versaoSdk.zip"
    $url = "https://www.nuget.org/api/v2/package/Microsoft.Windows.SDK.BuildTools/$versaoSdk"
    Write-Output "Baixando as ferramentas do Windows SDK ($versaoSdk)..."
    Invoke-WebRequest -Uri $url -OutFile $pacote -UseBasicParsing
    if (Test-Path $pastaSdk) { Remove-Item $pastaSdk -Recurse -Force }
    Expand-Archive -Path $pacote -DestinationPath $pastaSdk
    Remove-Item $pacote
}
$makeappx = Achar-Ferramenta "makeappx.exe"
$makepri = Achar-Ferramenta "makepri.exe"
foreach ($ferramenta in $makeappx, $makepri) {
    $assinatura = Get-AuthenticodeSignature $ferramenta
    if ($assinatura.Status -ne "Valid" -or $assinatura.SignerCertificate.Subject -notmatch "O=Microsoft Corporation") {
        throw "Assinatura inválida em $ferramenta ($($assinatura.Status))"
    }
}

# 2. Pasta do pacote
$saida = Join-Path $raiz $(if ($Teste) { "build\msix-teste" } else { "build\msix" })
if (Test-Path $saida) { Remove-Item $saida -Recurse -Force }
$layout = Join-Path $saida "layout"
$recursos = Join-Path $saida "recursos"
New-Item -ItemType Directory -Force $layout, $recursos | Out-Null
Copy-Item "dist\Sigra" (Join-Path $layout "Sigra") -Recurse

# Cliente Flet desempacotado (o flet-windows.zip do pacote já tem o ícone e a versão do Sigra)
$internos = Join-Path $layout "Sigra\_internal"
Expand-Archive -Path (Join-Path $internos "flet_desktop\app\flet-windows.zip") `
    -DestinationPath (Join-Path $internos "flet_client")
if (-not (Test-Path (Join-Path $internos "flet_client\flet\flet.exe"))) {
    throw "flet.exe não encontrado no cliente Flet desempacotado"
}

# Plugin de localização vazio (msix\plugin_vazio): o cliente Flet registra o geolocator ao abrir
# e, empacotado, o Windows pediria permissão de localização, que o app não usa. Compilado com o
# Zig do PyPI (pacote ziglang), instalado uma vez em .ferramentas\.
$zig = Join-Path $ferramentas "zig"
if (-not (Test-Path (Join-Path $zig "ziglang"))) {
    & $python -m pip install --quiet --target $zig "ziglang==0.16.0"
    if ($LASTEXITCODE -ne 0) { throw "Falha ao instalar o Zig (ziglang)" }
}
$pluginVazio = Join-Path $saida "plugin_vazio\geolocator_windows_plugin.dll"
New-Item -ItemType Directory -Force (Split-Path $pluginVazio) | Out-Null
$env:PYTHONPATH = $zig
& $python -m ziglang cc -shared -O2 -target x86_64-windows-gnu `
    (Join-Path $raiz "msix\plugin_vazio\geolocator_windows_plugin.c") -o $pluginVazio
$codigoZig = $LASTEXITCODE
Remove-Item Env:\PYTHONPATH
if ($codigoZig -ne 0) { throw "Falha ao compilar o plugin vazio ($codigoZig)" }
Copy-Item $pluginVazio (Join-Path $internos "flet_client\flet\geolocator_windows_plugin.dll") -Force

# 3. Imagens, manifesto e resources.pri. O makepri indexa só a pasta das imagens (indexar o
#    executável inteiro, com milhares de arquivos, só deixaria o .pri maior e mais lento).
& $python scripts\gerar_imagens_msix.py (Join-Path $recursos "Assets")
if ($LASTEXITCODE -ne 0) { throw "Falha ao gerar as imagens" }

$valores = @{
    "{{NOME_PACOTE}}" = $NomePacote
    "{{EDITOR}}" = $Editor
    "{{NOME_EDITOR}}" = $NomeEditor
    "{{VERSAO}}" = $Versao
}
$manifesto = [IO.File]::ReadAllText((Join-Path $raiz "msix\AppxManifest.xml"), [Text.Encoding]::UTF8)
foreach ($campo in $valores.Keys) {
    $manifesto = $manifesto.Replace($campo, [Security.SecurityElement]::Escape($valores[$campo]))
}
$arquivoManifesto = Join-Path $layout "AppxManifest.xml"
[IO.File]::WriteAllText($arquivoManifesto, $manifesto, (New-Object Text.UTF8Encoding $false))

$configPri = Join-Path $saida "priconfig.xml"
& $makepri createconfig /cf $configPri /dq pt-BR /pv 10.0.0 /o | Out-Null
if ($LASTEXITCODE -ne 0) { throw "makepri createconfig falhou ($LASTEXITCODE)" }
& $makepri new /pr $recursos /cf $configPri /mn $arquivoManifesto `
    /of (Join-Path $layout "resources.pri") /o | Out-Null
if ($LASTEXITCODE -ne 0) { throw "makepri new falhou ($LASTEXITCODE)" }
Copy-Item (Join-Path $recursos "Assets") (Join-Path $layout "Assets") -Recurse

# 4. Pacote
# O da Loja vai para dist\; o de teste fica em build\msix-teste (não pode ser enviado à Loja).
$msix = if ($Teste) { Join-Path $saida "Sigra-$Versao-x64-teste.msix" } else { Join-Path $raiz "dist\Sigra-$Versao-x64.msix" }
& $makeappx pack /d $layout /p $msix /o | Out-Null
if ($LASTEXITCODE -ne 0) { throw "makeappx pack falhou ($LASTEXITCODE)" }
Write-Output "Pacote: $msix ($([math]::Round((Get-Item $msix).Length / 1MB)) MB)"
Write-Output "Identidade: $NomePacote | $Editor | $NomeEditor | $Versao"

if ($Registrar) {
    Get-AppxPackage -Name $NomePacote | Remove-AppxPackage
    Add-AppxPackage -Register $arquivoManifesto
    $instalado = Get-AppxPackage -Name $NomePacote
    Write-Output "Registrado neste PC para teste: shell:AppsFolder\$($instalado.PackageFamilyName)!Sigra"
}
