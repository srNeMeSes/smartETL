# Gera o pacote MSIX do Signa para a Microsoft Store (dist\Signa-<versão>-x64.msix), a partir do
# executável já gerado por scripts\gerar_executavel.ps1. Rodar da raiz do projeto:
#
#   powershell -ExecutionPolicy Bypass -File scripts\gerar_msix.ps1 `
#       -NomePacote "<Package/Identity/Name>" -Editor "<Package/Identity/Publisher>" `
#       -NomeEditor "<Package/Properties/PublisherDisplayName>" -Versao 1.0.0.0
#
# Os três primeiros valores vêm do Partner Center (app > Gerenciamento de produto > Identidade do
# produto) e precisam ser exatamente iguais, senão a Loja recusa o pacote. Sem eles, o script usa
# valores de teste (o pacote serve para testar neste PC, não para enviar à Loja).
# A Loja assina o pacote na publicação: o .msix enviado vai sem assinatura.
#
# -Registrar instala a pasta do pacote neste PC (exige o Modo de Desenvolvedor), para testar o app
# como se tivesse vindo da Loja: Menu Iniciar, barra de tarefas, pasta somente leitura.
#
# Dentro do pacote: dist\Signa (executável), o cliente Flet já desempacotado em
# Signa\_internal\flet_client (main.usar_cliente_flet_embutido), as imagens geradas por
# scripts\gerar_imagens_msix.py e o resources.pri que indexa as escalas dessas imagens.

param(
    [string]$NomePacote = "Signa.Teste",
    [string]$Editor = "CN=Signa Teste",
    [string]$NomeEditor = "Evaniel Arcanjo Da Silva",
    [string]$Versao = "1.0.0.0",
    [switch]$Registrar
)

$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot
Set-Location $raiz

if ($Versao -notmatch '^\d+\.\d+\.\d+\.0$') {
    throw "Versão no formato X.Y.Z.0 (a Loja exige o último número 0): $Versao"
}
if (-not (Test-Path "dist\Signa\Signa.exe")) {
    throw "Gere o executável antes: scripts\gerar_executavel.ps1"
}
$python = Join-Path $raiz "venv\Scripts\python.exe"

# 1. makeappx e makepri: pacote NuGet oficial da Microsoft com as ferramentas do Windows SDK
#    (evita instalar o SDK inteiro). Guardado em build\, baixado uma vez.
$versaoSdk = "10.0.26100.9169"
$pastaSdk = Join-Path $raiz "build\msix-ferramentas\$versaoSdk"
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
$saida = Join-Path $raiz "build\msix"
if (Test-Path $saida) { Remove-Item $saida -Recurse -Force }
$layout = Join-Path $saida "layout"
$recursos = Join-Path $saida "recursos"
New-Item -ItemType Directory -Force $layout, $recursos | Out-Null
Copy-Item "dist\Signa" (Join-Path $layout "Signa") -Recurse

# Cliente Flet desempacotado (o flet-windows.zip do pacote já tem o ícone e a versão do Signa)
$internos = Join-Path $layout "Signa\_internal"
Expand-Archive -Path (Join-Path $internos "flet_desktop\app\flet-windows.zip") `
    -DestinationPath (Join-Path $internos "flet_client")
if (-not (Test-Path (Join-Path $internos "flet_client\flet\flet.exe"))) {
    throw "flet.exe não encontrado no cliente Flet desempacotado"
}

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
$msix = Join-Path $raiz "dist\Signa-$Versao-x64.msix"
& $makeappx pack /d $layout /p $msix /o | Out-Null
if ($LASTEXITCODE -ne 0) { throw "makeappx pack falhou ($LASTEXITCODE)" }
Write-Output "Pacote: $msix ($([math]::Round((Get-Item $msix).Length / 1MB)) MB)"
Write-Output "Identidade: $NomePacote | $Editor | $NomeEditor | $Versao"

if ($Registrar) {
    Get-AppxPackage -Name $NomePacote | Remove-AppxPackage
    Add-AppxPackage -Register $arquivoManifesto
    $instalado = Get-AppxPackage -Name $NomePacote
    Write-Output "Registrado neste PC para teste: shell:AppsFolder\$($instalado.PackageFamilyName)!Signa"
}
