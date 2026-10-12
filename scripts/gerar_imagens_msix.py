"""Imagens do pacote MSIX (ícones e blocos do Menu Iniciar e da Loja) a partir de assets/icon2.png.

Uso (chamado por scripts/gerar_msix.ps1):

    python scripts/gerar_imagens_msix.py <pasta de saída>

Gera cada imagem do manifesto nas escalas 100 a 400 (".scale-200" etc.) e o ícone pequeno também
por tamanho exato (".targetsize-32", com a variante "_altform-unplated" usada na barra de tarefas,
sem a placa de fundo). O `makepri` indexa os qualificadores e o Windows escolhe a melhor versão.
O ícone já é um quadrado arredondado laranja: nos ícones pequenos ocupa a imagem toda; nos blocos
fica centralizado com margem, sobre fundo transparente (a cor de destaque do Windows aparece).
"""

import sys
from pathlib import Path

from PIL import Image

ORIGEM = Path(__file__).resolve().parents[1] / "assets" / "icon2.png"
ESCALAS = (100, 125, 150, 200, 400)
TAMANHOS_ALVO = (16, 20, 24, 30, 32, 36, 40, 48, 60, 64, 72, 80, 96, 256)

# nome no manifesto: (largura, altura) na escala 100 e fração da altura ocupada pelo ícone
IMAGENS = {
    "Square44x44Logo": ((44, 44), 1.0),
    "StoreLogo": ((50, 50), 1.0),
    "SmallTile": ((71, 71), 0.62),
    "Square150x150Logo": ((150, 150), 0.6),
    "Wide310x150Logo": ((310, 150), 0.6),
    "LargeTile": ((310, 310), 0.6),
}


def desenhar(icone: Image.Image, largura: int, altura: int, fracao: float) -> Image.Image:
    """Ícone centralizado numa tela transparente, com lado = `fracao` da menor dimensão."""
    lado = max(1, round(min(largura, altura) * fracao))
    tela = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    reduzido = icone.resize((lado, lado), Image.Resampling.LANCZOS)
    tela.alpha_composite(reduzido, ((largura - lado) // 2, (altura - lado) // 2))
    return tela


def gerar(saida: Path) -> list[Path]:
    saida.mkdir(parents=True, exist_ok=True)
    icone = Image.open(ORIGEM).convert("RGBA")
    arquivos = []
    for nome, ((largura, altura), fracao) in IMAGENS.items():
        for escala in ESCALAS:
            imagem = desenhar(icone, largura * escala // 100, altura * escala // 100, fracao)
            arquivos.append(saida / f"{nome}.scale-{escala}.png")
            imagem.save(arquivos[-1])
    for tamanho in TAMANHOS_ALVO:
        imagem = desenhar(icone, tamanho, tamanho, 1.0)
        for sufixo in ("", "_altform-unplated"):
            arquivos.append(saida / f"Square44x44Logo.targetsize-{tamanho}{sufixo}.png")
            imagem.save(arquivos[-1])
    return arquivos


if __name__ == "__main__":
    print(f"{len(gerar(Path(sys.argv[1])))} imagens em {sys.argv[1]}")
