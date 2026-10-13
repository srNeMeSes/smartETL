"""Imagens da listagem na Microsoft Store (Partner Center > Listagem > Logotipos da Store).

    python scripts/gerar_imagens_loja.py

Saída em dist/loja/:
- poster_9x16_1440x2160.png e caixa_1x1_2160x2160.png: ícone + nome "Sigra" sobre o laranja suave
  da identidade (fundo sólido: a Loja recorta essas artes em vários layouts);
- bloco_300x300.png, bloco_150x150.png e bloco_71x71.png: só o ícone, fundo transparente.
Cores de app/ui/tema.py; fonte Inter (assets/fonts).
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from app.ui import tema  # noqa: E402

ICONE = RAIZ / "assets" / "icon2.png"
NEGRITO = RAIZ / "assets" / "fonts" / "Inter-Bold.ttf"
REGULAR = RAIZ / "assets" / "fonts" / "Inter-Regular.ttf"
SAIDA = RAIZ / "dist" / "loja"
SUBTITULO = "Testes de hipótese e análise estatística"


def _icone(lado: int) -> Image.Image:
    return Image.open(ICONE).convert("RGBA").resize((lado, lado), Image.Resampling.LANCZOS)


def _texto_centralizado(desenho, largura: int, y: int, texto: str, fonte, cor: str) -> int:
    caixa = desenho.textbbox((0, 0), texto, font=fonte)
    desenho.text(
        ((largura - (caixa[2] - caixa[0])) / 2 - caixa[0], y - caixa[1]),
        texto,
        font=fonte,
        fill=cor,
    )
    return y + caixa[3] - caixa[1]


def arte(largura: int, altura: int, com_subtitulo: bool) -> Image.Image:
    """Ícone e nome (e subtítulo) centralizados verticalmente sobre o laranja suave."""
    tela = Image.new("RGBA", (largura, altura), tema.LARANJA_SUAVE)
    desenho = ImageDraw.Draw(tela)
    lado = round(largura * 0.42)
    nome = ImageFont.truetype(str(NEGRITO), round(largura * 0.15))
    sub = ImageFont.truetype(str(REGULAR), round(largura * 0.042))
    espaco = round(largura * 0.06)
    alt_nome = desenho.textbbox((0, 0), "Sigra", font=nome)
    total = lado + espaco + (alt_nome[3] - alt_nome[1])
    if com_subtitulo:
        alt_sub = desenho.textbbox((0, 0), SUBTITULO, font=sub)
        total += round(espaco * 0.6) + (alt_sub[3] - alt_sub[1])
    y = (altura - total) // 2
    tela.alpha_composite(_icone(lado), ((largura - lado) // 2, y))
    y = _texto_centralizado(desenho, largura, y + lado + espaco, "Sigra", nome, tema.TEXTO)
    if com_subtitulo:
        _texto_centralizado(
            desenho, largura, y + round(espaco * 0.6), SUBTITULO, sub, tema.TEXTO_SECUNDARIO
        )
    return tela.convert("RGB")


def bloco(lado: int) -> Image.Image:
    """Ícone com 6% de margem em cada lado, fundo transparente."""
    tela = Image.new("RGBA", (lado, lado), (0, 0, 0, 0))
    interno = round(lado * 0.88)
    tela.alpha_composite(_icone(interno), ((lado - interno) // 2, (lado - interno) // 2))
    return tela


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    imagens = {
        "poster_9x16_1440x2160.png": arte(1440, 2160, com_subtitulo=True),
        "caixa_1x1_2160x2160.png": arte(2160, 2160, com_subtitulo=False),
        **{f"bloco_{n}x{n}.png": bloco(n) for n in (300, 150, 71)},
    }
    for nome, imagem in imagens.items():
        imagem.save(SAIDA / nome, optimize=True)
        print(f"{nome}: {imagem.size[0]}x{imagem.size[1]}")


if __name__ == "__main__":
    main()
