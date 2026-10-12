"""Gera as Inter estáticas do PDF (Regular e Bold) a partir da Inter variável da interface.

O fpdf2 e o matplotlib (gráficos do relatório) trabalham com um arquivo por peso: numa fonte
variável o matplotlib só enxerga o peso padrão e o negrito sairia regular. Rodar uma vez (o
resultado fica versionado em assets/fonts):

    python scripts/gerar_fontes_pdf.py
"""

from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

PASTA = Path(__file__).resolve().parents[1] / "assets" / "fonts"
TAMANHO_OPTICO = 14  # eixo opsz da Inter: desenho para texto corrido (o PDF usa 8 a 14 pt)
PESOS = {"Inter-Regular.ttf": 400, "Inter-Bold.ttf": 700}


def main() -> None:
    for nome, peso in PESOS.items():
        fonte = TTFont(PASTA / "Inter.ttf")
        estatica = instancer.instantiateVariableFont(
            fonte, {"wght": peso, "opsz": TAMANHO_OPTICO}, updateFontNames=True
        )
        estatica.save(PASTA / nome)
        print(f"{nome}: {(PASTA / nome).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
