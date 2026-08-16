import flet as flet
import pandas as pd



# função que importa a base de dados
import pandas as pd


# Função que importa a base de dados
def importar_dados(path, extensao):
    extensao = extensao.lower()
    if extensao == ".xlsx":
        df = pd.read_excel(path)
    elif extensao == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Formato de arquivo não suportado: {extensao}. "
            "Utilize apenas .xlsx ou .csv."
        )
    return df


