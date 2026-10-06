# smartETL — Protótipo de Interface (Flet)

Protótipo de interface desktop para um aplicativo de processamento e análise
de dados, inspirado em ferramentas modernas de ETL/BI. **Neste momento é uma
interface estática**: todos os dados são fictícios e gerados em memória, e os
botões (Gerar relatório, Filtros, Visualizar, Exportar, itens da barra
lateral) são apenas visuais — a paginação da tabela é a única interação
funcional, incluída como demonstração de como os dados fluem para os
componentes.

## Requisitos

- Python 3.9 ou superior
- Pacote `flet` (testado com a versão `0.86.2`)

## Instalação e execução

```bash
# 1. (opcional, mas recomendado) crie um ambiente virtual
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. instale as dependências
pip install -r requirements.txt

# 3. execute o aplicativo
python smartetl_app.py
```

Isso abrirá o smartETL como uma janela desktop nativa (via Flet/Flutter).

> Caso o ambiente não consiga abrir uma janela nativa (ex.: SO sem suporte),
> troque a última linha do arquivo para
> `ft.run(main, view=ft.AppView.WEB_BROWSER)` para rodar no navegador.

## Estrutura do projeto

```
smartetl_app.py     # aplicativo completo (interface + dados fictícios)
requirements.txt     # dependências
README.md            # este arquivo
```

## O que está incluído

- **Barra lateral**: logo "smartETL", itens Arquivo / Histórico / Logs e um
  cartão de status ("67 bases de dados — Carregadas com sucesso").
- **Cabeçalho**: título "Processamento de dados" + botão "Gerar relatório".
- **Descrição geral**: resumo executivo fictício + lista de variáveis
  analisadas.
- **Visualização geral**: gráfico de barras (construído sem bibliotecas
  externas de gráfico) mostrando o faturamento total das 5 filiais com
  melhor desempenho — calculado a partir dos mesmos dados da tabela.
- **Tabela geral**: 67 registros fictícios (Filial, Mês, Faturamento, Taxa de
  Conversão, Qtd. de Vendas), com paginação de 10 em 10 linhas.

## Personalizando os dados

Os dados fictícios são gerados pela função `gerar_dados()` no topo do
arquivo. Para gerar mais ou menos registros, ou alterar filiais/meses,
edite as listas `FILIAIS`, `MESES` e o parâmetro `total_registros`.

## Próximos passos (fora do escopo deste protótipo)

- Implementar upload real de bases de dados na opção "Arquivo".
- Persistir histórico de análises e logs de processamento.
- Conectar "Gerar relatório", "Filtros" e "Exportar" a lógica real.
- Implementar o pipeline de ETL propriamente dito.
