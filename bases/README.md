# Bases de teste

Bases usadas para conferir cada teste no app real (`python main.py` → **Arquivo**). As que
terminam em `_br` estão no formato brasileiro: encoding Windows (cp1252), separador `;` e
vírgula decimal. A leitura detecta tudo isso automaticamente.

Toda base nova usada nos testes manuais deve ser salva nesta pasta e listada aqui.

| Base | Conteúdo | Teste | Como preencher | Resultado esperado (α = 0,05) |
|------|----------|-------|----------------|-------------------------------|
| `dados_exemplo.csv` | 6 linhas (id, users, qtd, valor, total); CSV padrão (vírgula, UTF-8) | Teste t (uma amostra) | Variável `qtd`, μ₀ = 5 | Prévia simples; base do print de referência |
| `vendas_br.csv` | 5 vendas por região/cidade; milhar com ponto ("1.234,50"); a coluna `nota` tem um "n/d" | Teste t (uma amostra) | Variável `receita`, μ₀ = 1000 | Não rejeita H₀ (p = 0,286); aviso de que `nota` mistura números e texto |
| `turmas_br.csv` | 13 alunos: nota por turma (Manhã/Noite) e cidade (3 valores) | Teste t (duas amostras) | Variável `nota`, Grupo `turma`, Welch | Não rejeita (p = 0,201); Mann-Whitney 0,051 no card; boxplot com outlier (4,1) |
| `pressao_br.csv` | 10 pacientes, pressão antes/depois; uma linha sem "depois" | Teste t (pareado) | Medida 1 `antes`, Medida 2 `depois` | Rejeita (p = 0,002), 9 pares; aviso de 1 linha descartada |
| `pesquisa_br.csv` | 60 clientes: `comprou` (0/1), `satisfeito` (Sim/Não), `região` (3 valores) | Teste Z (uma proporção) | Variável `satisfeito`, sucesso `Sim`, p₀ = 0,5 | Rejeita (p = 0,002): 70,0% satisfeitos |
| `pesquisa_br.csv` | (mesma base) | Teste Z (duas proporções) | Resposta `satisfeito`, sucesso `Sim`, Grupo `comprou` | Não rejeita (p = 1,000): 70% nos dois grupos |
| `lojas_br.csv` | 95 clientes de duas lojas: comprou (Sim/Não) e valor | Teste Z (duas proporções) | Resposta `comprou`, sucesso `Sim`, Grupo `loja` | Não rejeita no bilateral (p = 0,052); rejeita em p₁ > p₂ (0,026); odds ratio 2,25 |
| `conceitos_br.csv` | 90 alunos: turno (Manhã/Noite) × conceito (A/B/C); a coluna `aluno` é identificador | Qui-quadrado | Independência: Variável 1 `turno`, Variável 2 `conceito` | Rejeita (p < 0,001); barras agrupadas; `aluno` não aparece na lista |
| `conceitos_br.csv` | (mesma base) | Qui-quadrado | Aderência: Variável 1 `conceito` | Não rejeita (p = 1,000): 30 alunos em cada conceito (1/3 cada) |
| `fumo_br.csv` | 32 pacientes: hábito (Fumante/Não fumante) × condição (Doente/Saudável) | Teste exato de Fisher | Variável 1 `hábito` (evento Fumante), Variável 2 `condição` (evento Doente) | Rejeita (p = 0,032); odds ratio amostral 6,6, condicional 6,17, IC [1,14; 41,61] |
| `campanha_br.csv` | 50 clientes: intenção de compra antes/depois de uma campanha (Sim/Não) | McNemar | Medida 1 `antes`, Medida 2 `depois`, evento `Sim` | Rejeita (p = 0,031, exato: b + c = 18 < 25); qui-quadrado 0,034 no card |
