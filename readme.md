# Análise Comparativa entre Implementações em R e Python para Previsão do Consumo de Energia Elétrica e Efeito do El Niño

## Resumo

Este trabalho tem caráter conceitual e compara duas implementações de um mesmo estudo, uma em **R** (`predict_elnino.R`) e outra em **Python** (`predict_ElNino.py`), que analisam o consumo mensal de energia elétrica no Brasil por região geográfica, buscando identificar padrões associados ao fenômeno El Niño. Ambos os scripts seguem a mesma sequência lógica: carregamento dos dados, transformação para formato longo, tratamento de valores ausentes, testes estatísticos, ajuste de três modelos preditivos (ARIMA, SVM e Random Forest), avaliação de erro (RMSE/MAE) e geração de previsões futuras comparadas a um cenário de consumo "habitual". Apesar da equivalência conceitual entre os dois scripts, as saídas numéricas divergem de forma significativa em pontos específicos, o que é explorado em detalhe neste documento como estudo de caso sobre os riscos de portar código estatístico entre linguagens sem validação cuidadosa de que os "valores padrão" (*defaults*) das bibliotecas realmente correspondem.

---

## 1. Introdução

O objetivo original do estudo é avaliar se o consumo de energia elétrica nas cinco regiões geográficas do Brasil se desvia do padrão histórico em anos associados a eventos El Niño, e comparar três abordagens de modelagem preditiva quanto à sua capacidade de captar esse desvio: um modelo de séries temporais (ARIMA), um modelo de aprendizado de máquina baseado em margem (SVM com kernel radial) e um modelo de conjunto baseado em árvores (Random Forest).

O script em R (`predict_elnino.R`) é a implementação original. O script em Python (`predict_ElNino.py`) é uma conversão que busca reproduzir a mesma lógica utilizando `pandas`, `scikit-learn`, `pmdarima`/`statsmodels` e `matplotlib`/`seaborn`. As saídas de execução de ambos (`saída_R.txt` e `Saída_Python.txt`) foram comparadas linha a linha para este relatório.

---

## 2. Metodologia e Dados

- **Fonte de dados:** `CONSUMO_MENSAL_DE_ENERGIA_ELÉTRICA_POR_CLASSE.xlsx`, aba `TOTAL`.
- **Estrutura original:** uma linha por ano/região, com uma coluna para cada mês (formato largo).
- **Período coberto:** conforme o gráfico de evolução (item 15 de ambos os scripts), de 2003 a 2030.
- **Regiões:** Norte, Nordeste, Sudeste, Sul e Centro-Oeste.
- **Divisão treino/teste:** anos `< 2020` como treino, anos `>= 2020` como teste — idêntica nos dois scripts, resultando em **360 observações de teste** em ambos os casos (confirmado nas duas saídas).
- **Modelos comparados:** ARIMA (`auto.arima` em R / `auto_arima` do `pmdarima` ou `ARIMA` do `statsmodels` em Python), SVM com kernel radial (`e1071::svm` em R / `sklearn.svm.SVR` em Python) e Random Forest (`randomForest` em R / `RandomForestRegressor` do `scikit-learn` em Python).
- **Métricas de erro:** RMSE e MAE, calculadas por funções customizadas equivalentes em ambas as linguagens (mesma fórmula matemática, com tratamento de `NA`/`NaN`).

---

## 3. Comparação dos Códigos (R vs. Python)

### 3.1 Carregamento e preparo dos dados

| Etapa | R | Python | Observação |
|---|---|---|---|
| Seleção do arquivo | `file.choose()` | `tkinter.filedialog.askopenfilename()` | Equivalentes funcionalmente |
| Leitura do Excel | `read_excel(file_path, sheet="TOTAL")` | `pd.read_excel(file_path, sheet_name="TOTAL")` | Idênticas |
| Renomeação de colunas | `rename()` | `.rename(columns={...})` | Idênticas |
| Formato largo → longo | `pivot_longer()` | `dados.melt()` | **Divergência relevante** (ver 3.1.1) |
| Tratamento de NAs | Substituição pela média global | Substituição pela média global (`fillna`) | Idênticas na lógica; nas saídas, ambos reportam **0 valores ausentes** antes e depois, ou seja, para este conjunto de dados essa etapa não teve efeito prático em nenhuma das duas linguagens |

#### 3.1.1 `pivot_longer()` vs. `melt()`

Embora produzam o mesmo conjunto de valores, as duas funções **não geram necessariamente a mesma ordem de linhas**: `pivot_longer()` do R tende a preservar a ordem das linhas originais, expandindo os meses em sequência para cada combinação ano/região; já `melt()` do pandas, por padrão, varre coluna a coluna (todos os anos/regiões para "JANEIRO", depois todos para "FEVEREIRO", e assim por diante). Isso não altera testes estatísticos que independem de ordem (como Shapiro-Wilk ou Kruskal-Wallis), mas **é crítico para qualquer modelo que dependa da ordenação temporal**, como o ARIMA construído diretamente a partir do vetor de valores (ver Seção 5.2).

O script Python tenta mitigar isso ordenando explicitamente por `ANO_DE_REFERENCIA` e `Mês` antes de montar a série usada no ARIMA, mas essa ordenação não distingue região, de modo que, para um mesmo par (ano, mês), os valores das cinco regiões continuam intercalados de forma não determinística. Esse é um problema estrutural presente, em graus diferentes, **nas duas linguagens**: tratar uma série multi-regional como uma única série mensal (`frequency = 12`) mistura conceitualmente cinco séries distintas em uma só.

### 3.2 Testes estatísticos

A lógica de decisão é idêntica em ambos os scripts:

1. Teste de normalidade de Shapiro-Wilk sobre `Consumo_MWh`.
2. Se normal → Teste t pareado entre Sudeste e Nordeste.
3. Se não normal e mais de 2 grupos → Teste de Kruskal-Wallis entre as cinco regiões.
4. Se não normal e apenas 2 grupos → Teste de Wilcoxon/Mann-Whitney.

Essa etapa é a que apresenta **maior fidelidade entre as duas linguagens**, funcionando como uma boa validação de que os valores brutos de `Consumo_MWh`, apesar da ordem diferente, são os mesmos nos dois lados (ver Seção 4.1).

### 3.3 Modelagem preditiva

Esta é a etapa com **maiores divergências de comportamento**, mesmo com código aparentemente equivalente:

| Modelo | R | Python | Divergência |
|---|---|---|---|
| SVM | `e1071::svm(kernel="radial")` | `sklearn.svm.SVR(kernel="rbf")` | `e1071::svm()` **normaliza (escala) automaticamente** as variáveis preditoras e a resposta antes do ajuste; `SVR` do scikit-learn **não escala nada por padrão**. O código Python não inclui nenhum `StandardScaler`/`MinMaxScaler`. |
| ARIMA | `auto.arima()` sobre uma série com `frequency = 12` | `auto_arima(..., seasonal=False, ...)` (ou fallback `ARIMA(1,1,1)` do statsmodels) | `auto.arima()`, com `frequency=12`, testa e tipicamente ativa componente **sazonal** (SARIMA); a função `rodar_auto_arima()` do Python **desativa explicitamente a sazonalidade** (`seasonal=False`). São, na prática, duas classes de modelo diferentes. |
| Random Forest | `randomForest(Consumo_MWh ~ ., data=..., ntree=100)`, com fatores tratados internamente | `RandomForestRegressor(n_estimators=100, random_state=123)` sobre variáveis categóricas **one-hot codificadas** (`pd.get_dummies`) | O R particiona diretamente sobre os níveis do fator (`REGIAO_GEOGRAFICA`, `Mês`); o Python transforma cada categoria em uma coluna binária antes do ajuste. A árvore resultante e a forma como o espaço de decisão é particionado diferem estruturalmente. |

Vale notar que `ntree`/`n_estimators = 100` e a semente aleatória (`set.seed(123)` / `random_state=123`) foram mantidos equivalentes, a intenção de paridade existe no código, mas os *defaults* silenciosos de cada biblioteca (escala no SVM, sazonalidade no ARIMA) subvertem essa intenção.

### 3.4 Visualização

Ambos os scripts geram os mesmos seis gráficos conceituais (itens 15 a 21), usando `ggplot2` no R e `matplotlib`/`seaborn` no Python. A lógica de destaque dos pontos de "efeito El Niño" (marcadores triangulares) é replicada em Python via `scatter()` manual sobre o subconjunto filtrado, em vez do `scale_shape_manual()` do `ggplot2`, funcionalmente equivalente, mas com pequenas diferenças estéticas de estilo de linha e legenda.

---

## 4. Resultados

### 4.1 Testes de Normalidade e Comparação entre Regiões

| Teste | R | Python |
|---|---|---|
| Shapiro-Wilk (estatística) | W = 0,78313 | 0,7831 |
| Shapiro-Wilk (p-valor) | < 2,2e-16 | 1,185e-38 |
| Kruskal-Wallis (estatística) | χ² = 1141,5 | 1141,5465 |
| Kruskal-Wallis (p-valor) | < 2,2e-16 | 7,474e-246 |

As estatísticas de teste **coincidem para todas as casas decimais reportadas**, o que confirma que, apesar da diferença de ordenação discutida na Seção 3.1.1, o **conjunto de valores** de `Consumo_MWh` é idêntico entre as duas implementações (já que esses testes não dependem da ordem das observações). A diferença aparente nos p-valores é apenas de precisão de exibição: o R trunca em `< 2.2e-16` (limite de precisão de ponto flutuante de dupla precisão do ambiente), enquanto o Python exibe o valor efetivamente calculado, que é ainda menor. Em ambos os casos a conclusão é a mesma: os dados **não seguem distribuição normal**, e há diferença estatisticamente significativa entre as cinco regiões (Kruskal-Wallis).

### 4.2 Avaliação dos Modelos (RMSE / MAE)

| Modelo | RMSE (R) | MAE (R) | RMSE (Python) | MAE (Python) |
|---|---|---|---|---|
| ARIMA | 6.743.109 | 4.624.574 | 1.009.273 | 722.452 |
| SVM | 2.017.963 | 1.251.186 | 7.407.070 | 4.955.697 |
| Random Forest | 2.042.628 | 1.375.926 | 1.019.225 | 774.088 |

O resultado mais notável da comparação é a **inversão de ranking entre os modelos**:

- Em **R**, o ARIMA é de longe o pior modelo (RMSE ≈ 6,7 milhões), enquanto SVM e Random Forest têm desempenho próximo e muito superior (RMSE ≈ 2,0 milhões).
- Em **Python**, o ARIMA é o **melhor** modelo (RMSE ≈ 1,0 milhão), o Random Forest fica muito próximo dele (RMSE ≈ 1,0 milhão), e o SVM passa a ser **o pior** de todos (RMSE ≈ 7,4 milhões), pior inclusive que o ARIMA em R.

Essa inversão não reflete uma diferença real de capacidade preditiva entre as linguagens, mas sim os desvios de configuração discutidos na Seção 3.3 e detalhados na Seção 5.

### 4.3 Classificação do Efeito "El Niño" nas Previsões Futuras (2026–2030)

Cada script gera, para os cinco anos seguintes ao fim da base histórica, uma previsão por modelo e classifica cada linha como `"El Niño"` quando a previsão diverge do consumo médio histórico ("habitual") em mais de 10%.

| Modelo | El Niño / Normal — R | El Niño / Normal — Python |
|---|---|---|
| SVM | 15 / 10 | **25 / 0** |
| ARIMA (Séries Temporais) | 25 / 0 | 25 / 0 |
| Random Forest | 19 / 6 | 8 / 17 |

#  *SVM*

<img width="1141" height="545" alt="image" src="https://github.com/user-attachments/assets/f60e9f35-b3ec-46a0-b460-2f72364faac6" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/da5d5865-f2c2-4c96-81a2-536a65a9c3b8" />

#  *ARIMA*

<img width="1141" height="545" alt="image" src="https://github.com/user-attachments/assets/4038f014-b2bb-4b8f-9fd2-e34c6d49f5b8" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/5e37d915-a53f-4d24-9e08-e5da59c0aac7" />

#  *Random Forest*

<img width="1141" height="545" alt="image" src="https://github.com/user-attachments/assets/12833203-52ff-4df1-88aa-8d53f9c8b044" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/41f064af-7d56-4809-bfaa-1a58ec032482" />


O achado mais expressivo é o comportamento do **SVM em Python**: todas as 25 linhas previstas recebem exatamente o **mesmo valor previsto** (5.273.047 MWh), independentemente do ano ou da região. Isso é um sintoma clássico de um SVR com kernel RBF ajustado **sem escalonamento de variáveis**: quando as variáveis (ano, região codificada, mês codificado) têm escalas muito diferentes e a variável resposta tem magnitude na casa dos milhões, o modelo tende a colapsar para uma previsão próxima da média da variável resposta, perdendo a capacidade de diferenciar as observações. Isso é discutido em detalhe na Seção 5.1.

---

## 5. Análise Comparativa das Divergências

### 5.1 SVM: o efeito do escalonamento ausente

`e1071::svm()`, por padrão, escalona (centraliza e normaliza) tanto as variáveis preditoras quanto a variável resposta antes do ajuste, e desfaz essa transformação na hora de gerar as previsões, de forma transparente para quem usa a função. `sklearn.svm.SVR` **não faz isso**: a documentação é explícita quanto a essa responsabilidade ser do usuário. Como o script Python não insere nenhum `StandardScaler` no fluxo, o SVM recebe variáveis com escalas de grandeza muito diferentes (ano ≈ 2000, colunas binárias ∈ {0,1}) e uma resposta na casa das dezenas de milhões, cenário em que o kernel RBF, sensível a distância euclidiana, praticamente ignora a estrutura da variável preditora. O resultado observado (previsão constante ≈ 5,27 milhões para todas as combinações de ano/região/mês) é consistente com esse diagnóstico. É também o motivo direto pelo qual o SVM passa de "melhor modelo" em R para "pior modelo" em Python.

### 5.2 ARIMA: sazonalidade desativada e ordenação da série

Dois fatores compostos explicam a diferença nos resultados de ARIMA:

1. **Sazonalidade:** `auto.arima()` em R, ao receber uma série com `frequency = 12`, testa e frequentemente seleciona um componente sazonal (SARIMA), capturando o padrão anual de consumo de energia (que tende a ser fortemente sazonal). O `rodar_auto_arima()` do Python passa `seasonal=False` explicitamente, descartando essa possibilidade por construção, o modelo Python é necessariamente um ARIMA não sazonal, estruturalmente mais simples.
2. **Ordenação da série de treino:** como descrito na Seção 3.1.1, a ordem das observações na série usada para ajustar o ARIMA depende de como os dados foram transformados de largo para longo, e essa ordem difere entre `pivot_longer()` e `melt()` (mesmo com a reordenação parcial aplicada no Python). Como o ARIMA depende inteiramente da estrutura sequencial dos dados, qualquer diferença de ordenação, combinada com a mistura de cinco regiões em uma única série "mensal", leva a ajustes completamente diferentes nos dois lados.

A combinação desses dois fatores é suficiente para explicar por que o ARIMA muda de "pior modelo" (R) para "melhor modelo" (Python), não porque um seja estatisticamente mais correto que o outro, mas porque **não são o mesmo modelo aplicado à mesma sequência de dados**.

### 5.3 Random Forest: codificação de variáveis categóricas

O `randomForest` do R lida com uma variável categórica (fator) encontrando o melhor subconjunto de níveis para dividir os dados em cada nó (um único ponto de divisão pode separar, por exemplo, {Sudeste, Sul} de {Norte, Nordeste, Centro-Oeste}). O `RandomForestRegressor` do scikit-learn, alimentado por variáveis *one-hot* (uma coluna binária por categoria), só pode dividir uma categoria de cada vez (Sudeste = 1 vs. Sudeste = 0). Isso torna as árvores estruturalmente diferentes, ainda que ambas usem o mesmo número de árvores (100) e a mesma semente aleatória, a semente controla a aleatoriedade do algoritmo, mas não torna equivalentes representações de dados diferentes. Esse é o motivo mais provável para a diferença de desempenho entre R (RMSE ≈ 2,0 milhões) e Python (RMSE ≈ 1,0 milhão) neste modelo — aqui, curiosamente, a divergência favoreceu a implementação Python, ao contrário do que ocorreu com SVM e ARIMA.

### 5.4 Síntese

| Fonte da divergência | Modelo afetado | Direção do impacto em Python |
|---|---|---|
| Ausência de escalonamento em `SVR` | SVM | Fortemente prejudicial (colapso para previsão quase constante) |
| `seasonal=False` explícito + ordenação da série | ARIMA | Resultado numericamente melhor, mas conceitualmente é **outro modelo** |
| *One-hot encoding* vs. fatores nativos | Random Forest | Moderadamente favorável, porém não diretamente comparável |

O padrão geral que emerge é que **nenhuma das discrepâncias reflete um erro de programação isolado** (não há bug de sintaxe ou de fórmula); todas decorrem de **valores padrão (*defaults*) diferentes entre bibliotecas equivalentes em conceito, mas não em implementação**. Esse é, em si, um resultado relevante do ponto de vista metodológico: replicar um pipeline estatístico entre linguagens exige validar explicitamente cada *default*, não apenas a assinatura da função.

---

## 6. Comparação Visual dos Gráficos

Esta seção reserva espaço para a inclusão manual das imagens geradas por cada script, permitindo uma inspeção visual das divergências discutidas nas seções anteriores.

### 6.1 Evolução do Consumo por Região (2003–2023) — item 15

<img width="1141" height="545" alt="image" src="https://github.com/user-attachments/assets/14c25f49-c039-42d6-809b-80603e5c4d74" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/67258ccc-6796-421a-a4e9-654a9c40c265" />

### 6.2 Distribuição Mensal do Consumo (Boxplot) — item 17

<img width="1141" height="545" alt="image" src="https://github.com/user-attachments/assets/377a6c64-d12a-4e2a-bc00-bd0f5ff33018" />
<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/5c42d856-f217-4c05-9c4e-d278eed99b24" />

### 6.3 Comparação entre Modelos Preditivos Nacionais (2024–2028) — itens 20 e 21

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/af3d078e-8220-407b-9cc9-0b131c9116ca" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/3e86a8df-ac3b-45b7-8376-580ccca4df30" />


---

## 7. Limitações Metodológicas Comuns às Duas Implementações

Independentemente das divergências entre R e Python, alguns pontos são limitações do desenho experimental original, presentes nos dois scripts:

- **Mistura de regiões em uma única série temporal:** ao construir `serie_treino`/`serie_temporal` a partir de `dados_treino$Consumo_MWh` (ou seu equivalente Python) sem separar por região antes de ajustar o primeiro modelo ARIMA (item 13), cinco séries distintas com magnitudes muito diferentes (ex.: Sudeste consome bem mais que Norte) são tratadas como uma única sequência de "meses", o que compromete a interpretabilidade desse ARIMA específico (diferente do ARIMA por região do item 18, que já corrige isso).
- **Anos de "impacto do El Niño" definidos por exemplo, não por dado real:** `anos_impacto <- c(2025, 2027)` (e seu equivalente Python) é declarado como exemplo ilustrativo nos comentários do próprio código, não com base em um índice climático real (como o ONI - *Oceanic Niño Index*). A robustez da conclusão "El Niño" depende inteiramente do critério arbitrário de divergência de 10% em relação à média histórica, não de uma variável climática observada.
- **Ausência de validação cruzada:** a divisão treino/teste é única (por ano de corte), sem repetição ou validação cruzada temporal, o que limita a confiabilidade estatística das métricas de RMSE/MAE reportadas.

---

## 8. Conclusão

A comparação entre `predict_elnino.R` e `predict_ElNino.py` demonstra que **paridade sintática entre linguagens não implica paridade numérica**. Os dois scripts implementam, no papel, a mesma sequência metodológica, mas produzem tabelas de erro e classificações de "efeito El Niño" substancialmente diferentes, a ponto de inverter qual modelo é considerado o melhor. As causas identificadas (ausência de escalonamento no SVM, desativação da sazonalidade no ARIMA combinada à ordenação diferente da série, e diferenças na codificação de variáveis categóricas no Random Forest) são todas rastreáveis a **decisões de configuração implícitas** nas bibliotecas usadas, reforçando a importância de tornar esses parâmetros explícitos — e de validar numericamente cada etapa da portabilidade, em qualquer trabalho que pretenda comparar ou substituir um pipeline estatístico entre R e Python.
