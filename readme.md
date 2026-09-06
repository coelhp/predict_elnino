# Comparação R vs. Python — Resultados Reais de Ambas as Execuções
---

## 1. Base de dados e testes estatísticos — resultado: **praticamente idênticos**

| Item | R | Python | Divergência |
|---|---|---|---|
| Shapiro-Wilk (estatística) | 0,78313 | 0,78310 | Nenhuma relevante |
| Shapiro-Wilk (p-valor) | < 2,2 × 10⁻¹⁶ | 1,19 × 10⁻³⁸ | Nenhuma (ambos "extremamente não normal"; R só arredonda o p-valor mínimo exibível) |
| Kruskal-Wallis (estatística) | 1141,5 | 1141,5465 | Nenhuma relevante |
| Kruskal-Wallis (p-valor) | < 2,2 × 10⁻¹⁶ | 7,47 × 10⁻²⁴⁶ | Nenhuma |
| Consumo médio histórico (ex.: Norte/Janeiro) | 2.500.116 | 2.500.116 | **Idêntico** |

Como esperado: `scipy.stats` e as funções nativas do R (`shapiro.test`, `kruskal.test`) implementam a mesma fórmula, então os números batem. Essa parte da migração está **100% correta**.

---

## 2. Avaliação dos modelos (RMSE / MAE) — resultado: **rankings praticamente invertidos**

| Modelo | RMSE (R) | RMSE (Python) | MAE (R) | MAE (Python) |
|---|---:|---:|---:|---:|
| ARIMA | **6.743.109** | 1.008.352 | **4.624.574** | 721.672 |
| SVM | 2.017.963 | **7.407.070** | 1.251.186 | **4.955.697** |
| Random Forest | 2.042.628 | 1.019.507 | 1.375.926 | 774.462 |

Isso é o achado mais importante deste relatório: **no R, o ARIMA foi o pior modelo; no Python, o ARIMA foi o melhor. No R, o SVM foi um dos melhores; no Python, foi disparado o pior.** Investiguei a causa raiz de cada uma dessas duas inversões e consegui **confirmar experimentalmente** ambas (não é só suposição — reproduzi os números).

### 2.1 Por que o SVM é catastrófico em Python (confirmado, não apenas suspeitado)

No relatório anterior eu já suspeitava de falta de escalonamento. Agora, com a saída real do R, dá para **provar isso linha a linha**:

| Ano | Região | `Consumo_SVM` no R | `Consumo_SVM` no Python |
|---|---|---:|---:|
| 2026 | Norte | 3.509.437 | 5.273.047 |
| 2026 | Nordeste | 6.530.542 | 5.273.047 |
| 2026 | Sudeste | 15.736.801 | 5.273.047 |
| 2026 | Sul | 6.932.098 | 5.273.047 |
| 2027 | Sudeste | 14.390.395 | 5.273.047 |
| 2030 | Sudeste | 12.955.908 | 5.273.047 |

**O SVM em Python retorna exatamente o mesmo valor (5.273.047) para as 25 linhas futuras, não importa a região ou o ano.** Na prática, o modelo "aprendeu" a prever a média geral do treino e ignora completamente as variáveis de entrada. O SVM do R, em contraste, varia corretamente por região (Sudeste bem mais alto que Norte, coerente com os dados históricos).

**Causa confirmada:** o pacote `e1071` do R escalona (`scale = TRUE`) X e y automaticamente antes de treinar o SVM. O `scikit-learn` não faz isso por padrão. Como o alvo está na casa dos milhões e o kernel RBF depende de distância entre pontos, o modelo em Python nunca converge para algo útil — ele efetivamente "desiste" e prevê a média. Refazendo o treino em Python com `StandardScaler` (replicando o comportamento padrão do R), o RMSE cai de 7,4 milhões para 1,6 milhão — ainda não é idêntico ao R (2,0 milhões), mas fica na mesma ordem de grandeza, confirmando a causa.

### 2.2 Por que o ARIMA "geral" (Seção 13) está invertido — duas causas reais, ambas confirmadas

Esta foi mais sutil. Encontrei **duas** diferenças reais introduzidas na conversão, e testei a combinação das duas reproduzindo o resultado do R quase exatamente:

**Causa 1 — `pivot_longer()` (R) e `melt()` (Python) não produzem a mesma ordem de linhas.**

O `pivot_longer()` do R "desempacota" os 12 meses de cada linha original em sequência (ano→região→Jan...Dez, um bloco de 12 meses de cada vez):

```
2004  Norte     JANEIRO
2004  Norte     FEVEREIRO
...
2004  Norte     DEZEMBRO
2004  Nordeste  JANEIRO
...
```

Já o `pandas.melt()` (usado na conversão) empilha **coluna por coluna**: todos os Janeiros de todos os anos/regiões primeiro, depois todos os Fevereiros, etc.:

```
2004  Norte      JANEIRO
2004  Nordeste   JANEIRO
2004  Sudeste    JANEIRO
...
2005  Norte      JANEIRO
...
```

Isso não é um detalhe cosmético: como a "série" que alimenta o `ts()`/ARIMA é simplesmente o vetor de consumo na ordem em que as linhas aparecem, **as duas linguagens acabam treinando o ARIMA sobre sequências de números completamente diferentes**, mesmo com os mesmos dados de origem.

**Causa 2 — sazonalidade: ligada por padrão no R, desligada no Python.**

No R, `ts(..., frequency = 12)` + `auto.arima()` habilita busca de modelo **sazonal** por padrão. Na conversão Python, o script usa `pmdarima.auto_arima(..., seasonal=False)` — desligando essa busca explicitamente.

**Confirmação experimental:** refiz o treino em Python usando (a) a ordem de linhas do `pivot_longer` (via `stack()` em vez de `melt()`) e (b) sazonalidade habilitada (`seasonal=True, m=12`), exatamente como o R faz. Resultado:

| | RMSE | MAE |
|---|---:|---:|
| R (real) | 6.743.109 | 4.624.574 |
| Python replicando ordem + sazonalidade do R | **6.744.091** | **4.631.830** |
| Diferença | 0,01% | 0,16% |

Praticamente idêntico — confirma que essas duas diferenças, juntas, explicam por completo a inversão de desempenho do ARIMA. Vale notar que esse "bom" resultado do R é, na verdade, um **artefato problemático**: como `dados_treino` mistura as 5 regiões dentro de cada bloco de 12 meses, o `auto.arima()` acaba enxergando "saltos" a cada 12 passos (a troca de região) como se fossem sazonalidade genuína — o que é estatisticamente questionável, mesmo que o número de RMSE pareça bom. Isso é discutido mais na Seção 4.

### 2.3 Random Forest — divergência moderada (~2×), causa provável mas não 100% confirmada

| | RMSE | MAE |
|---|---:|---:|
| R | 2.042.628 | 1.375.926 |
| Python | 1.019.507 | 774.462 |

Diferente do SVM e do ARIMA, aqui não há uma "quebra" — ambos os modelos funcionam e capturam a tendência das regiões (dá para ver comparando as previsões futuras linha a linha, tabela abaixo). A diferença de ~2× provavelmente vem de:

- **Codificação de categorias diferente:** a interface de fórmula do R (`Consumo_MWh ~ .`) deixa o `randomForest()` tratar `REGIAO_GEOGRAFICA` e `Mês` como fatores nativos, podendo particionar múltiplos níveis de uma vez em cada divisão de nó. Na conversão Python, apliquei one-hot encoding (uma coluna binária por região/mês) porque o `scikit-learn` não aceita categorias nativas — isso é estritamente menos expressivo a cada divisão da árvore.
- **Hiperparâmetros padrão diferentes:** `randomForest()` usa `mtry = p/3` (regressão) por padrão; `RandomForestRegressor` usa todas as features em cada split por padrão. `ntree=100` foi igual nos dois scripts, então isso não é fator aqui.

Não testei isolar cada causa (o SVM e o ARIMA já explicavam achados maiores e mais críticos), mas ambas são explicações plausíveis e bem documentadas na literatura de comparação R vs. scikit-learn para Random Forest.

**Exemplo linha a linha (previsão futura, Random Forest):**

| Ano | Região | R | Python | Consumo Normal (histórico) |
|---|---|---:|---:|---:|
| 2026 | Norte | 3.588.377 | 2.637.089 | 2.500.116 |
| 2026 | Sudeste | 17.514.686 | 20.694.110 | 19.638.254 |
| 2027 | Sul | 8.262.777 | 7.129.186 | 6.467.012 |

Interessante notar que, para o Sudeste 2026, a previsão do Python (20,69 milhões) ficou mais próxima do consumo histórico normal (19,64 milhões) do que a do R (17,51 milhões) — consistente com o RMSE geral mais baixo do Random Forest em Python.

---

## 3. Comparação visual — SVM achatado em Python vs. SVM funcional no R

Os gráficos gerados na execução Python confirmam visualmente o problema do SVM: a previsão (linha tracejada) fica achatada, sem distinguir as regiões — exatamente o que os números da Seção 2.1 mostram.

![Comparação SVM em Python](figuras/02_comparacao_svm.png)

No R, pela tabela de valores impressos (`Consumo_SVM` variando de ~3,5 milhões no Norte a ~15,7 milhões no Sudeste), o mesmo gráfico mostraria as linhas tracejadas acompanhando corretamente cada região — não temos a imagem do R (ele gera a saída via `ggplot2` na tela, não salva arquivo), mas os números deixam claro que o comportamento visual seria saudável.

O Random Forest, que não depende de escala, funciona bem nos dois lados:

![Comparação Random Forest em Python](figuras/05_comparacao_random_forest.png)

---

## 4. Limitação de desenho identificada em ambos os scripts (não é bug de conversão)

Tanto no R quanto no Python, a Seção 18 (ARIMA por região, para os "próximos 5 anos") tem um problema conceitual: `h = length(anos_futuros) = 5` é interpretado pelo `forecast()`/`predict()` como "**5 passos à frente da série mensal**", ou seja, **5 meses**, não 5 anos. Como esses 5 valores previstos coincidem em quantidade com as 5 linhas (uma por "ano" 2026–2030) reservadas para aquela região, eles são atribuídos a essas linhas sem que ninguém perceba a inconsistência — o que está rotulado como "previsão para 2030" é, na prática, apenas o 5º mês após o fim dos dados de treino.

Isso explica por que as previsões do ARIMA regional (Figura abaixo) crescem muito pouco de um "ano" para o outro — na real elas quase não avançam no tempo:

![Comparação ARIMA regional](figuras/04_comparacao_arima.png)

Essa limitação **existe igualmente no R e no Python** — não é um problema de tradução, é uma característica (falha) do desenho original do script.

---

## 5. Resumo das causas de divergência

| Componente | Divergência | Causa raiz | Status |
|---|---|---|---|
| Testes estatísticos (Shapiro, Kruskal) | Nenhuma | Mesma fórmula matemática | ✅ Confirmado |
| ARIMA (Seção 13) | RMSE invertido (R pior, Python melhor) | (1) `melt()` ≠ `pivot_longer()` na ordenação das linhas; (2) sazonalidade ligada no R / desligada no Python | ✅ Confirmado experimentalmente (reprodução bateu com 0,01% de diferença) |
| SVM | RMSE 3,7× pior em Python | `e1071::svm()` escalona automaticamente; `sklearn.svm.SVR` não | ✅ Confirmado (modelo Python prevê constante; teste com `StandardScaler` reduz RMSE em 4,5×) |
| Random Forest | RMSE ~2× pior em R | Provável: one-hot encoding (Python) vs. fatores nativos (R) + `mtry` padrão diferente | ⚠️ Explicação plausível, não isolada experimentalmente |
| Horizonte do ARIMA regional (Seção 18) | Previsões "achatadas" | `h` conta meses, mas é tratado como anos — bug do desenho original | ⚠️ Presente em ambas as linguagens, não é problema de conversão |

---

## 6. Recomendações para deixar as duas versões comparáveis "de verdade"

1. **SVM:** adicionar `StandardScaler` em X e y no Python antes de treinar o `SVR`, desfazendo a transformação ao interpretar as previsões — isso é obrigatório para a comparação fazer sentido.
2. **ARIMA (Seção 13):** decidir conscientemente a ordenação da série e a sazonalidade, em vez de herdar isso "de graça" do R. Duas alternativas honestas:
   - Manter a mesma ordem region-block/ano do R (com `stack()` em vez de `melt()`) e sazonalidade ligada, se o objetivo é replicar o comportamento do R; ou
   - Construir uma série verdadeiramente única por região (a forma estatisticamente correta), o que provavelmente supera ambas as versões atuais.
3. **Random Forest:** se a paridade exata com o R importa, testar `mtry`/`max_features` equivalente (`max_features = n_features/3` no `RandomForestRegressor`) para se aproximar do padrão do `randomForest()`.
4. **Seção 18 (ARIMA regional):** corrigir `h` para refletir meses reais (ex.: `h = 60` para 5 anos completos) ou agregar por ano antes de rodar o ARIMA, como já é feito corretamente nas Seções 20–21.

---

## 7. Conclusão

A migração de R para Python **não é "só trocar a sintaxe"** — o exercício de comparar as saídas reais revelou duas divergências de comportamento silenciosas (ordem de linhas no `melt`/`pivot_longer`, e escalonamento automático do SVM no R) que mudam completamente o resultado, a ponto de inverter qual modelo parece "o melhor". A boa notícia é que ambas as causas foram identificadas e confirmadas com precisão, e as correções propostas na Seção 6 são diretas de aplicar.
