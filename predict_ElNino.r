# 1. Selecionar o arquivo de dados manualmente

library(tidyverse)
library(forecast)
library(e1071)
library(randomForest)
library(ggpubr)
library(readxl)
library(zoo)
library(ggplot2)

# Solicitar ao usuário que escolha o arquivo
file_path <- file.choose()
print(paste("Arquivo selecionado:", file_path))


# 2. Carregar os dados do arquivo Excel corretamente

dados <- read_excel(file_path, sheet = "TOTAL")


# 3. Verificar os nomes das colunas

print("Nomes das colunas:")
print(colnames(dados))


# 4. Renomear colunas para evitar erros, se necessário

dados <- dados %>%
  rename(
    ANO_DE_REFERENCIA = "ANO DE REFERÊNCIA",
    REGIAO_GEOGRAFICA = "REGIÃO GEOGRÁFICA"
  )


# 5. Carregar os dados do arquivo Excel corretamente

dados <- read_excel(file_path, sheet = "TOTAL")


# 6. Verificar os nomes das colunas

print("Nomes das colunas:")
print(colnames(dados))


# 7. Renomear colunas para evitar erros, se necessário

dados <- dados %>%
  rename(
    ANO_DE_REFERENCIA = "ANO DE REFERÊNCIA",
    REGIAO_GEOGRAFICA = "REGIÃO GEOGRÁFICA"
  )


# 8. Transformar os dados para formato longo

dados_long <- dados %>%
  pivot_longer(
    cols = -c(ANO_DE_REFERENCIA, REGIAO_GEOGRAFICA),
    names_to = "Mês",
    values_to = "Consumo_MWh"
  )


# 9. Converter os nomes dos meses para fator ordenado

meses_ordenados <- c(
  "JANEIRO",
  "FEVEREIRO",
  "MARÇO",
  "ABRIL",
  "MAIO",
  "JUNHO",
  "JULHO",
  "AGOSTO",
  "SETEMBRO",
  "OUTUBRO",
  "NOVEMBRO",
  "DEZEMBRO"
)

dados_long$Mês <- factor(
  dados_long$Mês,
  levels = meses_ordenados,
  ordered = TRUE
)


# 10. TRATAR VALORES AUSENTES (NAs)

print("Valores ausentes antes da correção:")
print(colSums(is.na(dados_long)))

# Método escolhido: Preencher NAs com a média
dados_long$Consumo_MWh[is.na(dados_long$Consumo_MWh)] <- 
  mean(dados_long$Consumo_MWh, na.rm = TRUE)

# Verificar novamente
print("Valores ausentes depois da correção:")
print(colSums(is.na(dados_long)))

# 11. Teste de Normalidade (Shapiro-Wilk)
shapiro_test <- shapiro.test(dados_long$Consumo_MWh)
print(shapiro_test)

# 12. Escolher o Teste Estatístico adequado
if (shapiro_test$p.value > 0.05) {
  
  print("Os dados seguem distribuição normal. Aplicando Teste t Pareado...")
  
  # Escolher apenas duas regiões para rodar o Teste t Pareado
  dados_t_test <- dados_long %>%
    filter(REGIAO_GEOGRAFICA %in% c("Sudeste", "Nordeste")) %>%
    spread(REGIAO_GEOGRAFICA, Consumo_MWh)
  
  if (all(c("Sudeste", "Nordeste") %in% colnames(dados_t_test))) {
    resultado_t <- t.test(
      dados_t_test$Sudeste,
      dados_t_test$Nordeste,
      paired = TRUE
    )
    print(resultado_t)
  } else {
    print("Erro: Dados insuficientes para Teste t Pareado.")
  }
  
} else {
  
  print("Os dados NÃO seguem distribuição normal.")
  
  if (length(unique(dados_long$REGIAO_GEOGRAFICA)) > 2) {
    
    print("Mais de dois grupos detectados. Aplicando Teste de Kruskal-Wallis...")
    
    kruskal_test <- kruskal.test(
      Consumo_MWh ~ REGIAO_GEOGRAFICA,
      data = dados_long
    )
    
    print(kruskal_test)
    
  } else {
    
    print("Aplicando Teste de Wilcoxon para duas regiões específicas...")
    
    resultado_wilcox <- wilcox.test(
      Consumo_MWh ~ REGIAO_GEOGRAFICA,
      data = dados_long
    )
    
    print(resultado_wilcox)
  }
}

# 13. Criar Modelos Preditivos

# Separar dados para treinamento (80%) e teste (20%)

set.seed(123)

dados_treino <- dados_long %>%
  filter(ANO_DE_REFERENCIA < 2020)

dados_teste <- dados_long %>%
  filter(ANO_DE_REFERENCIA >= 2020)


# TRATAR NAs ANTES DO TREINAMENTO

print("Valores ausentes no conjunto de treino antes da correção:")

print(colSums(is.na(dados_treino)))

dados_treino$Consumo_MWh[is.na(dados_treino$Consumo_MWh)] <-
  mean(dados_treino$Consumo_MWh, na.rm = TRUE)

dados_treino <- na.omit(dados_treino)

print("Valores ausentes no conjunto de treino depois da correção:")

print(colSums(is.na(dados_treino)))

# Modelo ARIMA para Séries Temporais

serie_temporal <- ts(
  dados_treino$Consumo_MWh,
  start = min(dados_treino$ANO_DE_REFERENCIA),
  frequency = 12
)

modelo_arima <- auto.arima(serie_temporal)

previsao_arima <- forecast(
  modelo_arima,
  h = nrow(dados_teste)
)

# Modelo SVM (Máquinas de Vetores de Suporte)

modelo_svm <- svm(
  Consumo_MWh ~ .,
  data = dados_treino,
  kernel = "radial"
)

previsao_svm <- predict(
  modelo_svm,
  newdata = dados_teste
)

# Modelo Random Forest (Árvores de Decisão)

modelo_rf <- randomForest(
  Consumo_MWh ~ .,
  data = dados_treino,
  ntree = 100
)

previsao_rf <- predict(
  modelo_rf,
  newdata = dados_teste
)

# 14. Avaliação dos Modelos (RMSE e MAE)

# Verificar o tamanho das previsões

print(length(dados_teste$Consumo_MWh))

print(length(previsao_arima$mean))

print(length(previsao_svm))

print(length(previsao_rf))

# Ajustar o tamanho das previsões se necessário

previsao_arima <-
  previsao_arima$mean[1:length(dados_teste$Consumo_MWh)]

previsao_svm <-
  previsao_svm[1:length(dados_teste$Consumo_MWh)]

previsao_rf <-
  previsao_rf[1:length(dados_teste$Consumo_MWh)]

# Funções de Erro

rmse <- function(real, previsto)
  sqrt(mean((real - previsto)^2, na.rm = TRUE))

mae <- function(real, previsto)
  mean(abs(real - previsto), na.rm = TRUE)

# Calcular os erros dos modelos

erros <- tibble(
  Modelo = c("ARIMA", "SVM", "Random Forest"),
  
  RMSE = c(
    rmse(dados_teste$Consumo_MWh, previsao_arima),
    rmse(dados_teste$Consumo_MWh, previsao_svm),
    rmse(dados_teste$Consumo_MWh, previsao_rf)
  ),
  
  MAE = c(
    mae(dados_teste$Consumo_MWh, previsao_arima),
    mae(dados_teste$Consumo_MWh, previsao_svm),
    mae(dados_teste$Consumo_MWh, previsao_rf)
  )
)

print(erros)

# 15. Gráfico de Consumo ao longo dos anos por Região

ggplot(
  dados_long,
  aes(
    x = ANO_DE_REFERENCIA,
    y = Consumo_MWh,
    color = REGIAO_GEOGRAFICA
  )
) +
  geom_line() +
  geom_point() +
  labs(
    title = "Evolução do Consumo de Energia Elétrica (2003-2023)",
    x = "Ano",
    y = "Consumo (MWh)"
  ) +
  theme_minimal()

# 16. Comparação do Consumo Habitual vs. Consumo Previsto pelo Modelo SVM

# Criar um dataframe para armazenar as previsões futuras

anos_futuros <- seq(
  max(dados_long$ANO_DE_REFERENCIA) + 1,
  max(dados_long$ANO_DE_REFERENCIA) + 5
)

dados_futuros <- data.frame(
  ANO_DE_REFERENCIA = rep(
    anos_futuros,
    each = length(unique(dados_long$REGIAO_GEOGRAFICA))
  ),
  
  REGIAO_GEOGRAFICA = rep(
    unique(dados_long$REGIAO_GEOGRAFICA),
    times = length(anos_futuros)
  )
)

# Adicionar a variável 'Mês' ao dataframe de previsão

dados_futuros <- dados_futuros %>%
  mutate(
    Mês = factor(
      rep(meses_ordenados, length.out = nrow(dados_futuros)),
      levels = meses_ordenados,
      ordered = TRUE
    )
  )

# Prever com o modelo SVM

dados_futuros$Consumo_SVM <- predict(
  modelo_svm,
  newdata = dados_futuros
)

# Criar um cenário de consumo normal baseado na média histórica

consumo_normal <- dados_long %>%
  group_by(REGIAO_GEOGRAFICA, Mês) %>%
  summarise(
    Consumo_Normal = mean(Consumo_MWh, na.rm = TRUE),
    .groups = "drop"
  )

dados_futuros <- left_join(
  dados_futuros,
  consumo_normal,
  by = c("REGIAO_GEOGRAFICA", "Mês")
)

# Identificar períodos de El Niño onde a previsão SVM diverge
# significativamente do consumo habitual

dados_futuros <- dados_futuros %>%
  mutate(
    Efeito_El_Nino = ifelse(
      abs(Consumo_SVM - Consumo_Normal) / Consumo_Normal > 0.1,
      "El Niño",
      "Normal"
    )
  )

# Exibir comparativo de consumo habitual vs. consumo previsto

print("Comparação entre Consumo Habitual e Consumo Previsto pelo Modelo SVM:")

print(dados_futuros)

# Gráfico Comparativo destacando o efeito El Niño

ggplot(
  dados_futuros,
  aes(x = ANO_DE_REFERENCIA, color = REGIAO_GEOGRAFICA)
) +
  geom_line(
    aes(y = Consumo_Normal, linetype = "Consumo Habitual"),
    size = 1
  ) +
  geom_line(
    aes(y = Consumo_SVM, linetype = "Previsão SVM"),
    size = 1,
    linetype = "dashed"
  ) +  geom_point(
    aes(y = Consumo_SVM, shape = Efeito_El_Nino),
    size = 3
  ) +
  
  scale_shape_manual(
    values = c("El Niño" = 17, "Habitual" = NA)
  ) +
  
  labs(
    title = "Comparação entre Consumo Habitual e Previsão pelo Modelo SVM",
    subtitle = "Destaque para os períodos de impacto do El Niño",
    x = "Ano",
    y = "Consumo Previsto (MWh)",
    color = "Região Geográfica",
    linetype = "Modelo",
    shape = "Efeito El Niño"
  ) +
  
  theme_minimal()

# 17. Distribuição do Consumo de Energia por Mês

# Criar o gráfico de boxplot

ggplot(
  dados_long,
  aes(x = Mês, y = Consumo_MWh, fill = Mês)
) +
  
  geom_boxplot(
    outlier.color = "black",
    outlier.shape = 16,
    outlier.size = 2,
    alpha = 0.6
  ) +
  
  labs(
    title = "Distribuição do Consumo de Energia por Mês",
    x = "Mês",
    y = "Consumo (MWh)"
  ) +
  
  theme_minimal() +
  
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# 18. Comparação do Consumo Habitual vs. Consumo Previsto pelo Modelo de Séries Temporais

# Criar um dataframe para armazenar as previsões futuras

anos_futuros <- seq(
  max(dados_long$ANO_DE_REFERENCIA) + 1,
  max(dados_long$ANO_DE_REFERENCIA) + 5
)

dados_futuros <- data.frame(
  ANO_DE_REFERENCIA = rep(
    anos_futuros,
    each = length(unique(dados_long$REGIAO_GEOGRAFICA))
  ),
  
  REGIAO_GEOGRAFICA = rep(
    unique(dados_long$REGIAO_GEOGRAFICA),
    times = length(anos_futuros)
  )
)

# Adicionar a variável 'Mês' ao dataframe de previsão

dados_futuros <- dados_futuros %>%
  mutate(
    Mês = factor(
      rep(meses_ordenados, length.out = nrow(dados_futuros)),
      levels = meses_ordenados,
      ordered = TRUE
    )
  )

# Prever com o modelo de Séries Temporais ARIMA

for (regiao in unique(dados_long$REGIAO_GEOGRAFICA)) {
  
  serie_temporal <- ts(
    dados_long$Consumo_MWh[
      dados_long$REGIAO_GEOGRAFICA == regiao
    ],
    start = min(dados_long$ANO_DE_REFERENCIA),
    frequency = 12
  )
  
  modelo_arima <- auto.arima(serie_temporal)
  
  previsao_arima_futuro <- forecast(
    modelo_arima,
    h = length(anos_futuros)
  )
  
  dados_futuros$Consumo_ARIMA[
    dados_futuros$REGIAO_GEOGRAFICA == regiao
  ] <- previsao_arima_futuro$mean
}

# Criar um cenário de consumo normal baseado na média histórica

consumo_normal <- dados_long %>%
  group_by(REGIAO_GEOGRAFICA, Mês) %>%
  summarise(
    Consumo_Normal = mean(Consumo_MWh, na.rm = TRUE),
    .groups = "drop"
  )

dados_futuros <- left_join(
  dados_futuros,
  consumo_normal,
  by = c("REGIAO_GEOGRAFICA", "Mês")
)

# Identificar períodos de El Niño onde a previsão ARIMA diverge
# significativamente do consumo normal

dados_futuros <- dados_futuros %>%
  mutate(
    Efeito_El_Nino = ifelse(
      abs(Consumo_ARIMA - Consumo_Normal) / Consumo_Normal > 0.1,
      "El Niño",
      "Habitual"
    )
  )

# Exibir comparativo de consumo habitual vs. consumo previsto

print("Comparação entre Consumo Habitual e Consumo Previsto pelo Modelo de Séries Temporais:")

print(dados_futuros)

# Gráfico Comparativo destacando o efeito El Niño

ggplot(
  dados_futuros,
  aes(
    x = ANO_DE_REFERENCIA,
    color = REGIAO_GEOGRAFICA
  )
) +
  
  geom_line(
    aes(y = Consumo_Normal, linetype = "Consumo Habitual"),
    size = 1
  ) +
  
  geom_line(
    aes(y = Consumo_ARIMA, linetype = "Previsão ARIMA"),
    size = 1,
    linetype = "dashed"
  ) +
  
  geom_point(
    aes(y = Consumo_ARIMA, shape = Efeito_El_Nino),
    size = 3
  ) +
  
  scale_shape_manual(
    values = c("El Niño" = 17, "Habitual" = NA)
  ) +
  
  labs(
    title = "Comparação entre Consumo Habitual e Previsão pelo Modelo de Séries Temporais",
    subtitle = "Destaque para os períodos de impacto do El Niño",
    x = "Ano",
    y = "Consumo Previsto (MWh)",
    color = "Região Geográfica",
    linetype = "Modelo",
    shape = "Efeito El Niño"
  ) +
  theme_minimal()

# 19. Comparação do Consumo Normal vs. Consumo Previsto pelo Modelo Random Forest

# Criar um dataframe para armazenar as previsões futuras

anos_futuros <- seq(
  max(dados_long$ANO_DE_REFERENCIA) + 1,
  max(dados_long$ANO_DE_REFERENCIA) + 5
)

dados_futuros <- data.frame(
  ANO_DE_REFERENCIA = rep(
    anos_futuros,
    each = length(unique(dados_long$REGIAO_GEOGRAFICA))
  ),
  
  REGIAO_GEOGRAFICA = rep(
    unique(dados_long$REGIAO_GEOGRAFICA),
    times = length(anos_futuros)
  )
)

# Adicionar a variável 'Mês' ao dataframe de previsão

dados_futuros <- dados_futuros %>%
  mutate(
    Mês = factor(
      rep(meses_ordenados, length.out = nrow(dados_futuros)),
      levels = meses_ordenados,
      ordered = TRUE
    )
  )

# Prever com o modelo Random Forest

dados_futuros$Consumo_RandomForest <- predict(
  modelo_rf,
  newdata = dados_futuros
)

# Criar um cenário de consumo habitual baseado na média histórica

consumo_normal <- dados_long %>%
  group_by(REGIAO_GEOGRAFICA, Mês) %>%
  summarise(
    Consumo_Normal = mean(Consumo_MWh, na.rm = TRUE),
    .groups = "drop"
  )

dados_futuros <- left_join(
  dados_futuros,
  consumo_normal,
  by = c("REGIAO_GEOGRAFICA", "Mês")
)

# Identificar períodos de El Niño onde a previsão Random Forest diverge
# significativamente do consumo habitual

dados_futuros <- dados_futuros %>%
  mutate(
    Efeito_El_Nino = ifelse(
      abs(Consumo_RandomForest - Consumo_Normal) / Consumo_Normal > 0.1,
      "El Niño",
      "Normal"
    )
  )

# Exibir comparativo de consumo habitual vs. consumo previsto

print("Comparação entre Consumo Habitual e Consumo Previsto pelo Modelo Random Forest:")

print(dados_futuros)

# Gráfico Comparativo destacando o efeito El Niño

ggplot(
  dados_futuros,
  aes(
    x = ANO_DE_REFERENCIA,
    color = REGIAO_GEOGRAFICA
  )
) +
  
  geom_line(
    aes(y = Consumo_Normal, linetype = "Consumo Habitual"),
    size = 1
  ) +
  
  geom_line(
    aes(y = Consumo_RandomForest, linetype = "Previsão Random Forest"),
    size = 1,
    linetype = "dashed"
  ) +
  
  geom_point(
    aes(y = Consumo_RandomForest, shape = Efeito_El_Nino),
    size = 3
  ) +
  
  scale_shape_manual(
    values = c("El Niño" = 17, "Habitual" = NA)
  ) +
  
  labs(
    title = "Comparação entre Consumo Habitual e Previsão pelo Modelo Random Forest",
    subtitle = "Destaque para os períodos de impacto do El Niño",
    x = "Ano",
    y = "Consumo Previsto (MWh)",
    color = "Região Geográfica",
    linetype = "Modelo",
    shape = "Efeito El Niño"
  ) +
  
  theme_minimal()

# 20. Comparação entre Modelos Preditivos para os Próximos 5 Anos,
# Destacando o Efeito El Niño

# Criar um dataframe para armazenar as previsões futuras

anos_futuros <- seq(2024, 2028)

dados_futuros <- data.frame(
  ANO_DE_REFERENCIA = anos_futuros
)

# Criar o dataframe consumo_geral somando o consumo total por ano

consumo_geral <- dados_long %>%
  group_by(ANO_DE_REFERENCIA) %>%
  summarise(
    Consumo_Total = sum(Consumo_MWh, na.rm = TRUE)
  )

# Definir a frequência da série temporal

data_frequency <-
  ifelse(n_distinct(diff(consumo_geral$ANO_DE_REFERENCIA)) > 1, 1, 12)

# Identificar anos com impacto do El Niño

anos_impacto <- c(2025, 2027)  # Exemplo de anos que podem ter impacto significativo

# Prever com o modelo de Séries Temporais (ARIMA)

serie_temporal <- ts(
  consumo_geral$Consumo_Total,
  start = min(consumo_geral$ANO_DE_REFERENCIA),
  frequency = data_frequency
)

modelo_arima <- auto.arima(
  serie_temporal,
  stepwise = FALSE,
  approximation = FALSE
)

previsao_arima <- forecast(
  modelo_arima,
  h = length(anos_futuros)
)

dados_futuros$Consumo_Arima <- previsao_arima$mean

# Prever com o modelo de Máquinas de Vetores de Suporte (SVM)

modelo_svm <- svm(
  Consumo_Total ~ ANO_DE_REFERENCIA,
  data = consumo_geral,
  kernel = "radial"
)

dados_futuros$Consumo_SVM <- predict(
  modelo_svm,
  newdata = dados_futuros
)

# Prever com o modelo Baseado em Árvores de Decisão (Random Forest)

modelo_rf <- randomForest(
  Consumo_Total ~ ANO_DE_REFERENCIA,
  data = consumo_geral,
  ntree = 100
)

dados_futuros$Consumo_DecisionTree <- predict(
  modelo_rf,
  newdata = dados_futuros
)

# Exibir previsões comparando apenas os modelos preditivos,
# destacando anos de impacto do El Niño

ggplot(dados_futuros, aes(x = ANO_DE_REFERENCIA)) +
  
  geom_line(
    aes(y = Consumo_Arima, color = "Previsão ARIMA"),
    size = 1,
    linetype = "dashed"
  ) +
  
  geom_line(
    aes(y = Consumo_SVM, color = "Previsão SVM"),
    size = 1,
    linetype = "dotted"
  ) +
  
  geom_line(
    aes(y = Consumo_DecisionTree, color = "Previsão Decision Trees"),
    size = 1,
    linetype = "dotdash"
  ) +
  
  geom_point(
    data = dados_futuros %>%
      filter(ANO_DE_REFERENCIA %in% anos_impacto),
    aes(
      x = ANO_DE_REFERENCIA,
      y = Consumo_Arima,
      color = "Impacto ARIMA"
    ),
    shape = 17,
    size = 3
  ) +
  
  geom_point(
    data = dados_futuros %>%
      filter(ANO_DE_REFERENCIA %in% anos_impacto),
    aes(
      x = ANO_DE_REFERENCIA,
      y = Consumo_SVM,
      color = "Impacto SVM"
    ),
    shape = 17,
    size = 3
  ) +
  
  geom_point(
    data = dados_futuros %>%
      filter(ANO_DE_REFERENCIA %in% anos_impacto),
    aes(
      x = ANO_DE_REFERENCIA,
      y = Consumo_DecisionTree,
      color = "Impacto Decision Trees"
    ),
    shape = 17,
    size = 3
  ) +
  
  labs(
    title = "Comparação entre Modelos Preditivos para 2024-2028 Destacando o Impacto do El Niño",
    x = "Ano",
    y = "Consumo Total (MWh)",
    color = "Legenda"
  ) +
  
  theme_minimal()

# 21. Comparação entre Modelos Preditivos para os Próximos 5 Anos,
# Destacando o Efeito El Niño e Consumo Normal

# Criar um dataframe para armazenar as previsões futuras

anos_futuros <- seq(2024, 2028)

dados_futuros <- data.frame(
  ANO_DE_REFERENCIA = anos_futuros
)

# Criar o dataframe consumo_geral somando o consumo total por ano

consumo_geral <- dados_long %>%
  group_by(ANO_DE_REFERENCIA) %>%
  summarise(
    Consumo_Total = sum(Consumo_MWh, na.rm = TRUE)
  )

# Definir a frequência da série temporal

data_frequency <-
  ifelse(
    n_distinct(diff(consumo_geral$ANO_DE_REFERENCIA)) > 1,
    1,
    12
  )

# Identificar anos com impacto do El Niño

anos_impacto <- c(2025, 2027)  # Exemplo de anos que podem ter impacto significativo

# Calcular o consumo normal utilizando Suavização Exponencial Simples (SES)
# para capturar tendências

modelo_ses <- ses(
  ts(
    consumo_geral$Consumo_Total,
    start = min(consumo_geral$ANO_DE_REFERENCIA),
    frequency = data_frequency
  ),
  h = length(anos_futuros)
)

dados_futuros$Consumo_Normal <- modelo_ses$mean
# Prever com o modelo de Séries Temporais (ARIMA)

serie_temporal <- ts(
  consumo_geral$Consumo_Total,
  start = min(consumo_geral$ANO_DE_REFERENCIA),
  frequency = data_frequency
)

modelo_arima <- auto.arima(
  serie_temporal,
  stepwise = FALSE,
  approximation = FALSE
)

previsao_arima <- forecast(
  modelo_arima,
  h = length(anos_futuros)
)

dados_futuros$Consumo_Arima <- previsao_arima$mean

# Prever com o modelo de Máquinas de Vetores de Suporte (SVM)

modelo_svm <- svm(
  Consumo_Total ~ ANO_DE_REFERENCIA,
  data = consumo_geral,
  kernel = "radial"
)

dados_futuros$Consumo_SVM <- predict(
  modelo_svm,
  newdata = dados_futuros
)

# Prever com o modelo Baseado em Árvores de Decisão (Random Forest)

modelo_rf <- randomForest(
  Consumo_Total ~ ANO_DE_REFERENCIA,
  data = consumo_geral,
  ntree = 100
)

dados_futuros$Consumo_DecisionTree <- predict(
  modelo_rf,
  newdata = dados_futuros
)

# Exibir previsões comparando apenas os modelos preditivos,
# destacando anos de impacto do El Niño e adicionando Consumo Habitual

ggplot(
  dados_futuros,
  aes(x = ANO_DE_REFERENCIA)
) +
  
  geom_line(
    aes(y = Consumo_Normal, color = "Consumo Habitual"),
    size = 1
  ) +
  
  geom_line(
    aes(y = Consumo_Arima, color = "Previsão ARIMA"),
    size = 1,
    linetype = "dashed"
  ) +
  
  geom_line(
    aes(y = Consumo_SVM, color = "Previsão SVM"),
    size = 1,
    linetype = "dotted"
  ) +
  
  geom_line(
    aes(y = Consumo_DecisionTree, color = "Previsão Decision Trees"),
    size = 1,
    linetype = "dotdash"
  ) +
  
  geom_point(
    data = dados_futuros %>%
      filter(ANO_DE_REFERENCIA %in% anos_impacto),
    aes(
      x = ANO_DE_REFERENCIA,
      y = Consumo_Arima,
      color = "Impacto ARIMA"
    ),
    shape = 17,
    size = 3
  ) +
  
  geom_point(
    data = dados_futuros %>%
      filter(ANO_DE_REFERENCIA %in% anos_impacto),
    
    aes(
      x = ANO_DE_REFERENCIA,
      y = Consumo_SVM,
      color = "Impacto SVM"
    ),
    shape = 17,
    size = 3
  ) +
  
  geom_point(
    data = dados_futuros %>%
      filter(ANO_DE_REFERENCIA %in% anos_impacto),
    
    aes(
      x = ANO_DE_REFERENCIA,
      y = Consumo_DecisionTree,
      color = "Impacto Decision Trees"
    ),
    shape = 17,
    size = 3
  ) +
  
  labs(
    title = "Comparação entre Modelos Preditivos para 2024-2028 Destacando o Impacto do El Niño e Consumo Habitual",
    x = "Ano",
    y = "Consumo Total (MWh)",
    color = "Legenda"
  ) +
  
  theme_minimal()
