# -*- coding: utf-8 -*-
"""
Previsão de Consumo de Energia Elétrica e Efeito do El Niño
Conversão do script original em R para Python.

Dependências (instalar com pip):
    pip install pandas numpy scipy scikit-learn statsmodels pmdarima
    openpyxl matplotlib seaborn

Observação sobre o ARIMA:
    O R usa forecast::auto.arima(), que escolhe automaticamente (p,d,q).
    O equivalente mais próximo em Python é pmdarima.auto_arima(). Caso o
    pacote não esteja disponível, o script cai para um ARIMA(1,1,1) fixo
    via statsmodels como alternativa.
"""

import warnings
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except ImportError:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_PMDARIMA = False

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

MESES_ORDENADOS = [
    "JANEIRO", "FEVEREIRO", "MARÇO", "ABRIL", "MAIO", "JUNHO",
    "JULHO", "AGOSTO", "SETEMBRO", "OUTUBRO", "NOVEMBRO", "DEZEMBRO",
]


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def rmse(real, previsto):
    real = np.asarray(real, dtype=float)
    previsto = np.asarray(previsto, dtype=float)
    return np.sqrt(np.nanmean((real - previsto) ** 2))


def mae(real, previsto):
    real = np.asarray(real, dtype=float)
    previsto = np.asarray(previsto, dtype=float)
    return np.nanmean(np.abs(real - previsto))


def rodar_auto_arima(serie, h):
    """Ajusta um ARIMA e devolve a previsão com h passos à frente."""
    if HAS_PMDARIMA:
        modelo = auto_arima(serie, seasonal=False, suppress_warnings=True)
        previsao = modelo.predict(n_periods=h)
        return np.asarray(previsao)
    else:
        modelo = ARIMA(serie, order=(1, 1, 1)).fit()
        previsao = modelo.forecast(steps=h)
        return np.asarray(previsao)


def preparar_features(df, colunas_categoricas, categorias_referencia):
    """
    Faz one-hot encoding das colunas categóricas usando um conjunto FIXO de
    categorias (categorias_referencia), garantindo que o dataframe de treino
    e o de previsão futura fiquem com as mesmas colunas — equivalente ao que
    o R faz automaticamente com 'factor(levels = ...)' dentro de svm()/randomForest().
    """
    df = df.copy()
    for col in colunas_categoricas:
        df[col] = pd.Categorical(df[col], categories=categorias_referencia[col])
    return pd.get_dummies(df, columns=colunas_categoricas, drop_first=False)


def criar_dados_futuros(dados_long, meses_ordenados, n_anos=5):
    """Cria o dataframe de anos futuros x regiões x meses, igual ao R."""
    ano_max = dados_long["ANO_DE_REFERENCIA"].max()
    anos_futuros = list(range(ano_max + 1, ano_max + 1 + n_anos))
    regioes = dados_long["REGIAO_GEOGRAFICA"].unique().tolist()

    linhas = [
        {"ANO_DE_REFERENCIA": ano, "REGIAO_GEOGRAFICA": regiao}
        for ano in anos_futuros
        for regiao in regioes
    ]
    dados_futuros = pd.DataFrame(linhas)

    # Repetir o ciclo de meses para preencher todas as linhas (como rep(..., length.out=))
    n = len(dados_futuros)
    meses_repetidos = [meses_ordenados[i % len(meses_ordenados)] for i in range(n)]
    dados_futuros["Mês"] = pd.Categorical(
        meses_repetidos, categories=meses_ordenados, ordered=True
    )
    return dados_futuros, anos_futuros, regioes


def consumo_normal_por_regiao_mes(dados_long):
    return (
        dados_long.groupby(["REGIAO_GEOGRAFICA", "Mês"], observed=True)["Consumo_MWh"]
        .mean()
        .reset_index()
        .rename(columns={"Consumo_MWh": "Consumo_Normal"})
    )


def marcar_efeito_el_nino(df, col_previsto, col_normal="Consumo_Normal",
                           rotulo_evento="El Niño", rotulo_normal="Normal"):
    df = df.copy()
    diff_relativa = (df[col_previsto] - df[col_normal]).abs() / df[col_normal]
    df["Efeito_El_Nino"] = np.where(diff_relativa > 0.1, rotulo_evento, rotulo_normal)
    return df


# ---------------------------------------------------------------------------
# 1. Selecionar o arquivo de dados manualmente
# ---------------------------------------------------------------------------

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Selecione o arquivo de dados (Excel)",
    filetypes=[("Excel files", "*.xlsx *.xls")],
)
if not file_path:
    raise SystemExit("Nenhum arquivo selecionado. Encerrando.")
print(f"Arquivo selecionado: {file_path}")


# ---------------------------------------------------------------------------
# 2-4. Carregar os dados do Excel e renomear colunas
# ---------------------------------------------------------------------------

dados = pd.read_excel(file_path, sheet_name="TOTAL")

print("Nomes das colunas:")
print(dados.columns.tolist())

dados = dados.rename(
    columns={
        "ANO DE REFERÊNCIA": "ANO_DE_REFERENCIA",
        "REGIÃO GEOGRÁFICA": "REGIAO_GEOGRAFICA",
    }
)


# ---------------------------------------------------------------------------
# 8. Transformar os dados para formato longo (pivot_longer -> melt)
# ---------------------------------------------------------------------------

dados_long = dados.melt(
    id_vars=["ANO_DE_REFERENCIA", "REGIAO_GEOGRAFICA"],
    var_name="Mês",
    value_name="Consumo_MWh",
)


# ---------------------------------------------------------------------------
# 9. Converter os nomes dos meses para categoria ordenada
# ---------------------------------------------------------------------------

dados_long["Mês"] = pd.Categorical(
    dados_long["Mês"], categories=MESES_ORDENADOS, ordered=True
)


# ---------------------------------------------------------------------------
# 10. Tratar valores ausentes (NAs)
# ---------------------------------------------------------------------------

print("Valores ausentes antes da correção:")
print(dados_long.isna().sum())

media_consumo = dados_long["Consumo_MWh"].mean()
dados_long["Consumo_MWh"] = dados_long["Consumo_MWh"].fillna(media_consumo)

print("Valores ausentes depois da correção:")
print(dados_long.isna().sum())


# ---------------------------------------------------------------------------
# 11. Teste de Normalidade (Shapiro-Wilk)
# ---------------------------------------------------------------------------

shapiro_stat, shapiro_p = stats.shapiro(dados_long["Consumo_MWh"])
print(f"Shapiro-Wilk: estatística={shapiro_stat:.4f}, p-valor={shapiro_p:.4g}")


# ---------------------------------------------------------------------------
# 12. Escolher o teste estatístico adequado
# ---------------------------------------------------------------------------

if shapiro_p > 0.05:
    print("Os dados seguem distribuição normal. Aplicando Teste t Pareado...")

    dados_t_test = (
        dados_long[dados_long["REGIAO_GEOGRAFICA"].isin(["Sudeste", "Nordeste"])]
        .pivot_table(
            index=["ANO_DE_REFERENCIA", "Mês"],
            columns="REGIAO_GEOGRAFICA",
            values="Consumo_MWh",
        )
        .reset_index()
    )

    if {"Sudeste", "Nordeste"}.issubset(dados_t_test.columns):
        resultado_t = stats.ttest_rel(
            dados_t_test["Sudeste"], dados_t_test["Nordeste"], nan_policy="omit"
        )
        print(resultado_t)
    else:
        print("Erro: Dados insuficientes para Teste t Pareado.")

else:
    print("Os dados NÃO seguem distribuição normal.")

    if dados_long["REGIAO_GEOGRAFICA"].nunique() > 2:
        print("Mais de dois grupos detectados. Aplicando Teste de Kruskal-Wallis...")
        grupos = [
            grupo["Consumo_MWh"].values
            for _, grupo in dados_long.groupby("REGIAO_GEOGRAFICA")
        ]
        kruskal_stat, kruskal_p = stats.kruskal(*grupos)
        print(f"Kruskal-Wallis: estatística={kruskal_stat:.4f}, p-valor={kruskal_p:.4g}")
    else:
        print("Aplicando Teste de Wilcoxon (Mann-Whitney) para duas regiões específicas...")
        regioes_unicas = dados_long["REGIAO_GEOGRAFICA"].unique()
        grupo_a = dados_long.loc[
            dados_long["REGIAO_GEOGRAFICA"] == regioes_unicas[0], "Consumo_MWh"
        ]
        grupo_b = dados_long.loc[
            dados_long["REGIAO_GEOGRAFICA"] == regioes_unicas[1], "Consumo_MWh"
        ]
        wilcox_stat, wilcox_p = stats.mannwhitneyu(grupo_a, grupo_b)
        print(f"Wilcoxon/Mann-Whitney: estatística={wilcox_stat:.4f}, p-valor={wilcox_p:.4g}")


# ---------------------------------------------------------------------------
# 13. Criar modelos preditivos (treino 80% / teste 20% por ano)
# ---------------------------------------------------------------------------

np.random.seed(123)

dados_treino = dados_long[dados_long["ANO_DE_REFERENCIA"] < 2020].copy()
dados_teste = dados_long[dados_long["ANO_DE_REFERENCIA"] >= 2020].copy()

print("Valores ausentes no conjunto de treino antes da correção:")
print(dados_treino.isna().sum())

dados_treino["Consumo_MWh"] = dados_treino["Consumo_MWh"].fillna(
    dados_treino["Consumo_MWh"].mean()
)
dados_treino = dados_treino.dropna()

print("Valores ausentes no conjunto de treino depois da correção:")
print(dados_treino.isna().sum())

# Categorias de referência para one-hot encoding consistente entre treino/teste/futuro
CATEGORIAS_REF = {
    "REGIAO_GEOGRAFICA": sorted(dados_long["REGIAO_GEOGRAFICA"].unique().tolist()),
    "Mês": MESES_ORDENADOS,
}

# --- Modelo ARIMA para séries temporais (série agregada do treino) ---
serie_treino = dados_treino.sort_values(
    ["ANO_DE_REFERENCIA", "Mês"]
)["Consumo_MWh"].values

previsao_arima = rodar_auto_arima(serie_treino, h=len(dados_teste))

# --- Modelo SVM (equivalente a Consumo_MWh ~ .) ---
X_treino = preparar_features(
    dados_treino[["ANO_DE_REFERENCIA", "REGIAO_GEOGRAFICA", "Mês"]],
    ["REGIAO_GEOGRAFICA", "Mês"],
    CATEGORIAS_REF,
)
y_treino = dados_treino["Consumo_MWh"].values

X_teste = preparar_features(
    dados_teste[["ANO_DE_REFERENCIA", "REGIAO_GEOGRAFICA", "Mês"]],
    ["REGIAO_GEOGRAFICA", "Mês"],
    CATEGORIAS_REF,
)

modelo_svm = SVR(kernel="rbf")
modelo_svm.fit(X_treino, y_treino)
previsao_svm = modelo_svm.predict(X_teste)

# --- Modelo Random Forest ---
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=123)
modelo_rf.fit(X_treino, y_treino)
previsao_rf = modelo_rf.predict(X_teste)


# ---------------------------------------------------------------------------
# 14. Avaliação dos modelos (RMSE e MAE)
# ---------------------------------------------------------------------------

print(len(dados_teste["Consumo_MWh"]))
print(len(previsao_arima))
print(len(previsao_svm))
print(len(previsao_rf))

n_teste = len(dados_teste["Consumo_MWh"])
previsao_arima = previsao_arima[:n_teste]
previsao_svm = previsao_svm[:n_teste]
previsao_rf = previsao_rf[:n_teste]

erros = pd.DataFrame(
    {
        "Modelo": ["ARIMA", "SVM", "Random Forest"],
        "RMSE": [
            rmse(dados_teste["Consumo_MWh"], previsao_arima),
            rmse(dados_teste["Consumo_MWh"], previsao_svm),
            rmse(dados_teste["Consumo_MWh"], previsao_rf),
        ],
        "MAE": [
            mae(dados_teste["Consumo_MWh"], previsao_arima),
            mae(dados_teste["Consumo_MWh"], previsao_svm),
            mae(dados_teste["Consumo_MWh"], previsao_rf),
        ],
    }
)
print(erros)


# ---------------------------------------------------------------------------
# 15. Gráfico de consumo ao longo dos anos por região
# ---------------------------------------------------------------------------

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=dados_long, x="ANO_DE_REFERENCIA", y="Consumo_MWh",
    hue="REGIAO_GEOGRAFICA", marker="o", errorbar=None,
)
plt.title("Evolução do Consumo de Energia Elétrica (2003-2023)")
plt.xlabel("Ano")
plt.ylabel("Consumo (MWh)")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# 16. Comparação do Consumo Habitual vs. Previsto pelo Modelo SVM
# ---------------------------------------------------------------------------

dados_futuros, anos_futuros, regioes = criar_dados_futuros(dados_long, MESES_ORDENADOS)

X_futuro = preparar_features(
    dados_futuros[["ANO_DE_REFERENCIA", "REGIAO_GEOGRAFICA", "Mês"]],
    ["REGIAO_GEOGRAFICA", "Mês"],
    CATEGORIAS_REF,
)
dados_futuros["Consumo_SVM"] = modelo_svm.predict(X_futuro)

consumo_normal = consumo_normal_por_regiao_mes(dados_long)
dados_futuros = dados_futuros.merge(
    consumo_normal, on=["REGIAO_GEOGRAFICA", "Mês"], how="left"
)

dados_futuros = marcar_efeito_el_nino(
    dados_futuros, "Consumo_SVM", rotulo_evento="El Niño", rotulo_normal="Normal"
)

print("Comparação entre Consumo Habitual e Consumo Previsto pelo Modelo SVM:")
print(dados_futuros)

plt.figure(figsize=(10, 6))
ax = plt.gca()
for regiao, grupo in dados_futuros.groupby("REGIAO_GEOGRAFICA"):
    grupo = grupo.sort_values("ANO_DE_REFERENCIA")
    cor = ax.plot(grupo["ANO_DE_REFERENCIA"], grupo["Consumo_Normal"],
                   label=f"{regiao} - Habitual")[0].get_color()
    ax.plot(grupo["ANO_DE_REFERENCIA"], grupo["Consumo_SVM"], "--",
            color=cor, label=f"{regiao} - Previsão SVM")
    destaque = grupo[grupo["Efeito_El_Nino"] == "El Niño"]
    ax.scatter(destaque["ANO_DE_REFERENCIA"], destaque["Consumo_SVM"],
               marker="^", s=80, color=cor)
plt.title("Comparação entre Consumo Habitual e Previsão pelo Modelo SVM")
plt.suptitle("Destaque para os períodos de impacto do El Niño", y=0.94, fontsize=10)
plt.xlabel("Ano")
plt.ylabel("Consumo Previsto (MWh)")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# 17. Distribuição do Consumo de Energia por Mês (boxplot)
# ---------------------------------------------------------------------------

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=dados_long, x="Mês", y="Consumo_MWh",
    hue="Mês", order=MESES_ORDENADOS, legend=False,
)
plt.title("Distribuição do Consumo de Energia por Mês")
plt.xlabel("Mês")
plt.ylabel("Consumo (MWh)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# 18. Comparação do Consumo Habitual vs. Previsto pelo Modelo de Séries Temporais
# ---------------------------------------------------------------------------

dados_futuros, anos_futuros, regioes = criar_dados_futuros(dados_long, MESES_ORDENADOS)
dados_futuros["Consumo_ARIMA"] = np.nan

for regiao in dados_long["REGIAO_GEOGRAFICA"].unique():
    serie_regiao = dados_long.loc[
        dados_long["REGIAO_GEOGRAFICA"] == regiao
    ].sort_values(["ANO_DE_REFERENCIA", "Mês"])["Consumo_MWh"].values

    previsao_arima_futuro = rodar_auto_arima(serie_regiao, h=len(anos_futuros))

    idx = dados_futuros["REGIAO_GEOGRAFICA"] == regiao
    # distribui a previsão nas linhas dessa região, na ordem em que aparecem
    dados_futuros.loc[idx, "Consumo_ARIMA"] = np.resize(
        previsao_arima_futuro, idx.sum()
    )

consumo_normal = consumo_normal_por_regiao_mes(dados_long)
dados_futuros = dados_futuros.merge(
    consumo_normal, on=["REGIAO_GEOGRAFICA", "Mês"], how="left"
)

dados_futuros = marcar_efeito_el_nino(
    dados_futuros, "Consumo_ARIMA", rotulo_evento="El Niño", rotulo_normal="Habitual"
)

print("Comparação entre Consumo Habitual e Consumo Previsto pelo Modelo de Séries Temporais:")
print(dados_futuros)

plt.figure(figsize=(10, 6))
ax = plt.gca()
for regiao, grupo in dados_futuros.groupby("REGIAO_GEOGRAFICA"):
    grupo = grupo.sort_values("ANO_DE_REFERENCIA")
    cor = ax.plot(grupo["ANO_DE_REFERENCIA"], grupo["Consumo_Normal"],
                   label=f"{regiao} - Habitual")[0].get_color()
    ax.plot(grupo["ANO_DE_REFERENCIA"], grupo["Consumo_ARIMA"], "--",
            color=cor, label=f"{regiao} - Previsão ARIMA")
    destaque = grupo[grupo["Efeito_El_Nino"] == "El Niño"]
    ax.scatter(destaque["ANO_DE_REFERENCIA"], destaque["Consumo_ARIMA"],
               marker="^", s=80, color=cor)
plt.title("Comparação entre Consumo Habitual e Previsão pelo Modelo de Séries Temporais")
plt.suptitle("Destaque para os períodos de impacto do El Niño", y=0.94, fontsize=10)
plt.xlabel("Ano")
plt.ylabel("Consumo Previsto (MWh)")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# 19. Comparação do Consumo Normal vs. Previsto pelo Modelo Random Forest
# ---------------------------------------------------------------------------

dados_futuros, anos_futuros, regioes = criar_dados_futuros(dados_long, MESES_ORDENADOS)

X_futuro = preparar_features(
    dados_futuros[["ANO_DE_REFERENCIA", "REGIAO_GEOGRAFICA", "Mês"]],
    ["REGIAO_GEOGRAFICA", "Mês"],
    CATEGORIAS_REF,
)
dados_futuros["Consumo_RandomForest"] = modelo_rf.predict(X_futuro)

consumo_normal = consumo_normal_por_regiao_mes(dados_long)
dados_futuros = dados_futuros.merge(
    consumo_normal, on=["REGIAO_GEOGRAFICA", "Mês"], how="left"
)

dados_futuros = marcar_efeito_el_nino(
    dados_futuros, "Consumo_RandomForest", rotulo_evento="El Niño", rotulo_normal="Normal"
)

print("Comparação entre Consumo Habitual e Consumo Previsto pelo Modelo Random Forest:")
print(dados_futuros)

plt.figure(figsize=(10, 6))
ax = plt.gca()
for regiao, grupo in dados_futuros.groupby("REGIAO_GEOGRAFICA"):
    grupo = grupo.sort_values("ANO_DE_REFERENCIA")
    cor = ax.plot(grupo["ANO_DE_REFERENCIA"], grupo["Consumo_Normal"],
                   label=f"{regiao} - Habitual")[0].get_color()
    ax.plot(grupo["ANO_DE_REFERENCIA"], grupo["Consumo_RandomForest"], "--",
            color=cor, label=f"{regiao} - Previsão Random Forest")
    destaque = grupo[grupo["Efeito_El_Nino"] == "El Niño"]
    ax.scatter(destaque["ANO_DE_REFERENCIA"], destaque["Consumo_RandomForest"],
               marker="^", s=80, color=cor)
plt.title("Comparação entre Consumo Habitual e Previsão pelo Modelo Random Forest")
plt.suptitle("Destaque para os períodos de impacto do El Niño", y=0.94, fontsize=10)
plt.xlabel("Ano")
plt.ylabel("Consumo Previsto (MWh)")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# 20. Comparação entre modelos preditivos (2024-2028), consumo nacional agregado
# ---------------------------------------------------------------------------

anos_futuros_5 = list(range(2024, 2029))
dados_futuros_nac = pd.DataFrame({"ANO_DE_REFERENCIA": anos_futuros_5})

consumo_geral = (
    dados_long.groupby("ANO_DE_REFERENCIA")["Consumo_MWh"]
    .sum()
    .reset_index()
    .rename(columns={"Consumo_MWh": "Consumo_Total"})
    .sort_values("ANO_DE_REFERENCIA")
)

anos_impacto = [2025, 2027]  # anos com possível impacto significativo (exemplo)

# ARIMA
previsao_arima_nac = rodar_auto_arima(
    consumo_geral["Consumo_Total"].values, h=len(anos_futuros_5)
)
dados_futuros_nac["Consumo_Arima"] = previsao_arima_nac

# SVM (Consumo_Total ~ ANO_DE_REFERENCIA)
X_treino_nac = consumo_geral[["ANO_DE_REFERENCIA"]].values
y_treino_nac = consumo_geral["Consumo_Total"].values
X_futuro_nac = dados_futuros_nac[["ANO_DE_REFERENCIA"]].values

modelo_svm_nac = SVR(kernel="rbf")
modelo_svm_nac.fit(X_treino_nac, y_treino_nac)
dados_futuros_nac["Consumo_SVM"] = modelo_svm_nac.predict(X_futuro_nac)

# Random Forest (Consumo_Total ~ ANO_DE_REFERENCIA)
modelo_rf_nac = RandomForestRegressor(n_estimators=100, random_state=123)
modelo_rf_nac.fit(X_treino_nac, y_treino_nac)
dados_futuros_nac["Consumo_DecisionTree"] = modelo_rf_nac.predict(X_futuro_nac)

plt.figure(figsize=(10, 6))
plt.plot(dados_futuros_nac["ANO_DE_REFERENCIA"], dados_futuros_nac["Consumo_Arima"],
          "--", label="Previsão ARIMA")
plt.plot(dados_futuros_nac["ANO_DE_REFERENCIA"], dados_futuros_nac["Consumo_SVM"],
          ":", label="Previsão SVM")
plt.plot(dados_futuros_nac["ANO_DE_REFERENCIA"], dados_futuros_nac["Consumo_DecisionTree"],
          "-.", label="Previsão Decision Trees")

destaque = dados_futuros_nac[dados_futuros_nac["ANO_DE_REFERENCIA"].isin(anos_impacto)]
plt.scatter(destaque["ANO_DE_REFERENCIA"], destaque["Consumo_Arima"],
            marker="^", s=80, label="Impacto ARIMA")
plt.scatter(destaque["ANO_DE_REFERENCIA"], destaque["Consumo_SVM"],
            marker="^", s=80, label="Impacto SVM")
plt.scatter(destaque["ANO_DE_REFERENCIA"], destaque["Consumo_DecisionTree"],
            marker="^", s=80, label="Impacto Decision Trees")

plt.title("Comparação entre Modelos Preditivos para 2024-2028\nDestacando o Impacto do El Niño")
plt.xlabel("Ano")
plt.ylabel("Consumo Total (MWh)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# 21. Comparação entre modelos preditivos (2024-2028) + Consumo Habitual (SES)
# ---------------------------------------------------------------------------

dados_futuros_nac2 = pd.DataFrame({"ANO_DE_REFERENCIA": anos_futuros_5})

# Consumo habitual via Suavização Exponencial Simples (SES)
modelo_ses = SimpleExpSmoothing(
    consumo_geral["Consumo_Total"].values, initialization_method="estimated"
).fit()
dados_futuros_nac2["Consumo_Normal"] = modelo_ses.forecast(len(anos_futuros_5))

# ARIMA
previsao_arima_nac2 = rodar_auto_arima(
    consumo_geral["Consumo_Total"].values, h=len(anos_futuros_5)
)
dados_futuros_nac2["Consumo_Arima"] = previsao_arima_nac2

# SVM
modelo_svm_nac2 = SVR(kernel="rbf")
modelo_svm_nac2.fit(X_treino_nac, y_treino_nac)
dados_futuros_nac2["Consumo_SVM"] = modelo_svm_nac2.predict(
    dados_futuros_nac2[["ANO_DE_REFERENCIA"]].values
)

# Random Forest
modelo_rf_nac2 = RandomForestRegressor(n_estimators=100, random_state=123)
modelo_rf_nac2.fit(X_treino_nac, y_treino_nac)
dados_futuros_nac2["Consumo_DecisionTree"] = modelo_rf_nac2.predict(
    dados_futuros_nac2[["ANO_DE_REFERENCIA"]].values
)

plt.figure(figsize=(10, 6))
plt.plot(dados_futuros_nac2["ANO_DE_REFERENCIA"], dados_futuros_nac2["Consumo_Normal"],
          label="Consumo Habitual")
plt.plot(dados_futuros_nac2["ANO_DE_REFERENCIA"], dados_futuros_nac2["Consumo_Arima"],
          "--", label="Previsão ARIMA")
plt.plot(dados_futuros_nac2["ANO_DE_REFERENCIA"], dados_futuros_nac2["Consumo_SVM"],
          ":", label="Previsão SVM")
plt.plot(dados_futuros_nac2["ANO_DE_REFERENCIA"], dados_futuros_nac2["Consumo_DecisionTree"],
          "-.", label="Previsão Decision Trees")

destaque2 = dados_futuros_nac2[dados_futuros_nac2["ANO_DE_REFERENCIA"].isin(anos_impacto)]
plt.scatter(destaque2["ANO_DE_REFERENCIA"], destaque2["Consumo_Arima"],
            marker="^", s=80, label="Impacto ARIMA")
plt.scatter(destaque2["ANO_DE_REFERENCIA"], destaque2["Consumo_SVM"],
            marker="^", s=80, label="Impacto SVM")
plt.scatter(destaque2["ANO_DE_REFERENCIA"], destaque2["Consumo_DecisionTree"],
            marker="^", s=80, label="Impacto Decision Trees")

plt.title(
    "Comparação entre Modelos Preditivos para 2024-2028\n"
    "Destacando o Impacto do El Niño e Consumo Habitual"
)
plt.xlabel("Ano")
plt.ylabel("Consumo Total (MWh)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()