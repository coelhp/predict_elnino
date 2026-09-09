# Comparação de Resultados — R vs. Python (versão corrigida)

Comparação direta entre a saída real de `predict_elnino.R` e a saída real de `predict_elnino_v2.py`, ambos rodados sobre a mesma base de dados (`CONSUMO_MENSAL_DE_ENERGIA_ELÉTRICA_POR_CLASSE.xlsx`).

---

## 1. Testes estatísticos

| Teste | R | Python | Diferença |
|---|---|---|---|
| Shapiro-Wilk (estatística) | 0,78313 | 0,7831 | Nenhuma relevante |
| Kruskal-Wallis (estatística) | 1141,5 | 1141,5465 | Nenhuma relevante |
| Consumo médio histórico (ex.: Norte/Janeiro) | 2.500.116 | 2.500.116 | **Idêntico** |

---

## 2. RMSE e MAE (conjunto de teste, anos ≥ 2020)

| Modelo | RMSE (R) | RMSE (Python) | Diferença | MAE (R) | MAE (Python) |
|---|---:|---:|---:|---:|---:|
| ARIMA | 6.743.109 | 6.744.091 | **0,01%** | 4.624.574 | 4.631.830 |
| SVM | 2.017.963 | 1.638.563 | -18,8% | 1.251.186 | 1.026.021 |
| Random Forest | 2.042.628 | 1.038.021 | **-49,2%** | 1.375.926 | 787.078 |

- **ARIMA:** praticamente idêntico entre as duas linguagens.
- **SVM:** Python ficou com erro 18,8% menor que o R, mais próximo, mas não igual.
- **Random Forest:** Python ficou com erro 49,2% menor que o R, a maior divergência das três.

---

## 3. Previsão futura — Modelo SVM

| Ano | Região | Mês | Consumo_SVM (R) | Consumo_SVM (Python) | Dif. relativa |
|---|---|---|---:|---:|---:|
| 2026 | Norte | Janeiro | 3.509.437 | 3.958.178 | +12,8% |
| 2026 | Nordeste | Fevereiro | 6.530.542 | 7.186.508 | +10,0% |
| 2026 | Sudeste | Março | 15.736.801 | 17.329.030 | +10,1% |
| 2026 | Sul | Abril | 6.932.098 | 7.577.347 | +9,3% |
| 2026 | Centro-Oeste | Maio | 4.008.238 | 4.581.466 | +14,3% |
| 2027 | Sudeste | Agosto | 14.390.395 | 14.913.570 | +3,6% |
| 2028 | Sudeste | Janeiro | 14.342.998 | 16.370.190 | +14,1% |
| 2029 | Sudeste | Junho | 13.315.362 | 14.287.750 | +7,3% |
| 2030 | Sudeste | Novembro | 12.955.908 | 14.561.540 | +12,4% |
| 2030 | Centro-Oeste | Janeiro | 4.140.176 | 4.958.533 | +19,8% |

Diferença relativa consistente, na faixa de 3–20%, sem colapso do modelo (diferente da versão anterior, em que o Python previa um valor constante para todas as linhas).

---

## 4. Previsão futura — Modelo ARIMA

| Ano | Região | Mês | Consumo_ARIMA (R) | Consumo_ARIMA (Python) | Dif. relativa |
|---|---|---|---:|---:|---:|
| 2026 | Norte | Janeiro | 3.783.664 | 3.845.392 | +1,6% |
| 2026 | Sudeste | Março | 22.719.733 | 23.024.830 | +1,3% |
| 2026 | Sul | Abril | 9.342.735 | 9.173.634 | -1,8% |
| 2027 | Sudeste | Agosto | 22.721.202 | 22.705.430 | -0,1% |
| 2028 | Sudeste | Janeiro | 23.712.515 | 23.122.620 | -2,5% |
| 2029 | Sul | Julho | 9.251.288 | 10.344.140 | +11,8% |
| 2030 | Norte | Setembro | 3.858.399 | 4.267.510 | +10,6% |
| 2030 | Sudeste | Novembro | 22.252.536 | 23.498.840 | +5,6% |
| 2030 | Sul | Dezembro | 8.688.756 | 11.167.480 | +28,5% |

Efeito El Niño: **em ambos R e Python, as 25 linhas foram classificadas como "El Niño"** - 100% de concordância na classificação, mesmo com pequenas diferenças nos valores absolutos.

---

## 5. Previsão futura — Modelo Random Forest

| Ano | Região | Mês | Consumo_RF (R) | Consumo_RF (Python) | Dif. relativa |
|---|---|---|---:|---:|---:|
| 2026 | Norte | Janeiro | 3.588.377 | 2.663.934 | -25,8% |
| 2026 | Nordeste | Fevereiro | 6.211.514 | 6.718.214 | +8,2% |
| 2026 | Sudeste | Março | 17.514.686 | 20.661.450 | +18,0% |
| 2027 | Sudeste | Agosto | 17.421.465 | 19.167.100 | +10,0% |
| 2028 | Sudeste | Janeiro | 16.295.247 | 20.706.860 | +27,1% |
| 2029 | Sul | Julho | 7.944.662 | 7.074.252 | -11,0% |
| 2030 | Sul | Dezembro | 9.243.646 | 7.401.321 | -19,9% |
| 2030 | Centro-Oeste | Janeiro | 4.000.459 | 3.086.440 | -22,8% |

Diferenças em ambas as direções, sem viés sistemático único (ora Python prevê mais alto, ora mais baixo que o R).

Efeito El Niño: o R marca proporcionalmente **mais linhas** como "El Niño" que o Python, consistente com o Python estar, em média, mais próximo do consumo histórico normal (erro menor cruza o limiar de 10% com menos frequência).

---

## 6. Resumo

| Modelo | RMSE — quão perto do R | Previsões futuras — quão perto do R |
|---|---|---|
| ARIMA | Praticamente idêntico (0,01%) | Muito próximo (a maioria < 5% de diferença) |
| SVM | 18,8% menor | Consistente, 3–20% de diferença |
| Random Forest | 49,2% menor | Maior variação, sem padrão fixo (-26% a +27%) |
