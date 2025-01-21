# Análise do Preço de Fechamento do Bitcoin com Mínimos Quadrados Descontados e Técnicas de Suavização

![Bitcoin](https://img.icons8.com/color/96/000000/bitcoin.png)

## 📈 Visão Geral

Este projeto tem como objetivo analisar o preço de fechamento do Bitcoin utilizando **Mínimos Quadrados Descontados (MQD)** e diversas **Técnicas de Suavização (Médias Móveis Simples, Suavização Exponencial e Suavização Exponencial Dupla)**. Através dessa análise, buscamos entender as tendências e comportamentos do mercado de Bitcoin, oferecendo insights valiosos para investidores e entusiastas.

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- Bibliotecas:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `statsmodels`
  - `scikit-learn`

## 🧠 Abordagens Utilizadas
1. Mínimos Quadrados Descontados (MQD)
O MQD ajusta os parâmetros de regressão de forma a dar menos peso aos erros conforme o tempo avança, capturando tendências recentes de maneira mais eficaz.

2. Técnicas de Suavização
- Médias Móveis Simples (SMA): Calcula a média de um número fixo de períodos para identificar tendências de curto e longo prazo.
- Suavização Exponencial: Dá mais peso aos dados mais recentes, ideal para séries com tendências ou sazonalidades leves.
- Suavização Exponencial Dupla: Captura tanto a tendência quanto a sazonalidade dos dados, adequada para séries temporais com tendências significativas.

## 📊 Resultados
Após executar o script, você obterá:

- Visualizações Gráficas:
  - Preço de fechamento do Bitcoin ao longo do tempo.
  - Comparação entre os dados originais e as previsões do MQD.
  - Aplicação das diferentes técnicas de suavização (Médias Móveis Simples, Suavização Exponencial e Suavização Exponencial Dupla).

- Métricas de Erro:
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - Resumo da Regressão OLS: Coeficientes β₀ e β₁ ajustados.

- Exemplos de Gráficos
![Figure_1](https://github.com/user-attachments/assets/ae3b789e-f9f5-4097-ba2a-1657b356b3c0)
![Figure_2](https://github.com/user-attachments/assets/e12b6ccb-659c-4afa-95a7-949ef1e728f8)
![Figure_3](https://github.com/user-attachments/assets/a7f4ce02-c7d8-49cf-ae7e-6d0a80e3d4c5)
![Figure_4](https://github.com/user-attachments/assets/8c128538-bcce-414c-a075-aaac8b5e5613)
![Figure_5](https://github.com/user-attachments/assets/9416511c-ba17-452c-b214-5f55b99a8bcd)


