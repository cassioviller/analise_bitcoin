# =============================================================================
# Mínimos Quadrados Descontados e Técnicas de Suavização para Análise do
# Preço de Fechamento do Bitcoin
# =============================================================================

# =============================================================================
# 1. Importação das Bibliotecas Necessárias
# =============================================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Configurando o estilo dos gráficos com Seaborn
sns.set_style('whitegrid')

# =============================================================================
# 2. Carregamento e Pré-processamento dos Dados
# =============================================================================

# Carregando os dados do arquivo CSV
df = pd.read_csv("dados_bitcoin.csv")

# Exibindo as primeiras linhas do DataFrame para verificação
print("Primeiras linhas do DataFrame:")
print(df.head())

# Convertendo a coluna "Date" para o formato datetime
df["Date"] = pd.to_datetime(df["Date"])

# Verificando o tipo de dados após a conversão
print("\nTipos de dados após a conversão:")
print(df.dtypes)

# Normalizando a variável dependente para estabilizar as atualizações
scaler = StandardScaler()
df['Close_scaled'] = scaler.fit_transform(df[['Close']])

# =============================================================================
# 3. Visualização Inicial dos Dados
# =============================================================================

# Plotando o gráfico do preço de fechamento do Bitcoin ao longo do tempo
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Close"], color='blue', linewidth=1)
plt.xlabel("Data")
plt.ylabel("Preço de Fechamento (USD)")
plt.title("Preço de Fechamento do Bitcoin")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# 4. Regressão de Mínimos Quadrados Ordinários (MQO) para Chute Inicial
# =============================================================================

# Definindo as variáveis para a regressão
x = df.index.values  # Utilizamos o índice como variável independente (tempo)
y = df['Close_scaled'].values  # Preço de fechamento normalizado como variável dependente

# Adicionando o termo constante para o intercepto (β₀)
x = sm.add_constant(x)

# Realizando a regressão OLS
model = sm.OLS(y, x).fit()

# Exibindo o resumo da regressão
print("\nResumo da Regressão OLS:")
print(model.summary())

# Obtendo os parâmetros β₀ e β₁
beta0 = model.params[0]
beta1 = model.params[1]
print(f"Valor de beta_0 = {beta0}")
print(f"Valor de beta_1 = {beta1}")

# =============================================================================
# 5. Implementação de Mínimos Quadrados Descontados (MQD) - Completa
# =============================================================================

def mqd(dataframe, beta0_init, beta1_init, theta, pred_steps=1):
    """
    Implementa Mínimos Quadrados Descontados (MQD).

    Parameters:
    - dataframe: DataFrame contendo os dados.
    - beta0_init: Valor inicial de β₀.
    - beta1_init: Valor inicial de β₁.
    - theta: Fator de desconto (0 < theta < 1).
    - pred_steps: Número de passos à frente para previsão.

    Returns:
    - data: DataFrame com as previsões e erros.
    - beta0: β₀ final ajustado.
    - beta1: β₁ final ajustado.
    - history: DataFrame com o histórico de beta0 e beta1.
    """
    beta0 = beta0_init
    beta1 = beta1_init
    var1 = 1 - theta**2  # 0.19 para theta=0.9
    var2 = (1 - theta)**2  # 0.01 para theta=0.9

    # Listas para armazenar previsões, erros e histórico dos parâmetros
    yhat_list = []
    list_erros = []
    beta0_history = []
    beta1_history = []

    # Iterando sobre os dados
    for i in range(len(dataframe.Close_scaled)):
        yhat = beta0 + beta1  # Predição conforme o exemplo original
        yhat_list.append(yhat)
        yreal = dataframe['Close_scaled'].iloc[i]
        error = yreal - yhat
        list_erros.append(error)

        # Armazenando o histórico dos parâmetros
        beta0_history.append(beta0)
        beta1_history.append(beta1)

        # Atualizando os parâmetros
        beta0 = beta0 + beta1 + var1 * error
        beta1 = beta1 + var2 * error

        # Limitação dos parâmetros para evitar explosão
        beta0 = max(min(beta0, 1e6), -1e6)
        beta1 = max(min(beta1, 1e3), -1e3)

    # Previsão para os próximos passos
    for p in range(1, pred_steps + 1):
        yhat_pred = beta0 + beta1
        yhat_list.append(yhat_pred)
        list_erros.append(None)  # Sem erro para previsões futuras

    # Criando um DataFrame para armazenar os resultados
    data = pd.DataFrame({
        'y_pred_scaled': yhat_list,
        'error': list_erros
    })

    # Criando um DataFrame para o histórico dos parâmetros
    history = pd.DataFrame({
        'beta0': beta0_history,
        'beta1': beta1_history
    })

    # Desfazendo a normalização para as previsões
    data['y_pred'] = scaler.inverse_transform(data[['y_pred_scaled']])

    return data, beta0, beta1, history

# Aplicando a função MQD com theta=0.9 e previsão de 1 passo à frente
data_mqd, beta0_final, beta1_final, history = mqd(df, beta0, beta1, theta=0.9, pred_steps=1)

# Exibindo as primeiras linhas do resultado do MQD
print("\nPrimeiras linhas do resultado do MQD:")
print(data_mqd.head())

# Exibindo os valores finais de beta0 e beta1
print(f"\nValor final de beta_0 = {beta0_final}")
print(f"Valor final de beta_1 = {beta1_final}")

# Exibindo o histórico dos parâmetros
print("\nHistórico dos parâmetros (primeiras 5 linhas):")
print(history.head())


# =============================================================================
# 6. Plotagem das Previsões do MQD vs Dados Originais - Completa
# =============================================================================

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Close"], label="Original", color='black')
plt.plot(df["Date"], data_mqd["y_pred"][:-1], label="MQD", color='red')  # Excluindo a última previsão futura
plt.xlabel("Data")
plt.ylabel("Preço de Fechamento (USD)")
plt.title("Comparação entre Dados Originais e Previsões MQD")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# 7. Cálculo das Métricas de Erro
# =============================================================================

# Calculando o erro entre as previsões e os dados reais
mae = mean_absolute_error(df['Close'], data_mqd['y_pred'][:-1])
mse = mean_squared_error(df['Close'], data_mqd['y_pred'][:-1])
rmse = np.sqrt(mse)

print(f"\nMétricas de Erro do MQD:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# =============================================================================
# 8. Implementação das Técnicas de Suavização
# =============================================================================

# 8.1. Médias Móveis Simples (SMA)
def moving_average_smoothing(X, k):
    """
    Calcula a média móvel simples.

    Parameters:
    - X: Array de valores.
    - k: Período da média móvel.

    Returns:
    - S: Array com a média móvel calculada.
    """
    S = np.zeros(len(X))
    for t in range(len(X)):
        if t < k:
            S[t] = np.mean(X[:t+1])
        else:
            S[t] = np.mean(X[t-k+1:t+1])
    return S

# Calculando médias móveis com diferentes períodos
time_series = df['Close'].values
sma_4 = moving_average_smoothing(time_series, 4)
sma_12 = moving_average_smoothing(time_series, 12)
sma_50 = moving_average_smoothing(time_series, 50)
sma_100 = moving_average_smoothing(time_series, 100)

# 8.2. Suavização Exponencial
def exponential_smoothing(X, alpha):
    """
    Aplica a suavização exponencial.

    Parameters:
    - X: Array de valores.
    - alpha: Fator de suavização (0 < alpha < 1).

    Returns:
    - S: Array com a série suavizada.
    """
    S = np.zeros(len(X))
    S[0] = X[0]  # Inicializa com o primeiro valor
    for t in range(1, len(X)):
        S[t] = alpha * X[t] + (1 - alpha) * S[t-1]
    return S

# Aplicando a suavização exponencial com diferentes alphas
e_s_0_1 = exponential_smoothing(time_series, 0.1)
e_s_0_3 = exponential_smoothing(time_series, 0.3)
e_s_0_5 = exponential_smoothing(time_series, 0.5)

# 8.3. Suavização Exponencial Dupla
def double_exponential_smoothing(X, alpha, beta):
    """
    Aplica a suavização exponencial dupla.

    Parameters:
    - X: Array de valores.
    - alpha: Fator de suavização para o nível.
    - beta: Fator de suavização para a tendência.

    Returns:
    - S: Array com a série suavizada.
    """
    S = np.zeros(len(X))
    b = np.zeros(len(X))
    S[0] = X[0]
    b[0] = X[1] - X[0] if len(X) > 1 else 0

    for t in range(1, len(X)):
        S[t] = alpha * X[t] + (1 - alpha) * (S[t-1] + b[t-1])
        b[t] = beta * (S[t] - S[t-1]) + (1 - beta) * b[t-1]
    return S

# Aplicando a suavização exponencial dupla com diferentes parâmetros
d_e_s_0_5_0_1 = double_exponential_smoothing(time_series, alpha=0.5, beta=0.1)
d_e_s_0_5_0_5 = double_exponential_smoothing(time_series, alpha=0.5, beta=0.5)
d_e_s_0_1_0_5 = double_exponential_smoothing(time_series, alpha=0.1, beta=0.5)

# =============================================================================
# 9. Plotagem das Técnicas de Suavização
# =============================================================================

# 9.1. Plotando as Médias Móveis
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], time_series, label="Original", color='black')
plt.plot(df["Date"], sma_4, label="SMA k=4", color="blue")
plt.plot(df["Date"], sma_12, label="SMA k=12", color="red")
plt.plot(df["Date"], sma_50, label="SMA k=50", color="green")
plt.plot(df["Date"], sma_100, label="SMA k=100", color="orange")
plt.xlabel("Data")
plt.ylabel("Preço de Fechamento (USD)")
plt.title("Médias Móveis Simples (SMA)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9.2. Plotando a Suavização Exponencial
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], time_series, label="Original", color="black")
plt.plot(df["Date"], e_s_0_1, label="Suavização Exponencial α=0.1", color="#88CCEE")
plt.plot(df["Date"], e_s_0_3, label="Suavização Exponencial α=0.3", color="red")
plt.plot(df["Date"], e_s_0_5, label="Suavização Exponencial α=0.5", color="green")
plt.xlabel("Data")
plt.ylabel("Preço de Fechamento (USD)")
plt.title("Suavização Exponencial")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9.3. Plotando a Suavização Exponencial Dupla
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], time_series, label="Original", color="black")
plt.plot(df["Date"], d_e_s_0_5_0_1, label="Suavização Exponencial Dupla α=0.5, β=0.1", color="red")
plt.plot(df["Date"], d_e_s_0_5_0_5, label="Suavização Exponencial Dupla α=0.5, β=0.5", color="green")
plt.plot(df["Date"], d_e_s_0_1_0_5, label="Suavização Exponencial Dupla α=0.1, β=0.5", color="#117733")
plt.xlabel("Data")
plt.ylabel("Preço de Fechamento (USD)")
plt.title("Suavização Exponencial Dupla")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# 10. Comparação das Técnicas de Suavização
# =============================================================================

# Aplicando as técnicas de suavização com parâmetros escolhidos
sma_final = moving_average_smoothing(time_series, k=12)
e_s_final = exponential_smoothing(time_series, alpha=0.5)
d_e_s_final = double_exponential_smoothing(time_series, alpha=0.5, beta=0.1)

# Plotando todas as técnicas de suavização junto com os dados originais e MQD
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], time_series, label="Original", color='black')
plt.plot(df["Date"], data_mqd["y_pred"][:-1], label="MQD", color='red')
plt.plot(df["Date"], e_s_final, label="Suavização Exponencial α=0.5", color="#88CCEE")
plt.plot(df["Date"], sma_final, label="Média Móvel Simples k=12", color="orange")
plt.plot(df["Date"], d_e_s_final, label="Suavização Exponencial Dupla α=0.5, β=0.1", color="#117733")
plt.xlabel("Data")
plt.ylabel("Preço de Fechamento (USD)")
plt.title("Comparação das Técnicas de Suavização")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# 11. Considerações Finais
# =============================================================================

print("\nConsiderações Finais:")
print("""
- **MQD (Mínimos Quadrados Descontados):** Técnica que ajusta os parâmetros de regressão de forma a dar menos peso aos erros conforme o tempo avança, ajudando a capturar tendências recentes.

- **Médias Móveis:** Suavizam os dados ao calcular a média de um número fixo de períodos, ajudando a identificar tendências de curto e longo prazo.

- **Suavização Exponencial:** Dá mais peso aos dados mais recentes, sendo útil para dados com tendências ou sazonalidades leves.

- **Suavização Exponencial Dupla:** Além de suavizar, também captura a tendência dos dados, sendo adequada para séries temporais com tendência.

Cada técnica tem suas vantagens e pode ser escolhida com base nas características específicas dos dados e nos objetivos da análise.
""")
