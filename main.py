from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams
import statsmodels.api as sm

rcParams['figure.figsize'] = 10, 6

#upando os dados a partir do arquivo
path = "/Users/joaokasprowicz/Desktop/archive/annual_csv.csv"
dataset = pd.read_csv(path)
dataset = pd.DataFrame(dataset)
dataset['Date'] = pd.to_datetime(dataset['Date'])

print(dataset.head())
print(dataset.tail())

#Colocando os dados em um grafico de linha, análise exploratoria
plt.plot(dataset['Date'], dataset['Price'])
plt.xlabel('Data')
plt.ylabel('Preço')
plt.title('Preço Durante o Tempo')
plt.show()

#calculando a média e o desvio padrão dos dados no determinado frame de tempo
rolling_mean = dataset['Price'].rolling(window=2).mean()
rolling_std = dataset['Price'].rolling(window=2).std()

# Colocando os dados originais e as médias e SD calculados no gráfico de linha
plt.plot(dataset['Date'], dataset['Price'], label='Dados Originais')
plt.plot(dataset['Date'], rolling_mean, label='Média', linestyle='--')
plt.plot(dataset['Date'], rolling_std, label='Desvio Padrão', linestyle='--')

#adicionando títulos e legenda
plt.xlabel('Data')
plt.ylabel('Preço')
plt.title('Preço durante e com as estatísticas calculadas')
plt.legend()
plt.show()

#performar o teste de Dickey-Fuller para investigar se a série possui estacionariedade ou não para aplicação do modelo ARIMA
result = adfuller(dataset['Price'])

# Resultados
print('Estatística ADF:', result[0])
print('P-valor:', result[1])
print('Valores Críticos:')
for key, value in result[4].items():
    print(f'\t{key}: {value}')

indexedDataset_logScale = np.log(dataset['Price'])
plt.plot(indexedDataset_logScale)
plt.title('Transformação dos dados')
plt.show()

# Transformação Log
dataset['Log_Price'] = np.log(dataset['Price'])

# Determinar as estatísticas com a transformação Log
movingAverage = dataset['Log_Price'].rolling(window=2).mean()
movingSTD = dataset['Log_Price'].rolling(window=2).std()

# Colocar em gráfico
plt.plot(dataset['Log_Price'], color='blue', label='Original')
plt.plot(movingAverage, color='red', label='Médias Móveis')
plt.plot(movingSTD, color='black', label='Desvio Padrão Móvel')
plt.legend(loc='best')
plt.title('Média Móvel e Desvio Padrão')
plt.show()

# Diferenciando para formação dos dados estacionários
datasetLogScaleMinusMovingAverage = dataset['Log_Price'] - movingAverage

# Remoção de valores nulos (null values)
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
print(datasetLogScaleMinusMovingAverage.head(10))


#APOS REALIZAÇÃO DA DIFERENCIAÇÃO REALIZAMOS A FUNÇÃO E DEPOIS CHAMAMOS A FUNÇÃO EM CIMA DO RESULTADO PARA VERIFICAR A ESTACIONARIDADE.
def test_stationarity(timeseries):
    # Determine rolling statistics
    movingAverage = timeseries.rolling(window=2).mean()
    movingSTD = timeseries.rolling(window=2).std()

    # Plot rolling statistics
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(movingAverage, color='red', label='Média Móvel')
    plt.plot(movingSTD, color='black', label='Desvio Padrão Móvel')
    plt.legend(loc='best')
    plt.title('Média e Desvio Padrão Móvel')
    plt.show()

    # Perform Dickey–Fuller test
    result = adfuller(timeseries)
    print('Estatística ADF:', result[0])
    print('P-valor:', result[1])
    print('Valores Críticos:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# aplicar a função para avaliação
test_stationarity(datasetLogScaleMinusMovingAverage)

#transformção log e diferenciação
indexedDataset_logScale = np.log(dataset['Price'])
datasetLogDiff = indexedDataset_logScale - indexedDataset_logScale.shift()
datasetLogDiff.dropna(inplace=True)


# ACF and PACF plots
lag_acf = acf(datasetLogDiff, nlags=20)
lag_pacf = pacf(datasetLogDiff, nlags=20, method='ols')

# Grafico ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(datasetLogDiff)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(datasetLogDiff)), linestyle='--', color='gray')
plt.title('Função de Autocorrelação')

# Grafico PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(datasetLogDiff)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(datasetLogDiff)), linestyle='--', color='gray')
plt.title('Função Parcial de Autocorrelação')
plt.show()



# Fit AR Model
model_ar = sm.tsa.arima.ARIMA(dataset['Price'], order=(2, 1, 0))  # 'p' is the order of the AR component
results_ar = model_ar.fit()  # Use disp parameter here

# Plotting the original and predicted values
plt.plot(dataset['Price'])
plt.plot(results_ar.fittedvalues, color='red')
plt.title('Modelo AR - Atual vs. Predito')
plt.show()

# Print Model Summary
print(results_ar.summary())


# Fit MA Model
model_ma = sm.tsa.arima.ARIMA(dataset['Price'], order=(0, 1, 2))  # 'q' is the order of the MA component
results_ma = model_ma.fit()

# Plotting the original and predicted values
plt.plot(dataset['Price'])
plt.plot(results_ma.fittedvalues, color='orange')
plt.title('Modelo MA - Atual vs. Predito')
plt.show()

# Print Model Summary
print(results_ma.summary())

# AR Model Residuals
residuals_ar = results_ar.resid
plt.plot(residuals_ar)
plt.title('Residuos - Modelo AR')
plt.show()

# MA Model Residuals
residuals_ma = results_ma.resid
plt.plot(residuals_ma)
plt.title('Residuos - Modelo MA')
plt.show()


# Fit ARIMA Model
model_arima = sm.tsa.arima.ARIMA(dataset['Price'], order=(2, 1, 2))  # Replace with your chosen order (p, d, q)
results_arima = model_arima.fit()

# Plotting the original and predicted values
plt.plot(dataset['Price'])
plt.plot(results_arima.fittedvalues, color='red')
plt.title('Modelo Arima - Atual vs. Preditos')
plt.show()

# Print Model Summary
print(results_arima.summary())


# ARIMA Model Residuals
residuals_arima = results_arima.resid
plt.plot(residuals_arima)
plt.title('Residuos do Modelo ARIMA')
plt.show()

# Fit ARIMA Model
model_arima = sm.tsa.arima.ARIMA(dataset['Price'], order=(2, 1, 2))  # Replace with your chosen order (p, d, q)
results_arima = model_arima.fit()

# Forecast for the next 10 years (assuming your data has annual frequency)
forecast_steps = 10
forecast = results_arima.get_forecast(steps=forecast_steps)

# Extracting forecast values and confidence intervals
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()

# Plotting the original and predicted values along with the forecast
plt.plot(dataset['Date'], dataset['Price'], label='Dados Originais')
plt.plot(dataset['Date'], results_arima.fittedvalues, color='red', label='Valores preditos')
forecast_dates = pd.date_range(start=dataset['Date'].iloc[-1], periods=forecast_steps + 1, freq='A')[1:]
plt.plot(forecast_dates, forecast_values, color='green', label='Previsão')
plt.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='green', alpha=0.1, label='Intervalo de confiança de 95%')
plt.title('Modelo Arima - Atual, Predito, e Previsão')
plt.legend()
plt.show()

# Print Model Summary
print(results_arima.summary())
