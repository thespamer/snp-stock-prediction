#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Se for "prophet":
from prophet import Prophet
# Se for "fbprophet" (versões antigas):
# from fbprophet import Prophet

# Lista de símbolos-alvo
TARGET_SYMBOLS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "TSLA",  # Tesla
    "GOOG",  # Alphabet
    "META",  # Meta
]

def load_full_data():
    """
    Carrega o CSV com todos os dados do S&P 500.
    Retorna um DataFrame.
    Se não encontrar o arquivo, faz sys.exit().
    """
    csv_path = os.path.join("sp500_stocks", "sp500_stocks.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"[ERRO] Arquivo CSV não encontrado em: {csv_path}")

    print(f"[INFO] Carregando dataset de: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Formato do DataFrame: {df.shape}")
    print(df.head(3))
    return df

def filter_by_symbol(df, symbol):
    """
    Retorna apenas as linhas para o símbolo desejado.
    """
    df_symbol = df[df['Symbol'] == symbol].copy()
    if df_symbol.empty:
        print(f"[AVISO] Não há dados para {symbol}.")
    else:
        print(f"[INFO] {symbol}: {len(df_symbol)} linhas encontradas.")
    return df_symbol

def preprocess_data_prophet(df_symbol):
    """
    Prepara o DataFrame no formato exigido pelo Prophet:
      - 'ds' (datetime) no lugar de 'Date'
      - 'y' (valor) no lugar de 'Close'
    """
    # Se houver valores faltantes, faça forward-fill
    df_symbol.ffill(inplace=True)

    # Converte 'Date' para datetime
    if 'Date' in df_symbol.columns:
        df_symbol['Date'] = pd.to_datetime(df_symbol['Date'])

    # Ordena por data
    df_symbol.sort_values(by='Date', inplace=True)

    # Renomeia colunas para Prophet
    # Precisamos apenas de 2 colunas: ds e y
    # ds = data, y = valor a prever
    df_prophet = df_symbol[['Date', 'Close']].rename(
        columns={'Date': 'ds', 'Close': 'y'}
    )

    return df_prophet

def prophet_forecast(df_prophet, periods=365*2):
    """
    Treina o Prophet e gera previsão para 'periods' dias (por padrão 2 anos).
    Retorna:
      - forecast: DataFrame com previsões
      - model: objeto Prophet treinado
    """
    # Inicializa Prophet
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(df_prophet)

    # Cria um DataFrame de datas futuras
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)

    return forecast, model

def main():
    # 1. Carrega dataset
    df = load_full_data()

    # 2. Loop pelos símbolos
    for symbol in TARGET_SYMBOLS:
        print(f"\n[INFO] Processando símbolo: {symbol}")

        df_symbol = filter_by_symbol(df, symbol)
        if df_symbol.empty:
            continue

        # 3. Prepara dados no formato Prophet
        df_prophet = preprocess_data_prophet(df_symbol)

        # Verifica se temos dados suficientes
        if len(df_prophet) < 50:
            print(f"[AVISO] {symbol}: poucos dados após preprocessamento.")
            continue

        # 4. Ajusta o modelo Prophet e faz forecast para 2 anos
        forecast, model = prophet_forecast(df_prophet, periods=365*2)

        print(f"[INFO] Exibindo parte do forecast de {symbol}:")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))

        # 5. Plot principal
        fig1 = model.plot(forecast, xlabel='Data', ylabel='Preço (Close)')
        fig1.suptitle(f"Previsão de 2 anos para {symbol}", fontsize=14)
        out_file1 = f"{symbol}_forecast.png"
        fig1.savefig(out_file1)
        print(f"[INFO] Gráfico de previsão salvo em: {out_file1}")
        plt.close(fig1)

        # 6. Plot componentes (tendência e sazonalidades)
        fig2 = model.plot_components(forecast)
        fig2.suptitle(f"Componentes de previsão para {symbol}", fontsize=14)
        out_file2 = f"{symbol}_forecast_components.png"
        fig2.savefig(out_file2)
        print(f"[INFO] Gráfico de componentes salvo em: {out_file2}")
        plt.close(fig2)

if __name__ == "__main__":
    main()

