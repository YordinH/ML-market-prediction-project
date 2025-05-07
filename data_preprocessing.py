import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
#DOWNLOADED DATA TUESDAY, MAY 6TH, 2025

def format(data):
    if data.get("s") != "ok" or not data.get("t"):
        return pd.DataFrame()
    
    market_df = pd.DataFrame({
        "datetime": pd.to_datetime(data["t"], unit="s"),
        "Open": data["o"],
        "High": data["h"],
        "Low": data["l"],
        "Close": data["c"],
        "Volume": data["v"]
    })
    market_df.set_index("datetime", inplace=True)
    return market_df


def download_data(symbols, interval, start, end, token):
    rawData = {}
    headers = {
        "Authorization": f"Bearer {token}"
    }

    symbol_conversion = {
        "QQQ": "NQ",
        "SPY": "ES",
        "DIA": "YM"
    }

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    current = start_dt
    while current < end_dt:
        next_week = min(current + timedelta(days=7), end_dt)
        date_start = current.strftime("%Y-%m-%d")
        date_end = next_week.strftime("%Y-%m-%d")

        for symbol in symbols: #Names in format "Open/Close_symbol"
            try:
                url = f"https://api.marketdata.app/v1/stocks/candles/{interval}/{symbol}/?from={date_start}&to={date_end}&extended=true"
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                market_df = format(response.json())

                if market_df.empty:
                    print("Empty dataset")
                    continue

                market_df.columns = [f"{col}_{symbol_conversion[symbol]}" for col in market_df.columns] 
                if symbol not in rawData:
                    rawData[symbol] = []
                rawData[symbol].append(market_df)

            except Exception as e:
                print(f"Data download FAILED: {e}")
        current = next_week

    merged = []
    for symbol in symbols:
        if rawData[symbol]:
            combined_symbol_df = pd.concat(rawData[symbol], axis=0)
            combined_symbol_df = combined_symbol_df[~combined_symbol_df.index.duplicated(keep='last')]
            rawData[symbol] = combined_symbol_df
        else:
            return None
    
    combinedData = pd.concat([rawData[symbol] for symbol in symbols], axis=1).dropna()
    return combinedData

def features(combinedData):
    #Change in price from the 1m interval
    combinedData['Change_NQ'] = (combinedData['Close_NQ'] - combinedData['Open_NQ']) / combinedData['Open_NQ']
    combinedData['Change_ES'] = (combinedData['Close_ES'] - combinedData['Open_ES']) / combinedData['Open_ES']
    combinedData['Change_YM'] = (combinedData['Close_YM'] - combinedData['Open_YM']) / combinedData['Open_YM']


    #SMT Finder
    combinedData['SMT_NQ_ES'] = (np.sign(combinedData['Change_NQ']) != np.sign(combinedData['Change_ES'])).astype(int)
    combinedData['SMT_NQ_YM'] = (np.sign(combinedData['Change_NQ']) != np.sign(combinedData['Change_YM'])).astype(int)

    #Relative Strength to determine SMT
    combinedData['RS'] = pd.DataFrame({
        'NQ': combinedData['Change_NQ'],
        'ES': combinedData['Change_ES'],
        'YM': combinedData['Change_YM']
    }).rank(axis=1, method='min', ascending=False).idxmin(axis=1).map({'NQ':0,'ES':1,'YM':2})

    #FVG
    highOne = combinedData['High_NQ'].shift(2)
    lowTwo = combinedData['Low_NQ'].shift(1)
    lowThree = combinedData['Low_NQ']

    combinedData['FVG_Bullish'] = ((lowTwo > highOne) & (lowThree > highOne)).astype(int)

    lowOne = combinedData['Low_NQ'].shift(2)
    highTwo = combinedData['High_NQ'].shift(1)
    highThree = combinedData['High_NQ']

    combinedData['FVG_Bearish'] = ((highTwo < lowOne) & (highThree < lowOne)).astype(int)

    #Liquidity
    hourly_highs = combinedData['High_NQ'].resample('1H').max()
    hourly_lows = combinedData['Low_NQ'].resample('1H').min()

    combinedData['Hourly_High'] = hourly_highs.reindex(combinedData.index, method='ffill')
    combinedData['Hourly_Low'] = hourly_lows.reindex(combinedData.index, method='ffill')

    combinedData['Sweep_High'] = (combinedData['High_NQ'] > combinedData['Hourly_High'].shift(1)).astype(int)
    combinedData['Sweep_Low'] = (combinedData['Low_NQ'] < combinedData['Hourly_Low'].shift(1)).astype(int)

    #Combine
    combinedData['Bullish_Setup'] = (
    (combinedData['Sweep_High'] == 1) &
    (combinedData['SMT_NQ_ES'] == 1) &
    (combinedData['FVG_Bearish'] == 1)
    ).astype(int)

    combinedData['Bearish_Setup'] = (
    (combinedData['Sweep_Low'] == 1) &
    (combinedData['SMT_NQ_ES'] == 1) &
    (combinedData['FVG_Bullish'] == 1)
    ).astype(int)

    #Target for price
    combinedData['Target'] = (combinedData['Close_NQ'].shift(-5) > combinedData['Close_NQ']).astype(int)


    columns_csv = ['Change_NQ', 'Change_ES', 'Change_YM', 'SMT_NQ_ES', 'SMT_NQ_YM', 'RS', 'FVG_Bullish','FVG_Bearish','Sweep_High', 'Sweep_Low','Bullish_Setup', 'Bearish_Setup','Target']
    print(f"Final dataset shape: {combinedData.dropna().shape}")
    return combinedData.dropna()[columns_csv]

def save_csv(df): #Save data to ict_dataset.csv
    df.to_csv("ict_dataset.csv")

def main():
    symbols = ["QQQ", "SPY", "DIA"] 
    time_interval = "5"
    start = "2024-01-01"
    end = "2024-05-01"
    token = "Token HIDDEN. Will provide if needed to download data" #Token is PAID, valid till beginning of June

    data = download_data(symbols,time_interval, start, end, token)

    if data is None or data.empty:
        quit()

    print(f"Combined raw data shape: {data.shape}")
    print(f"First few timestamps:\n{data.head(3)}")
    market_df = features(data)
    save_csv(market_df)

if __name__ == "__main__":
    main()