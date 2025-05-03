import pandas as pd
import numpy as np
import yfinance as yf

symbols = ["NQ=F", "ES=F", "YM=F"] 
time_interval = "5m"
rangeStart = "2024-12-01"
rangeEnd = "2025-04-01"

rawData = {
    symbol: yf.download(symbol, interval="5m", start=rangeStart, end=rangeEnd) for symbol in symbols
}