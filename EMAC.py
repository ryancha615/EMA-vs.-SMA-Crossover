import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  

TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2023-01-01"

LOOKBACK_DAYS = 100

#Download data from yfinance
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.fillna(0, inplace=True)
    return df
# Calculate moving averages for given short and long window
def calculate_moving_averages(df, short_window, long_window, ticker=TICKER):
    df[f"{long_window}_LARGE"] = df["Close"].rolling(window=long_window).mean()
    df[f'{short_window}_SMALL'] = df["Close"].ewm(span=short_window, adjust=False).mean()
    df.fillna(0, inplace=True)
    return df

#Find values where the short moving average crosses the long moving average
def Strategy(df, small_window, large_window):
    df["STRATEGY"] = np.where(df[f'{small_window}_SMALL'] > df[f'{large_window}_LARGE'], 1, -1)
    df.fillna(0, inplace=True)
    df["STRATEGY_RETURN"] = df["STRATEGY"].shift(1) * df["Close"][TICKER].pct_change()
    return df

# Backtest the strategy by calculating cumulative returns
def backtest(df,small_window, large_window):
    df["CUMULATIVE_STRATEGY"] = (1 + df["STRATEGY_RETURN"]).cumprod() - 1
    df["CUMULATIVE_MARKET"] = (1 + df["Close"].pct_change()).cumprod() - 1
    #plot results
    plt.plot(df["CUMULATIVE_STRATEGY"], label=f"Cumulative Strategy Return [{small_window},{large_window}]")
    plt.title(f"{TICKER} Cumulative Returns vs Strategy Cross Overs")

    return df
#Iterate through different combinations of short and long window sizes to find the best performing strategy
def main():
    SMALL_WINDOW = list(range(3,LOOKBACK_DAYS))
    LARGE_WINDOW = list(range(4,LOOKBACK_DAYS))
    df = fetch_data(TICKER, START_DATE, END_DATE)
    plt.figure(figsize=(14, 7))
    best_return = -np.inf
    best_params = None
    df_og = df
    count = 0
    for l in LARGE_WINDOW:
        for s in SMALL_WINDOW:
            if l > s:
                df = calculate_moving_averages(df, s, l, TICKER)
                df = Strategy(df,s,l)
                df = backtest(df,s,l)
                if df["CUMULATIVE_STRATEGY"].iloc[-1] > best_return:
                    best_return = df["CUMULATIVE_STRATEGY"].iloc[-1]
                    best_params = (s, l)
                count += 1
                if count % 10 == 0:
                    df = df_og.copy()  
    print(f"Best parameters: Small Window = {best_params[0]}, Large Window = {best_params[1]} with Return = {best_return}")
    return df

main()