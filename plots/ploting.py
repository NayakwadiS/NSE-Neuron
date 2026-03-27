import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_candlestick_with_forecast(df, details, pred_LSTM):
    # Candlestick Plot for last 100 days
    # Prepare historical OHLC data with DatetimeIndex
    df_ohlc = df[['Date', 'high', 'low', 'close']].copy()
    df_ohlc['Date'] = pd.to_datetime(df_ohlc['Date'], errors='coerce')
    for col in ['high', 'low', 'close']:
        df_ohlc[col] = pd.to_numeric(df_ohlc[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
    df_ohlc = df_ohlc.dropna(subset=['Date', 'high', 'low', 'close'])
    # Sort by Date column ascending BEFORE setting as index, then take last 100 rows
    df_ohlc = df_ohlc.sort_values(by='Date', ascending=True).tail(100)
    # Synthesize 'Open' as previous day's close (mplfinance requires Open column)
    df_ohlc['open'] = df_ohlc['close'].shift(1)
    df_ohlc = df_ohlc.dropna(subset=['open'])
    df_ohlc = df_ohlc.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
    df_ohlc = df_ohlc.set_index('Date')
    df_ohlc.index = pd.DatetimeIndex(df_ohlc.index)

    close_pred = pred_LSTM[:, 2]   # index 2 = close
    last_date = df_ohlc.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)

    # Use len(df_ohlc) for correct alignment
    nan_pad = [np.nan] * len(df_ohlc)
    close_line = nan_pad + list(close_pred)
    high_line  = nan_pad + list(pred_LSTM[:, 0])   # index 0 = high
    low_line   = nan_pad + list(pred_LSTM[:, 1])   # index 1 = low

    # Extend df_ohlc with forecast rows (NaN OHLC) so mplfinance covers full date range
    df_forecast = pd.DataFrame(
        {'Open': np.nan, 'High': np.nan, 'Low': np.nan, 'Close': np.nan},
        index=future_dates
    )
    df_full = pd.concat([df_ohlc, df_forecast])

    ap = [
        mpf.make_addplot(close_line, type='line', color='blue', label='Forecast Close', panel=0),
        mpf.make_addplot(high_line, type='line', color='green', label='Forecast High', panel=0),
        mpf.make_addplot(low_line, type='line', color='red', label='Forecast Low', panel=0),
    ]

    fig, axes = mpf.plot(
        df_full,
        type='candle',
        style='charles',
        ylabel='Price in Rupees',
        title=f"Predictions for {details['scheme_name']}",
        volume=False,
        addplot=ap,
        warn_too_much_data=200,
        returnfig=True
    )
    plt.show()
