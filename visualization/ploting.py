import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import config


def plot_candlestick_with_forecast(df, details, pred_LSTM, signals=None):
    # ── Historical OHLC (last N days from config) ─────────────────────────────
    df_ohlc = df[['Date', 'high', 'low', 'close']].copy()
    df_ohlc['Date'] = pd.to_datetime(df_ohlc['Date'], errors='coerce')
    for col in ['high', 'low', 'close']:
        df_ohlc[col] = pd.to_numeric(df_ohlc[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
    df_ohlc = df_ohlc.dropna(subset=['Date', 'high', 'low', 'close'])
    # Sort by Date column ascending BEFORE setting as index, then take last N rows
    df_ohlc = df_ohlc.sort_values(by='Date', ascending=True).tail(config.PLOT_HISTORICAL_DAYS)
    # Synthesize 'Open' as previous day's close (mplfinance requires Open column)
    df_ohlc['open'] = df_ohlc['close'].shift(1)
    df_ohlc = df_ohlc.dropna(subset=['open'])
    df_ohlc = df_ohlc.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
    df_ohlc = df_ohlc.set_index('Date')
    df_ohlc.index = pd.DatetimeIndex(df_ohlc.index)

    close_pred = pred_LSTM[:, 2]   # index 2 = close
    high_pred  = pred_LSTM[:, 0]   # index 0 = high
    low_pred   = pred_LSTM[:, 1]   # index 1 = low

    last_date    = df_ohlc.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=config.FORECAST_DAYS)

    # Use len(df_ohlc) for correct alignment
    nan_pad = [np.nan] * len(df_ohlc)
    close_line = nan_pad + list(close_pred)
    high_line  = nan_pad + list(high_pred)
    low_line   = nan_pad + list(low_pred)

    # Extend df_ohlc with NaN forecast rows so mplfinance covers full date range
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

    ax = axes[0]

    # ── Overlay BUY / SELL / HOLD signal markers ─────────────────────────────
    if signals:
        COLORS  = {k: v for k, v in config.SIGNAL_COLORS.items()}
        COLORS_NAMED  = {'BUY': COLORS[2], 'SELL': COLORS[0], 'HOLD': COLORS[1]}
        MARKERS = {k: v for k, v in config.SIGNAL_MARKERS.items()}
        MARKERS_NAMED = {'BUY': MARKERS[2], 'SELL': MARKERS[0], 'HOLD': MARKERS[1]}
        OFFSETS = config.SIGNAL_OFFSETS

        # x-axis positions: historical bars occupy 0 … len(df_ohlc)-1,
        # forecast bars start at len(df_ohlc)
        base_x = len(df_ohlc)

        for i, sig in enumerate(signals):
            x_pos  = base_x + i
            label  = sig['label']
            conf   = sig['confidence']
            color  = COLORS_NAMED[label]
            marker = MARKERS_NAMED[label]
            # Place marker slightly above/below the forecast close price
            y_price = close_pred[i]
            y_offset = y_price * (1 + OFFSETS[label] * 2)

            ax.plot(x_pos, y_offset, marker=marker, color=color,
                    markersize=12, zorder=5, markeredgecolor='white', markeredgewidth=0.8)

            # Confidence label
            ax.annotate(
                f"{label}\n{conf}%",
                xy=(x_pos, y_offset),
                xytext=(6, 0),
                textcoords='offset points',
                fontsize=7,
                color=color,
                fontweight='bold',
                va='center'
            )

        # ── Custom legend entries for signals ─────────────────────────────────
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#22c55e',
                   markersize=9, label='BUY Signal',  markeredgecolor='white'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='#ef4444',
                   markersize=9, label='SELL Signal', markeredgecolor='white'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#f59e0b',
                   markersize=9, label='HOLD Signal', markeredgecolor='white'),
        ]
        existing_legend = ax.get_legend()
        existing_handles, existing_labels = [], []
        if existing_legend:
            existing_handles = existing_legend.legend_handles
            existing_labels  = [t.get_text() for t in existing_legend.get_texts()]

        ax.legend(
            handles=existing_handles + legend_elements,
            labels=existing_labels + [e.get_label() for e in legend_elements],
            loc='upper left', fontsize=8,
            facecolor='#1a1d27', labelcolor='white', framealpha=0.8
        )

    plt.tight_layout()
    plt.show()
