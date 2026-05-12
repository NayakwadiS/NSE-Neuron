import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU as KerasGRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from config import (
    SIGNAL_COLORS,
    SIGNAL_LABELS,
    SIGNAL_MARKERS,
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BB_PERIOD,
    STOCH_K_PERIOD,
    STOCH_D_PERIOD,
    ATR_PERIOD,
    WILLIAMS_R_PERIOD,
    EMA_FAST,
    EMA_MID,
    EMA_SLOW,
    CLASSIFIER_THRESHOLD,
    CLASSIFIER_TIME_STEP,
    CLASSIFIER_TRAIN_SPLIT,
    CLASSIFIER_LSTM_UNITS_1,
    CLASSIFIER_LSTM_UNITS_2,
    CLASSIFIER_DROPOUT,
    CLASSIFIER_DENSE_UNITS,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_PATIENCE,
    CLASSIFIER_MIN_DELTA,
    FORECAST_DAYS,
)

SIGNAL_COLORS  = SIGNAL_COLORS
SIGNAL_LABELS  = SIGNAL_LABELS
SIGNAL_MARKERS = SIGNAL_MARKERS


# ─────────────────────────── indicator helpers ────────────────────────────────
def _rsi(series, period=None):
    period = period or RSI_PERIOD
    delta  = series.diff()
    gain   = delta.clip(lower=0).rolling(period).mean()
    loss   = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def _macd(series, fast=None, slow=None, sig=None):
    fast  = fast or MACD_FAST
    slow  = slow or MACD_SLOW
    sig   = sig  or MACD_SIGNAL
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    line  = ema_f - ema_s
    signal = line.ewm(span=sig, adjust=False).mean()
    return line, signal


def _bollinger_pct(series, period=None):
    period = period or BB_PERIOD
    sma    = series.rolling(period).mean()
    std    = series.rolling(period).std()
    upper  = sma + 2 * std
    lower  = sma - 2 * std
    return (series - lower) / ((upper - lower).replace(0, np.nan))


def _stochastic(high, low, close, k=None, d=None):
    k = k or STOCH_K_PERIOD
    d = d or STOCH_D_PERIOD
    lowest_low   = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    k_pct = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    d_pct = k_pct.rolling(d).mean()
    return k_pct, d_pct


def _atr(high, low, close, period=None):
    period = period or ATR_PERIOD
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean() / (close + 1e-9)


def _williams_r(high, low, close, period=None):
    period = period or WILLIAMS_R_PERIOD
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll + 1e-9)


def _build_features(df):
    c = df['close']
    h = df['high']
    l = df['low']

    f = pd.DataFrame(index=df.index)

    f['ema_9']    = c.ewm(span=EMA_FAST, adjust=False).mean()
    f['ema_21']   = c.ewm(span=EMA_MID,  adjust=False).mean()
    f['ema_50']   = c.ewm(span=EMA_SLOW, adjust=False).mean()
    f['ema_diff'] = f['ema_9'] - f['ema_21']

    f['rsi_14']    = _rsi(c, 14)
    macd, macd_sig = _macd(c)
    f['macd']      = macd
    f['macd_sig']  = macd_sig
    f['macd_hist'] = macd - macd_sig

    stoch_k, stoch_d = _stochastic(h, l, c)
    f['stoch_k'] = stoch_k
    f['stoch_d'] = stoch_d

    f['williams_r'] = _williams_r(h, l, c, 14)
    f['bb_pct']     = _bollinger_pct(c, 20)
    f['atr']        = _atr(h, l, c, 14)

    f['ret_1'] = c.pct_change(1)
    f['ret_3'] = c.pct_change(3)
    f['ret_5'] = c.pct_change(5)

    f['hl_spread'] = (h - l) / (c + 1e-9)

    return f.dropna()


def _make_labels(close_series, threshold=None):
    nr = close_series.pct_change(1).shift(-1)
    return pd.Series(
        np.where(nr > threshold, 2, np.where(nr < -threshold, 0, 1)),
        index=close_series.index
    )


# ─────────────────────────── main function ────────────────────────────────────
def gru_classifier(df, pred_prices):
    """
    Train a GRU BUY/SELL/HOLD classifier on technical indicators,
    then predict one signal per forecast day.

    Parameters
    ----------
    df          : full historical OHLC DataFrame (from getData)
    pred_prices : np.ndarray (FORECAST_DAYS, 4)  — [high, low, close, prev_close] per day

    Returns
    -------
    signals : list of dicts with label, signal, confidence, probs, color, marker
    """

    # ── 1. Clean OHLC ────────────────────────────────────────────────────────
    raw = df[['close', 'high', 'low']].copy()
    for col in raw.columns:
        raw[col] = pd.to_numeric(
            raw[col].astype(str).str.replace(',', '', regex=False), errors='coerce'
        )
    raw = raw.dropna().reset_index(drop=True)

    # ── 2. Build features ────────────────────────────────────────────────────
    feat_df      = _build_features(raw)
    feat_columns = list(feat_df.columns)

    # ── 3. Labels ────────────────────────────────────────────────────────────
    close_aligned = raw['close'].loc[feat_df.index]
    labels        = _make_labels(close_aligned, threshold=CLASSIFIER_THRESHOLD)

    feat_vals  = feat_df.values.astype(np.float32)
    label_vals = labels.values.astype(int)

    X_for_train = feat_vals[:-1]
    y_for_train = label_vals[:-1]

    unique, counts = np.unique(y_for_train, return_counts=True)
    label_dist = dict(zip([SIGNAL_LABELS[u] for u in unique], counts))
    print(f"\n  [GRU Classifier] Label distribution: {label_dist}")

    # ── 4. Scale ─────────────────────────────────────────────────────────────
    feat_scaler = MinMaxScaler(feature_range=(0, 1))
    feat_scaler.fit(X_for_train)
    X_scaled_train = feat_scaler.transform(X_for_train)

    # ── 5. Sequences ─────────────────────────────────────────────────────────
    time_step = CLASSIFIER_TIME_STEP

    def make_sequences(X, y, ts):
        Xs, ys = [], []
        for i in range(len(X) - ts):
            Xs.append(X[i:i + ts])
            ys.append(y[i + ts])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = make_sequences(X_scaled_train, y_for_train, time_step)

    print(f"  [GRU Classifier] Total sequences: {len(X_seq)}")

    # ── 6. Train / val split ─────────────────────────────────────────────────
    split      = int(len(X_seq) * CLASSIFIER_TRAIN_SPLIT)
    Xtr, Xval  = X_seq[:split], X_seq[split:]
    ytr, yval  = y_seq[:split], y_seq[split:]

    ytr_cat  = to_categorical(ytr,  num_classes=3)
    yval_cat = to_categorical(yval, num_classes=3)

    cw = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=ytr)
    cw_dict = {0: float(cw[0]), 1: float(cw[1]), 2: float(cw[2])}

    print(f"  [GRU Classifier] Train: {len(Xtr)}  Val: {len(Xval)}")
    print(f"  [GRU Classifier] Class weights — SELL:{cw[0]:.2f}  HOLD:{cw[1]:.2f}  BUY:{cw[2]:.2f}\n")

    # ── 7. Build GRU model ───────────────────────────────────────────────────
    n_feat = X_seq.shape[2]

    model = Sequential([
        KerasGRU(CLASSIFIER_LSTM_UNITS_1, return_sequences=True, input_shape=(time_step, n_feat)),
        Dropout(CLASSIFIER_DROPOUT),
        KerasGRU(CLASSIFIER_LSTM_UNITS_2),
        Dropout(CLASSIFIER_DROPOUT),
        Dense(CLASSIFIER_DENSE_UNITS, activation='relu'),
        Dense(3, activation='softmax'),
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=CLASSIFIER_PATIENCE,
        restore_best_weights=True,
        verbose=1,
        min_delta=CLASSIFIER_MIN_DELTA
    )

    model.fit(
        Xtr, ytr_cat,
        validation_data=(Xval, yval_cat),
        epochs=CLASSIFIER_EPOCHS,
        batch_size=CLASSIFIER_BATCH_SIZE,
        verbose=1,
        class_weight=cw_dict,
        callbacks=[early_stop]
    )

    # ── 8. Rolling forecast ──────────────────────────────────────────────────
    x_seed    = X_scaled_train[-time_step:].copy()
    ohlc_hist = raw[['close', 'high', 'low']].copy().reset_index(drop=True)
    signals   = []

    for i in range(FORECAST_DAYS):
        probs      = model.predict(x_seed.reshape(1, time_step, n_feat), verbose=0)[0]
        signal_idx = int(np.argmax(probs))
        confidence = float(probs[signal_idx]) * 100

        print(f"  [GRU Classifier] Day {i+1} probs — "
              f"SELL:{probs[0]*100:.1f}%  HOLD:{probs[1]*100:.1f}%  BUY:{probs[2]*100:.1f}%"
              f"  → {SIGNAL_LABELS[signal_idx]}")

        signals.append({
            'label':      SIGNAL_LABELS[signal_idx],
            'signal':     signal_idx,
            'confidence': round(confidence, 1),
            'probs': {
                'SELL': round(float(probs[0]) * 100, 1),
                'HOLD': round(float(probs[1]) * 100, 1),
                'BUY':  round(float(probs[2]) * 100, 1),
            },
            'color':  SIGNAL_COLORS[signal_idx],
            'marker': SIGNAL_MARKERS[signal_idx],
        })

        # Roll window forward with predicted OHLC
        p_close = float(pred_prices[i, 2])
        p_high  = float(pred_prices[i, 0])
        p_low   = float(pred_prices[i, 1])

        new_row   = pd.DataFrame(
            {'close': [p_close], 'high': [p_high], 'low': [p_low]},
            index=[ohlc_hist.index[-1] + 1]
        )
        ohlc_hist = pd.concat([ohlc_hist, new_row])

        feat_ext = _build_features(ohlc_hist)
        if feat_ext.empty:
            continue

        new_row_ind = feat_ext[feat_columns].iloc[-1].values.astype(np.float32).reshape(1, -1)
        new_scaled  = feat_scaler.transform(new_row_ind)[0]
        x_seed      = np.vstack([x_seed[1:], new_scaled])

    return signals

