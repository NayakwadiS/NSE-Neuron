"""
Configuration file for NSE-Neuron project
Contains all hyperparameters, thresholds, and settings used across models
"""

# ══════════════════════════════════════════════════════════════════════════════
# FORECASTING PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Number of future days to predict
FORECAST_DAYS = 5

# Train/test split ratio
TRAIN_TEST_SPLIT = 0.80

# Time step (lookback window) for sequence models
TIME_STEP = 10

# Feature columns used for modeling
FEATURE_COLUMNS = ['close', 'high', 'low', 'prev_close']


# ══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# LSTM / BiLSTM / GRU Units
# 64 units is the standard for financial time series
# - 32: too small, underfits price patterns
# - 64: good balance of capacity vs training speed
# - 128+: diminishing returns, slower, risk of overfitting on daily stock data
RNN_UNITS = 64

# Training epochs (with early stopping, actual epochs may be less)
EPOCHS = 100

# Batch size for training
BATCH_SIZE = 64

# Early stopping patience (epochs to wait for improvement)
EARLY_STOPPING_PATIENCE = 5

# Validation metric to monitor for early stopping
EARLY_STOPPING_MONITOR = 'val_loss'


# ══════════════════════════════════════════════════════════════════════════════
# CNN-LSTM SPECIFIC PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# CNN-LSTM uses a subsequence approach:
# the full time_step window is split into N_SEQ subsequences of SUB_STEPS each
# e.g. time_step=20, n_seq=4, sub_steps=5 -> 4 windows of 5 days each
CNN_N_SEQ = 4           # number of subsequences (LSTM time steps)
CNN_SUB_STEPS = 5       # timesteps per subsequence (CNN input length)
CNN_TIME_STEP = CNN_N_SEQ * CNN_SUB_STEPS  # total lookback = 20

# CNN filter sizes
CNN_FILTERS_1 = 64      # first conv layer
CNN_FILTERS_2 = 32      # second conv layer
CNN_KERNEL_SIZE = 3     # kernel size for conv layers


# ══════════════════════════════════════════════════════════════════════════════
# LSTM CLASSIFIER PARAMETERS (BUY/SELL/HOLD)
# ══════════════════════════════════════════════════════════════════════════════

# Classifier time step (longer lookback for better context)
CLASSIFIER_TIME_STEP = 20

# Classifier training parameters
CLASSIFIER_EPOCHS = 200
CLASSIFIER_BATCH_SIZE = 16  # small batch for more weight updates per epoch
CLASSIFIER_PATIENCE = 15    # patience for early stopping
CLASSIFIER_MIN_DELTA = 0.001  # minimum improvement to count

# LSTM units for classifier
CLASSIFIER_LSTM_UNITS_1 = 64
CLASSIFIER_LSTM_UNITS_2 = 32
CLASSIFIER_DENSE_UNITS = 32
CLASSIFIER_DROPOUT = 0.2

# Signal threshold for BUY/SELL classification
# BUY:  next day return > +CLASSIFIER_THRESHOLD
# SELL: next day return < -CLASSIFIER_THRESHOLD
# HOLD: otherwise
CLASSIFIER_THRESHOLD = 0.01  # 1% threshold

# Train/val split for classifier
CLASSIFIER_TRAIN_SPLIT = 0.80


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS PARAMETERS (used in classifier)
# ══════════════════════════════════════════════════════════════════════════════

# EMA periods
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50

# RSI period
RSI_PERIOD = 14

# MACD parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands period
BB_PERIOD = 20

# ATR (Average True Range) period
ATR_PERIOD = 14

# Stochastic Oscillator parameters
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3

# Williams %R period
WILLIAMS_R_PERIOD = 14


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Number of historical days to show in candlestick chart
PLOT_HISTORICAL_DAYS = 100

# Signal colors for BUY/SELL/HOLD
SIGNAL_COLORS = {
    0: '#ef4444',  # SELL - red
    1: '#f59e0b',  # HOLD - orange/yellow
    2: '#22c55e'   # BUY - green
}

SIGNAL_LABELS = {
    0: 'SELL',
    1: 'HOLD',
    2: 'BUY'
}

SIGNAL_MARKERS = {
    0: 'v',   # down triangle for SELL
    1: 'o',   # circle for HOLD
    2: '^'    # up triangle for BUY
}

# Signal marker offsets (as fraction of price)
SIGNAL_OFFSETS = {
    'BUY': -0.03,   # below the price
    'SELL': 0.03,   # above the price
    'HOLD': 0.0     # at the price
}


# ══════════════════════════════════════════════════════════════════════════════
# SCALER PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# MinMaxScaler range
SCALER_FEATURE_RANGE = (0, 1)

