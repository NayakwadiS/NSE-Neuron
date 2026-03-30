from models import *
from .base_model import BaseModel
from config import (
    FORECAST_DAYS,
    FEATURE_COLUMNS,
    TRAIN_TEST_SPLIT,
    RNN_UNITS,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    BATCH_SIZE, CNN_FILTERS_1, CNN_KERNEL_SIZE, CNN_FILTERS_2, CNN_N_SEQ, CNN_SUB_STEPS, CNN_TIME_STEP
)


def cnn_lstm(df):
    days = FORECAST_DAYS
    # Select relevant columns and ensure numeric types (strip commas first)
    feature_cols = FEATURE_COLUMNS
    df_features = df[feature_cols].copy()
    for col in feature_cols:
        df_features[col] = pd.to_numeric(
            df_features[col].astype(str).str.replace(',', '', regex=False),
            errors='coerce'
        )
    df_features = df_features.dropna()

    # Compute spreads: model learns spread instead of absolute high/low
    df_model = df_features.copy()
    df_model['high_spread'] = df_features['high'] - df_features['close']
    df_model['low_spread']  = df_features['close'] - df_features['low']
    df_model = df_model[['close', 'high_spread', 'low_spread', 'prev_close']]

    # Apply MinMax scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df_model)

    # Split dataset into train 80% and test 20%
    training_size = int(len(df_scaled) * TRAIN_TEST_SPLIT)
    train_data = df_scaled[0:training_size, :]
    test_data  = df_scaled[training_size:len(df_scaled), :]

    # CNN-LSTM uses a subsequence approach:
    # the full time_step window is split into n_seq subsequences of sub_steps each.
    # CNN extracts local features from each subsequence,
    # then LSTM captures temporal dependencies across subsequences.
    n_seq     = CNN_N_SEQ   # number of subsequences (LSTM time steps)
    sub_steps = CNN_SUB_STEPS   # timesteps per subsequence (CNN input length)
    time_step = CNN_TIME_STEP   # total lookback

    n_features = df_model.shape[1]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), :]
            dataX.append(a)
            dataY.append(dataset[i + time_step, :])
        return np.array(dataX), np.array(dataY)

    X_train, y_train = create_dataset(train_data, time_step)
    X_test,  ytest   = create_dataset(test_data,  time_step)

    # Reshape to [samples, n_seq, sub_steps, n_features] for TimeDistributed CNN
    X_train = X_train.reshape((X_train.shape[0], n_seq, sub_steps, n_features))
    X_test  = X_test.reshape((X_test.shape[0],   n_seq, sub_steps, n_features))

    # Build CNN-LSTM model
    # CNN block (TimeDistributed): extracts local spatial/pattern features from each sub-sequence
    # LSTM block: learns long-range temporal dependencies across the CNN feature maps
    model = Sequential([
        # CNN feature extractor applied independently to each of the n_seq sub-sequences
        TimeDistributed(Conv1D(filters=CNN_FILTERS_1, kernel_size=CNN_KERNEL_SIZE,
                              activation='relu', padding='same'),
                        input_shape=(n_seq, sub_steps, n_features)),
        TimeDistributed(MaxPooling1D(pool_size=2, padding='same')),
        TimeDistributed(Conv1D(filters=CNN_FILTERS_2, kernel_size=CNN_KERNEL_SIZE,
                              activation='relu', padding='same')),
        TimeDistributed(Flatten()),

        # LSTM reads the sequence of CNN feature vectors across n_seq windows
        KerasLSTM(RNN_UNITS, return_sequences=True),
        KerasLSTM(RNN_UNITS),

        # Output layer predicts all features at once
        Dense(n_features),
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    # EarlyStopping: stop when val_loss doesn't improve for patience epochs, restore best weights
    early_stop = EarlyStopping(monitor=EARLY_STOPPING_MONITOR,
                               patience=EARLY_STOPPING_PATIENCE,
                               restore_best_weights=True, verbose=1)
    model.fit(
        X_train, y_train,
        validation_data=(X_test, ytest),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
        callbacks=[early_stop]
    )

    # Prediction and performance metrics
    train_predict = model.predict(X_train)
    train_predict = scaler.inverse_transform(train_predict)

    # RMSE on close only
    rmse = {}
    rmse['close'] = math.sqrt(mean_squared_error(
        scaler.inverse_transform(y_train)[:, 0], train_predict[:, 0]
    ))

    # Forecast future 5 days
    # Keep a rolling buffer of (time_step) rows in scaled space
    x_input = test_data[len(test_data) - time_step:].copy()   # shape: (time_step, n_features)
    lst_output = []
    last_known_close = df_features['close'].iloc[-1]

    for i in range(days):
        # Reshape buffer -> [1, n_seq, sub_steps, n_features]
        x_input_seq = x_input.reshape(1, n_seq, sub_steps, n_features)
        yhat     = model.predict(x_input_seq, verbose=0)[0]
        yhat_inv = scaler.inverse_transform(yhat.reshape(1, -1))[0]

        # yhat_inv: [close, high_spread, low_spread, prev_close]
        pred_close       = yhat_inv[0]
        pred_high_spread = abs(yhat_inv[1])   # ensure spread is always positive
        pred_low_spread  = abs(yhat_inv[2])   # ensure spread is always positive
        pred_high        = pred_close + pred_high_spread
        pred_low         = pred_close - pred_low_spread

        # Chain prev_close: Day 1 uses last known close, subsequent days use predicted close
        pred_prev_close = last_known_close if i == 0 else lst_output[i - 1][2]

        # Output order: [high, low, close, prev_close] — matches table/plot expectations
        result = np.array([pred_high, pred_low, pred_close, pred_prev_close])
        lst_output.append(result)

        # Roll the input buffer forward by one step
        next_row_raw    = np.array([[pred_close, pred_high_spread, pred_low_spread, pred_prev_close]])
        next_row_scaled = scaler.transform(next_row_raw)[0]
        x_input = np.vstack([x_input[1:], next_row_scaled])

    forecasted_stock_price = np.array(lst_output)
    return forecasted_stock_price, rmse
