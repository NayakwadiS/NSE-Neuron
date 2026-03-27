from Algorithms import *


def bilstm(df):
    days = 5
    # Select relevant columns and ensure numeric types (strip commas first)
    feature_cols = ['close', 'high', 'low', 'prev_close']
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
    df_model['low_spread'] = df_features['close'] - df_features['low']
    df_model = df_model[['close', 'high_spread', 'low_spread', 'prev_close']]

    # Apply MinMax scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df_model)

    # Split dataset into train 80% and test 20%
    training_size = int(len(df_scaled) * 0.80)
    train_data, test_data = df_scaled[0:training_size, :], df_scaled[training_size:len(df_scaled), :]

    # Convert array of values into a dataset matrix for multivariate
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), :]
            dataX.append(a)
            dataY.append(dataset[i + time_step, :])
        return np.array(dataX), np.array(dataY)

    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    units = 64

    # Create the Bidirectional LSTM model
    # Bidirectional reads sequence in both forward and backward directions
    # giving the model context from both past and future within the time window
    model = Sequential()
    model.add(Bidirectional(KerasLSTM(units, return_sequences=True), input_shape=(time_step, df_model.shape[1])))
    model.add(Bidirectional(KerasLSTM(units, return_sequences=True)))
    model.add(Bidirectional(KerasLSTM(units)))
    model.add(Dense(df_model.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # EarlyStopping: stop when val_loss doesn't improve for 5 epochs, restore best weights
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, ytest),
        epochs=100,
        batch_size=64,
        verbose=1,
        callbacks=[early_stop],
    )

    # Prediction and performance metrics
    train_predict = model.predict(X_train)
    train_predict = scaler.inverse_transform(train_predict)

    # RMSE on close only
    rmse = {}
    rmse['close'] = math.sqrt(mean_squared_error(
        scaler.inverse_transform(y_train)[:, 0], train_predict[:, 0]
    ))

    # Forecast future values
    x_input = test_data[len(test_data) - time_step:].copy()
    lst_output = []
    last_known_close = df_features['close'].iloc[-1]

    for i in range(days):
        x_input_seq = x_input.reshape(1, time_step, df_model.shape[1])
        yhat = model.predict(x_input_seq, verbose=0)[0]
        yhat_inv = scaler.inverse_transform(yhat.reshape(1, -1))[0]

        # yhat_inv: [close, high_spread, low_spread, prev_close]
        pred_close = yhat_inv[0]
        pred_high_spread = abs(yhat_inv[1])  # ensure spread is always positive
        pred_low_spread = abs(yhat_inv[2])  # ensure spread is always positive
        pred_high = pred_close + pred_high_spread
        pred_low = pred_close - pred_low_spread

        # Chain prev_close correctly
        pred_prev_close = last_known_close if i == 0 else lst_output[i - 1][2]

        # Output order: [high, low, close, prev_close] to match table/plot expectations
        result = np.array([pred_high, pred_low, pred_close, pred_prev_close])
        lst_output.append(result)

        # Prepare next scaled input row using spread representation
        next_row_raw = np.array([[pred_close, pred_high_spread, pred_low_spread, pred_prev_close]])
        next_row_scaled = scaler.transform(next_row_raw)[0]
        x_input = np.vstack([x_input[1:], next_row_scaled])

    forecasted_stock_price = np.array(lst_output)
    return forecasted_stock_price, rmse
