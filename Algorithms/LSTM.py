from Algorithms import *


def lstm(df):
    days = 5
    # Select relevant columns and ensure numeric types (strip commas first)
    feature_cols = ['high', 'low', 'close', 'prev_close']
    df_features = df[feature_cols].copy()
    for col in feature_cols:
        df_features[col] = pd.to_numeric(
            df_features[col].astype(str).str.replace(',', '', regex=False),
            errors='coerce'
        )
    df_features = df_features.dropna()

    # Apply MinMax scaler to all features
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df_features)

    # Split dataset into train 80% and test 20%
    training_size = int(len(df_scaled) * 0.80)
    test_size = len(df_scaled) - training_size
    train_data, test_data = df_scaled[0:training_size, :], df_scaled[training_size:len(df_scaled), :]

    # Convert array of values into a dataset matrix for multivariate
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), :]  # shape: (time_step, features)
            dataX.append(a)
            dataY.append(dataset[i + time_step, :])  # Predict all features
        return np.array(dataX), np.array(dataY)

    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # Create the Stacked LSTM model
    model = Sequential()
    model.add(LSTM(10, return_sequences=True, input_shape=(time_step, len(feature_cols))))
    model.add(LSTM(10, return_sequences=True))
    model.add(LSTM(10))
    model.add(Dense(len(feature_cols)))  # Output all features
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=5, batch_size=64, verbose=1)

    # Prediction and performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform all columns
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Calculate RMSE for each feature
    rmse = {}
    for idx, col in enumerate(feature_cols):
        rmse[col] = math.sqrt(mean_squared_error(y_train[:, idx], train_predict[:, idx]))

    # Forecast future values with prev_close chaining
    x_input = test_data[len(test_data) - time_step:].copy()
    lst_output = []
    n_steps = time_step

    # Get last known close from the original data for Day 1 prev_close
    last_known_close = df_features['close'].iloc[-1]

    for i in range(days):
        x_input_seq = x_input.reshape(1, n_steps, len(feature_cols))
        yhat = model.predict(x_input_seq, verbose=0)[0]
        # Inverse transform prediction
        yhat_inv = scaler.inverse_transform(yhat.reshape(1, -1))[0]

        # Force prev_close to be the actual previous day's close (chaining)
        if i == 0:
            yhat_inv[3] = last_known_close          # Day 1: use last real close
        else:
            yhat_inv[3] = lst_output[i - 1][2]     # Day N: use Day N-1 predicted close

        # Enforce logical constraints: high >= close >= low
        high, low, close = yhat_inv[0], yhat_inv[1], yhat_inv[2]
        yhat_inv[0] = max(high, close)   # high must be >= close
        yhat_inv[1] = min(low, close)    # low must be <= close

        lst_output.append(yhat_inv)

        # Prepare next input: shift window forward, set prev_close in scaled input
        next_row = scaler.transform(yhat_inv.reshape(1, -1))[0]
        x_input = np.vstack([x_input[1:], next_row])

    forecasted_stock_price = np.array(lst_output)
    return forecasted_stock_price, rmse
