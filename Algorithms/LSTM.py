from Algorithms import *


def lstm(df):
    days = 5
    # Select relevant columns and ensure numeric types
    feature_cols = ['open', 'high', 'low', 'close', 'prev_close']
    df_features = df[feature_cols].apply(pd.to_numeric, errors='coerce').dropna()

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
    for i in range(days):
        x_input_seq = x_input.reshape(1, n_steps, len(feature_cols))
        yhat = model.predict(x_input_seq, verbose=0)[0]
        # Inverse transform prediction
        yhat_inv = scaler.inverse_transform(yhat.reshape(1, -1))[0]
        lst_output.append(yhat_inv)
        # Prepare next input: shift, update prev_close
        x_input = np.vstack([x_input[1:], scaler.transform(yhat_inv.reshape(1, -1))])
        # Set prev_close for next step to predicted close
        x_input[-1, 4] = scaler.transform(yhat_inv.reshape(1, -1))[0, 3]  # prev_close = predicted close (scaled)

    forecasted_stock_price = np.array(lst_output)
    return forecasted_stock_price, rmse