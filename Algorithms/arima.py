from Algorithms import *
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

def arima(df):
    days = 5
    # Convert 'close' to numeric and drop NaNs
    df_new = pd.to_numeric(df['close'], errors='coerce').dropna()

    X = df_new.values
    size = int(len(X) * 0.90)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    prediction = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        prediction.append(yhat)
        obs = test[t]
        history.append(obs)
    # Remove NaNs from test and prediction
    test_clean = np.array(test)[~np.isnan(test)]
    prediction_clean = np.array(prediction)[~np.isnan(prediction)]
    # Ensure lengths match
    min_len = min(len(test_clean), len(prediction_clean))
    rmse = math.sqrt(mean_squared_error(test_clean[:min_len], prediction_clean[:min_len]))

    # Actual 30 days Forecasting
    history = [x for x in X]
    forecasting = []
    for i in range(days):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        forecasting.append(yhat)    #future day
        history.append(yhat)
    print("ARIMA",forecasting)

    # plot
    # pyplot.plot(Y[-5:])                      # last 5 days
    # pyplot.plot(test[], color='green')        # 10% data used as test
    # plt.plot(forecasting, color='red')        # 5 days forecasting
    # plt.show()
    return forecasting, rmse


def arima_new(df):
    # Convert 'close' to numeric and handle any non-numeric values
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    # Convert date to datetime and set as index, removing duplicates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.drop_duplicates(subset='date', keep='first')

    # Drop rows with NaN in 'close' column before setting index
    df = df.dropna(subset=['close'])
    df.index = pd.DatetimeIndex(df['date'])

    df['rolling_av'] = df['close'].rolling(10).mean()

    # creating the model
    MA_model = ARIMA(endog=df['close'], order=(0, 0, 55))

    # fitting data to the model
    results = MA_model.fit()
    # summary of the model
    print(results.summary())

    # prediction data
    start_date = df.index[-5]
    end_date = df.index[-1]
    predictions = results.predict(start=start_date, end=end_date)

    # Align predictions with dataframe index
    df.loc[predictions.index, 'prediction'] = predictions.values

    # Calculate RMSE only on rows where both close and prediction are not NaN
    last_5 = df[['close', 'prediction']].tail(5).dropna()

    if len(last_5) > 0:
        rmse = math.sqrt(mean_squared_error(last_5['close'], last_5['prediction']))
    else:
        rmse = float('nan')

    # printing last 5 values of the prediction with original and rolling avg value
    print(df[['close', 'prediction', 'rolling_av']].tail(5))

    # Forecast future closing prices
    forecast_steps = 5  # Forecasting for the next 5 days
    forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='D')[1:]
    forecast = results.forecast(steps=forecast_steps)

    # plotting the end results
    # df[['close', 'rolling_av', 'prediction']].plot()
    # plt.plot(forecast_index, forecast, color='red', label='Forecast')

    return forecast.tolist(), rmse
