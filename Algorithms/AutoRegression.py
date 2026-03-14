from Algorithms import *


def AutoR(df):
    days = 5
    df_new = df.copy()
    # Ensure 'close' is numeric and drop NaNs
    df_new['close'] = pd.to_numeric(df_new['close'], errors='coerce')
    df_new = df_new.dropna(subset=['close'])

    # Use only the 'close' column for AutoReg
    close_series = df_new['close']
    X = close_series.dropna()
    train, test = X[1:len(X)-days], X[len(X)-days:]

    # train autoregression
    window = 50
    model = AutoReg(train.values, lags=window)
    model_fit = model.fit()
    coef = model_fit.params
    print('Coefficients in Auto Regression: %s' % coef)

    predictions = model_fit.predict(start=len(train),end=len(train) + len(test)-1,dynamic=False)
    # print('NAV Predicted -', predictions)

    #### below code can be use for prediction without using '.predict' method

    # history = train[len(train)-window:]
    # history = [history[i] for i in range(len(history))]
    # predictions = list()
    #
    # for t in range(len(test)):
    #     length = len(history)
    #     lag = [history[i] for i in range(length-window,length)]
    #     yhat = coef[0]
    #     for d in range(window):
    #         yhat += coef[d+1] * lag[window-d-1]
    #     obs = test[t]
    #     predictions.append(yhat)
    #     history.append(obs)
    #     print('predicted=%f, expected=%f' % (yhat, obs))
    rmse = math.sqrt(mean_squared_error(test, predictions))
    # print('Test RMSE: %.3f' % rmse)

    # Plot prediction
    # last_date = df_new.iloc[-1].name
    # next_date =dt.date(last_date.year,last_date.month,last_date.day)
    # df_new['predict'] = np.nan
    #
    # for y in predictions:
    #     next_date += dt.timedelta(days=1)
    #     df_new.loc[next_date] = [np.nan] + [y]
    #
    # # print(df_predict.tail(15))
    # df_new['nav'].tail(15).plot(color='blue')
    # df_new['predict'].tail(15).plot(color='red')
    # plt.ylabel('NAV')
    # plt.show()
    return predictions, rmse