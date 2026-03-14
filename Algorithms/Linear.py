from Algorithms import *


def linear(df):
    days = 5
    last_week = df.iloc[-days:]
    last_day = df.iloc[-1:]

    df_new = df.copy()
    # Robust date conversion
    try:
        df_new['date'] = pd.to_datetime(df_new['date'], format='mixed', dayfirst=True, errors='coerce')
        # Convert str 'nav' column to float
        df_new['close'] = pd.to_numeric(df_new['close'], errors='coerce')
    except Exception as e:
        print(f"Date conversion error: {e}")
        raise
    # Drop rows with invalid dates and nav values
    df_new = df_new.dropna(subset=['date', 'close'])

    df_new['Prev Close'] = df_new['close']
    df_new['CLOSE'] = df_new['close'].shift(1)
    df_new.set_index('date', inplace=True)

    X = df_new[['Prev Close']][1:]
    y = df_new['CLOSE'][1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # To Train model on previous data
    model = LinearRegression().fit(X_train, y_train)
    r_sq = model.score(X_test, y_test)
    print('confidence of determination in Linear:', r_sq)
    # print('intercept:', model.intercept_)
    # print('slope:', model.coef_)

    # for Test purpose to check confidence of model
    y_pred_test = model.predict(X_test)
    #RMSE
    rmse= math.sqrt(mean_squared_error(y_test,y_pred_test))

    # Actual prediction
    pre_date = date.today()
    day_index = [(pre_date + dt.timedelta(days=i)) for i in range(1,days + 1)]
    # Use only numeric values for last_week
    last_week = [row['close'] for index, row in df_new.tail(days).iterrows() if pd.notnull(row['close'])]

    # 1 day forecasting
    X_new = pd.DataFrame([last_week[-1]], columns=['Prev Close'], index=[pre_date])
    y_pred = model.predict(X_new)
    # print("1 day prediction :",y_pred)

    y_pred_days = [y_pred]
    # month forecasting
    for i in range(0,days - 1):
        X_new = pd.DataFrame(y_pred,columns =['Prev Close'],index=[day_index[i]])
        y_pred = model.predict(X_new)
        y_pred_days.append(y_pred)

    return y_pred_days, rmse