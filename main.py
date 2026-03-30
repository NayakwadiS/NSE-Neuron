from models import *
from visualization.ploting import plot_candlestick_with_forecast
import config

scheme_code = input('Enter the NSE Share Symbol:- ')
algorithm = input('Select the algorithm for forecasting '
                  '1.LSTM '
                  '2.BiLSTM '
                  '3.GRU '
                  '4.CNN-LSTM:- ')


@getDataFrame(scheme_code)
def forecasting_mutual_fund(df, details, algorithm):
    signals = None  # only populated for LSTM

    match algorithm:
        case '1':
            pred, rmse = lstm(df)
            print("\n  Running LSTM Classifier for BUY/SELL/HOLD signals...\n")
            signals = lstm_classifier(df, pred)
        case '2': pred, rmse = bilstm(df)
        case '3': pred, rmse = gru(df)
        case '4': pred, rmse = cnn_lstm(df)
        case _: raise ValueError('Invalid algorithm selection. Choose 1.LSTM 2.BiLSTM 3.GRU 4.CNN-LSTM')

    # Generate next N business days from last date in df (N from config)
    last_date = pd.to_datetime(df['Date']).max()
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=config.FORECAST_DAYS)

    # Prepare table data for tabulate
    data = []
    for i in range(config.FORECAST_DAYS):
        row = [future_dates[i].strftime('%d-%b-%Y')] + list(pred[i])
        if signals:
            row.append(f"{signals[i]['label']} ({signals[i]['confidence']}%)")
        data.append(row)
    # Calculate min/max for each column
    min_vals = ['Min'] + list(np.min(pred, axis=0))
    max_vals = ['Max'] + list(np.max(pred, axis=0))
    if signals:
        min_vals.append('-')
        max_vals.append('-')
    data.append(min_vals)
    data.append(max_vals)

    headers = ["Date", "High", "Low", "Close", "Prev_Close"]
    if signals:
        headers.append("Signal")

    print("\n  Time Series Forecasting for " + details['scheme_name'] + " (" + str(details['scheme_code']) + ")\n")
    print(tabulate(data, headers=headers, tablefmt='orgtbl'))

    # Plotting
    plot_candlestick_with_forecast(df, details, pred, signals)

forecasting_mutual_fund(algorithm)
