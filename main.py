from Algorithms import *
from plots.plloting import plot_candlestick_with_forecast

scheme_code = input('Enter the NSE Share Symbol:- ')
algorithm = input('Select the algorithm for forecasting 1.LSTM 2.BiLSTM:- ')


@getDataFrame(scheme_code)
def forecasting_mutual_fund(df, details, algorithm):
    match algorithm:
        case '1': pred, rmse = lstm(df)
        case '2': pred, rmse = bilstm(df)
        case _: raise ValueError('Invalid algorithm selection. Choose 1 for LSTM or 2 for BiLSTM.')
    # Generate next 5 business days from last date in df
    last_date = pd.to_datetime(df['Date']).max()
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)

    # Prepare table data for tabulate
    data = []
    for i in range(5):
        row = [future_dates[i].strftime('%d-%b-%Y')] + list(pred[i])
        data.append(row)
    # Calculate min/max for each column
    min_vals = ['Min'] + list(np.min(pred, axis=0))
    max_vals = ['Max'] + list(np.max(pred, axis=0))
    data.append(min_vals)
    data.append(max_vals)
    print("\n  Time Series Forecasting for " + details['scheme_name'] + " (" + str(details['scheme_code']) + ")\n")
    print(tabulate(data, headers=["Date", "High", "Low", "Close", "Prev_Close"], tablefmt='orgtbl'))
    # Plotting
    plot_candlestick_with_forecast(df, details, pred)

forecasting_mutual_fund(algorithm)
