from Algorithms import *
from plots.plloting import plot_candlestick_with_forecast

scheme_code = input('Enter the NSE Share Symbol:- ')


@getDataFrame(scheme_code)
def forecasting_mutual_fund(df, details):
    pred_LSTM, rmse_lstm = lstm(df)
    plot_candlestick_with_forecast(df, details, pred_LSTM)
    # Prepare table data for tabulate
    data = []
    for i in range(5):
        row = [f'Day {i+1}'] + list(pred_LSTM[i])
        data.append(row)
    # Calculate min/max for each column
    min_vals = ['Min'] + list(np.min(pred_LSTM, axis=0))
    max_vals = ['Max'] + list(np.max(pred_LSTM, axis=0))
    data.append(min_vals)
    data.append(max_vals)
    print("\n  Time Series Forecasting for " + details['scheme_name'] + " (" + str(details['scheme_code']) + ")\n")
    print(tabulate(data, headers=["Day", "Open", "High", "Low", "Close", "Prev_Close"], tablefmt='orgtbl'))

    # Plotting
    plot_candlestick_with_forecast(df, details, pred_LSTM)

forecasting_mutual_fund()
