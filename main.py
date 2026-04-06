from models import *
from models.bilstm import bilstm
from models.cnn_lstm import cnn_lstm
from models.gru import gru
from models.lstm import lstm
from models.lstm_classifier import lstm_classifier
from visualization.ploting import plot_candlestick_with_forecast, plot_all_algos_forecast
import config

scheme_code = input('Enter the NSE Share Symbol:- ')
choice = input("Select the algorithm for forecasting:\n1. LSTM with Classifier\n2. BiLSTM\n3. GRU\n4. CNN-LSTM\n5. Run All\nSelection: ")


@getDataFrame(scheme_code)
def forecasting_mutual_fund(df, details, choice):
    signals = None  # only populated for LSTM
    pred    = None
    rmse    = None

    match choice:
        case '1':
            pred, rmse = lstm(df)
            print("\n  Running LSTM Classifier for BUY/SELL/HOLD signals...\n")
            signals = lstm_classifier(df, pred)
        case '2': pred, rmse = bilstm(df)
        case '3': pred, rmse = gru(df)
        case '4': pred, rmse = cnn_lstm(df)
        case '5':
            # ── Run All Algorithms
            ALGO_NAMES = ['LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM']
            algo_funcs = [lstm, bilstm, gru, cnn_lstm]
            all_preds  = {}
            all_rmse   = {}

            for name, func in zip(ALGO_NAMES, algo_funcs):
                print(f"\n  Running {name}...")
                p, r = func(df)
                all_preds[name] = p
                all_rmse[name]  = r['close'] if isinstance(r, dict) else r

            last_date    = pd.to_datetime(df['Date']).max()
            future_dates = pd.bdate_range(
                start=last_date + pd.Timedelta(days=1), periods=config.FORECAST_DAYS
            )
            date_labels = [d.strftime('%d-%b-%Y') for d in future_dates]

            # ── Forecast comparison table (rows = algos, cols = days) ─────────
            print(f"\n\n  Close Price Forecast Comparison — {details['scheme_name']} ({details['scheme_code']})\n")
            day_headers = ["Algorithm"] + [f"Day {i+1}\n{date_labels[i]}" for i in range(config.FORECAST_DAYS)]
            comp_data = []
            for name in ALGO_NAMES:
                close_vals = [round(all_preds[name][i][2], 2) for i in range(config.FORECAST_DAYS)]
                comp_data.append([name] + close_vals)
            print(tabulate(comp_data, headers=day_headers, tablefmt='orgtbl'))

            # ── Benchmark table (RMSE per algo) ───────────────────────────────
            print(f"\n\n  Model Benchmark — RMSE\n")
            bench_data = [[name, f"{all_rmse[name]:.6f}"] for name in ALGO_NAMES]
            best_algo  = min(all_rmse, key=all_rmse.get)
            bench_data.append(['Best Model', f"{best_algo}  ✅"])
            print(tabulate(bench_data, headers=["Algorithm", "RMSE"], tablefmt='orgtbl'))

            # ── Plot all algos together ───────────────────────────────────────
            plot_all_algos_forecast(df, details, all_preds, ALGO_NAMES)
            return
        case _: raise ValueError('Invalid algorithm selection. Choose 1-5')

    # ── Single-algo output (cases 1-4) ───────────────────────────────────────
    last_date    = pd.to_datetime(df['Date']).max()
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

forecasting_mutual_fund(choice)
