from models import *
from models.bilstm import bilstm
from models.cnn_lstm import cnn_lstm
from models.gru import gru
from models.lstm import lstm
from models.lstm_classifier import lstm_classifier
from visualization.ploting import plot_candlestick_with_forecast, plot_all_algos_forecast
from utils.regime_detector import detect_regime, apply_regime_confidence, regime_summary_line
import config

scheme_code = input('Enter the NSE Share Symbol:- ')
choice = input(
    "Select the algorithm for forecasting:\n"
    "1. LSTM with Classifier\n"
    "2. BiLSTM\n"
    "3. GRU\n"
    "4. CNN-LSTM\n"
    "5. Run All\n"
    "6. Regime Analysis (detect market regime, then choose model)\n"
    "Selection: "
)


@getDataFrame(scheme_code)
def forecasting_nse_stocks(df, details, choice):
    signals = None
    pred    = None
    rmse    = None

    match choice:
        case '1':
            pred, rmse = lstm(df)
            print("\nRunning LSTM Classifier...")
            signals = lstm_classifier(df, pred)
        case '2': pred, rmse = bilstm(df)
        case '3': pred, rmse = gru(df)
        case '4': pred, rmse = cnn_lstm(df)
        case '5':
            ALGO_NAMES = ['LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM']
            algo_funcs = [lstm, bilstm, gru, cnn_lstm]
            all_preds  = {}
            all_rmse   = {}

            for name, func in zip(ALGO_NAMES, algo_funcs):
                print(f"\n[{name}] Training...")
                p, r = func(df)
                all_preds[name] = p
                all_rmse[name]  = r['close'] if isinstance(r, dict) else r

            last_date    = pd.to_datetime(df['Date']).max()
            future_dates = pd.bdate_range(
                start=last_date + pd.Timedelta(days=1), periods=config.FORECAST_DAYS
            )
            date_labels = [d.strftime('%d-%b-%Y') for d in future_dates]

            print(f"\n\nClose Price Forecast — {details['scheme_name']} ({details['scheme_code']})\n")
            day_headers = ["Algorithm"] + [f"Day {i+1}\n{date_labels[i]}" for i in range(config.FORECAST_DAYS)]
            comp_data = []
            for name in ALGO_NAMES:
                close_vals = [round(all_preds[name][i][2], 2) for i in range(config.FORECAST_DAYS)]
                comp_data.append([name] + close_vals)
            print(tabulate(comp_data, headers=day_headers, tablefmt='orgtbl'))

            print(f"\n\nModel Benchmark — RMSE\n")
            bench_data = [[name, f"{all_rmse[name]:.6f}"] for name in ALGO_NAMES]
            best_algo  = min(all_rmse, key=all_rmse.get)
            bench_data.append(['Best Model', f"{best_algo} "])
            print(tabulate(bench_data, headers=["Algorithm", "RMSE"], tablefmt='orgtbl'))

            plot_all_algos_forecast(df, details, all_preds, ALGO_NAMES)
            return

        case '6':
            regime = detect_regime(df)

            print(f"\n{'─'*60}")
            print(f"  Regime Analysis — {details['scheme_name']} ({details['scheme_code']})")
            print(f"{'─'*60}")
            if regime['sufficient_data']:
                emoji = {'BULL': '🟢', 'BEAR': '🔴', 'SIDEWAYS': '🟡'}.get(regime['regime'], '⚪')
                print(f"  Regime     : {emoji}  {regime['regime']}")
                print(f"  SMA{config.REGIME_SMA_FAST}       : {regime['sma_fast']:.2f}")
                print(f"  SMA{config.REGIME_SMA_SLOW}      : {regime['sma_slow']:.2f}")
                print(f"  Suggestion : {regime['recommended_model']} works best in this regime")
            else:
                print(f"  {regime['description']}")
            print(f"{'─'*60}\n")

            # Step 3: ask user to pick model (1–5 only, no recursive 6)
            model_choice = input(
                "  Select algorithm to run:\n"
                "  1. LSTM with Classifier\n"
                "  2. BiLSTM\n"
                "  3. GRU\n"
                "  4. CNN-LSTM\n"
                "  5. Run All & Compare\n"
                "  Selection: "
            ).strip()

            if model_choice not in ('1', '2', '3', '4', '5'):
                raise ValueError('Invalid selection. Choose 1–5.')

            # Step 4: run chosen model(s)
            if model_choice == '5':
                ALGO_NAMES = ['LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM']
                algo_funcs = [lstm, bilstm, gru, cnn_lstm]
                all_preds  = {}
                all_rmse   = {}

                for name, func in zip(ALGO_NAMES, algo_funcs):
                    print(f"\n[{name}] Training...")
                    p, r = func(df)
                    all_preds[name] = p
                    all_rmse[name]  = r['close'] if isinstance(r, dict) else r

                last_date    = pd.to_datetime(df['Date']).max()
                future_dates = pd.bdate_range(
                    start=last_date + pd.Timedelta(days=1), periods=config.FORECAST_DAYS
                )
                date_labels = [d.strftime('%d-%b-%Y') for d in future_dates]

                print(f"\n\nClose Price Forecast — {details['scheme_name']} ({details['scheme_code']})")
                print(f"{regime_summary_line(regime)}\n")
                day_headers = ["Algorithm"] + [f"Day {i+1}\n{date_labels[i]}" for i in range(config.FORECAST_DAYS)]
                comp_data = []
                for name in ALGO_NAMES:
                    tag        = '  ←' if name == regime['recommended_model'] else ''
                    close_vals = [round(all_preds[name][i][2], 2) for i in range(config.FORECAST_DAYS)]
                    comp_data.append([f"{name}{tag}"] + close_vals)
                print(tabulate(comp_data, headers=day_headers, tablefmt='orgtbl'))

                print(f"\n\nModel Benchmark — RMSE\n")
                bench_data = [[name, f"{all_rmse[name]:.6f}"] for name in ALGO_NAMES]
                best_algo  = min(all_rmse, key=all_rmse.get)
                bench_data.append(['Best Model', f"{best_algo} "])
                print(tabulate(bench_data, headers=["Algorithm", "RMSE"], tablefmt='orgtbl'))
                print(f"\n{regime_summary_line(regime)}")

                plot_all_algos_forecast(df, details, all_preds, ALGO_NAMES)
                return

            model_map   = {'1': lstm, '2': bilstm, '3': gru, '4': cnn_lstm}
            name_map    = {'1': 'LSTM', '2': 'BiLSTM', '3': 'GRU', '4': 'CNN-LSTM'}
            chosen_func = model_map[model_choice]
            chosen_name = name_map[model_choice]

            pred, rmse = chosen_func(df)

            # Run classifier only when LSTM is selected — mirrors case '1' behaviour
            signals = None
            if model_choice == '1':
                print("\nRunning LSTM Classifier...")
                signals = lstm_classifier(df, pred)
                if regime['sufficient_data'] and signals:
                    signals = apply_regime_confidence(signals, regime)

            last_date    = pd.to_datetime(df['Date']).max()
            future_dates = pd.bdate_range(
                start=last_date + pd.Timedelta(days=1), periods=config.FORECAST_DAYS
            )
            data = []
            for i in range(config.FORECAST_DAYS):
                row = [future_dates[i].strftime('%d-%b-%Y')] + list(pred[i])
                if signals:
                    sig = signals[i]
                    adj = f" {sig.get('regime_direction', '')}" if sig.get('regime_adjusted') else ''
                    row.append(f"{sig['label']} ({sig['confidence']}%{adj})")
                data.append(row)

            min_vals = ['Min'] + list(np.min(pred, axis=0))
            max_vals = ['Max'] + list(np.max(pred, axis=0))
            if signals:
                min_vals.append('-')
                max_vals.append('-')
            data.extend([min_vals, max_vals])

            headers = ["Date", "High", "Low", "Close", "Prev_Close"]
            if signals:
                headers.append("Signal")

            print(f"\nForecast — {details['scheme_name']} ({details['scheme_code']})  [{chosen_name}]\n")
            print(tabulate(data, headers=headers, tablefmt='orgtbl'))
            print(f"\n{regime_summary_line(regime)}")

            plot_candlestick_with_forecast(df, details, pred, signals)
            return

        case _:
            raise ValueError('Invalid selection. Choose 1–6.')

    # ── cases 1–4 output ─────────────────────────────────────────────────────
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

    plot_candlestick_with_forecast(df, details, pred, signals)

forecasting_nse_stocks(choice)
