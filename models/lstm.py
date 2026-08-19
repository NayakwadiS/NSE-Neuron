from models import *
from models.base_model import BaseModel
from utils import model_registry
from config import (
    FORECAST_DAYS,
    FEATURE_COLUMNS,
    TIME_STEP,
    TRAIN_TEST_SPLIT,
    RNN_UNITS,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    BATCH_SIZE
)


class LSTMModel(BaseModel):
    """
    Stacked LSTM model for multi-variate stock price forecasting.
    Derives from BaseModel and implements fit, predict, and evaluate.
    """

    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.time_step = TIME_STEP
        self.units = RNN_UNITS
        self.n_features = None          # set during fit
        self.test_data = None           # kept for forecasting window
        self.df_features = None         # kept for last known close
        self.cache_status = 'miss'      # 'miss' | 'fresh' | 'warm'
        self.cache_meta = {}

    # ------------------------------------------------------------------
    # Helper: turn raw DataFrame into scaled numpy arrays
    # ------------------------------------------------------------------
    def _prepare_data(self, df):
        feature_cols = FEATURE_COLUMNS
        df_features = df[feature_cols].copy()
        for col in feature_cols:
            df_features[col] = pd.to_numeric(
                df_features[col].astype(str).str.replace(',', '', regex=False),
                errors='coerce'
            )
        df_features = df_features.dropna()

        # Use spread representation so high/low stay consistent with close
        df_model = df_features.copy()
        df_model['high_spread'] = df_features['high'] - df_features['close']
        df_model['low_spread']  = df_features['close'] - df_features['low']
        df_model = df_model[['close', 'high_spread', 'low_spread', 'prev_close']]

        return df_features, df_model

    # ------------------------------------------------------------------
    # Helper: sliding-window dataset
    # ------------------------------------------------------------------
    @staticmethod
    def _create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            dataX.append(dataset[i:(i + time_step), :])
            dataY.append(dataset[i + time_step, :])
        return np.array(dataX), np.array(dataY)

    # ------------------------------------------------------------------
    # BaseModel: fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the LSTM on pre-built (X_train, y_train) arrays."""
        early_stop = EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        self.model.fit(
            X, y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=1,
            callbacks=[early_stop]
        )

    # ------------------------------------------------------------------
    # BaseModel: predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return raw scaled predictions from the trained model."""
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # BaseModel: evaluate
    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return RMSE for the close column on scaled values (fair cross-model comparison)."""
        preds   = self.predict(X)        # scaled predictions
        return math.sqrt(mean_squared_error(y[:, 0], preds[:, 0]))

    # ------------------------------------------------------------------
    # Architecture definition (kept separate so it can be skipped on cache hit)
    # ------------------------------------------------------------------
    def _build(self):
        # 64 units: good balance of capacity vs speed for daily stock data
        model = Sequential()
        model.add(LSTM(self.units, return_sequences=True,
                       input_shape=(self.time_step, self.n_features)))
        model.add(LSTM(self.units, return_sequences=True))
        model.add(LSTM(self.units))
        model.add(Dense(self.n_features))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    # ------------------------------------------------------------------
    # Cache signature — everything that must match for weight reuse
    # ------------------------------------------------------------------
    def _signature(self, df):
        return model_registry.build_signature('LSTM', df, {
            'time_step':  self.time_step,
            'units':      self.units,
            'n_features': self.n_features,
            'arch':       'stacked-lstm-3x',
        })

    # ------------------------------------------------------------------
    # Main entry: train model + forecast future days
    # ------------------------------------------------------------------
    def run(self, df, symbol=None, force_retrain=False):
        """
        Full pipeline: preprocess → load-or-build model → fit → forecast.

        Parameters
        ----------
        symbol        : NSE symbol — enables the per-symbol weight cache
        force_retrain : ignore any cached weights and train from scratch

        Returns (forecasted_stock_price, rmse) — unchanged shape.
        """
        df_features, df_model = self._prepare_data(df)
        self.df_features = df_features
        self.n_features = df_model.shape[1]

        # ── Try the weight cache ─────────────────────────────────────────────
        sig = self._signature(df)
        cached, cached_scaler, status, meta = model_registry.load(
            symbol, 'LSTM', sig, force_retrain=force_retrain
        )
        self.cache_status = status
        self.cache_meta   = meta

        # Scale — CRITICAL: only fit a new scaler on a cache miss, otherwise the
        # cached weights would be fed data in a different numeric space.
        if status == 'miss':
            df_scaled = self.scaler.fit_transform(df_model)
        else:
            self.scaler = cached_scaler
            self.model  = cached
            df_scaled   = self.scaler.transform(df_model)

        # Train / test split
        training_size = int(len(df_scaled) * TRAIN_TEST_SPLIT)
        train_data = df_scaled[:training_size, :]
        test_data  = df_scaled[training_size:, :]
        self.test_data = test_data

        # Sliding-window datasets
        X_train, y_train = self._create_dataset(train_data, self.time_step)
        X_test,  y_test  = self._create_dataset(test_data,  self.time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], self.n_features)
        X_test  = X_test.reshape(X_test.shape[0],   X_test.shape[1],  self.n_features)

        # ── Train / fine-tune / skip ─────────────────────────────────────────
        if status == 'miss':
            self.model = self._build()
            self.fit(X_train, y_train)
        elif status == 'warm':
            print("  [Cache] Warm-starting LSTM on newest windows…")
            model_registry.warm_start(self.model, X_train, y_train,
                                      batch_size=BATCH_SIZE)
        else:
            print("  [Cache] Using cached LSTM weights as-is (no training).")

        # ── RMSE on full dataset for fair cross-model benchmarking ────────────
        # Using only test set gives noisy results because the test window is small
        # (~20% of data). Evaluating on the full sequence gives a stable, comparable
        # number across all models regardless of their time_step size.
        # NOTE: always recomputed, even on a 'fresh' cache hit, so the Run-All
        # benchmark never compares a stale RMSE against freshly trained ones.
        X_all, y_all = self._create_dataset(df_scaled, self.time_step)
        X_all = X_all.reshape(X_all.shape[0], X_all.shape[1], self.n_features)
        rmse = {'close': self.evaluate(X_all, y_all)}

        # ── Persist weights (skip when nothing changed) ───────────────────────
        if status != 'fresh':
            model_registry.save(symbol, 'LSTM', self.model, self.scaler, sig,
                                {'rmse': rmse['close']})

        # Forecast FORECAST_DAYS into the future
        forecasted_stock_price = self._forecast(df_features)

        return forecasted_stock_price, rmse

    # ------------------------------------------------------------------
    # Iterative multi-step forecast
    # ------------------------------------------------------------------
    def _forecast(self, df_features):
        days = FORECAST_DAYS
        x_input = self.test_data[len(self.test_data) - self.time_step:].copy()
        lst_output = []
        last_known_close = df_features['close'].iloc[-1]

        for i in range(days):
            x_input_seq = x_input.reshape(1, self.time_step, self.n_features)
            yhat = self.predict(x_input_seq)[0]                         # uses BaseModel.predict
            yhat_inv = self.scaler.inverse_transform(yhat.reshape(1, -1))[0]

            # yhat_inv: [close, high_spread, low_spread, prev_close]
            pred_close       = yhat_inv[0]
            pred_high_spread = abs(yhat_inv[1])   # spread must be positive
            pred_low_spread  = abs(yhat_inv[2])
            pred_high        = pred_close + pred_high_spread
            pred_low         = pred_close - pred_low_spread

            # Chain prev_close: Day-1 uses last real close, Day-N uses Day-(N-1) close
            pred_prev_close = last_known_close if i == 0 else lst_output[i - 1][2]

            # Output: [high, low, close, prev_close]
            result = np.array([pred_high, pred_low, pred_close, pred_prev_close])
            lst_output.append(result)

            # Roll the input window forward
            next_row_raw    = np.array([[pred_close, pred_high_spread, pred_low_spread, pred_prev_close]])
            next_row_scaled = self.scaler.transform(next_row_raw)[0]
            x_input = np.vstack([x_input[1:], next_row_scaled])

        return np.array(lst_output)


# ----------------------------------------------------------------------
# Standalone function — calling interface from main.py stays unchanged
# ----------------------------------------------------------------------
def lstm(df, symbol=None, force_retrain=False, return_model=False):
    """
    Entry point called from main.py / the API. Internally uses LSTMModel.

    `symbol` enables the per-symbol weight cache; omit it to keep the original
    always-train-from-scratch behaviour.
    """
    model = LSTMModel()
    pred, rmse = model.run(df, symbol=symbol, force_retrain=force_retrain)
    if return_model:
        return pred, rmse, model
    return pred, rmse
