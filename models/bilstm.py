from models import *
from models.base_model import BaseModel
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


class BiLSTMModel(BaseModel):
    """
    Bidirectional LSTM model for multi-variate stock price forecasting.
    Reads sequence in both forward and backward directions for richer context.
    Derives from BaseModel and implements fit, predict, and evaluate.
    """

    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.time_step = TIME_STEP
        self.units = RNN_UNITS
        self.n_features = None
        self.test_data = None
        self.df_features = None

    # ------------------------------------------------------------------
    # Helper: turn raw DataFrame into spread-based model DataFrame
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
        """Train the BiLSTM on pre-built (X_train, y_train) arrays."""
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
        """Return RMSE for the close column on the given dataset."""
        preds   = self.scaler.inverse_transform(self.predict(X))
        actuals = self.scaler.inverse_transform(y)
        return math.sqrt(mean_squared_error(actuals[:, 0], preds[:, 0]))

    # ------------------------------------------------------------------
    # Main entry: train model + forecast future days
    # ------------------------------------------------------------------
    def run(self, df):
        """
        Full pipeline: preprocess → build model → fit → forecast.
        Returns (forecasted_stock_price, rmse) same shape as standalone bilstm().
        """
        df_features, df_model = self._prepare_data(df)
        self.df_features = df_features
        self.n_features = df_model.shape[1]

        # Scale
        df_scaled = self.scaler.fit_transform(df_model)

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

        # Build Bidirectional LSTM
        # Bidirectional reads sequence in both forward and backward directions
        # giving the model context from both past and future within the time window
        self.model = Sequential()
        self.model.add(Bidirectional(KerasLSTM(self.units, return_sequences=True),
                                     input_shape=(self.time_step, self.n_features)))
        self.model.add(Bidirectional(KerasLSTM(self.units, return_sequences=True)))
        self.model.add(Bidirectional(KerasLSTM(self.units)))
        self.model.add(Dense(self.n_features))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        # Train (uses BaseModel.fit)
        self.fit(X_train, y_train)

        # RMSE on training set (uses BaseModel.evaluate)
        rmse = {'close': self.evaluate(X_train, y_train)}

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
            yhat     = self.predict(x_input_seq)[0]                      # uses BaseModel.predict
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
def bilstm(df):
    """Entry point called from main.py. Internally uses BiLSTMModel class."""
    model = BiLSTMModel()
    return model.run(df)
