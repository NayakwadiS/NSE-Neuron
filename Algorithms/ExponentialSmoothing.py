from statsmodels.tsa.holtwinters import ExponentialSmoothing

from Algorithms import *
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)


def exponential(df):
    df.index = df['Date']
    df_new = pd.to_numeric(df['close'], errors='coerce').dropna()
    fitted_model = ExponentialSmoothing(df_new, trend='add', seasonal='add', seasonal_periods=2).fit()
    test_predictions = fitted_model.forecast(5)
    return test_predictions.tolist(), 1
