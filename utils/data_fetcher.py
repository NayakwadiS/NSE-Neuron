from nselib import capital_market
import pandas as pd
from datetime import date
import os
from utils.preprocessor import preprocess_nse_df

# Directory where raw fetched data is cached as CSV
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')


def _get_cache_path(symbol: str, today: str) -> str:
    """Return the expected CSV cache path for a given symbol and date."""
    return os.path.join(RAW_DATA_DIR, f"{symbol}_{today}.csv")


def _load_or_fetch(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Check if a CSV for (symbol, today) exists in /data/raw/.
    - If YES  → load and return it directly (skip API call).
    - If NO   → fetch from NSE API, save as CSV, then return raw DataFrame.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    cache_path = _get_cache_path(symbol, to_date)

    if os.path.exists(cache_path):
        print(f"  [Cache] Loading cached data from: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"  [Cache] No cache found for {symbol} on {to_date}. Fetching from NSE API...")
    df = capital_market.price_volume_and_deliverable_position_data(
        symbol=symbol,
        from_date=from_date,
        to_date=to_date
    )
    df.to_csv(cache_path, index=False)
    print(f"  [Cache] Data saved to: {cache_path}")
    return df


def getDataFrame(SYMBOL):
    def decorate(func):
        def decorated(*args, **kwargs):
            # accept algorithm either as first positional arg or as kwarg
            algorithm = None
            if args:
                algorithm = args[0]
            algorithm = kwargs.get('algorithm', algorithm)

            ticker_info = capital_market.equity_list()
            ticker_info = ticker_info[ticker_info['SYMBOL'] == SYMBOL]

            from_date = ticker_info[' DATE OF LISTING'].values[0]
            from_date = pd.to_datetime(from_date, dayfirst=True).strftime('%d-%m-%Y')
            to_date = date.today().strftime('%d-%m-%Y')

            # Load from cache or fetch from NSE API
            df = _load_or_fetch(SYMBOL, from_date, to_date)

            # Preprocessing delegated to preprocessor.py
            df = preprocess_nse_df(df)

            details = {
                'scheme_name': ticker_info['NAME OF COMPANY'].values[0],
                'scheme_code': str(SYMBOL)
            }

            if algorithm is None:
                # Backward compatible: if wrapped func still expects only (df, details)
                return func(df, details)

            return func(df, details, algorithm)

        return decorated
    return decorate
