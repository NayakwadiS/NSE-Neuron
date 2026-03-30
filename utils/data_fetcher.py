from nselib import capital_market
import pandas as pd
from datetime import date
from utils.preprocessor import preprocess_nse_df


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

            df = capital_market.price_volume_and_deliverable_position_data(
                symbol=SYMBOL,
                from_date=from_date,
                to_date=to_date
            )

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
