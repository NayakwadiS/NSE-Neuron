from nselib import capital_market
import pandas as pd
import yfinance as yf
from datetime import date

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
            df = df[['Date', 'ClosePrice', 'PrevClose', 'HighPrice', 'LowPrice']]
            df = df.rename(
                columns={
                    'ClosePrice': 'close',
                    'PrevClose': 'prev_close',
                    'HighPrice': 'high',
                    'LowPrice': 'low'
                }
            )

            # Strip commas from price columns and convert to numeric
            for col in ['close', 'prev_close', 'high', 'low']:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
            print(df.head())
            df.reset_index(inplace=True)
            # Ensure the 'Date' column is in datetime format
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # Sort oldest to newest so time series models get correct temporal order
            df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
            df['date'] = df['Date'].dt.strftime('%Y-%m-%d')

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


def data_frame(func):
    def decorated(*args, **kwargs):
        df = yf.download(str(*args) + ".BO", period='max')
        df = df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'])
        df = df.rename(columns={'Close': 'nav'})
        df.reset_index(inplace=True)
        df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
        info = yf.Ticker(str(*args) + ".BO").get_info()
        details = {'scheme_name': info['longName'], 'scheme_code': str(*args)}
        return func(df, details)

    return decorated
