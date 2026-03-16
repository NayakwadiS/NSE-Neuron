from nselib import capital_market
import pandas as pd
import yfinance as yf

def getDataFrame(SYMBOL):
    def decorate(func):
        def decorated(*args,**kwargs):
            df = capital_market.price_volume_and_deliverable_position_data(symbol='SBIN', period='1Y')
            df = df[['Date', 'ClosePrice']]
            df = df.rename(columns={'ClosePrice': 'close'})
            df.reset_index(inplace=True)
            # Ensure the 'Date' column is in datetime format
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['date'] = df['Date'].dt.strftime('%Y-%m-%d')

            info = capital_market.equity_list()
            info = info[info['SYMBOL'] == SYMBOL]
            details = {'scheme_name': info['NAME OF COMPANY'].values[0], 'scheme_code': str(SYMBOL)}
            return func(df,details)
        return decorated
    return decorate


def data_frame(func):
    def decorated(*args,**kwargs):
        df = yf.download(str(*args) + ".BO", period='max')
        df = df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'])
        df = df.rename(columns={'Close': 'nav'})
        df.reset_index(inplace=True)
        df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
        info = yf.Ticker(str(*args) + ".BO").get_info()
        details = {'scheme_name': info['longName'], 'scheme_code': str(*args)}
        return func(df,details)
    return decorated
