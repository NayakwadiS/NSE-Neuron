import pandas as pd


# Price columns that need numeric conversion
PRICE_COLUMNS = ['close', 'prev_close', 'high', 'low']


def rename_columns(df):
    """Rename raw NSE API columns to standard lowercase names."""
    return df.rename(
        columns={
            'ClosePrice': 'close',
            'PrevClose':  'prev_close',
            'HighPrice':  'high',
            'LowPrice':   'low'
        }
    )


def select_columns(df):
    """Keep only required OHLC columns from raw NSE DataFrame."""
    return df[['Date', 'ClosePrice', 'PrevClose', 'HighPrice', 'LowPrice']]


def convert_price_columns(df):
    """Strip commas and convert price columns to numeric."""
    for col in PRICE_COLUMNS:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '', regex=False),
            errors='coerce'
        )
    return df


def parse_and_sort_dates(df):
    """Parse Date column to datetime, sort ascending (oldest → newest)."""
    df = df.copy()
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df


def preprocess_nse_df(df):
    """
    Full preprocessing pipeline for NSE equity price data.
    Steps:
        1. Select required columns
        2. Rename to standard names
        3. Convert price columns to numeric
        4. Parse dates, sort ascending, add formatted 'date' column
    """
    df = select_columns(df)
    df = rename_columns(df)
    df = convert_price_columns(df)
    df = parse_and_sort_dates(df)
    print(df.head())
    return df
