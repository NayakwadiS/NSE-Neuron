import pandas as pd
import config

# Price columns that need numeric conversion
PRICE_COLUMNS = ['open','close', 'prev_close', 'high', 'low']

# Volume columns that need numeric conversion (kept separate from prices so a
# missing/'-' volume never invalidates an otherwise perfectly good price row).
VOLUME_COLUMNS = ['volume', 'deliverable_qty']

# Raw NSE column -> standard lowercase name
COLUMN_RENAME_MAP = {
    'ClosePrice':          'close',
    'PrevClose':           'prev_close',
    'HighPrice':           'high',
    'LowPrice':            'low',
    'OpenPrice':           'open',
    'TotalTradedQuantity': 'volume',
    'DeliverableQty':      'deliverable_qty',
}

# Columns that must exist in the raw NSE payload
REQUIRED_RAW_COLUMNS = [
    'Date', 'ClosePrice', 'PrevClose', 'HighPrice', 'LowPrice', 'OpenPrice',
]

# Columns we take when present but never require (older cached CSVs may lack them)
OPTIONAL_RAW_COLUMNS = ['TotalTradedQuantity', 'DeliverableQty']

# Columns stripped before the frame is handed to the models. They are useful for
# charting / pattern detection but are not part of config.FEATURE_COLUMNS, and
# leaving them in risks NaN rows silently shrinking the training set.
NON_FEATURE_COLUMNS = ['open', 'volume', 'deliverable_qty']


def rename_columns(df):
    """Rename raw NSE API columns to standard lowercase names."""
    return df.rename(columns=COLUMN_RENAME_MAP)


def select_columns(df):
    """Keep required OHLC columns plus any optional volume columns that exist."""
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Raw NSE data is missing required column(s): {missing}")
    optional = [c for c in OPTIONAL_RAW_COLUMNS if c in df.columns]
    return df[REQUIRED_RAW_COLUMNS + optional]


def convert_price_columns(df):
    """Strip commas and convert price columns to numeric."""
    df = df.copy()
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '', regex=False),
            errors='coerce'
        )
    return df


def convert_volume_columns(df):
    """Strip commas and convert traded-quantity columns to numeric.

    NSE returns '-' for days with no reported quantity, and thousands
    separators for large numbers — both are coerced to NaN / int respectively.
    """
    df = df.copy()
    for col in VOLUME_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '', regex=False),
            errors='coerce'
        )
    return df


def parse_and_sort_dates(df):
    """Parse Date column to datetime, dedupe, sort ascending (oldest → newest).

    NSE's API sometimes returns overlapping rows for the same trading day
    (e.g. when date-range windows overlap during pagination), which produces
    duplicate 'Date' entries. Downstream chart rendering (lightweight-charts)
    requires strictly ascending, unique timestamps — a duplicate causes it to
    throw and silently blank the whole chart. We de-duplicate here, keeping
    the last occurrence (assumed most authoritative / most recently fetched).
    """
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values(by='Date', ascending=True)
    df = df.drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
    df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df


def removed_open_coulmn(df):
    """Remove 'open' column from DataFrame if it exists.

    Kept for backward compatibility — prefer remove_non_feature_columns().
    """
    if 'open' in df.columns:
        df = df.drop(columns=['open'])
    return df


def remove_non_feature_columns(df):
    """Drop chart-only columns (open, volume, ...) before modelling."""
    present = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    return df.drop(columns=present) if present else df


def preprocess_nse_df(df):
    """
    Full preprocessing pipeline for NSE equity price data.
    Steps:
        1. Select required columns (+ optional volume columns)
        2. Rename to standard names
        3. Convert price and volume columns to numeric
        4. Parse dates, sort ascending, add formatted 'date' column
    """
    df = select_columns(df)
    df = rename_columns(df)
    df = convert_price_columns(df)
    df = convert_volume_columns(df)
    df = parse_and_sort_dates(df)
    # Save AFTER renaming/parsing but BEFORE removing chart-only columns
    # so pattern_detector / charting get correct column names including
    # 'open' and 'volume'.
    config.HISTORIC_DATA = df.copy()
    df = remove_non_feature_columns(df)
    return df
