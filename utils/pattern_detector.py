import talib
from config import PATTERN_COLS


def detect_patterns(df):

    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values

    df['HAMMER']       = talib.CDLHAMMER(o, h, l, c)        # +100 bullish
    df['ENGULFING']    = talib.CDLENGULFING(o, h, l, c)     # +100 bullish / -100 bearish
    df['DOJI']         = talib.CDLDOJI(o, h, l, c)          # +100 neutral/indecision
    df['SHOOTING_STAR']= talib.CDLSHOOTINGSTAR(o, h, l, c)  # -100 bearish
    df['MORNING_STAR'] = talib.CDLMORNINGSTAR(o, h, l, c)   # +100 bullish

    # TA-Lib already returns +100 for bullish and -100 for bearish signals.
    # Simply sum all — no manual sign flipping needed.
    df['Pattern_Score'] = (
        df['HAMMER'] + df['ENGULFING'] + df['DOJI'] +
        df['SHOOTING_STAR'] + df['MORNING_STAR']
    ) / 100

    # Scan last 10 trading days — TA-Lib patterns are rare on any single day
    LOOKBACK = 10
    recent = df.tail(LOOKBACK)
    active_pats = []
    for _, row in recent.iterrows():
        for col in PATTERN_COLS:
            if row[col] != 0:
                active_pats.append((col, int(row[col]),
                                    row['Date'].strftime('%d-%b-%Y') if hasattr(row['Date'], 'strftime') else str(
                                        row['Date'])))

    # Remove duplicates keeping most recent
    seen = set()
    unique_pats = []
    for pat, val, date_str in reversed(active_pats):
        if pat not in seen:
            seen.add(pat)
            unique_pats.append((pat, val, date_str))
    active_pats = list(reversed(unique_pats))
    return df, active_pats

def combine_regime_patterns(regime, active_pats):
    if regime['sufficient_data'] and active_pats:
        bullish_pats = [p for p, v, d in active_pats if v > 0]
        bearish_pats = [p for p, v, d in active_pats if v < 0]
        reg = regime['regime']
        print()
        if reg == 'BEAR' and bullish_pats:
            print("  ⚠️  Insight : Bullish pattern detected inside a BEAR trend.")
            print("              This may indicate a short-term reversal or dead-cat bounce.")
            print("              Wait for confirmation before buying — trend is still down.")
        elif reg == 'BULL' and bearish_pats:
            print("  ⚠️  Insight : Bearish pattern detected inside a BULL trend.")
            print("              This may indicate a short-term pullback in an uptrend.")
            print("              Consider holding — trend is still up.")
        elif reg == 'BULL' and bullish_pats:
            print("  ✅  Insight : Bullish pattern confirms the BULL regime.")
            print("              Strong buy signal — trend and candles are aligned.")
        elif reg == 'BEAR' and bearish_pats:
            print("  🔴  Insight : Bearish pattern confirms the BEAR regime.")
            print("              Strong sell signal — trend and candles are aligned.")
        elif reg == 'SIDEWAYS' and bullish_pats:
            print("  🟡  Insight : Bullish pattern in a SIDEWAYS market.")
            print("              Possible breakout upward — watch volume for confirmation.")
        elif reg == 'SIDEWAYS' and bearish_pats:
            print("  🟡  Insight : Bearish pattern in a SIDEWAYS market.")
            print("              Possible breakout downward — watch volume for confirmation.")
