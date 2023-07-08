import talib
import numpy as np
import pandas as pd
import sys

sys.path.append("../")
import conf

# CONFIGS information
LONG_TERM_MA = conf.backtest_conf["long_term_ma"]
SHORT_TERM_MA = conf.backtest_conf["short_term_ma"]

def ensemble(security):
# Identify long term trend with golden cross and death cross
    df = security.copy()
    dates = cross(df)
    if len(dates) == 0: #no golden/death cross
        return
    else:
        period = cross_period_long(dates)
        
# Identify suitable entry points with MACD indicator 
    macd_result = MACD(df)  
    macd_result = macd_result.reset_index()
    col = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', '200ma','50ma', 'long_positions', 'short_positions', 
    'positions', '21ma', 'histogram', 'macd', 'long_signal', 'short_signal', 'long_position', 'short_position', 'position', 'rsi',                             
    'entry_date', 'exit_long', 'exit_short','exit_date','profit','winlose'] 
    macd_dates = pd.DataFrame(columns = col)
    for j in range(len(macd_result)):
        for k in range(len(period)):
            if period[k][0] <= macd_result['Date'].iloc[j] < period[k][1]:
                macd_dates = macd_dates.append(macd_result.loc[j])
                

# Identify specific entry points with hammer candlestick pattern
    hammer_dates = hammer(df)
    col = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', '200ma','50ma', 'long_positions', 'short_positions', 
    'positions', '21ma', 'histogram', 'macd', 'long_signal', 'short_signal', 'long_position', 'short_position', 'position', 'rsi',                             
    'entry_date', 'exit_long', 'exit_short','exit_date','profit','winlose'] 
    trade_dates = pd.DataFrame(columns = col)
    macd_dates = macd_dates.reset_index()
    for f in range(len(hammer_dates)):
        for g in range(len(macd_dates)):
            if hammer_dates[f][0] == macd_dates['Date'].iloc[g]:
                trade_dates = trade_dates.append(macd_dates.loc[g])

## returns (golden/death cross + MACD strategy) (0), and with candlestick pattern (1)
    return (macd_dates,trade_dates)