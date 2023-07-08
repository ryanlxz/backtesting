import talib
import numpy as np
import pandas as pd
import sys

sys.path.append("../")
import conf

# CONFIGS information
LONG_TERM_MA = conf.backtest_conf["long_term_ma"]
SHORT_TERM_MA = conf.backtest_conf["short_term_ma"]

class Candlestick():
    def __init__(self, stock_data):
        self.stock_data = stock_data

    def hammer():


        
        security_price['21ma'] = talib.SMA(security_price['Close'],timeperiod=21) 
        hammer_list = talib.CDLHAMMER(security_price['Open'], security_price['High'], security_price['Low'], security_price['Close'])
        tradelist = []
        for i in range(len(security_price[:-21])): # void the last 21 trades as not enough time to measure 21MA. 
            if hammer_list[i] == 100: # Hammer (0 = not hammer, 100= hammer)
                ma = security_price['21ma'].iloc[i+21]
                if ma > security_price['Close'].iloc[i]:
                    winlose = 1
                else:
                    winlose = 0
                date = security_price['Date'].iloc[i]
                profit = (ma-security_price['Close'].iloc[i])/security_price['Close'].iloc[i]*100
                tradelist.append((date,winlose,profit))
        return tradelist
    # returns the date where hammer occurs, winlose, profit
