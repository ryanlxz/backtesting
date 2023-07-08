"""
Stock trend following system
Entry: Long when stock is in an uptrend (uptrend defined as 15 out of 20 days stock price is above 100 day MA) and hits 10 day low, place 3% limit buy order of previous day closing price. 
E.g 10 day low is 100 then buy at 97 or less.
Exit: Short when 2 period RSI crosses above 50 or after 5 trading days (whichever comes first).  
"""
import talib
import numpy as np
import pandas as pd
import sys

sys.path.append("../")
import conf

# CONFIGS information
UPTREND_MA = conf.backtest_conf["uptrend_ma"] # 100 day MA


class StockTrend:
    def __init__(self, stock_df: pd.DataFrame):
        self.stock_df = stock_df

    def generate_buy_dates(self):
        self.stock_df['100ma'] = talib.SMA(self.stock_df['Close'],timeperiod=UPTREND_MA) 
        # 1: close price above 100MA, 0: close price below 100MA
        self.stock_df['above_ma'] = np.where(self.stock_df['Close']>=self.stock_df['100ma'],1,0) 
        # 1 = uptrend if past 15/20 days above 100MA
        self.stock_df['uptrend'] = np.where(self.stock_df['above_ma'].rolling(20).sum()>=15,1,0) 
        # 1 = 10-day low
        self.stock_df['10_day_low'] = np.where((self.stock_df['uptrend']==1) & (self.stock_df['Low']==self.stock_df['Low'].rolling(10).min()),1,0)
        # Buy within 3 days on the following day, after 10-day low at 3% limit buy order
        self.stock_df['entry'] = np.where((self.stock_df['10_day_low'].shift(1)==1) & (self.stock_df['Low'].shift(-2).rolling(3).min()<= 0.97*self.stock_df['Close'].shift(1)),1,0)
        self.stock_df['entry_date'] = ''
        for i in self.stock_df.loc[self.stock_df['entry']==1].index:
            if self.stock_df['Low'].iloc[i] <= 0.97*self.stock_df['Close'].iloc[i-1]:
                self.stock_df['entry_date'].iloc[i]=self.stock_df['Date'].iloc[i]
            elif self.stock_df['Low'].iloc[i+1] <= 0.97*self.stock_df['Close'].iloc[i-1]:
                self.stock_df['entry_date'].iloc[i]=self.stock_df['Date'].iloc[i+1]
            elif self.stock_df['Low'].iloc[i+2] <= 0.97*self.stock_df['Close'].iloc[i-1]:
                self.stock_df['entry_date'].iloc[i]=self.stock_df['Date'].iloc[i+2] 

    ### Exit
    def generate_sell_dates(self):
        self.stock_df['rsi'] = talib.RSI(self.stock_df['Close'],timeperiod=2) #2period rsi 
        self.stock_df['exit_date'] = ''
        for i in self.stock_df.loc[self.stock_df['entry_date']!=''].index:
            day = 1
            date = self.stock_df.loc[self.stock_df['Date']==self.stock_df['entry_date'].iloc[i]].index[0]
            while day<5:
                try:
                    if self.stock_df['rsi'].iloc[date+day] > 50:
                        self.stock_df['exit_date'].iloc[i] = self.stock_df['Date'].iloc[date+day]
                        break
                    else:
                        day+=1
                except: #Index out of bounds
                    self.stock_df['exit_date'].iloc[i] = self.stock_df['Date'].iloc[-1] #use last date in dataset to exit
                    break
            if self.stock_df['exit_date'].iloc[i] == '':
                try:
                    self.stock_df['exit_date'].iloc[i] = self.stock_df['Date'].iloc[date+5]
                except: #Index out of bounds;use last date in dataset to exit
                    self.stock_df['exit_date'].iloc[i] = self.stock_df['Date'].iloc[-1]