# Methodology

## Motivation of study:
There are many trading strategies available online, and this study aims to backtest some of the strategies to see which are the most effective. 

## Data:
Historical daily stock data from Yahoo Finance from 2006-2021. 
Tested on 2 separate datasets:
1.	Top 100 stocks in terms of market capitalization in the S&P 500 
2.	2211 companies in the New York Stock Exchange (NYSE), with a minimum of 1 year of trading history and an average of 11 years of trading history. 


## Methodology:
List of strategies:
1. MACD strategy
2. Ensemble strategy
3. Stock trend following system
4. RSI strategy

### MACD strategy
The Moving Average Convergence/Divergence (MACD) is a technical indicator which uses the difference between two exponential moving averages to determine the momentum and the direction of the market. The MACD crossover occurs when the MACD line and the signal line intercept, often indicating a change in the momentum/trend of the market. 

MACD line: 12 day EMA – 26 day EMA 
Signal line: 9 day EMA of MACD line
MACD histogram: MACD line – signal line

Entry: A bullish signal is present when the MACD line crosses ABOVE the signal line and a bearish signal is present when the MACD line crosses BELOW the signal line.

Exit: 2-day RSI of single day is greater than 65 for long positions, and less than 35 for short positions

### Ensemble strategy
Combines golden/death cross, MACD, Hammer candlestick pattern.

First part contains golden/death cross and MACD only. Second part uses golden/death cross, MACD, and Hammer candlestick pattern.

### Stock trend following system
Entry: Long when stock is in an uptrend (uptrend defined as 15 out of 20 days stock price is above 100 day MA) and hits 10 day low, place 3% limit buy order of previous day closing price. E.g 10 day low is 100 then buy at 97 or less.

Exit: Short when 2 period RSI crosses above 50 or after 5 trading days (whichever comes first).

### RSI strategy
Take the cumulative 2-day RSI of X days, and if less than Y, buy the stock. Sell when 2-day RSI > 65

E.g X = 3, Y = 30

## Conclusion:
Based on average annual returns, the RSI strategy performed the best and achieved 16% returns for the NYSE dataset and 13% returns for the S&P500 dataset. 
While the ensemble strategy with hammer candlestick pattern performed the worst with 0.18% returns for the NYSE dataset and 0% returns for the S&P500 dataset. 
Simple and clear trading strategies with few rules tend to perform better than strategies which incorporate more rules, based on this relatively small sample of strategies. 
Moving forward, robustness of the strategies can be tested by changing the parameters slightly of each strategy and testing out new strategies. 