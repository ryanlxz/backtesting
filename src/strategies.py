import talib
import numpy as np
import pandas as pd


def profit():
    profit_macd_df = pd.DataFrame(columns=["profit", "win_rate"])


def MACD(ticker: pd.Series) -> pd.Series:
    """MACD strategy which will be applied to stock dataframe

    Args:
        ticker (pd.Series): Close price of ticker

    Returns:
        pd.DataFrame: Dataframe with trading positions
    """
    # Initialize macd histogram
    macd_histogram = talib.MACD(ticker, fastperiod=12, slowperiod=26, signalperiod=9)
    macd_df = pd.DataFrame()
    macd_df["macd_histogram"] = macd_histogram[2]

    # # entry
    macd_df["long_signal"] = np.where(
        (macd_df["macd_histogram"] > 0) & (macd_df["macd_histogram"].shift(1) <= 0),
        1,
        0,
    )
    macd_df["short_signal"] = np.where(
        (macd_df["macd_histogram"] < 0) & (macd_df["macd_histogram"].shift(1) >= 0),
        1,
        0,
    )
    macd_df["long_position"] = np.where(macd_df["long_signal"].shift(1) == 1, 1, 0)
    macd_df["short_position"] = np.where(macd_df["short_signal"].shift(1) == 1, 1, 0)
    position = np.where(
        macd_df["long_position"] == 1,
        "long",
        np.where(macd_df["short_position"] == 1, "short", np.nan),
    )
    return position

    # # exit (2-day RSI of single day is greater than 65 for long positions, and less than 35 for short positions)
    # macd_df["rsi"] = talib.RSI(macd_df["Close"], timeperiod=2)
    # macd_df["entry_date"] = ""
    # for i in macd_df.loc[macd_df["position"] == 1].index:
    #     macd_df["entry_date"].iloc[i] = macd_df["Date"].iloc[i]
    # macd_df["exit_long"] = np.where(
    #     macd_df["rsi"].shift(1) > 65, 1, 0
    # )  # signal to exit is generated when 2-period RSI above 65, but exit trade is only executed on following day.
    # macd_df["exit_short"] = np.where(macd_df["rsi"].shift(1) < 35, 1, 0)
    # macd_df["exit_date"] = ""
    # # exit long positions
    # for i in macd_df.loc[macd_df["long_position"] == 1].index:
    #     new_macd_df = macd_df.loc[i + 2 :]
    #     if len(new_macd_df) == 0:  # index out of range, so use last date as exit
    #         macd_df["exit_date"].iloc[i] = macd_df["Date"].iloc[-1]
    #     for j in range(len(new_macd_df)):
    #         if new_macd_df["exit_long"].iloc[j] == 1:
    #             macd_df["exit_date"].iloc[i] = new_macd_df["Date"].iloc[j]
    #             break
    #         else:
    #             macd_df["exit_date"].iloc[i] = macd_df["Date"].iloc[-1]

    # # exit short positions
    # for i in macd_df.loc[macd_df["short_position"] == 1].index:
    #     new_macd_df = macd_df.loc[i + 2 :]
    #     if len(new_macd_df) == 0:  # index out of range, so use last date as exit
    #         macd_df["exit_date"].iloc[i] = macd_df["Date"].iloc[-1]
    #     for j in range(len(new_macd_df)):
    #         if new_macd_df["exit_short"].iloc[j] == 1:
    #             macd_df["exit_date"].iloc[i] = new_macd_df["Date"].iloc[j]
    #             break
    #         else:
    #             macd_df["exit_date"].iloc[i] = macd_df["Date"].iloc[-1]

    # ### Profit
    # macd_df["profit"] = ""
    # macd_df["winlose"] = ""

    # # long profit
    # for i in macd_df[macd_df["long_position"] == 1].index:
    #     start_price = float(
    #         macd_df[macd_df["Date"] == macd_df["entry_date"].iloc[i]]["Close"]
    #     )
    #     end_price = float(
    #         macd_df[macd_df["Date"] == macd_df["exit_date"].iloc[i]]["Close"]
    #     )
    #     profit = 1 + (end_price - start_price) / start_price
    #     if profit > 1:
    #         winlose = 1
    #     else:
    #         winlose = 0
    #     macd_df["profit"].iloc[i] = profit
    #     macd_df["winlose"].iloc[i] = winlose

    # # short profit
    # for i in macd_df[macd_df["short_position"] == 1].index:
    #     start_price = float(
    #         macd_df[macd_df["Date"] == macd_df["exit_date"].iloc[i]]["Close"]
    #     )
    #     end_price = float(
    #         macd_df[macd_df["Date"] == macd_df["entry_date"].iloc[i]]["Close"]
    #     )
    #     profit = 1 + (end_price - start_price) / start_price
    #     if profit > 1:
    #         winlose = 1
    #     else:
    #         winlose = 0
    #     macd_df["profit"].iloc[i] = profit
    #     macd_df["winlose"].iloc[i] = winlose

    # return macd_df[macd_df["position"] == 1]


class exit:
    pass

    def rsi():
        pass

    def profit():
        pass
        # a certain profit has been reached e.g 10%
