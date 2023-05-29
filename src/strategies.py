import talib
import numpy as np
import pandas as pd


class MACD:
    """MACD strategy which involves getting the entry and exit trade positions"""

    def __init__(self, stock_data):
        self.stock_data = stock_data

    def apply(self):
        entry_position = self.stock_data.iloc[:, 1:].apply(lambda x: self.macd_entry(x))
        # entry_position["Date"] = self.stock_data["Date"]
        entry_position.rename(
            columns=lambda x: x.replace("_Close", "_Entry"), inplace=True
        )
        exit_position = self.stock_data.iloc[:, 1:].apply(lambda x: self.rsi_exit(x))
        exit_position.rename(
            columns=lambda x: x.replace("_Close", "_Exit"), inplace=True
        )
        positions = pd.concat([entry_position, exit_position], axis=1)
        positions = positions.reindex(columns=sorted(positions.columns))
        return positions

    def get_exit_position(
        self, ticker_enter: pd.Series, ticker_exit: pd.Series
    ) -> pd.Series:
        """get exit positions for each entry position

        Args:
            ticker_enter (pd.Series): entry positions
            ticker_exit (pd.Series): exit signals

        Returns:
            pd.Series: index of exit position for each entry position
        """
        # invert the logic for np.where
        exit_list = []
        entry_idx = ticker_enter[ticker_enter == "enter_long"].index
        for i in entry_idx:
            if (
                len(ticker_exit[(ticker_exit.index > i) & (ticker_exit == "exit_long")])
                > 0
            ):
                # case 1: entry and exit positions are available
                first_occurrence_index = ticker_exit[
                    (ticker_exit.index > i) & (ticker_exit == "exit_long")
                ].index[0]
                exit_list.append(first_occurrence_index)
            elif len(ticker_exit[(ticker_exit.index > i)]) == 0:
                # case 2: enter trade on the last date of the dataframe so not possible to exit
                exit_list.append("exit_na")
            else:
                # case 3: no exit position after entry position. Exit at last available date.
                exit_list.append(ticker_exit.loc[i:].iloc[-1])
        exit_idx = pd.Series(exit_list)
        return exit_idx

    def macd_entry(self, ticker: pd.Series) -> pd.Series:
        """MACD entry strategy which will be applied to stock dataframe

        Args:
            ticker (pd.Series): Close price of ticker

        Returns:
            pd.Series: Series with entry positions
        """
        # Initialize macd histogram
        macd_histogram = talib.MACD(
            ticker, fastperiod=12, slowperiod=26, signalperiod=9
        )
        macd_df = pd.DataFrame()
        macd_df["macd_histogram"] = macd_histogram[2]

        # entry
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
        macd_df["short_position"] = np.where(
            macd_df["short_signal"].shift(1) == 1, 1, 0
        )
        entry_position = np.where(
            macd_df["long_position"] == 1,
            "enter_long",
            np.where(macd_df["short_position"] == 1, "enter_short", np.nan),
        )
        return entry_position

    def rsi_exit(self, ticker: pd.Series) -> pd.Series:
        """RSI exit strategy
        2-day RSI of single day is greater than 65 for long positions, and less than 35 for short positions

        Args:
            ticker (pd.Series): Close price of ticker

        Returns:
            pd.Series: Series with exit positions
        """
        exit_df = pd.DataFrame()
        exit_df["rsi"] = talib.RSI(ticker, timeperiod=2)
        # signal to exit is generated when 2-period RSI above 65, but exit trade is only executed on the following day.
        exit_df["exit_long"] = np.where(exit_df["rsi"].shift(1) > 65, 1, 0)
        exit_df["exit_short"] = np.where(exit_df["rsi"].shift(1) < 35, 1, 0)
        exit_position = np.where(
            exit_df["exit_long"] == 1,
            "exit_long",
            np.where(exit_df["exit_short"] == 1, "exit_short", np.nan),
        )
        return exit_position

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


def profit():
    profit_macd_df = pd.DataFrame(columns=["profit", "win_rate"])


def close_exit():
    # stop loss exit. when close price falls below or exceeds a threshold, generate exit signal
    pass
