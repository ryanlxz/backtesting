import talib
import numpy as np
import pandas as pd


class MACD:
    """MACD strategy which involves getting the entry and exit trade positions"""

    def __init__(self, stock_data):
        self.stock_data = stock_data

    def get_entry_exit_signal(self) -> pd.DataFrame:
        """Get both entry positions and rsi exit signals by running get_entry_position and get_rsi_exit_signal

        Returns:
            pd.DataFrame: dataframe containing the entry positions and rsi exit signals
        """
        # skip the first column that is Date column
        entry_position = self.stock_data.iloc[:, 1:].apply(
            lambda x: self.get_entry_position(x)
        )

        entry_position.rename(
            columns=lambda x: x.replace("_Close", "_Entry"), inplace=True
        )
        exit_position = self.stock_data.iloc[:, 1:].apply(
            lambda x: self.get_rsi_exit_signal(x)
        )
        exit_position.rename(
            columns=lambda x: x.replace("_Close", "_Exit"), inplace=True
        )
        positions = pd.concat([entry_position, exit_position], axis=1)
        # sort columns to prepare for calling the next method get_entry_exit_position
        positions = positions.reindex(columns=sorted(positions.columns))
        positions.set_index(self.stock_data["Date"], inplace=True)
        return positions

    def get_entry_exit_position(self, enter: pd.Series, exit: pd.Series) -> pd.Series:
        """get exit positions for each entry position

        Args:
            enter (pd.Series): entry signal
            exit (pd.Series): exit signal

        Returns:
            pd.Series: index of exit position for each entry position
        """
        long_position = []
        # get exit positions for long positions
        entry_long_idx = enter[enter == "enter_long"].index
        for i in entry_long_idx:
            if len(exit[(exit.index > i) & (exit == "exit_long")]) > 0:
                # case 1: entry and exit positions are available
                first_occurrence_index = exit[
                    (exit.index > i) & (exit == "exit_long")
                ].index[0]
                long_position.append(("long", i, first_occurrence_index))
            else:
                # case 2 and case 3
                # case 2: enter trade on the last date of the dataframe so not possible to exit
                # case 3: no exit position after entry position. Exit at last available date.
                long_position.append(("long", i, "exit_na"))

        # get exit positions for short positions
        short_position = []
        entry_short_idx = enter[enter == "enter_short"].index
        for i in entry_short_idx:
            if len(exit[(exit.index > i) & (exit == "exit_short")]) > 0:
                # case 1: entry and exit positions are available
                first_occurrence_index = exit[
                    (exit.index > i) & (exit == "exit_short")
                ].index[0]
                short_position.append(("short", i, first_occurrence_index))
            else:
                # case 2 and case 3
                # case 2: enter trade on the last date of the dataframe so not possible to exit
                # case 3: no exit position after entry position. Exit at last available date.
                short_position.append(
                    (
                        "short",
                        i,
                        "exit_na",
                    )
                )
        position_df = pd.DataFrame(
            long_position + short_position, columns=["Position", "Entry", "Exit"]
        )
        position_df.sort_values("Entry")
        return position_df

    def get_entry_position(self, ticker: pd.Series) -> pd.Series:
        """MACD entry strategy that returns the entry positions for both long and short trades.

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

    def get_rsi_exit_signal(self, ticker: pd.Series) -> pd.Series:
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
        exit_signal = np.where(
            exit_df["exit_long"] == 1,
            "exit_long",
            np.where(exit_df["exit_short"] == 1, "exit_short", np.nan),
        )
        return exit_signal

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
