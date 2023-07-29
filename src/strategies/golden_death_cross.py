import talib
import numpy as np
import pandas as pd
import sys

sys.path.append("../")
import conf

# CONFIGS information
LONG_TERM_MA = conf.backtest_conf["long_term_ma"]
SHORT_TERM_MA = conf.backtest_conf["short_term_ma"]


class GoldenDeathCross:
    """
    Gets the dates of golden and death crosses.
    - Golden cross indicates a long-term bull market.
    - Death cross indicates a long-term bear market.
    Short-term moving average crossing over a major long-term moving average.
    Swing traders use longer time frames, such as five hours or 10 hours to calculate the moving averages.
    """

    def __init__(self, stock_data):
        self.stock_data = stock_data

    def get_cross_dates(self):
        return self.stock_data.iloc[:, 1:].apply(lambda x: self.get_cross(x))

    def get_cross(self, ticker: pd.Series) -> pd.DataFrame:
        """get the dates of golden and death crosses

        Args:
            ticker (pd.Series): close price of stock

        Returns:
            pd.DataFrame: dataframe containing the dates and crosses (golden or death)
        """
        crossover_df = pd.DataFrame()
        crossover_df["Date"] = self.stock_data["Date"]
        crossover_df["long_term_ma"] = talib.SMA(ticker, timeperiod=LONG_TERM_MA)
        crossover_df["short_term_ma"] = talib.SMA(ticker, timeperiod=SHORT_TERM_MA)
        crossover_df["long_positions"] = np.where(
            crossover_df["short_term_ma"] > crossover_df["long_term_ma"], 1, 0
        )
        crossover_df["short_positions"] = np.where(
            crossover_df["short_term_ma"] < crossover_df["long_term_ma"], -1, 0
        )
        crossover_golden = crossover_df[
            (crossover_df["long_positions"] == 1)
            & (crossover_df["long_positions"].shift(1) == 0)
        ]["Date"]
        crossover_death = crossover_df[
            (crossover_df["short_positions"] == -1)
            & (crossover_df["short_positions"].shift(1) == 0)
        ]["Date"]
        crossover_df["cross"] = np.where(
            crossover_df["Date"].isin(crossover_golden),
            "golden",
            np.where(crossover_df["Date"].isin(crossover_death), "death", ""),
        )
        # check if there was any golden or death cross for the ticker
        if "golden" not in crossover_df["cross"]:
            return crossover_df[["Date", "cross", "long_term_ma"]]

# logging = get_logger(LOG_DIR, include_debug=True)
