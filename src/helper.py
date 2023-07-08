import pandas as pd



class Profit:
    def __init__(self, positions_df: pd.DataFrame, price_df: pd.DataFrame):
        """Calculate profit for both long and short strategies 

        Args:
            positions_df (pd.DataFrame): dataframe containing 2 columns - entry and exit dates
            price_df (pd.DataFrame): dataframe containing the relevant price columns. 
            If a strategy requires only the close price then the price_df will only contain the close price. 
        """
        self.positions_df = positions_df
        self.price_df = price_df

    def calculate_long_profit(self):
        pass


    def calculate_short_profit(self):
        pass
