import pandas as pd
from datetime import datetime

def get_column_name(dataframe:pd.DataFrame,suffix:str)->str:
        """helper function to get column name which ends with the specified suffix

        Args:
            dataframe (pd.DataFrame): dataframe which contains the needed column name 
            suffix (str): ending of column name which follows the format: _suffix. e.g _Exit

        Returns:
            str: column name
        """
        return dataframe.columns[dataframe.columns.str.endswith(suffix)][0]

class Profit:
    def __init__(self, positions_df: pd.DataFrame, price_df: pd.DataFrame):
        """Calculate profit for both long and short strategies 

        Args:
            positions_df (pd.DataFrame): dataframe containing 2 columns - entry and exit dates. dataframe should only contain 1 type of trading position - either long or short. 
            price_df (pd.DataFrame): dataframe containing the relevant price columns. 
            If a strategy requires only the close price then the price_df will only contain the close price. 
        """
        self.positions_df = positions_df
        self.price_df = price_df
    
    def calculate_profit(self,position:str):
        """calculate trading profits. Merge the positions and price_df to get the Close price for entry and exit dates, and calculate the profit according to the trade position. 

        Args:
            position (str): trade position, either long or short. 

        Returns:
            pd.DataFrame: dataframe containing the entry and exit dates and prices, and profits
        """
        # check if position input is in the correct format
        if position not in ['long','short']:
            print('Position is an incorrect value. It has to be either long or short.')
        self.entry_column = get_column_name(self.positions_df,'_Entry')
        self.exit_column = get_column_name(self.positions_df,'_Exit')
        #merge positions and price df to get the Close price for entry and exit dates
        merged_df = pd.merge(self.positions_df, self.price_df, left_on=self.entry_column, right_on='Date', how='left')
        merged_df = pd.merge(merged_df, self.price_df, left_on=self.exit_column, right_on='Date', suffixes=('_Entry', '_Exit'), how='left')
        # drop unnecessary columns
        merged_df.drop(['Date_Entry','Date_Exit'],axis=1,inplace=True)
        # calculate profits
        entry_price = get_column_name(merged_df,'Close_Entry')
        exit_price = get_column_name(merged_df,'Close_Exit')
        if position == 'long':
            merged_df['profit_percent'] = (merged_df[exit_price]-merged_df[entry_price])/merged_df[entry_price] + 1
            merged_df['profit_absolute'] = (merged_df[exit_price]-merged_df[entry_price])

        elif position == 'short':
            merged_df['profit_percent'] = (merged_df[entry_price]-merged_df[exit_price])/merged_df[entry_price] + 1
            merged_df['profit_absolute'] = (merged_df[entry_price]-merged_df[exit_price])
        return merged_df
    
    def calculate_long_term_profit(self,profit_df:pd.DataFrame, profit_type:str = 'cumulative',start_year:int=0, end_year:int=9999)->float:
        """calculate long term profit over the year range. Year range defaults to 1900-2200 which is the entire duration of the dataframe. 

        Args:
            profit_df (pd.DataFrame): profit dataframe which is returned from calculate_profit method.
            start_year (int, optional): Starting year. Defaults to 1900.
            end_year (int, optional): ending year. Defaults to 2200.

        Returns:
            float: compounded return in percentage over the specified year range. 
        """
        year_range = (start_year,end_year)
        profit_df[self.entry_column] = pd.to_datetime(profit_df[self.entry_column])
        profit_df[self.exit_column] = pd.to_datetime(profit_df[self.exit_column])
        profit_df['entry_year'] = profit_df[self.entry_column].dt.year
        profit_df['exit_year'] = profit_df[self.exit_column].dt.year
        if profit_type == 'cumulative':
            return profit_df.loc[(profit_df['entry_year'].between(year_range[0], year_range[1])) & (profit_df['exit_year'].between(year_range[0], year_range[1])), 'profit_percent'].prod()
        elif profit_type == 'sum':
            return profit_df.loc[(profit_df['entry_year'].between(year_range[0], year_range[1])) & (profit_df['exit_year'].between(year_range[0], year_range[1])), 'profit_absolute'].sum()
        else:
            print('profit_type is not the right value')

