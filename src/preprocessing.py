import pandas as pd
import numpy as np
from strategies.macd import MACD
from strategies.stock_trend_following_system import StockTrend
import sys
sys.path.append('../..')
from conf import constants

class Preprocessing:
    """Prepares the stock data according to the trading strategy. For example, some strategies may require Low and Close price while others require only the Close price. Also the stock_trend strategy requires the Date column while the macd strategy has Date as the index.  
    """
    def __init__(self, preprocessed_df:pd.DataFrame, strategy:str) -> None:
        """Initialize with the relevant stock dataframe that corresponds to the trading strategy. 
        - If macd strategy is chosen, stock dataframe will contain only Close price.  
        - If stock_trend strategy is chosen, stock dataframe will contain Low and Close price. 

        Args:
            preprocessed_df (pd.DataFrame): preprocessed stock dataframe 
            strategy (str): trading strategy

        Raises:
            KeyError: Invalid strategy that is not found in the strategy list. 
        """
        if strategy not in constants.strategy_list:
            raise KeyError("Invalid strategy")
        self.df = preprocessed_df
        self.strategy = strategy
    
    def split_dataset(self, num_partitions: int, df: pd.DataFrame)-> list:
        """Splits dataset into equal parts column-wise. This is to break down a large dataframe into smaller parts to bypass out-of-memory error. 

        Args:
            num_partitions (int): number of partitions for dataset.
            df (pd.DataFrame): dataframe to split into smaller partitions.

        Returns:
            list: list of partitioned dataframes 
        """
        if self.strategy == 'macd':
            # after calling get_entry_exit_signal() method, dataset is in the form of consecutive column pairs, with date index: 
            # - ticker_entry, ticker_exit 

            num_pairs = df.shape[1] // 2    
            # Calculate the number of pairs in each partition
            pairs_per_partition = num_pairs // num_partitions
            partitioned_df_list = [df.iloc[:, i:i+pairs_per_partition*2] for i in range(0, df.shape[1], pairs_per_partition*2)]
        
        elif self.strategy == 'stock_trend':
            pass
    
        return partitioned_df_list

    def prepare_data(self,num_partitions:int)->None:
        """process the data according to the trading strategy, and split the dataset into N number of partitions.

        Args:
            num_partitions (int): number of partitions for dataset.

        Returns:
            None
        """
        if self.strategy == 'macd':
            self.macd_positions = MACD(self.df) 
            self.prepared_data_df = self.macd_positions.get_entry_exit_signal()
        elif self.strategy == 'stock_trend':
            # rearrange 'Date' column to be the first column 
            self.prepared_data_df = pd.concat([self.df['Date'], self.df.drop('Date', axis=1)], axis=1)
            print(self.prepared_data_df)
        self.prepared_data_list  = self.split_dataset(num_partitions,self.prepared_data_df)
        return None
    
    def get_column_names(self,df:pd.DataFrame)->list:
        """helper function to get column names. The column names will be used to filter the stock dataframe for an individual stock where the trading strategy will be applied. For e.g, ('AAIC_Entry', 'AAIC_Exit') for macd and ['Date', 'AIR_Low', 'AIR_Close'] for stock_trend strategy. 
        
        Args:
            df (pd.DataFrame): 1 of the partitioned dataframe

        Returns:
            list: list of relevant column names for an individual stock
        """
        if self.strategy == 'macd':
            stock_columns = [(df.columns[i], df.columns[i+1]) for i in range(0, len(df.columns), 2)]
        elif self.strategy == 'stock_trend':
            stock_columns = [['Date', self.prepared_data_df.columns[i], self.prepared_data_df.columns[i+1]]for i in range(1, len(self.prepared_data_df.columns[1:]), 2)]
        return stock_columns

