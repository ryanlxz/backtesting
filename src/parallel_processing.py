import dask
import pandas as pd
import numpy as np
from strategies.macd import MACD
from strategies.stock_trend_following_system import StockTrend
import sys
sys.path.append('../..')
from conf import constants

class PrepareData:
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
    
    def split_dataset(self,num_partitions:int,df:pd.DataFrame):
        """Splits dataset into equal parts column-wise. This is to break down a large dataframe into smaller parts to bypass out-of-memory error. 

        Args:
            num_partitions (int): number of partitions for dataset.
            df (pd.DataFrame): dataframe to split into smaller partitions.

        Returns:
            list: list of partitioned dataframes 
        """
        split_points = np.array_split(np.arange(df.shape[1]), num_partitions)
        split_df_list = [df.iloc[:, split_points[i]] for i in range(num_partitions)]
        return split_df_list

    def prepare_data(self,df:pd.DataFrame):
        """process the data according to the trading strategy. 

        Returns:
            None
        """
        if self.strategy == 'macd':
            self.macd_positions = MACD(self.df) 
            self.prepared_data_df = self.macd_positions.get_entry_exit_signal()
        elif self.strategy == 'stock_trend':
            # rearrange 'Date' column to be the first column 
            self.prepared_data_df = pd.concat([self.df['Date'], self.df.drop('Date', axis=1)], axis=1)
            self.prepared_data_df  = self.split_dataset(4,self.prepared_data_df)
            # rearranged_df = rearranged_df.iloc[:,:715]
        return None
    
    def get_column_names(self):
        """helper function to get column names. The column names will be used to filter the stock dataframe for an individual stock where the trading strategy will be applied. For e.g, ('AAIC_Entry', 'AAIC_Exit') for macd and ['Date', 'AIR_Low', 'AIR_Close'] for stock_trend strategy. 

        Returns:
            list: list of relevant column names for an individual stock
        """
        if self.strategy == 'macd':
            stock_columns = [(self.prepared_data_df.columns[i], self.prepared_data_df.columns[i+1]) for i in range(0, len(self.prepared_data_df.columns), 2)]
        elif self.strategy == 'stock_trend':
            stock_columns = [['Date', self.prepared_data_df.columns[i], self.prepared_data_df.columns[i+1]]for i in range(1, len(self.prepared_data_df.columns[1:]), 2)]
        return stock_columns


class ParallelProcess(PrepareData):
    def __init__(self,preprocessed_df:pd.DataFrame,strategy:str) -> None:
        super().__init__(preprocessed_df,strategy)

    def lazy_function(self,stock):
        if self.strategy == 'macd':
            col1,col2 = stock[0],stock[1]
            exit_position = self.macd_positions.get_entry_exit_position(self.prepared_data_df[col1],self.prepared_data_df[col2])
        elif self.strategy == 'stock_trend':
            stocktrend = StockTrend(self.prepared_data_df[stock])
            stocktrend.generate_buy_dates()
            exit_position = stocktrend.generate_sell_dates()
        return exit_position

    def parallel_processing(self):
        # create empty list to store lazy Dask collection 
        lazy_list = []
        stock_columns = self.get_column_names()
        for stock in stock_columns:
            exit_position = dask.delayed(self.lazy_function)(stock)
            lazy_list.append(exit_position)
        exit_position = dask.compute(*lazy_list)
        return exit_position