import dask
import pandas as pd
import numpy as np
# from strategies.macd import MACD
from strategies.stock_trend_following_system import StockTrend
from preprocessing import Preprocessing
import sys
# from utils import get_logger
# logging = get_logger(__file__)
sys.path.append('../..')
import conf 

NUM_PARTITIONS = conf.backtest_conf['preprocessing']['num_partitions']
MACD_DEST_FILEPATH = conf.backtest_conf['preprocessing']['macd_dest_filepath']
STOCKTREND_DEST_FILEPATH = conf.backtest_conf['preprocessing']['stocktrend_dest_filepath']

class ParallelProcess(Preprocessing):
    def __init__(self, preprocessed_df: pd.DataFrame, strategy: str) -> None:
        super().__init__(preprocessed_df, strategy)

    def lazy_function(self,stock:list,prepared_df:pd.DataFrame):
        """function that generates the entry and exit positions of the trading strategy. Will be applied in parallel. 

        Args:
            stock (list): relevant column names necessary to run the lazy function.
            prepared_df (pd.DataFrame): prepared dataframe that has been partitioned. 

        Returns:
            pd.DataFrame: dataframe containing the entry and exit positions of the chosen trading strategy. 
        """
        if self.strategy == 'macd':
            col1,col2 = stock[0],stock[1]
            exit_position = self.macd_positions.get_entry_exit_position(prepared_df[col1],prepared_df[col2])
        elif self.strategy == 'stock_trend':
            stocktrend = StockTrend(prepared_df[stock])
            stocktrend.generate_buy_dates()
            exit_position = stocktrend.generate_sell_dates()
        return exit_position

    def parallel_processing(self,prepared_df:pd.DataFrame):
        # create empty list to store lazy Dask collection 
        lazy_list = []
        stock_columns = self.get_column_names(prepared_df)
        for stock in stock_columns:
            exit_position = dask.delayed(self.lazy_function)(stock,prepared_df)
            lazy_list.append(exit_position)
        # dask.compute triggers the actual computation and turns a lazy Dask collection into its in-memory equivalent, which returns a tuple of dataframes in this case.  
        exit_position = dask.compute(*lazy_list)
        
        # add back the ticker columns to the dateframes in exit_position
        if self.strategy == 'macd':
            tickers = [col.split('_')[0] for col in prepared_df.iloc[:].columns]
            for i in range(len(exit_position)):
                exit_position[i].rename(columns=lambda x: tickers[i] + '_' + x, inplace=True)
            merged_df = pd.concat(exit_position, axis=1)
        elif self.strategy == 'stock_trend':
            tickers = [col.split('_')[0] for col in prepared_df.iloc[:, 1::2].columns]
            for i in range(len(exit_position)):
                exit_position[i].rename(columns=lambda x: tickers[i] + '_' + x, inplace=True)
                merged_df = pd.concat(exit_position, axis=1)
        return merged_df
    
    def merge_partitions(self, exit_position_list: list) -> pd.DataFrame:
        """Merge dataframes in exit_position_list 

        Args:
            exit_position_list (list): list containing the position dataframes to be merged

        Returns:
            pd.DataFrame: merged dataframe 
        """
        return pd.concat([df for df in exit_position_list],axis=1)
        
    def save_dataframe(self, merged_df: pd.DataFrame) -> None:
        """Save merged positions dataframe as a csv file

        Args:
            merged_df (pd.DataFrame): merged positions dataframe 
        """
        if self.strategy == 'macd':
            dest_filepath = MACD_DEST_FILEPATH
        elif self.strategy == 'stock_trend':
            dest_filepath = STOCKTREND_DEST_FILEPATH
        merged_df.to_csv(dest_filepath)
        return None

    def run_pipeline(self) -> None:
        # logging.info('Starting parallel processing pipeline...')
        self.prepare_data(NUM_PARTITIONS)
        exit_position_list = []
        for prepared_df in self.prepared_data_list:
            exit_position_df = self.parallel_processing(prepared_df)
            print('Complete parallel processing')
            exit_position_list.append(exit_position_df)
        merged_df = self.merge_partitions(exit_position_list)
        self.save_dataframe(merged_df)
        return None