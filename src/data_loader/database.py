# This file governs the database which contains historical data for running trading strategies during inference.

import pandas as pd
import yfinance as yf
import os
import conf
from conf import constants
from datetime import date, timedelta
from logs import logger
from data import get_stock_data


class Database:
    def __init__(self) -> None:
        self.num_days = conf.backtest_conf["required_days"]
        self.interval = conf.backtest_conf["data"]["interval"]

    def load_database(self) -> pd.DataFrame:
        """loads the database containing all the data for stock prices

        Returns:
            pd.DataFrame: dataframe containing all the data for stock prices
        """
        datapath = constants.database_file_path
        if os.path.exists(datapath):
            stock_df = pd.read_csv(datapath, index_col=[0])
        else:
            # create database
            directory = os.path.dirname(datapath)
            os.makedirs(directory, exist_ok=True)
            today = date.today()
            start_date = today - timedelta(days=self.num_days - 1)
            stock_df = get_stock_data(
                datapath=datapath,
                start_date=start_date,
                end_date=today,
                interval=self.interval,
            )
        logger.info("Loaded database")
        return stock_df

    def update(self):
        # check for the missing required data (before the start date of database and after the last date of database till today)
        stock_df = self.load_database()
        start_date = stock_df["Date"].iloc[0]
        last_date = stock_df["Date"].iloc[-1]
        today_date = date.today()
        total_days = (today_date - start_date).days
        if total_days >= self.num_days:
            # update the end of the database till today latest data
            updated_stock_df = self.append(
                stock_df=stock_df, start_date=last_date, end_date=today_date
            )
        else:
            # update the start of the database
            diff_days = self.num_days - total_days
            first_date = start_date - timedelta(days=diff_days)
            end_date = start_date - timedelta(days=start_date - 1)
            updated_start_df = self.append(
                stock_df=stock_df, start_date=first_date, end_date=end_date
            )
            # update the end of the database till today latest data
            updated_stock_df = self.append(
                stock_df=updated_start_df, start_date=last_date, end_date=today_date
            )
        return updated_stock_df

    def append(
        self, stock_df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """downloads the required stock data and append it to the database

        Args:
            stock_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        # get existing tickers from database
        ticker_list = stock_df
        stock_list = []
        # download stock data and merge to dataframe

        for ticker in ticker_list:
            stock = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=self.interval,
            )
        # append data


if __name__ == "__main__":
    database = Database()
    df = database.load_database()
