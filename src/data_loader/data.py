import json
import pathlib
import string
import sys
import os
import cloudscraper
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

# when running in notebook
sys.path.append("../")
# when running script on its own
# sys.path.append("../../")

import conf
from conf import constants

# CONFIGS information from import conf
START_DATE = conf.backtest_conf["data"]["start_date"]
END_DATE = conf.backtest_conf["data"]["end_date"]
INTERVAL = conf.backtest_conf["data"]["interval"]

# constants
RAW_DATAPATH = constants.raw_data_file_path
TICKERS_DATAPATH = constants.raw_data_tickers_file_path
PREPROCESSED_DATAPATH = constants.preprocessed_data_file_path
WEBSITE_URL = constants.raw_data_url

# def load_preprocessed_data(cfg: dict) -> pd.DataFrame:
#     """load preprocessed data

#     Args:
#         cfg (dict): config dictionary

#     Returns:
#         pd.DataFrame: preprocessed dataframe
#     """
#     datapath = pathlib.Path(cfg["preprocessed_datapath"])

#     if exists(datapath):
#         stock_data = pd.read_csv(datapath)
#         stock_data = convert_multiindex(stock_data)
#     else:
#         stock_data = get_stock_data(cfg)
#         # stock_data = convert_multiindex(stock_data)
#         stock_data = drop_tickers(stock_data)
#         stock_data.to_csv(datapath)
#     return stock_data


def preprocess_data(preprocessed_datapath: str, raw_datapath: str):
    """preprocess raw data by dropping tickers with more than 10 years of missing data.
    Save preprocessed data.

    Args:
        preprocessed_datapath (str): _description_
        raw_datapath (str): _description_

    Returns:
        _type_: _description_
    """
    raw_df = pd.read_csv(
        pathlib.Path(raw_datapath), parse_dates=["Date"], index_col="Date"
    )
    preprocessed_df = drop_tickers(raw_df)
    preprocessed_df.to_csv(preprocessed_datapath, index=True)
    return None


def load_data(datapath: str, columns: list = None) -> pd.DataFrame:
    """load specific columns from the dataset into a pandas dataframe.

    Args:
        datapath (str): file path of dataset.
        columns (list, optional): suffixes of column names in list format.
        E.g ['_Close, '_Low'] to only load the Close and Low prices. Defaults to None.

    Returns:
        Pd.DataFrame: dataframe containing selected columns.
    """
    if columns == None:
        return pd.read_csv(datapath)
    else:
        columns = columns + ["Date"]
        return pd.read_csv(datapath, usecols=lambda col: col.endswith(tuple(columns)))


# def load_close_data(datapath:str) -> pd.DataFrame:
#     """load Close data

#     Args:
#         cfg (dict): config dictionary

#     Returns:
#         pd.DataFrame: dataframe with Close price
#     """
#     datapath = pathlib.Path(datapath)

#     if exists(datapath):
#         df = pd.read_csv(datapath,index_col=[0])
#     else:
#         stock_data = load_preprocessed_data(cfg)
#         df = stock_data["Close"]
#         df.index = stock_data["Date"].iloc[:, 0]
#         df.index = df.index.rename("Date")
#         df.rename(columns=lambda x: str(x) + "_Close", inplace=True)
#         df.to_csv(datapath)
#     return df


def scrape_stock_symbols(website_url: str) -> pd.DataFrame:
    """Scrape stock ticker symbols from the new york stock exchange from this website.
    https://www.advfn.com/nyse/newyorkstockexchange.asp?companies=
    Then save the tickers in a tickers.json file.
    'https://randerson112358.medium.com/web-scraping-stock-tickers-using-python-3e5801a52c6d'
    Args:
        website_url (str): website to scrape stock tickers from

    Returns:
        pd.DataFrame: Pandas dataframe of company names and their tickers
    """
    company_name = []
    company_ticker = []
    for letter in string.ascii_uppercase:
        url = website_url + letter
        scraper = cloudscraper.create_scraper()
        info = scraper.get(url).text
        soup = BeautifulSoup(info, "html.parser")
        odd_rows = soup.find_all("tr", attrs={"class": "ts0"})
        even_rows = soup.find_all("tr", attrs={"class": "ts1"})
        for i in odd_rows:
            row = i.find_all("td")
            company_name.append(row[0].text.strip())
            company_ticker.append(row[1].text.strip())
        for i in even_rows:
            row = i.find_all("td")
            company_name.append(row[0].text.strip())
            company_ticker.append(row[1].text.strip())
    ticker_df = pd.DataFrame(columns=["company_name", "company_ticker"])
    ticker_df["company_name"] = company_name
    ticker_df["company_ticker"] = company_ticker
    ticker_df = ticker_df[ticker_df["company_name"] != ""]
    # save tickers into a json file
    with open(TICKERS_DATAPATH, "w") as file:  # overwrites the previous data
        json.dump(list(ticker_df["company_ticker"]), file)
    return ticker_df


def get_stock_data(
    datapath: str, start_date: str, end_date: str, interval: str
) -> pd.DataFrame:
    """Download stock data for the scraped stock tickers

    Args:
        cfg (dict): config datapipeline dictionary
        start_date: start date of stock data
        end_date: end date of stock data
        interval: time interval of stock data e.g 1D, 1H
        datapath: data path of stock data

    Returns:
        pd.DataFrame: Pandas dataframe of stock data sorted by their tickers
    """

    ticker_file = open(TICKERS_DATAPATH)
    ticker_list = json.load(ticker_file)
    # create dataframe with dates ranging from start to end date
    # date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    # stock_data = pd.DataFrame({'Date': date_range})

    stock_list = []
    # download stock data and merge to dataframe
    for ticker in ticker_list:
        stock = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
        )
        stock = stock.dropna().reset_index()
        # rename columns by adding ticker, except for date column which is the first index. e.g ticker_Close
        stock = stock.rename(
            columns=lambda x: "_".join((f"{ticker}", x))
            if stock.columns.get_loc(x) > 0
            else x
        )
        stock_list.append(stock)
    # merge stock dataframes
    stock_data = stock_list[0]
    for stock_df in stock_list[1:]:
        stock_data = pd.merge(stock_data, stock_df, how="outer", on="Date")
        # stock["symbol"] = i
        # stock_data = pd.concat([stock_data, stock])
    # stock_data = stock_data.pivot(columns="symbol")
    stock_data.to_csv(datapath)
    return stock_data


# def convert_multiindex(stock_data: pd.DataFrame) -> pd.DataFrame:
#     """convert dataframe into a multiindex

#     Args:
#         stock_data (pd.DataFrame): raw dataframe

#     Returns:
#         pd.DataFrame: multiindex
#     """
#     # check if all tickers are the same for all columns
#     test_df = pd.DataFrame()
#     for i in ["^Date", "^Open", "^High", "^Low", "^Close", "^Adj Close", "^Volume"]:
#         test_df = pd.concat(
#             [test_df, stock_data.filter(regex=i).iloc[0].reset_index(drop=True)], axis=1
#         )
#     test_df["matching"] = test_df.eq(test_df.iloc[:, 0], axis=0).all(1)
#     assert not False in test_df["matching"], "Tickers do not match in all columns"

#     # configure headers
#     stock_data.drop(["Unnamed: 0"], axis=1, inplace=True)
#     header_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
#     length = len(stock_data.filter(like="Date").columns.tolist())
#     tickers = stock_data.filter(like="Date").iloc[0].tolist()
#     header_1 = []
#     for i in [length * [i] for i in header_cols]:
#         for j in i:
#             header_1.append(j)
#     header = [header_1, len(header_cols) * tickers]
#     stock_data.columns = header
#     stock_data = stock_data.iloc[1:].reset_index(drop=True)
#     return stock_data


def drop_tickers(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Drop tickers with more than 10 years of missing data.
    252 trading days in a year so drop rows with >= 2520 missing values

    Args:
        stock_data (pd.DataFrame): dataframe

    Returns:
        pd.DataFrame: dataframe after dropping tickers with >=10 years of missing values
    """
    drop_ticker_list = []
    # filter for all the Close columns
    close_columns = stock_data.filter(like="_Close", axis=1).columns.tolist()
    for column in close_columns:
        if stock_data[column].isnull().sum() >= 2520:
            drop_ticker_list.append(column.split("_")[0] + "_")
    # get all column names that start with the tickers to drop
    columns_to_drop = [
        col for col in stock_data.columns if col.startswith(tuple(drop_ticker_list))
    ]
    stock_data = stock_data.drop(columns_to_drop, axis=1)
    return stock_data


def run_data_pipeline():
    # scrape_stock_symbols(WEBSITE_URL)
    # get_stock_data(RAW_DATAPATH, START_DATE, END_DATE, INTERVAL)
    preprocess_data(PREPROCESSED_DATAPATH, RAW_DATAPATH)


if __name__ == "__main__":
    # run_data_pipeline()
    sys.path.append("../../")
    print(os.getcwd())
