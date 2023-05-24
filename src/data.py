import string
import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
import yfinance as yf
import pathlib
from os.path import exists

def load_preprocessed_data(cfg:dict)->pd.DataFrame:
    """load preprocessed data

    Args:
        cfg (dict): config dictionary

    Returns:
        pd.DataFrame: preprocessed dataframe 
    """
    datapath = pathlib.Path(cfg['preprocessed_datapath'])

    if exists(datapath):
        stock_data = pd.read_csv(datapath)
        stock_data = convert_multiindex(stock_data)
    else:
        stock_data = get_stock_data(cfg)
        stock_data = convert_multiindex(stock_data)
        stock_data = drop_tickers(stock_data)
        stock_data.to_csv(datapath)
    return stock_data

def load_close_data(cfg:dict)->pd.DataFrame:
    """load Close data

    Args:
        cfg (dict): config dictionary

    Returns:
        pd.DataFrame: dataframe with Close price 
    """
    datapath = pathlib.Path(cfg['close_datapath'])

    if exists(datapath):
        df = pd.read_csv(datapath)
    else:
        stock_data = load_preprocessed_data(cfg)
        df = stock_data['Close']
        df.index = stock_data['Date'].iloc[:,0]
        df.index = df.index.rename('Date')
        df.rename(columns=lambda x: str(x) + "_Close", inplace=True)
        df.to_csv(datapath)
    return df


def scrape_stock_symbols(cfg:dict):
    """Scrape stock ticker symbols from the new york stock exchange 
    'https://randerson112358.medium.com/web-scraping-stock-tickers-using-python-3e5801a52c6d'
    Args:
        cfg (dict): config datapipeline dictionary
        URL: website to scrape stock tickers from

    Returns:
        pd.DataFrame: Pandas dataframe of company names and their tickers 
    """
    company_name =[]
    company_ticker = []
    for letter in string.ascii_uppercase:
        URL =  cfg['URL']+letter
        scraper = cloudscraper.create_scraper()
        info = scraper.get(URL).text
        soup = BeautifulSoup(info, "html.parser")
        odd_rows = soup.find_all('tr', attrs= {'class':'ts0'})
        even_rows = soup.find_all('tr', attrs= {'class':'ts1'})
        for i in odd_rows:
            row = i.find_all('td')
            company_name.append(row[0].text.strip())
            company_ticker.append(row[1].text.strip())
        for i in even_rows:
            row = i.find_all('td')
            company_name.append(row[0].text.strip())
            company_ticker.append(row[1].text.strip())
    ticker_df = pd.DataFrame(columns = ['company_name',  'company_ticker']) 
    ticker_df['company_name'] = company_name
    ticker_df['company_ticker'] = company_ticker
    ticker_df = ticker_df[ticker_df['company_name'] != '']
    return ticker_df

def get_stock_data(cfg:dict):
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
    datapath = pathlib.Path(cfg["raw_datapath"])

    if exists(datapath):
        stock_data = pd.read_csv(datapath)
    else:
        ticker_df = scrape_stock_symbols(cfg)
        stock_data = pd.DataFrame()
        for i in ticker_df['company_ticker']:
            stock = yf.download(i, start=cfg['start_date'], end=cfg['end_date'], interval=cfg['interval'])
            stock = stock.dropna().reset_index() 
            stock['symbol'] = i
            stock_data = pd.concat([stock_data,stock])
        stock_data = stock_data.pivot(columns='symbol')
        stock_data.to_csv(cfg['datapath'])
    return stock_data

def convert_multiindex(stock_data:pd.DataFrame)->pd.DataFrame:
    """convert dataframe into a multiindex 

    Args:
        stock_data (pd.DataFrame): raw dataframe

    Returns:
        pd.DataFrame: multiindex 
    """
    # check if all tickers are the same for all columns 
    test_df = pd.DataFrame()
    for i in ['^Date','^Open','^High','^Low','^Close','^Adj Close','^Volume']:
        test_df = pd.concat([test_df,stock_data.filter(regex=i).iloc[0].reset_index(drop=True)],axis=1)
    test_df['matching'] = test_df.eq(test_df.iloc[:, 0], axis=0).all(1)
    assert not False in test_df['matching'], 'Tickers do not match in all columns'

    # configure headers 
    stock_data.drop(['Unnamed: 0'], axis=1,inplace=True)
    header_cols = ['Date','Open','High','Low','Close','Adj Close','Volume']
    length = len(stock_data.filter(like='Date').columns.tolist())
    tickers = stock_data.filter(like='Date').iloc[0].tolist()
    header_1 = []
    for i in [length*[i] for i in header_cols]:
        for j in i:
            header_1.append(j)
    header = [header_1, 
            len(header_cols)*tickers]
    stock_data.columns=header
    stock_data = stock_data.iloc[1:].reset_index(drop=True) 
    return stock_data 

def drop_tickers(stock_data:pd.DataFrame)->pd.DataFrame:
    """Drop tickers with more than 10 years of missing data. 
    252 trading days in a year so drop rows with >= 2520 missing values

    Args:
        stock_data (pd.DataFrame): dataframe

    Returns:
        pd.DataFrame: dataframe after dropping tickers with >=10 years of missing values 
    """
    drop_ticker_list = []
    for i in stock_data['Date'].columns.tolist():
        if stock_data['Date'][i].isnull().sum() >= 2520:
            drop_ticker_list.append(i)
    stock_data = stock_data.drop(drop_ticker_list, axis=1, level = 1)
    return stock_data

