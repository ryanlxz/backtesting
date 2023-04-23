import string
import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
import yfinance as yf
import pathlib
from os.path import exists

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
    datapath = pathlib.Path(cfg["datapath"])

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
