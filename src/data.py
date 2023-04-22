import string
import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
import yfinance as yf

def scrape_stock_symbols(cfg:dict):
    """Scrape stock ticker symbols from the new york stock exchange 
    'https://randerson112358.medium.com/web-scraping-stock-tickers-using-python-3e5801a52c6d'
    Args:
        cfg (dict): config datapipeline dictionary

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
    ticker_df = scrape_stock_symbols()
    stock_data = pd.DataFrame()
    for i in ticker_df['company_ticker']:
        stock = yf.download(i, start=cfg['start_date'], end=cfg['end_date'], interval=cfg['interval'])
        stock = stock.dropna().reset_index() 
        nyse_dict[i] = security_price