import string
import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup

def scrape_stock_symbols():
    company_name =[]
    company_ticker = []
    for letter in string.ascii_uppercase:
        URL =  'https://www.advfn.com/nyse/newyorkstockexchange.asp?companies='+letter
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
    data = pd.DataFrame(columns = ['company_name',  'company_ticker']) 
    data['company_name'] = company_name
    data['company_ticker'] = company_ticker
    data = data[data['company_name'] != '']
    return data