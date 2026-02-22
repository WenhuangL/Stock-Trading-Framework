import math
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import yfinance as yf
from pprint import pprint
import csv
import time
import pandas as pd
import sqlite3
from datetime import date
import re
from typing import List, Optional
from datetime import date


SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
)

# Helpers
def _is_valid_ticker(symbol: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{1,5}", symbol.strip()))


def _clean(tickers: List[str]) -> List[str]:
    seen = set()
    result = []
    for t in tickers:
        t = t.strip().upper()
        if t and _is_valid_ticker(t) and t not in seen:
            seen.add(t)
            result.append(t)
    return sorted(result)


def scrape_nasdaq(today: str) -> List[str]:
    """
    Fetch earnings from Nasdaq's public API endpoint.
    *today* should be in 'YYYY-MM-DD' format.
    """
    url = (
        "https://api.nasdaq.com/api/calendar/earnings"
        f"?date={today}"
    )
    headers = {
        **SESSION.headers,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nasdaq.com/",
        "Origin": "https://www.nasdaq.com",
    }
    tickers: List[str] = []
    try:
        resp = SESSION.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        rows = (
            data.get("data", {})
                .get("rows", [])
        )
        for row in rows:
            symbol = row.get("symbol", "")
            if symbol:
                tickers.append(symbol)
        # log.info("Nasdaq         → %d raw tickers", len(tickers))
        print(f"Nasdaq         → {len(tickers)} raw tickers")
    except Exception as exc:
        # log.warning("Nasdaq scrape failed: %s", exc)
        print(f"Nasdaq scrape failed: {exc}")
    return tickers


def scrape_yahoo_fin_stocks() -> None:
    sort_condition = input("What would you like to sort by?")

    cats = [
        "Ticker", "Sector",
        "Current Price", "1D Change",
        "5D Change", "1M Change", "6M Change",
        "1Y Change", "5Y Change", "All Change"
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Referer": "https://finance.yahoo.com/",
        "Origin": "https://finance.yahoo.com"
    }



    # List to store the scraped data
    data = []

    def yahoo_api_request(url: str, params: dict) -> Optional[dict]:
        r = requests.get(url, headers=headers, params=params)
        if r.status_code != 200:
            return None
        return r.json()

    # Helper to calculate % change
    def calc_pct_change(current: float, old: float) -> str:
        if old == 0: return "N/A"
        change = ((current - old) / old) * 100
        return f"{change:.2f}%"

    def scrape_stock_details(symbol: str) -> Optional[List]:
        try:
            ticker = yf.Ticker(symbol)

            info = ticker.info
            sector = info.get("sector", "N/A")

            # Returns a Pandas DataFrame
            hist = ticker.history(period="max")

            if hist.empty:
                return None

            # Get latest closing price
            current_price = hist["Close"].iloc[-1]
            current_price_fmt = f"{current_price:.2f}"

            # 1 week = 5 days, 1 mo = 21 days, 6 mo = 126 days, 1 yr = 252 days
            total_rows = len(hist)

            change_1d = calc_pct_change(current_price, hist["Close"].iloc[-2]) if total_rows >= 2 else "N/A"

            change_5d = calc_pct_change(current_price, hist["Close"].iloc[-6]) if total_rows >= 6 else "N/A"

            change_1m = calc_pct_change(current_price, hist["Close"].iloc[-22]) if total_rows >= 22 else "N/A"

            change_6m = calc_pct_change(current_price, hist["Close"].iloc[-127]) if total_rows >= 127 else "N/A"

            change_1y = calc_pct_change(current_price, hist["Close"].iloc[-253]) if total_rows >= 253 else "N/A"

            change_5y = calc_pct_change(current_price, hist["Close"].iloc[-1261]) if total_rows >= 1261 else "N/A"

            change_all = calc_pct_change(current_price, hist["Close"].iloc[0])

            return [
                symbol, sector, current_price_fmt,
                change_1d, change_5d, change_1m,
                change_6m, change_1y, change_5y, change_all
            ]

        except Exception as e:
            print(f"Error scraping {symbol}: {e}")
            return None

    screener_url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    params = {"count": 25, "scrIds": "day_gainers" if sort_condition == "gainers" else "day_losers"}

    print(f"Fetching {sort_condition} list...")
    screener = yahoo_api_request(screener_url, params)

    if not screener:
        print("Screener failed.")
        return

    quotes = screener["finance"]["result"][0]["quotes"]

    raw_symbols = [q.get("symbol", "") for q in quotes]
    valid_symbols = _clean(raw_symbols)
    print(f"{len(raw_symbols)} tickers returned; {len(valid_symbols)} passed validation.")

    for symbol in valid_symbols:
        print(f"Processing {symbol}...")
        details: Optional[List] = scrape_stock_details(symbol)
        if details:
            data.append(details)
        time.sleep(0.2)


    save_to_sql(cats, data, "yahoo_table")
    with open("yahoofin_data.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(cats)
        writer.writerows(data)

    print("Success! Data saved to 'yahoofin_data.csv'.")


def scrape_msn_money_stocks() -> None:
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service)
    except ValueError:
        print("WebDriverManager failed. Trying to run ChromeDriver from path.")
        print("If this fails, download ChromeDriver and place it in your script's directory.")
        driver = webdriver.Chrome()

    # wait = WebDriverWait(driver, 15)

    sort_condition = "Losers"
    sort_condition_temp = input("What condition would you like to sort the data by?")

    if sort_condition_temp.lower() == "losers" or sort_condition_temp.lower() == "gainers":  # can add another 'or' if they want full shabang data or just ticker name
        sort_condition = sort_condition_temp.capitalize()
    else:
        print("Sort condition not recognized, defaulting to losers")

    driver.get(f"https://int1.msn.com/en-us/money/markets?tab=Top{sort_condition}")  # Capital L Loser

    time.sleep(3)

    stock_list = driver.find_elements(By.CLASS_NAME, "quoteTitle-DS-EntryPoint1-4")
    facts_list = ["Ticker", "Price", "Previous Close",
                  "Average Volume", "Shares Outstanding", "EPS (TTM)",
                  "P/E (TTM)", "Fwd Dividend (% Yield)", "Ex-Dividend Date"]
    facts_val_list = []

    count = math.inf
    while count > len(stock_list):
        count = int(input(f"How many stocks would you like to scrape? Max is {len(stock_list)}"))
        if count > len(stock_list):
            print("Input exceeds maximum. Please try again.")

    #print(len(stock_list))
    for stock in stock_list:
        # Stop early
        if stock_list.index(stock) == count:
            break

        # Accounts for bug where last list item isn't actually shown
        if stock.text == stock_list[len(stock_list)-1].text:
            break
        stock.click()
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", stock)
        time.sleep(1)

        symbol = driver.find_element(By.CLASS_NAME, "symbolWithBtn-DS-EntryPoint1-1").text
        print(f"Scraping {stock.text}")

        if not _is_valid_ticker(symbol):
            print(f"  ↳ Skipping '{symbol}' — failed ticker validation.")
            continue

        symbol = symbol.strip().upper()
        price = driver.find_element(By.CSS_SELECTOR, ".mainPrice").text

        facts_elements = driver.find_elements(By.CLASS_NAME, "factsRowKey-DS-EntryPoint1-1")
        facts_val_elements = driver.find_elements(By.CLASS_NAME, "factsRowValue-DS-EntryPoint1-1")

        facts_val_list.append([symbol, price])

        for n in range(len(facts_elements)):
            facts_val_list[len(facts_val_list)-1].append(facts_val_elements[n].text)

        #  For testing/displaying data
        # print(f"- {symbol}: {price}")
        # for n in range(len(facts_elements)):
        #     if facts_val_elements[n].text != "-":
        #         print(f"   {facts_elements[n].text}: {facts_val_elements[n].text}")
        #for n in range(0, len(facts_list), 2):
        #    print(f"    {facts_list[n].text}: {facts_list[n+1].text}")


    driver.quit()
    save_to_sql(facts_list, facts_val_list, "msn_table")
    with open("msn_money_data.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(facts_list)
        writer.writerows(facts_val_list)

    print("Scraping complete. Data saved to 'msn_money_data.csv'.")


# Returns just a list of all the ticker names of the sort condition
def scrape_msn_money_simple() -> List[str]:
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service)
    except ValueError:
        print("WebDriverManager failed. Trying to run ChromeDriver from path.")
        print("If this fails, download ChromeDriver and place it in your script's directory.")
        driver = webdriver.Chrome()

    wait = WebDriverWait(driver, 15)

    sort_condition = "Losers"
    """
    sort_condition_temp = input("What condition would you like to sort the data by?")

    if sort_condition_temp.lower() == "losers" or sort_condition_temp.lower() == "gainers":  # can add another 'or' if they want full shabang data or just ticker name
        sort_condition = sort_condition_temp.capitalize()
    else:
        print("Sort condition not recognized, defaulting to losers")"""

    driver.get(f"https://int1.msn.com/en-us/money/markets?tab=Top{sort_condition}")  # Capital L Loser

    time.sleep(3)

    stocktest = driver.find_element(By.CLASS_NAME, "secTitle-DS-EntryPoint1-3")

    while stocktest.text[0:5] == "Price":
        stocktest = driver.find_element(By.CLASS_NAME, "secTitle-DS-EntryPoint1-3")

    ticker_list = []

    for element in driver.find_elements(By.CLASS_NAME, "secTitle-DS-EntryPoint1-3"):
        ticker_list.append(element.text)

    driver.quit()

    return _clean(ticker_list)


def save_to_sql(cats: List[str], data: List[List], table_name: str) -> None:
    data_temp = []
    today = date.today()

    for n in range(len(data)):
        temp = {}
        #for cat in cats:
        for i in range(len(cats)):
            temp[cats[i]] = data[n][i]
        temp['date'] = today
        data_temp.append(temp)
    #pprint(data_temp)
    df = pd.DataFrame(data_temp)
    conn = sqlite3.connect('analysis.db')
    df.to_sql(table_name, conn, if_exists='append', index=False)


def main():

    #print(scrape_msn_money_simple())

    
    setting = input("Would you like to scrape yahoo finance(1) or MSN money(2) or MSN money simple(3)?")
    if setting == "1":
        scrape_yahoo_fin_stocks()
    elif setting == "2":
        scrape_msn_money_stocks()
    elif setting == "3":
        print(scrape_msn_money_simple())
    else:
        print("Invalid input")

if __name__ == "__main__":
    main()
