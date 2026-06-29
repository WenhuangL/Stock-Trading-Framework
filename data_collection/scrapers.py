"""
scrapers.py
-----------
Web scraping functions for Yahoo Finance and MSN Money stock data,
with type hinting and ticker validation applied throughout.

Requirements
------------
    pip install requests yfinance selenium webdriver-manager beautifulsoup4 pandas
"""

import csv
import math
import re
import sqlite3
import time
from datetime import date
from typing import List, Optional
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


# ── Ticker validation ─────────────────────────────────────────────────────────

def _is_valid_ticker(symbol: str) -> bool:
    """Return True if *symbol* looks like a real ticker (1–5 uppercase letters)."""
    return bool(re.fullmatch(r"[A-Z]{1,5}", symbol.strip()))


def _clean(tickers: List[str]) -> List[str]:
    """Deduplicate, validate, and sort a raw list of ticker strings."""
    seen: set[str] = set()
    result: List[str] = []
    for t in tickers:
        t = t.strip().upper()
        if t and _is_valid_ticker(t) and t not in seen:
            seen.add(t)
            result.append(t)
    return sorted(result)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _calc_pct_change(current: float, old: float) -> str:
    """Return a formatted percentage-change string, or 'N/A' if *old* is zero."""
    if old == 0:
        return "N/A"
    change = ((current - old) / old) * 100
    return f"{change:.2f}%"


def save_to_sql(cats: List[str], data: List[List], table_name: str) -> None:
    """Persist *data* rows to a local SQLite database, appending today's date."""
    data_temp: List[dict] = []
    today = date.today()

    for row in data:
        temp: dict = {}
        for i, cat in enumerate(cats):
            temp[cat] = row[i]
        temp["date"] = today
        data_temp.append(temp)

    df = pd.DataFrame(data_temp)
    db_path = Path("output/analysis.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="append", index=False)


# ── Yahoo Finance ─────────────────────────────────────────────────────────────

def scrape_yahoo_fin_stocks() -> None:
    """
    Fetch the top day-gainers or day-losers from Yahoo Finance, enrich each
    ticker with historical price-change data via yfinance, and save the results
    to both a CSV file and a local SQLite database.
    """
    sort_condition: str = input("What would you like to sort by? (gainers/losers): ").strip().lower()

    cats: List[str] = [
        "Ticker", "Sector",
        "Current Price", "1D Change",
        "5D Change", "1M Change", "6M Change",
        "1Y Change", "5Y Change", "All Change",
    ]

    headers: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://finance.yahoo.com/",
        "Origin": "https://finance.yahoo.com",
    }

    data: List[List] = []

    def yahoo_api_request(url: str, params: dict) -> Optional[dict]:
        r = requests.get(url, headers=headers, params=params)
        if r.status_code != 200:
            return None
        return r.json()

    def scrape_stock_details(symbol: str) -> Optional[List]:
        """Fetch sector and price-change history for a single *symbol*."""
        try:
            ticker = yf.Ticker(symbol)
            info: dict = ticker.info
            sector: str = info.get("sector", "N/A")

            hist = ticker.history(period="max")
            if hist.empty:
                return None

            current_price: float = hist["Close"].iloc[-1]
            current_price_fmt: str = f"{current_price:.2f}"
            total_rows: int = len(hist)

            # Trading-day approximations: 1w=5d, 1mo=21d, 6mo=126d, 1y=252d, 5y=1260d
            change_1d:  str = _calc_pct_change(current_price, hist["Close"].iloc[-2])   if total_rows >= 2    else "N/A"
            change_5d:  str = _calc_pct_change(current_price, hist["Close"].iloc[-6])   if total_rows >= 6    else "N/A"
            change_1m:  str = _calc_pct_change(current_price, hist["Close"].iloc[-22])  if total_rows >= 22   else "N/A"
            change_6m:  str = _calc_pct_change(current_price, hist["Close"].iloc[-127]) if total_rows >= 127  else "N/A"
            change_1y:  str = _calc_pct_change(current_price, hist["Close"].iloc[-253]) if total_rows >= 253  else "N/A"
            change_5y:  str = _calc_pct_change(current_price, hist["Close"].iloc[-1261]) if total_rows >= 1261 else "N/A"
            change_all: str = _calc_pct_change(current_price, hist["Close"].iloc[0])

            return [
                symbol, sector, current_price_fmt,
                change_1d, change_5d, change_1m,
                change_6m, change_1y, change_5y, change_all,
            ]
        except Exception as e:
            print(f"Error scraping {symbol}: {e}")
            return None

    screener_url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    params: dict[str, object] = {
        "count": 25,
        "scrIds": "day_gainers" if sort_condition == "gainers" else "day_losers",
    }

    print(f"Fetching {sort_condition} list...")
    screener: Optional[dict] = yahoo_api_request(screener_url, params)

    if not screener:
        print("Screener failed.")
        return

    quotes: List[dict] = screener["finance"]["result"][0]["quotes"]

    # ── Validate tickers before processing ────────────────────────────────────
    raw_symbols: List[str] = [q.get("symbol", "") for q in quotes]  # fixed: was q["symbol}"]
    valid_symbols: List[str] = _clean(raw_symbols)
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


# ── MSN Money (full detail) ───────────────────────────────────────────────────

def scrape_msn_money_stocks() -> None:
    """
    Use Selenium to scrape detailed stock facts (price, P/E, EPS, etc.) for the
    top gainers or losers from MSN Money, then save to CSV and SQLite.
    """
    try:
        service: Service = Service(ChromeDriverManager().install())
        driver: webdriver.Chrome = webdriver.Chrome(service=service)
    except ValueError:
        print("WebDriverManager failed. Trying to run ChromeDriver from path.")
        print("If this fails, download ChromeDriver and place it in your script's directory.")
        driver = webdriver.Chrome()

    _wait: WebDriverWait = WebDriverWait(driver, 15)  # kept for potential future use

    sort_condition: str = "Losers"
    sort_condition_temp: str = input("What condition would you like to sort the data by? (gainers/losers): ")

    if sort_condition_temp.lower() in ("losers", "gainers"):
        sort_condition = sort_condition_temp.capitalize()
    else:
        print("Sort condition not recognized, defaulting to Losers.")

    driver.get(f"https://int1.msn.com/en-us/money/markets?tab=Top{sort_condition}")
    time.sleep(3)

    stock_list = driver.find_elements(By.CLASS_NAME, "quoteTitle-DS-EntryPoint1-4")

    facts_list: List[str] = [
        "Ticker", "Price", "Previous Close",
        "Average Volume", "Shares Outstanding", "EPS (TTM)",
        "P/E (TTM)", "Fwd Dividend (% Yield)", "Ex-Dividend Date",
    ]
    facts_val_list: List[List] = []

    count: int = math.inf  # type: ignore[assignment]
    while count > len(stock_list):
        count = int(input(f"How many stocks would you like to scrape? Max is {len(stock_list)}: "))
        if count > len(stock_list):
            print("Input exceeds maximum. Please try again.")

    for stock in stock_list:
        if stock_list.index(stock) == count:
            break
        if stock.text == stock_list[len(stock_list) - 1].text:
            break

        stock.click()
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", stock)
        time.sleep(1)

        raw_symbol: str = driver.find_element(By.CLASS_NAME, "symbolWithBtn-DS-EntryPoint1-1").text
        print(f"Scraping {stock.text}")

        # ── Validate ticker before storing ────────────────────────────────────
        if not _is_valid_ticker(raw_symbol):
            print(f"  ↳ Skipping '{raw_symbol}' — failed ticker validation.")
            continue

        symbol: str = raw_symbol.strip().upper()
        price: str = driver.find_element(By.CSS_SELECTOR, ".mainPrice").text

        facts_elements = driver.find_elements(By.CLASS_NAME, "factsRowKey-DS-EntryPoint1-1")
        facts_val_elements = driver.find_elements(By.CLASS_NAME, "factsRowValue-DS-EntryPoint1-1")

        row: List = [symbol, price]
        for n in range(len(facts_elements)):
            row.append(facts_val_elements[n].text)
        facts_val_list.append(row)

    driver.quit()
    save_to_sql(facts_list, facts_val_list, "msn_table")
    with open("msn_money_data.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(facts_list)
        writer.writerows(facts_val_list)

    print("Scraping complete. Data saved to 'msn_money_data.csv'.")


# ── MSN Money (tickers only) ──────────────────────────────────────────────────

def scrape_msn_money_simple() -> List[str]:
    """
    Scrape MSN Money for top Losers and return a validated, deduplicated list
    of ticker symbols only.
    """
    try:
        service: Service = Service(ChromeDriverManager().install())
        driver: webdriver.Chrome = webdriver.Chrome(service=service)
    except ValueError:
        print("WebDriverManager failed. Trying to run ChromeDriver from path.")
        print("If this fails, download ChromeDriver and place it in your script's directory.")
        driver = webdriver.Chrome()

    _wait: WebDriverWait = WebDriverWait(driver, 15)  # kept for potential future use

    sort_condition: str = "Losers"

    driver.get(f"https://int1.msn.com/en-us/money/markets?tab=Top{sort_condition}")
    time.sleep(3)

    # Wait until the section title is no longer showing stale "Price" text
    stocktest = driver.find_element(By.CLASS_NAME, "secTitle-DS-EntryPoint1-3")
    while stocktest.text.startswith("Price"):
        stocktest = driver.find_element(By.CLASS_NAME, "secTitle-DS-EntryPoint1-3")

    # Collect raw ticker strings, then validate + deduplicate
    raw_tickers: List[str] = [
        element.text
        for element in driver.find_elements(By.CLASS_NAME, "secTitle-DS-EntryPoint1-3")
    ]

    driver.quit()

    # ── Apply validation ──────────────────────────────────────────────────────
    ticker_list: List[str] = _clean(raw_tickers)
    print(f"{len(raw_tickers)} raw tickers scraped; {len(ticker_list)} passed validation.")
    return ticker_list
