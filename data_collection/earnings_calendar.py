"""
earnings_calendar.py
--------------------
Scrapes multiple public earnings-calendar sources for today's date and returns
a deduplicated list of ticker symbols that have earnings reports today.

Sources used (no API key required):
  1. Yahoo Finance earnings calendar  (finance.yahoo.com)
  2. Nasdaq earnings calendar          (nasdaq.com)
  3. Stockanalysis.com earnings calendar

Usage
-----
    from earnings_calendar import get_todays_earnings

    tickers = get_todays_earnings()
    print(tickers)
    # ['AAPL', 'MSFT', 'NVDA', ...]

Requirements
------------
    pip install requests beautifulsoup4
"""

import re
import json
import logging
from datetime import date
from typing import List

import requests
from bs4 import BeautifulSoup

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Shared HTTP session ───────────────────────────────────────────────────────
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
TIMEOUT = 15  # seconds per request

# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_valid_ticker(symbol: str) -> bool:
    """Return True if *symbol* looks like a real ticker (1-5 uppercase letters)."""
    return bool(re.fullmatch(r"[A-Z]{1,5}", symbol.strip()))


def _clean(tickers: List[str]) -> List[str]:
    """Deduplicate, validate, and sort a raw list of ticker strings."""
    seen = set()
    result = []
    for t in tickers:
        t = t.strip().upper()
        if t and _is_valid_ticker(t) and t not in seen:
            seen.add(t)
            result.append(t)
    return sorted(result)


# ── Source 1 · Yahoo Finance ──────────────────────────────────────────────────

def _scrape_yahoo(today: str) -> List[str]:
    """
    Fetch earnings for *today* from Yahoo Finance's calendar endpoint.
    *today* should be in 'YYYY-MM-DD' format.
    Returns a raw list of ticker strings (may contain duplicates / junk).
    """
    url = (
        "https://finance.yahoo.com/calendar/earnings"
        f"?from={today}&to={today}&day={today}"
    )
    tickers: List[str] = []
    try:
        resp = SESSION.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Yahoo renders a JSON blob inside a <script> tag
        for script in soup.find_all("script"):
            text = script.string or ""
            if "earningsDate" in text and "ticker" in text.lower():
                # Try to pull ticker symbols with a regex
                found = re.findall(r'"symbol"\s*:\s*"([A-Z]{1,5})"', text)
                tickers.extend(found)
                if found:
                    break

        # Fallback: look for table cells that contain ticker links
        if not tickers:
            for a in soup.select("a[href*='/quote/']"):
                href = a.get("href", "")
                match = re.search(r"/quote/([A-Z]{1,5})/", href)
                if match:
                    tickers.append(match.group(1))

        log.info("Yahoo Finance  → %d raw tickers", len(tickers))
    except Exception as exc:  # noqa: BLE001
        log.warning("Yahoo Finance scrape failed: %s", exc)
    return tickers


# ── Source 2 · Nasdaq ─────────────────────────────────────────────────────────

def _scrape_nasdaq(today: str) -> List[str]:
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
        resp = SESSION.get(url, headers=headers, timeout=TIMEOUT)
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
        log.info("Nasdaq         → %d raw tickers", len(tickers))
    except Exception as exc:  # noqa: BLE001
        log.warning("Nasdaq scrape failed: %s", exc)
    return tickers


# ── Source 3 · StockAnalysis ──────────────────────────────────────────────────

def _scrape_stockanalysis(today: str) -> List[str]:
    """
    Fetch earnings from StockAnalysis.com for *today*.
    *today* should be in 'YYYY-MM-DD' format.
    """
    url = f"https://stockanalysis.com/earnings-calendar/?date={today}"
    tickers: List[str] = []
    try:
        resp = SESSION.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # StockAnalysis stores data in a Next.js __NEXT_DATA__ JSON blob
        script = soup.find("script", {"id": "__NEXT_DATA__"})
        if script and script.string:
            payload = json.loads(script.string)
            # Walk the nested structure to find earnings rows
            try:
                earnings_data = (
                    payload["props"]["pageProps"]["data"]
                )
                if isinstance(earnings_data, list):
                    for item in earnings_data:
                        symbol = item.get("s") or item.get("symbol", "")
                        if symbol:
                            tickers.append(symbol)
            except (KeyError, TypeError):
                pass

        # Fallback: parse HTML table rows
        if not tickers:
            for a in soup.select("table a[href*='/stocks/']"):
                href = a.get("href", "")
                match = re.search(r"/stocks/([A-Z]{1,5})/", href)
                if match:
                    tickers.append(match.group(1))

        log.info("StockAnalysis  → %d raw tickers", len(tickers))
    except Exception as exc:  # noqa: BLE001
        log.warning("StockAnalysis scrape failed: %s", exc)
    return tickers


# ── Public API ────────────────────────────────────────────────────────────────

def get_todays_earnings() -> List[str]:
    """
    Scrape multiple earnings-calendar sources for today's date.

    Returns
    -------
    List[str]
        A sorted, deduplicated list of ticker symbols (e.g. ['AAPL', 'MSFT'])
        that have earnings reports scheduled for today.
        Returns an empty list if all sources fail or no data is found.

    Notes
    -----
    - Results are merged across sources so a ticker is included even if only
      one source lists it.
    - Invalid-looking symbols (non-alphabetic, >5 chars) are filtered out.
    - No API key is required; all requests use public web endpoints.
    """
    today = date.today().isoformat()  # e.g. '2025-02-19'
    log.info("Fetching earnings for %s …", today)

    raw: List[str] = []
    raw.extend(_scrape_nasdaq(today))       # most reliable JSON API → run first
    raw.extend(_scrape_stockanalysis(today))
    raw.extend(_scrape_yahoo(today))

    result = _clean(raw)
    log.info("Total unique tickers found: %d", len(result))
    return result


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tickers = get_todays_earnings()
    if tickers:
        print(f"\nEarnings reports today ({date.today().isoformat()}):")
        for t in tickers:
            print(f"  {t}")
        print(f"\nTotal: {len(tickers)} companies")
    else:
        print("No earnings data found for today (all sources may have failed or returned no results).")
