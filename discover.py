import csv
import io
import json
import os
import re
import time
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
LIST_DIR = os.path.join(BASE, "lists")
REPORT_FILE = os.path.join(BASE, "discovery_report.json")
MAX_THAI = int(os.getenv("MAX_THAI_STOCKS", "150"))
MAX_US = int(os.getenv("MAX_US_STOCKS", "150"))
MAX_CRYPTO = int(os.getenv("MAX_CRYPTO_PAIRS", "100"))
MIN_CRYPTO_QUOTE_VOLUME = float(os.getenv("MIN_CRYPTO_QUOTE_VOLUME", "5000000"))
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MarketScanner/1.0; +https://github.com)"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


class VisibleTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ("script", "style", "noscript", "svg"):
            self.hidden_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in ("script", "style", "noscript", "svg") and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data):
        if not self.hidden_depth:
            self.parts.append(data)


def read_list(name):
    path = os.path.join(LIST_DIR, name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(dict.fromkeys(str(x).strip().upper() for x in data if str(x).strip()))
    except (OSError, ValueError, TypeError):
        return []


def write_list(name, symbols):
    path = os.path.join(LIST_DIR, name)
    os.makedirs(LIST_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(set(symbols)), f, ensure_ascii=False, indent=2)
        f.write("\n")


def fetch_set_universe():
    url = "https://www.set.or.th/th/market/get-quote/stock"
    response = SESSION.get(url, timeout=30)
    response.raise_for_status()
    parser = VisibleTextParser()
    parser.feed(response.text)
    visible = " ".join(parser.parts)
    # SET's public stock-quote page lists common-stock symbols in its visible content.
    tokens = re.findall(r"(?<![A-Z0-9])([A-Z0-9]{1,6})(?![A-Z0-9])", visible.upper())
    stop = {
        "SET", "TH", "THE", "STOCK", "STOCKS", "MARKET", "COMMON", "WARRANT",
        "PREFERRED", "ETF", "DR", "DW", "FUND", "SEARCH", "QUOTE", "LOGIN",
        "REGISTER", "LANG", "EN", "THAI", "SETTRADE", "HTTPS", "WWW", "OR"
    }
    symbols = {x for x in tokens if x not in stop and 1 <= len(x) <= 6 and re.search(r"[A-Z]", x)}
    # A valid response should contain a meaningful number of listed tickers.
    if len(symbols) < 100:
        raise RuntimeError(f"SET page parser returned only {len(symbols)} possible symbols")
    return symbols


def fetch_nasdaq_universe():
    urls = [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "nasdaq"),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "other"),
    ]
    symbols = set()
    for url, kind in urls:
        response = SESSION.get(url, timeout=30)
        response.raise_for_status()
        lines = response.text.splitlines()
        if not lines or "File Creation Time" not in lines[-1]:
            raise RuntimeError(f"Unexpected Nasdaq symbol file format: {url}")
        reader = csv.DictReader(lines, delimiter="|")
        for row in reader:
            if kind == "nasdaq":
                symbol = (row.get("Symbol") or "").strip().upper()
                test = (row.get("Test Issue") or "Y").strip().upper()
                etf = (row.get("ETF") or "Y").strip().upper()
                nextshares = (row.get("NextShares") or "Y").strip().upper()
            else:
                symbol = (row.get("ACT Symbol") or "").strip().upper()
                test = (row.get("Test Issue") or "Y").strip().upper()
                etf = (row.get("ETF") or "Y").strip().upper()
                nextshares = "N"
            if test != "N" or etf != "N" or nextshares == "Y":
                continue
            if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,6}", symbol):
                continue
            if symbol.endswith((".", "-", "$")) or "^" in symbol:
                continue
            symbols.add(symbol)
    if len(symbols) < 1000:
        raise RuntimeError(f"Nasdaq universe unexpectedly small: {len(symbols)}")
    return symbols


def fetch_yahoo_most_active(region, allowed, suffix=None, limit=250):
    url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    params = {
        "formatted": "false",
        "count": str(limit),
        "scrIds": "most_actives",
        "region": region,
    }
    response = SESSION.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("finance", {}).get("result") or []
    if not results:
        raise RuntimeError(f"Yahoo most_actives returned no result for region {region}")
    quotes = results[0].get("quotes") or []
    selected = []
    allowed = set(allowed)
    for quote in quotes:
        symbol = str(quote.get("symbol") or "").upper()
        quote_type = quote.get("quoteType")
        if quote_type and quote_type != "EQUITY":
            continue
        if suffix and not symbol.endswith(suffix):
            continue
        if symbol in allowed and symbol not in selected:
            selected.append(symbol)
    if not selected:
        raise RuntimeError(f"No Yahoo active symbols matched the official {region} universe")
    return selected


def fetch_binance_active_pairs():
    info_response = SESSION.get("https://api.binance.com/api/v3/exchangeInfo", timeout=30)
    info_response.raise_for_status()
    info = info_response.json()
    active = {
        row["symbol"] for row in info.get("symbols", [])
        if row.get("status") == "TRADING"
        and row.get("quoteAsset") == "USDT"
        and row.get("isSpotTradingAllowed", True)
        and row.get("symbol", "").endswith("USDT")
    }
    ticker_response = SESSION.get("https://api.binance.com/api/v3/ticker/24hr", timeout=30)
    ticker_response.raise_for_status()
    ranked = []
    for row in ticker_response.json():
        symbol = row.get("symbol")
        if symbol not in active:
            continue
        try:
            quote_volume = float(row.get("quoteVolume", 0))
        except (TypeError, ValueError):
            continue
        if quote_volume >= MIN_CRYPTO_QUOTE_VOLUME:
            ranked.append((quote_volume, symbol))
    ranked.sort(reverse=True)
    if not ranked:
        raise RuntimeError("Binance returned no active USDT pairs above the liquidity threshold")
    return [symbol for _, symbol in ranked[:MAX_CRYPTO]]


def main():
    os.makedirs(LIST_DIR, exist_ok=True)
    old = {
        "THAI": read_list("thai_stocks.json"),
        "US": read_list("us_stocks.json"),
        "CRYPTO": read_list("crypto.json"),
    }
    result = dict(old)
    sources = {}
    errors = {}

    # Each market is updated independently. A failed source never empties its old list.
    try:
        set_symbols = fetch_set_universe()
        sources["THAI"] = "SET public stock quote list + Yahoo Finance most-active ranking"
        try:
            active = fetch_yahoo_most_active("TH", {s + ".BK" for s in set_symbols}, suffix=".BK")
            result["THAI"] = active[:MAX_THAI]
        except Exception as exc:
            errors["THAI ranking"] = str(exc)
            # Keep the existing verified watchlist when ranking cannot be refreshed.
    except Exception as exc:
        errors["THAI discovery"] = str(exc)

    try:
        us_symbols = fetch_nasdaq_universe()
        sources["US"] = "Nasdaq Trader symbol directory + Yahoo Finance most-active ranking"
        try:
            active = fetch_yahoo_most_active("US", us_symbols)
            result["US"] = active[:MAX_US]
        except Exception as exc:
            errors["US ranking"] = str(exc)
    except Exception as exc:
        errors["US discovery"] = str(exc)

    try:
        result["CRYPTO"] = fetch_binance_active_pairs()
        sources["CRYPTO"] = "Binance Spot exchangeInfo + 24h quote volume (USDT pairs)"
    except Exception as exc:
        errors["CRYPTO discovery"] = str(exc)

    now = datetime.now(timezone(timedelta(hours=7)))
    report = {
        "updated_at": now.strftime("%Y-%m-%d %H:%M:%S (UTC+7)"),
        "limits": {"thai": MAX_THAI, "us": MAX_US, "crypto": MAX_CRYPTO},
        "min_crypto_quote_volume_usdt": MIN_CRYPTO_QUOTE_VOLUME,
        "sources": sources,
        "errors": errors,
        "markets": {}
    }
    for market, filename in [
        ("THAI", "thai_stocks.json"),
        ("US", "us_stocks.json"),
        ("CRYPTO", "crypto.json"),
    ]:
        # Never write an empty watchlist.
        if result[market]:
            write_list(filename, result[market])
        old_set, new_set = set(old[market]), set(result[market])
        report["markets"][market] = {
            "count": len(result[market]),
            "added": sorted(new_set - old_set),
            "removed": sorted(old_set - new_set),
            "source": sources.get(market, "Existing list retained; source unavailable")
        }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if len(result["THAI"]) + len(result["US"]) + len(result["CRYPTO"]) == 0:
        raise RuntimeError("All watchlists are empty; refusing to complete discovery")


if __name__ == "__main__":
    main()
