import os
import json
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta

# ดึง Webhook URL จาก GitHub Secrets
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")

# โหลดรายชื่อจากไฟล์แยกในโฟลเดอร์ lists/
def load_watchlist():
    return {
        "THAI": load_json("lists/thai_stocks.json", []),
        "US": load_json("lists/us_stocks.json", []),
        "CRYPTO": load_json("lists/crypto.json", [])
    }

STATE_FILE = "alert_state.json"
OUTPUT_FILE = "data.json"
COOLDOWN_HOURS = 4  # ป้องกันแจ้งเตือนตัวเดิมซ้ำภายใน 4 ชั่วโมง

def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return default
    return default

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def fetch_crypto_klines(symbol, interval="15m", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    res = requests.get(url, timeout=10)
    data = res.json()
    if not isinstance(data, list):
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=[
        "Time", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteVol", "Trades", "TakerBase", "TakerQuote", "Ignore"
    ])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
    return df

def fetch_stock_klines(ticker, interval="15m", period="5d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_indicators(df):
    if len(df) < 30:
        return None
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    # EMA 9 & 21
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    # RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # ATR 14 สำหรับคำนวณจุด TP / SL
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()

    # Volume Spike
    vol_avg = vol.rolling(window=20).mean()

    curr_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    curr_rsi = float(rsi.iloc[-1])
    curr_ema9 = float(ema9.iloc[-1])
    prev_ema9 = float(ema9.iloc[-2])
    curr_ema21 = float(ema21.iloc[-1])
    prev_ema21 = float(ema21.iloc[-2])
    curr_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else curr_close * 0.02
    vol_ratio = float(vol.iloc[-1] / vol_avg.iloc[-1]) if vol_avg.iloc[-1] > 0 else 1.0

    pct_change = ((curr_close - prev_close) / prev_close) * 100

    # วิเคราะห์สัญญาณซื้อ/ขาย
    signal = "WATCH"
    reason = "รอดูแนวโน้ม"
    tp = 0.0
    sl = 0.0

    # เงื่อนไข BUY
    if curr_rsi < 32:
        signal = "BUY"
        reason = f"RSI ต่ำโซน Oversold ({curr_rsi:.1f}) ลุ้นเด้งกลับ"
        sl = curr_close - (1.5 * curr_atr)
        tp = curr_close + (2.5 * curr_atr)
    elif prev_ema9 <= prev_ema21 and curr_ema9 > curr_ema21 and curr_rsi < 65:
        signal = "BUY"
        reason = "Golden Cross (EMA9 ตัดขึ้น EMA21)"
        sl = curr_close - (1.5 * curr_atr)
        tp = curr_close + (3.0 * curr_atr)
    elif vol_ratio > 2.2 and pct_change > 1.0 and curr_rsi < 68:
        signal = "BUY"
        reason = f"Volume เข้าผิดปกติ ({vol_ratio:.1f}x) ราคาเบรกขึ้น"
        sl = curr_close - (1.5 * curr_atr)
        tp = curr_close + (2.5 * curr_atr)

    # เงื่อนไข SELL / TAKE PROFIT
    elif curr_rsi > 72:
        signal = "SELL"
        reason = f"RSI สูงโซน Overbought ({curr_rsi:.1f}) ควรแบ่งขายทำกำไร"
        sl = curr_close + (1.5 * curr_atr)
        tp = curr_close - (2.5 * curr_atr)
    elif prev_ema9 >= prev_ema21 and curr_ema9 < curr_ema21 and curr_rsi > 35:
        signal = "SELL"
        reason = "Death Cross (EMA9 ตัดลง EMA21) เสี่ยงพักตัว"
        sl = curr_close + (1.5 * curr_atr)
        tp = curr_close - (2.5 * curr_atr)

    return {
        "price": round(curr_close, 4 if curr_close < 1 else 2),
        "change_pct": round(pct_change, 2),
        "rsi": round(curr_rsi, 1),
        "vol_ratio": round(vol_ratio, 2),
        "signal": signal,
        "reason": reason,
        "tp": round(tp, 4 if curr_close < 1 else 2) if tp > 0 else None,
        "sl": round(sl, 4 if curr_close < 1 else 2) if sl > 0 else None
    }

def send_discord_alert(item):
    if not DISCORD_WEBHOOK:
        return
    
    is_buy = item["signal"] == "BUY"
    color = 0x10B981 if is_buy else 0xEF4444  # เขียว / แดง
    emoji = "🟢 สัญญาณเข้าซื้อ (BUY)" if is_buy else "🔴 สัญญาณขายทำกำไร (SELL)"
    currency = "THB" if item["market"] == "THAI" else ("USDT" if item["market"] == "CRYPTO" else "USD")

    embed = {
        "title": f"{emoji} : {item['symbol']} ({item['market']})",
        "description": f"**เหตุผล:** {item['reason']}",
        "color": color,
        "fields": [
            {"name": "💰 ราคาปัจจุบัน", "value": f"`{item['price']} {currency}` ({item['change_pct']:+}%)", "inline": True},
            {"name": "📊 RSI (14)", "value": f"`{item['rsi']}`", "inline": True},
            {"name": "🔥 Volume Spike", "value": f"`{item['vol_ratio']}x`", "inline": True},
            {"name": "🎯 เป้าทำกำไร (TP)", "value": f"`{item['tp']}`", "inline": True},
            {"name": "🛡️ จุดตัดขาดทุน (SL)", "value": f"`{item['sl']}`", "inline": True}
        ],
        "footer": {"text": f"AI Market Radar • Timeframe 15m"}
    }
    try:
        requests.post(DISCORD_WEBHOOK, json={"embeds": [embed]}, timeout=10)
    except Exception as e:
        print(f"Discord Error: {e}")

def main():
    tz_bkk = timezone(timedelta(hours=7))
    now_bkk = datetime.now(tz_bkk)
    now_ts = time.time()

    watchlist = load_watchlist()
    state = load_json(STATE_FILE, {})
    results = []

    for market, symbols in watchlist.items():
        print(f"Scanning {market}...")
        for sym in symbols:
            try:
                if market == "CRYPTO":
                    df = fetch_crypto_klines(sym)
                    display_sym = sym.replace("USDT", "/USDT")
                else:
                    df = fetch_stock_klines(sym)
                    display_sym = sym.replace(".BK", "")

                metrics = calculate_indicators(df)
                if not metrics:
                    continue

                item = {
                    "symbol": display_sym,
                    "raw_symbol": sym,
                    "market": market,
                    **metrics
                }
                results.append(item)

                if item["signal"] in ["BUY", "SELL"]:
                    state_key = f"{sym}_{item['signal']}"
                    last_alert = state.get(state_key, 0)
                    if (now_ts - last_alert) > (COOLDOWN_HOURS * 3600):
                        send_discord_alert(item)
                        state[state_key] = now_ts
                        time.sleep(0.5)

            except Exception as e:
                print(f"Error scanning {sym}: {e}")

    priority = {"BUY": 0, "SELL": 1, "WATCH": 2}
    results.sort(key=lambda x: (priority[x["signal"]], -abs(x["change_pct"])))

    output_payload = {
        "updated_at": now_bkk.strftime("%Y-%m-%d %H:%M:%S (UTC+7)"),
        "summary": {
            "total": len(results),
            "buy": sum(1 for r in results if r["signal"] == "BUY"),
            "sell": sum(1 for r in results if r["signal"] == "SELL")
        },
        "items": results
    }

    save_json(OUTPUT_FILE, output_payload)
    save_json(STATE_FILE, state)
    print("Scan completed successfully.")

if __name__ == "__main__":
    main()