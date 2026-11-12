import os
import json
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
STATE_FILE = "alert_state.json"
OUTPUT_FILE = "data.json"
COOLDOWN_HOURS = 4
ATR_SL_MULTIPLIER = 1.5
TP_R_MULTIPLIERS = (1.0, 2.0, 3.0)

# Thai-stock position sizing defaults (the dashboard calculator can override these locally).
THAI_CAPITAL = 5000.0
THAI_COMMISSION_RATE = 0.0025  # per side; excludes VAT and other exchange fees
THAI_LOT_SIZE = 100
THAI_MAX_NET_LOSS = 100.0

def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return default
    return default

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_watchlist():
    return {
        "THAI": load_json("lists/thai_stocks.json", []),
        "US": load_json("lists/us_stocks.json", []),
        "CRYPTO": load_json("lists/crypto.json", [])
    }

def fetch_crypto_klines(symbol, interval="15m", limit=120):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    res = requests.get(url, timeout=15)
    res.raise_for_status()
    data = res.json()
    if not isinstance(data, list) or len(data) < 31:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=[
        "Time", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteVol", "Trades", "TakerBase", "TakerQuote", "Ignore"
    ])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Binance returns the currently-forming candle as the final row.
    now_ms = int(time.time() * 1000)
    df = df[pd.to_numeric(df["CloseTime"], errors="coerce") < now_ms].copy()
    return df.reset_index(drop=True)

def fetch_stock_klines(ticker, interval="15m", period="5d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Exclude the latest bar if its 15-minute interval has not finished.
    try:
        idx = pd.Timestamp(df.index[-1])
        idx_utc = idx.tz_localize("UTC") if idx.tzinfo is None else idx.tz_convert("UTC")
        if idx_utc + pd.Timedelta(minutes=15) > pd.Timestamp.now(tz="UTC"):
            df = df.iloc[:-1]
    except Exception:
        # Conservative fallback: don't analyse a potentially-live final candle.
        df = df.iloc[:-1]
    return df

def build_chart_candles(df, market, limit=60):
    """Return closed OHLC candles in Lightweight Charts format."""
    if df is None or df.empty:
        return []
    rows = []
    source = df.tail(limit)
    for index, row in source.iterrows():
        try:
            if market == "CRYPTO":
                timestamp = int(float(row["Time"]) / 1000)
            else:
                stamp = pd.Timestamp(index)
                if stamp.tzinfo is None:
                    stamp = stamp.tz_localize("UTC")
                timestamp = int(stamp.timestamp())
            candle = {
                "time": timestamp,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"])
            }
            if all(np.isfinite(candle[key]) for key in ("open", "high", "low", "close")):
                rows.append(candle)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    # Lightweight Charts requires unique, ascending timestamps.
    unique = {item["time"]: item for item in rows}
    return [unique[key] for key in sorted(unique)]



def calculate_thai_trade_plan(entry, tp1, tp2, tp3, sl):
    """Estimate Thai-stock position size and net P/L after buy/sell commission."""
    try:
        entry, tp1, tp2, tp3, sl = map(float, (entry, tp1, tp2, tp3, sl))
    except (TypeError, ValueError):
        return {"eligible": False, "reason": "ราคา Entry/TP/SL ไม่ถูกต้อง"}

    values = (entry, tp1, tp2, tp3, sl)
    if not all(np.isfinite(value) for value in values):
        return {"eligible": False, "reason": "ราคา Entry/TP/SL ไม่ใช่ตัวเลข"}
    if entry <= 0 or sl <= 0 or sl >= entry or min(tp1, tp2, tp3) <= entry:
        return {"eligible": False, "reason": "ระดับราคาไม่ถูกต้อง"}

    lot_size = int(THAI_LOT_SIZE)
    fee = float(THAI_COMMISSION_RATE)
    capital = float(THAI_CAPITAL)
    max_loss = float(THAI_MAX_NET_LOSS)
    if lot_size < 1 or fee < 0 or capital <= 0 or max_loss <= 0:
        return {"eligible": False, "reason": "ค่าตั้งค่าการลงทุนไม่ถูกต้อง"}

    cost_per_lot = entry * lot_size * (1.0 + fee)
    net_loss_per_share = (entry - sl) + (entry + sl) * fee
    if cost_per_lot <= 0 or net_loss_per_share <= 0:
        return {"eligible": False, "reason": "ไม่สามารถคำนวณขนาดการลงทุนได้"}

    affordable_lots = int(np.floor((capital + 1e-9) / cost_per_lot))
    risk_lots = int(np.floor((max_loss + 1e-9) / (net_loss_per_share * lot_size)))
    lots = max(0, min(affordable_lots, risk_lots))

    # Recheck actual rounded lots to avoid floating-point boundary overruns.
    while lots > 0:
        shares = lots * lot_size
        capital_used = entry * shares * (1.0 + fee)
        net_sl = (sl - entry) * shares - (entry + sl) * shares * fee
        if capital_used <= capital + 1e-7 and net_sl >= -max_loss - 1e-7:
            break
        lots -= 1

    if lots < 1:
        return {
            "eligible": False,
            "reason": "เงินทุนไม่พอซื้อขั้นต่ำ หรือขาดทุนสุทธิที่ SL เกินวงเงิน",
            "min_lot_cost": round(cost_per_lot, 2),
            "min_lot_loss": round(net_loss_per_share * lot_size, 2),
            "capital": capital,
            "max_loss": max_loss,
            "lot_size": lot_size
        }

    shares = lots * lot_size
    buy_value = entry * shares
    buy_fee = buy_value * fee
    capital_used = buy_value + buy_fee

    def net_pnl(target):
        return (target - entry) * shares - (entry + target) * shares * fee

    return {
        "eligible": True,
        "capital": capital,
        "lot_size": lot_size,
        "lots": lots,
        "shares": shares,
        "capital_used": round(capital_used, 2),
        "buy_fee": round(buy_fee, 2),
        "tp1_net": round(net_pnl(tp1), 2),
        "tp2_net": round(net_pnl(tp2), 2),
        "tp3_net": round(net_pnl(tp3), 2),
        "sl_net": round(net_pnl(sl), 2),
        "max_loss": max_loss,
        "commission_rate": fee
    }

def calculate_indicators(df, market="THAI"):
    if df is None or len(df) < 35:
        return None
    df = df.copy()
    for col in ["Close", "High", "Low", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close", "High", "Low", "Volume"])
    if len(df) < 35:
        return None

    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    vol_avg = vol.rolling(20).mean()

    price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    curr_rsi = float(rsi.iloc[-1])
    e9, e21 = float(ema9.iloc[-1]), float(ema21.iloc[-1])
    p9, p21 = float(ema9.iloc[-2]), float(ema21.iloc[-2])
    atr_value = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else price * 0.02
    atr_value = max(atr_value, price * 0.001)
    avg_volume = float(vol_avg.iloc[-1]) if pd.notna(vol_avg.iloc[-1]) else 0
    vol_ratio = float(vol.iloc[-1] / avg_volume) if avg_volume > 0 else 1.0
    pct_change = ((price - prev_close) / prev_close * 100) if prev_close else 0.0

    golden_cross = p9 <= p21 and e9 > e21
    death_cross = p9 >= p21 and e9 < e21
    trend_up = e9 > e21 and price > e21
    trend_down = e9 < e21 and price < e21
    oversold = curr_rsi < 32
    breakout = vol_ratio > 2.2 and pct_change > 1.0 and curr_rsi < 68

    score = 0
    score += 25 if trend_up else 0
    score += 25 if golden_cross else (15 if e9 > e21 else 0)
    score += 20 if 40 <= curr_rsi <= 65 else (15 if oversold else 0)
    score += 20 if vol_ratio >= 1.5 else (10 if vol_ratio >= 1.1 else 0)
    score += 10 if pct_change > 0 else 0
    score = min(100, score)

    signal, reason = "WATCH", "รอสัญญาณยืนยัน"
    if oversold:
        signal, reason = "BUY", f"RSI Oversold ({curr_rsi:.1f}) รอสัญญาณกลับตัว"
    elif golden_cross and curr_rsi < 65:
        signal, reason = "BUY", "EMA9 ตัดขึ้น EMA21 (Golden Cross)"
    elif breakout:
        signal, reason = "BUY", f"Volume สูงกว่าค่าเฉลี่ย {vol_ratio:.1f} เท่า พร้อมราคาเบรกขึ้น"
    elif curr_rsi > 72:
        signal, reason = "SELL", f"RSI Overbought ({curr_rsi:.1f}) พิจารณาขาย/ลดพอร์ต"
    elif death_cross and curr_rsi > 35:
        signal, reason = "SELL", "EMA9 ตัดลง EMA21 (Death Cross)"

    digits = 4 if price < 1 else 2
    entry = price if signal == "BUY" else None
    risk = ATR_SL_MULTIPLIER * atr_value
    sl = price - risk if signal == "BUY" else (price + risk if signal == "SELL" else None)
    tp1 = price + TP_R_MULTIPLIERS[0] * risk if signal == "BUY" else None
    tp2 = price + TP_R_MULTIPLIERS[1] * risk if signal == "BUY" else None
    tp3 = price + TP_R_MULTIPLIERS[2] * risk if signal == "BUY" else None

    return {
        "price": round(price, digits),
        "change_pct": round(pct_change, 2),
        "rsi": round(curr_rsi, 1),
        "vol_ratio": round(vol_ratio, 2),
        "signal": signal,
        "reason": reason,
        "score": score,
        "entry": round(entry, digits) if entry is not None else None,
        "tp1": round(tp1, digits) if tp1 is not None else None,
        "tp2": round(tp2, digits) if tp2 is not None else None,
        "tp3": round(tp3, digits) if tp3 is not None else None,
        "tp": round(tp2, digits) if tp2 is not None else None,
        "sl": round(sl, digits) if sl is not None else None,
        "risk_reward": "1:1 / 1:2 / 1:3" if signal == "BUY" else None,
        "trade_plan": calculate_thai_trade_plan(entry, tp1, tp2, tp3, sl)
                      if signal == "BUY" and market == "THAI" else None,
        "atr": round(atr_value, digits),
        "candle_closed": True
    }

def send_discord_alert(item):
    if not DISCORD_WEBHOOK:
        return
    if item["signal"] not in ("BUY", "SELL"):
        return
    is_buy = item["signal"] == "BUY"
    color = 0x10B981 if is_buy else 0xEF4444
    currency = "THB" if item["market"] == "THAI" else ("USDT" if item["market"] == "CRYPTO" else "USD")
    fields = [
        {"name": "ราคาล่าสุด", "value": f"`{item['price']} {currency}`", "inline": True},
        {"name": "คะแนน", "value": f"`{item['score']}/100`", "inline": True},
        {"name": "RSI", "value": f"`{item['rsi']}`", "inline": True}
    ]
    if is_buy:
        fields += [
            {"name": "Entry", "value": f"`{item['entry']}`", "inline": True},
            {"name": "TP1 / TP2 / TP3", "value": f"`{item['tp1']} / {item['tp2']} / {item['tp3']}`", "inline": False},
            {"name": "Stop Loss", "value": f"`{item['sl']}`", "inline": True},
            {"name": "Risk/Reward", "value": item["risk_reward"], "inline": True}
        ]
        plan = item.get("trade_plan")
        if item["market"] == "THAI" and plan:
            if plan.get("eligible"):
                fields += [
                    {"name": "ขนาดลงทุน", "value": f"`{plan['lots']} Lot / {plan['shares']:,} หุ้น`", "inline": True},
                    {"name": "ใช้เงินรวมค่าคอมฯ ซื้อ", "value": f"`{plan['capital_used']:,.2f} THB`", "inline": True},
                    {"name": "กำไรสุทธิ TP1 / TP2 / TP3", "value": f"`{plan['tp1_net']:,.2f} / {plan['tp2_net']:,.2f} / {plan['tp3_net']:,.2f} THB`", "inline": False},
                    {"name": "ขาดทุนสุทธิ SL", "value": f"`{plan['sl_net']:,.2f} THB`", "inline": True}
                ]
            else:
                fields.append({
                    "name": "แผนลงทุนหุ้นไทย",
                    "value": f"`ไม่ผ่านเงื่อนไข` — {plan.get('reason', 'ตรวจสอบเงินทุน/ความเสี่ยง')}",
                    "inline": False
                })
    embed = {
        "title": f"{'🟢 BUY' if is_buy else '🔴 SELL'}: {item['symbol']} ({item['market']})",
        "description": item["reason"] + "\nวิเคราะห์จากแท่งเทียนที่ปิดแล้วเท่านั้น",
        "color": color,
        "fields": fields,
        "footer": {"text": "Market Scanner • 15m • ATR-based levels"}
    }
    try:
        response = requests.post(DISCORD_WEBHOOK, json={"embeds": [embed]}, timeout=10)
        response.raise_for_status()
    except Exception as exc:
        print(f"Discord Error: {exc}")

def main():
    tz_bkk = timezone(timedelta(hours=7))
    now_bkk = datetime.now(tz_bkk)
    now_ts = time.time()
    watchlist = load_watchlist()
    state = load_json(STATE_FILE, {})
    results = []

    for market, symbols in watchlist.items():
        print(f"Scanning {market}: {len(symbols)} symbols")
        for sym in symbols:
            try:
                if market == "CRYPTO":
                    df = fetch_crypto_klines(sym)
                    display_sym = sym.replace("USDT", "/USDT")
                else:
                    df = fetch_stock_klines(sym)
                    display_sym = sym.replace(".BK", "")
                metrics = calculate_indicators(df, market=market)
                if not metrics:
                    print(f"Skip {sym}: insufficient/invalid candles")
                    continue
                item = {"symbol": display_sym, "raw_symbol": sym, "market": market, "candles": build_chart_candles(df, market), **metrics}
                results.append(item)

                if item["signal"] in ("BUY", "SELL"):
                    state_key = f"{sym}_{item['signal']}"
                    last_alert = state.get(state_key, 0)
                    if now_ts - last_alert > COOLDOWN_HOURS * 3600:
                        send_discord_alert(item)
                        state[state_key] = now_ts
                        time.sleep(0.3)
            except Exception as exc:
                print(f"Error scanning {sym}: {exc}")

    priority = {"BUY": 0, "SELL": 1, "WATCH": 2}
    results.sort(key=lambda x: (priority.get(x["signal"], 3), -x["score"], -abs(x["change_pct"])))
    buy_items = [x for x in results if x["signal"] == "BUY"]
    output_payload = {
        "updated_at": now_bkk.strftime("%Y-%m-%d %H:%M:%S (UTC+7)"),
        "timeframe": "15m",
        "refresh_interval_minutes": 15,
        "method": "Closed candles only; EMA9/21 + RSI14 + Volume + ATR14",
        "summary": {
            "total": len(results),
            "buy": sum(1 for x in results if x["signal"] == "BUY"),
            "sell": sum(1 for x in results if x["signal"] == "SELL"),
            "watch": sum(1 for x in results if x["signal"] == "WATCH")
        },
        "top10": buy_items[:10],
        "items": results
    }
    save_json(OUTPUT_FILE, output_payload)
    save_json(STATE_FILE, state)
    print(f"Scan completed: {len(results)} assets, BUY={output_payload['summary']['buy']}")

if __name__ == "__main__":
    main()
