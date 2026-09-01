# -*- coding: utf-8 -*-
"""
Sync Scalper - فقط فیلتر شکست خط‌روند (بدون هیچ فیلتر دیگه)
------------------------------------------------------------------
تغییرات نسبت به نسخه‌ی قبلی:
- همه‌ی فیلترهای قبلی (بایاس M15، ADX، RSI، فاصله از EMA، فیلتر کندل) حذف شدن
- فقط یک فیلتر ورود مونده: شکست خط‌روند - خط از دو سقف/کف سوینگ اخیر H1 رسم
  میشه، ولی تایید شکست روی یک کندل ۱۵ دقیقه‌ای انجام میشه: باید حداقل ۵۵٪ از
  بازه‌ی اون کندل (high-low) طرفِ شکسته‌شده‌ی خط باشه
- هر دو جهت (لانگ برای شکست به بالا، شورت برای شکست به پایین) آزادن - جهت
  معامله دقیقاً هم‌جهت با شکست تعیین میشه
- مقایسه‌ی دو تایم ورود: 3m و 5m
- SL: نسبت به قیمتی که شکست در آن تایید شده (کندل ۱۵ دقیقه)، فاصله‌ی ثابت
  SL_BREAKOUT_PCT = ۳٪
- TP: بر پایه‌ی ATR، با ضریب استاندارد جهانی TP_ATR_MULT = ۲× ATR (نه دیگه
  بر پایه‌ی نسبت ریسک/ریوارد نسبت به SL)

نحوه اجرا (روی Replit/Colab):
    pip install ccxt pandas numpy
    python sync_scalper_tf_compare.py
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ------------------ تنظیمات ------------------
SYMBOL = "XRP/USDT:USDT"  # فرمت یکپارچه‌ی ccxt برای فیوچرز پرپچوال OKX
TREND_TF = "1h"          # تایم‌فریم رسم خط‌روند (از سوینگ‌های H1)
BREAK_CONFIRM_TF = "15m"  # تایم‌فریم تایید شکست (کندل ۱۵ دقیقه)
ATR_TF = "30m"            # ATR فقط برای TP از این تایم‌فریم محاسبه میشه
BREAK_CONFIRM_PCT = 0.55  # حداقل درصدی از بازه‌ی کندل ۱۵ دقیقه که باید طرف شکسته باشه
MONTHS_BACK = 6

# تست Out-of-Sample: به‌جای «۶ ماه اخیر»، «۶ ماه قبل‌تر از ۶ ماه اخیر» رو می‌گیریم.
# برای رفتن به حالت عادی (۶ ماه اخیر)، این رو None کن.
OOS_END_DATE = (datetime.utcnow() - timedelta(days=30 * MONTHS_BACK)).strftime("%Y-%m-%dT%H:%M:%SZ")

INITIAL_CAPITAL = 100.0
COMMISSION_PCT = 0.05  # درصد کارمزد هر طرف معامله

RISK_PCT = 0.75          # Risk % per trade - برای سایز پوزیشن (بر اساس فاصله‌ی SL)
LEVERAGE = 3.0            # لوریج ۳ برابر - روی سایز پوزیشن (qty) ضرب میشه

ALLOW_LONG = True
ALLOW_SHORT = True

SL_BREAKOUT_PCT = 3.0     # SL نسبت به قیمت شکست، فاصله‌ی ثابت ۳٪
TP_ATR_MULT = 2.0         # TP = ورود ± ۲×ATR (ضریب استاندارد جهانی - قابل تغییر)

PIVOT_LEN = 5             # Pivot Left/Right bars (برای رسم خط‌روند H1)


def fetch_ohlcv(exchange, symbol, timeframe, months_back, end_date=None):
    """
    end_date: اگر داده شود (رشته‌ی ISO مثل "2026-03-01T00:00:00Z")، بازه‌ی
    months_back ماه *قبل از* همین تاریخ گرفته می‌شود، نه از "الان". برای تست
    out-of-sample روی یه بازه‌ی گذشته که موقع طراحی دیده نشده لازمه.
    """
    if end_date is not None:
        end_dt = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")
    else:
        end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=30 * months_back)
    since = exchange.parse8601(start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    until_ms = int(end_dt.timestamp() * 1000)

    all_candles = []
    limit = 100  # سقف OKX هر درخواست کمتر از بایننسه (که ۱۵۰۰ بود)
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not candles:
            break
        candles = [c for c in candles if c[0] <= until_ms]
        all_candles += candles
        if not candles or candles[-1][0] >= until_ms:
            break
        since = candles[-1][0] + 1
        if len(candles) < limit:
            break
    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# ------------------ اندیکاتورها (همه با هموارسازی وایلدر، مطابق TradingView) ------------------

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.where(avg_loss != 0, 100.0)


def atr(df, length):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def dmi_adx(df, length):
    """خروجی: (+DI, -DI, ADX) - دقیقاً مطابق ta.dmi() در Pine Script."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)

    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_mask = (up_move > down_move) & (up_move > 0)
    minus_mask = (down_move > up_move) & (down_move > 0)
    plus_dm[plus_mask] = up_move[plus_mask]
    minus_dm[minus_mask] = down_move[minus_mask]

    atr_w = tr.ewm(alpha=1 / length, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_w)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_w)
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx_val = dx.ewm(alpha=1 / length, adjust=False).mean()
    return plus_di, minus_di, adx_val


def pivot_points(df, left, right):
    """
    معادل ta.pivothigh/pivotlow: کندل i پیوت است اگر high/low آن نسبت به `left`
    کندل قبل و `right` کندل بعد بیشترین/کمترین باشد. غیرقابل‌تکرار (non-repainting):
    پیوت کندل i فقط از کندل i+right به بعد "دیده" می‌شود.
    """
    n = len(df)
    is_pivot_high = pd.Series(False, index=df.index)
    is_pivot_low = pd.Series(False, index=df.index)
    highs, lows = df["high"].values, df["low"].values
    for i in range(left, n - right):
        window_h = highs[i - left: i + right + 1]
        window_l = lows[i - left: i + right + 1]
        if highs[i] == window_h.max() and (highs[i] > highs[i - left:i]).all() \
                and (highs[i] > highs[i + 1:i + right + 1]).all():
            is_pivot_high.iloc[i] = True
        if lows[i] == window_l.min() and (lows[i] < lows[i - left:i]).all() \
                and (lows[i] < lows[i + 1:i + right + 1]).all():
            is_pivot_low.iloc[i] = True
    return is_pivot_high, is_pivot_low


def add_htf_indicators(df):
    """فقط pivot_high/pivot_low لازمه - برای رسم خط‌روند H1."""
    df["pivot_high"], df["pivot_low"] = pivot_points(df, PIVOT_LEN, PIVOT_LEN)
    return df


def add_ltf_indicators(df):
    """فقط ATR لازمه - برای محاسبه‌ی TP."""
    df["atr5"] = atr(df, 14)
    return df


def merge_atr_to_ltf(ltf_df, atr_tf_df):
    """
    ATR که برای TP استفاده میشه، دیگه از خودِ تایم‌فریم ورود نیست - از تایم‌فریم
    ATR_TF (پیش‌فرض ۳۰ دقیقه) میاد. برای هر کندل ورود، آخرین کندل ATR_TF ای که
    *قبل از* بسته‌شدنش تمام شده استفاده می‌شود (بدون نگاه به آینده).
    """
    atr_tf_df = atr_tf_df.set_index("timestamp").sort_index()
    atr_times = atr_tf_df.index.values
    atr_records = atr_tf_df.to_dict("records")
    atr_ptr = -1

    tp_atr_list = []
    for ts in ltf_df["timestamp"]:
        while atr_ptr + 1 < len(atr_times) and atr_times[atr_ptr + 1] < np.datetime64(ts):
            atr_ptr += 1
        if atr_ptr < 0:
            tp_atr_list.append(np.nan)
        else:
            tp_atr_list.append(atr_records[atr_ptr]["atr5"])

    ltf_df = ltf_df.copy()
    ltf_df["tp_atr"] = tp_atr_list
    return ltf_df


def _line_value_at(x1, y1, x2, y2, x):
    """معادله‌ی خط بین دو نقطه‌ی سوینگ، مقدارش را در ایندکس x برمی‌گرداند."""
    if x2 == x1:
        return y2
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (x - x1)


def compute_m15_break_confirm(m15_df, h1_df):
    """
    خط‌روند از دو سقف/کف سوینگ اخیر H1 رسم می‌شود (بدون نگاه به آینده).
    برای هر کندل ۱۵ دقیقه‌ای، ارزش خط‌روند در همون لحظه (بر اساس آخرین کندل H1
    کامل‌شده) محاسبه می‌شود. شکست وقتی تایید می‌شود که حداقل BREAK_CONFIRM_PCT
    (پیش‌فرض ۵۵٪) از بازه‌ی همون کندل ۱۵ دقیقه‌ای (high تا low) طرفِ شکسته‌شده‌ی
    خط باشد - نه فقط قیمت بسته.
    """
    h1_df = h1_df.set_index("timestamp").sort_index()
    sw_high1, sw_high2, sw_low1, sw_low2 = np.nan, np.nan, np.nan, np.nan
    sw_high1_idx, sw_high2_idx, sw_low1_idx, sw_low2_idx = None, None, None, None
    h1_ptr = -1
    h1_times = h1_df.index.values
    h1_records = h1_df.to_dict("records")

    break_up_list, break_down_list = [], []
    breakout_price_up_list, breakout_price_down_list = [], []

    for _, m15_row in m15_df.iterrows():
        ts = m15_row["timestamp"]
        while h1_ptr + 1 < len(h1_times) and h1_times[h1_ptr + 1] < np.datetime64(ts):
            h1_ptr += 1
            rec = h1_records[h1_ptr]
            if rec["pivot_high"]:
                sw_high2, sw_high2_idx = sw_high1, sw_high1_idx
                sw_high1, sw_high1_idx = rec["high"], h1_ptr
            if rec["pivot_low"]:
                sw_low2, sw_low2_idx = sw_low1, sw_low1_idx
                sw_low1, sw_low1_idx = rec["low"], h1_ptr

        m15_high, m15_low = m15_row["high"], m15_row["low"]
        candle_range = (m15_high - m15_low) if (m15_high - m15_low) > 0 else 1e-9

        if h1_ptr < 0:
            break_up_list.append(False)
            break_down_list.append(False)
            breakout_price_up_list.append(np.nan)
            breakout_price_down_list.append(np.nan)
            continue

        break_up = False
        break_down = False

        if sw_high1_idx is not None and sw_high2_idx is not None:
            line_val_up = _line_value_at(sw_high2_idx, sw_high2, sw_high1_idx, sw_high1, h1_ptr)
            frac_above = (m15_high - max(m15_low, line_val_up)) / candle_range
            frac_above = min(max(frac_above, 0.0), 1.0)
            break_up = frac_above >= BREAK_CONFIRM_PCT

        if sw_low1_idx is not None and sw_low2_idx is not None:
            line_val_down = _line_value_at(sw_low2_idx, sw_low2, sw_low1_idx, sw_low1, h1_ptr)
            frac_below = (min(m15_high, line_val_down) - m15_low) / candle_range
            frac_below = min(max(frac_below, 0.0), 1.0)
            break_down = frac_below >= BREAK_CONFIRM_PCT

        break_up_list.append(bool(break_up))
        break_down_list.append(bool(break_down))
        breakout_price_up_list.append(m15_row["close"] if break_up else np.nan)
        breakout_price_down_list.append(m15_row["close"] if break_down else np.nan)

    out = m15_df.copy()
    out["break_up"] = break_up_list
    out["break_down"] = break_down_list
    out["breakout_price_up"] = breakout_price_up_list
    out["breakout_price_down"] = breakout_price_down_list
    return out


def merge_break_to_ltf(ltf_df, m15_break_df):
    """
    سیگنال شکست (تایید‌شده روی کندل ۱۵ دقیقه) را به تایم‌فریم ورود (3m/5m)
    منتقل می‌کند - برای هر کندل ورود، آخرین کندل ۱۵ دقیقه‌ای که *قبل از*
    بسته‌شدنش تمام شده استفاده می‌شود (بدون نگاه به آینده).
    """
    m15_break_df = m15_break_df.set_index("timestamp").sort_index()
    m15_times = m15_break_df.index.values
    m15_records = m15_break_df.to_dict("records")
    m15_ptr = -1

    break_up_list, break_down_list = [], []
    breakout_price_up_list, breakout_price_down_list = [], []

    for ts in ltf_df["timestamp"]:
        while m15_ptr + 1 < len(m15_times) and m15_times[m15_ptr + 1] < np.datetime64(ts):
            m15_ptr += 1

        if m15_ptr < 0:
            break_up_list.append(False)
            break_down_list.append(False)
            breakout_price_up_list.append(np.nan)
            breakout_price_down_list.append(np.nan)
            continue

        rec = m15_records[m15_ptr]
        break_up_list.append(bool(rec["break_up"]))
        break_down_list.append(bool(rec["break_down"]))
        breakout_price_up_list.append(rec["breakout_price_up"])
        breakout_price_down_list.append(rec["breakout_price_down"])

    ltf_df = ltf_df.copy()
    ltf_df["break_up"] = break_up_list
    ltf_df["break_down"] = break_down_list
    ltf_df["breakout_price_up"] = breakout_price_up_list
    ltf_df["breakout_price_down"] = breakout_price_down_list
    return ltf_df


def backtest(ltf_df):
    equity = INITIAL_CAPITAL
    position = None  # dict: side, entry, sl, tp, qty
    trades = []

    n = len(ltf_df)
    for i in range(max(PIVOT_LEN * 2, 20), n):
        row = ltf_df.iloc[i]
        high, low, close = row["high"], row["low"], row["close"]

        # ---- مدیریت پوزیشن باز ----
        if position is not None:
            side = position["side"]
            exit_price, exit_reason = None, None
            if side == "long":
                if low <= position["sl"]:
                    exit_price, exit_reason = position["sl"], "stop_loss"
                elif high >= position["tp"]:
                    exit_price, exit_reason = position["tp"], "take_profit"
            else:
                if high >= position["sl"]:
                    exit_price, exit_reason = position["sl"], "stop_loss"
                elif low <= position["tp"]:
                    exit_price, exit_reason = position["tp"], "take_profit"

            if exit_price is not None:
                if side == "long":
                    pnl = position["qty"] * (exit_price - position["entry"])
                else:
                    pnl = position["qty"] * (position["entry"] - exit_price)
                fee = (position["qty"] * position["entry"] + position["qty"] * exit_price) * (COMMISSION_PCT / 100)
                pnl -= fee
                equity += pnl
                trades.append({
                    "side": side, "entry_time": position["entry_time"], "entry": position["entry"],
                    "exit_time": row["timestamp"], "exit": exit_price, "reason": exit_reason,
                    "pnl": pnl, "equity_after": equity,
                })
                position = None

        if pd.isna(row.get("tp_atr", np.nan)):
            continue

        # ---- بررسی سیگنال ورود جدید - فقط فیلتر شکست خط‌روند، بدون سقف روزانه ----
        if position is None:
            buy_cond = ALLOW_LONG and row.get("break_up", False)
            sell_cond = ALLOW_SHORT and row.get("break_down", False)

            breakout_price_up = row.get("breakout_price_up", np.nan)
            breakout_price_down = row.get("breakout_price_down", np.nan)
            atr_val = row["tp_atr"]  # ATR از تایم‌فریم ATR_TF (۳۰ دقیقه)، نه تایم ورود

            if buy_cond and not pd.isna(breakout_price_up):
                sl_price = breakout_price_up * (1 - SL_BREAKOUT_PCT / 100)
                risk = close - sl_price
                if risk > 0:
                    risk_amount = equity * (RISK_PCT / 100)
                    qty = (risk_amount / risk) * LEVERAGE
                    position = {
                        "side": "long", "entry": close, "sl": sl_price,
                        "tp": close + TP_ATR_MULT * atr_val, "qty": qty, "entry_time": row["timestamp"],
                    }
            elif sell_cond and not pd.isna(breakout_price_down):
                sl_price = breakout_price_down * (1 + SL_BREAKOUT_PCT / 100)
                risk = sl_price - close
                if risk > 0:
                    risk_amount = equity * (RISK_PCT / 100)
                    qty = (risk_amount / risk) * LEVERAGE
                    position = {
                        "side": "short", "entry": close, "sl": sl_price,
                        "tp": close - TP_ATR_MULT * atr_val, "qty": qty, "entry_time": row["timestamp"],
                    }

    trades_df = pd.DataFrame(trades)
    return equity, trades_df


LTF_CANDIDATES = ["15m"]  # فقط ۱۵ دقیقه - 3m/5m حذف شدن


def run_one_timeframe(exchange, ltf, m15_break_df, atr_tf_df):
    print(f"\n--- تایم‌فریم ورود: {ltf} ---")
    print(f"در حال دانلود دیتای {ltf} برای {SYMBOL}...")
    ltf_df = fetch_ohlcv(exchange, SYMBOL, ltf, MONTHS_BACK, end_date=OOS_END_DATE)
    print(f"⚠️ بازه‌ی واقعی دیتای دریافتی ({ltf}): از {ltf_df['timestamp'].min()} تا {ltf_df['timestamp'].max()} ({len(ltf_df)} کندل)")

    print("در حال هماهنگ‌سازی سیگنال شکست (تایید‌شده روی M15) با این تایم‌فریم...")
    ltf_df = merge_break_to_ltf(ltf_df, m15_break_df)

    print(f"در حال هماهنگ‌سازی ATR (از {ATR_TF}) برای محاسبه‌ی TP...")
    ltf_df = merge_atr_to_ltf(ltf_df, atr_tf_df)

    print("در حال اجرای بک‌تست...")
    final_equity, trades_df = backtest(ltf_df)

    safe_ltf = ltf.replace("/", "-")
    trades_df.to_csv(f"sync_scalper_trades_{safe_ltf}.csv", index=False)

    total_trades = len(trades_df)
    wins = (trades_df["pnl"] > 0).sum() if total_trades else 0
    win_rate = (wins / total_trades * 100) if total_trades else 0
    total_pnl = trades_df["pnl"].sum() if total_trades else 0

    return {
        "ltf": ltf,
        "final_equity": final_equity,
        "pnl": total_pnl,
        "trades": total_trades,
        "win_rate": win_rate,
    }


def main():
    exchange = ccxt.okx({"enableRateLimit": True})
    print(f"=== تست Out-of-Sample ===")
    print(f"بازه‌ی درخواستی: ۶ ماه منتهی به {OOS_END_DATE}")

    print(f"در حال دانلود دیتای {TREND_TF} (رسم خط‌روند) برای {SYMBOL}...")
    h1_df = fetch_ohlcv(exchange, SYMBOL, TREND_TF, MONTHS_BACK, end_date=OOS_END_DATE)
    print(f"⚠️ بازه‌ی واقعی دیتای دریافتی ({TREND_TF}): از {h1_df['timestamp'].min()} تا {h1_df['timestamp'].max()} ({len(h1_df)} کندل)")
    h1_df = add_htf_indicators(h1_df)

    print(f"در حال دانلود دیتای {BREAK_CONFIRM_TF} (تایید شکست) برای {SYMBOL}...")
    m15_df = fetch_ohlcv(exchange, SYMBOL, BREAK_CONFIRM_TF, MONTHS_BACK, end_date=OOS_END_DATE)
    print(f"⚠️ بازه‌ی واقعی دیتای دریافتی ({BREAK_CONFIRM_TF}): از {m15_df['timestamp'].min()} تا {m15_df['timestamp'].max()} ({len(m15_df)} کندل)")

    print("در حال محاسبه‌ی تایید شکست خط‌روند روی کندل‌های ۱۵ دقیقه...")
    m15_break_df = compute_m15_break_confirm(m15_df, h1_df)

    print(f"در حال دانلود دیتای {ATR_TF} (فقط برای محاسبه‌ی TP) برای {SYMBOL}...")
    atr_tf_df = fetch_ohlcv(exchange, SYMBOL, ATR_TF, MONTHS_BACK, end_date=OOS_END_DATE)
    print(f"⚠️ بازه‌ی واقعی دیتای دریافتی ({ATR_TF}): از {atr_tf_df['timestamp'].min()} تا {atr_tf_df['timestamp'].max()} ({len(atr_tf_df)} کندل)")
    atr_tf_df = add_ltf_indicators(atr_tf_df)

    results = []
    for ltf in LTF_CANDIDATES:
        res = run_one_timeframe(exchange, ltf, m15_break_df, atr_tf_df)
        results.append(res)

    print("\n\n=== مقایسه‌ی تایم‌فریم‌های ورود (فقط فیلتر شکست خط‌روند) ===")
    print(f"نماد: {SYMBOL} | بازه: {MONTHS_BACK} ماه | سرمایه اولیه: ${INITIAL_CAPITAL}")
    print(f"{'تایم‌فریم':<10}{'سرمایه نهایی':<15}{'سود/زیان':<12}{'معاملات':<10}{'نرخ برد':<10}")
    for r in results:
        print(f"{r['ltf']:<10}${r['final_equity']:<14.2f}${r['pnl']:<11.2f}{r['trades']:<10}{r['win_rate']:<9.2f}%")

    summary_df = pd.DataFrame([{
        "timeframe": r["ltf"], "final_equity": r["final_equity"], "pnl": r["pnl"],
        "trades": r["trades"], "win_rate": r["win_rate"],
    } for r in results])
    summary_df.to_csv("sync_scalper_timeframe_comparison.csv", index=False)
    print("\nفایل مقایسه ذخیره شد: sync_scalper_timeframe_comparison.csv")
    print("فایل معاملات هر تایم‌فریم هم جدا ذخیره شد: sync_scalper_trades_<تایم‌فریم>.csv")


if __name__ == "__main__":
    main()
