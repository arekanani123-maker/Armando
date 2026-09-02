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

# ══════════════════════════════════════════════════════════════════
# 🔒 تنظیمات قفل‌شده (استراتژی «۶» نهایی) - این بخش نباید تغییر کنه
# ══════════════════════════════════════════════════════════════════
SYMBOLS = ["XRP/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT:USDT", "ADA/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT"]
TREND_TF = "1h"          # تایم‌فریم رسم خط‌روند (از سوینگ‌های H1)
ATR_TF = "15m"            # ATR برای TP - از ۱۵ دقیقه‌ای محاسبه می‌شه
MONTHS_BACK = 6           # طول خودِ بازه‌ی تست (۶ ماه)

OOS_OFFSET_MONTHS = 0     # حالت عادی: ۶ ماه اخیر (ختم به الان)
OOS_END_DATE = (datetime.utcnow() - timedelta(days=30 * OOS_OFFSET_MONTHS)).strftime("%Y-%m-%dT%H:%M:%SZ")

INITIAL_CAPITAL = 10000.0
COMMISSION_PCT = 0.05  # درصد کارمزد هر طرف معامله

SYMBOL_DIRECTION = {
    "XRP/USDT:USDT": "short",  # فقط شورت - قفل قبلی
    "ETH/USDT:USDT": "long",   # فقط لانگ - قفل قبلی
    "BTC/USDT:USDT": "both",   # هردو جهت آزاد
    "ADA/USDT:USDT": "both",   # هردو جهت آزاد
    "SOL/USDT:USDT": "both",   # هردو جهت آزاد
    "DOGE/USDT:USDT": "both",  # هردو جهت آزاد
}

SL_BREAKOUT_PCT = 3.0     # SL نسبت به قیمت شکست، فاصله‌ی ثابت ۳٪
TP_ATR_MULT = 3.04        # TP اصلی (نیمه‌ی اول پوزیشن) = ورود ± ۳.۰۴×ATR
TP3_ATR_MULT = 5.04       # تارگت۳ (نیمه‌ی دوم پوزیشن) = ورود ± ۵.۰۴×ATR

ADX_LEN = 14
ADX_MIN = 21               # فیلتر ADX - فقط بازار غیر-رنج
ADX_ENABLED = True

PIVOT_LEN = 5             # Pivot Left/Right bars (برای رسم خط‌روند H1)

# ══════════════════════════════════════════════════════════════════
# ⚙️ تنظیمات قابل‌تغییر - فقط این دو تا آزادن
# ══════════════════════════════════════════════════════════════════
POSITION_MARGIN = 500.0  # مارجین ثابت هر پوزیشن - قابل تنظیم
LEVERAGE = 3.0            # لوریج - قابل تنظیم
# ══════════════════════════════════════════════════════════════════


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


def compute_h1_break(h1_df):
    """
    شکست بر پایه‌ی بالاترین/پایین‌ترین قیمت کندل H1 نسبت به خط چک می‌شه (نه
    فقط قیمت بسته) - یعنی اگه در طول کندل قیمت از خط رد شده باشه، شکست ثبت
    می‌شه. قیمت مبنا (breakout_price) دقیقاً قیمت خودِ خط‌روند در همون لحظه
    است - نه قیمت بسته یا هر قیمت دیگه‌ی کندل. برای هر کندل H1، سوینگ‌پوینت‌ها
    فقط از کندل‌های *قبلی* (نه خودش) استفاده می‌کنن - بدون نگاه به آینده.
    """
    h1_df = h1_df.reset_index(drop=True).copy()
    sw_high1, sw_high2, sw_low1, sw_low2 = np.nan, np.nan, np.nan, np.nan
    sw_high1_idx, sw_high2_idx, sw_low1_idx, sw_low2_idx = None, None, None, None

    break_up_list, break_down_list = [], []
    breakout_price_up_list, breakout_price_down_list = [], []
    line_val_up_list, line_val_down_list = [], []

    for i, row in h1_df.iterrows():
        break_up = False
        break_down = False
        line_val_up = np.nan
        line_val_down = np.nan

        if sw_high1_idx is not None and sw_high2_idx is not None:
            line_val_up = _line_value_at(sw_high2_idx, sw_high2, sw_high1_idx, sw_high1, i)
            break_up = row["high"] > line_val_up

        if sw_low1_idx is not None and sw_low2_idx is not None:
            line_val_down = _line_value_at(sw_low2_idx, sw_low2, sw_low1_idx, sw_low1, i)
            break_down = row["low"] < line_val_down

        break_up_list.append(bool(break_up))
        break_down_list.append(bool(break_down))
        breakout_price_up_list.append(line_val_up if break_up else np.nan)
        breakout_price_down_list.append(line_val_down if break_down else np.nan)
        line_val_up_list.append(line_val_up)
        line_val_down_list.append(line_val_down)

        # سوینگ‌پوینت‌های این کندل رو *بعد* از محاسبه‌ی بالا آپدیت می‌کنیم -
        # یعنی این کندل نمی‌تونه خودش رو به‌عنوان سوینگ ببینه (بدون نگاه به آینده).
        if row["pivot_high"]:
            sw_high2, sw_high2_idx = sw_high1, sw_high1_idx
            sw_high1, sw_high1_idx = row["high"], i
        if row["pivot_low"]:
            sw_low2, sw_low2_idx = sw_low1, sw_low1_idx
            sw_low1, sw_low1_idx = row["low"], i

    out = h1_df.copy()
    out["break_up"] = break_up_list
    out["break_down"] = break_down_list
    out["breakout_price_up"] = breakout_price_up_list
    out["breakout_price_down"] = breakout_price_down_list
    out["line_val_up"] = line_val_up_list
    out["line_val_down"] = line_val_down_list
    return out


def compute_reversal_triggers(h1_break_df):
    """
    تارگت دوم: مانیتورینگش به یک‌ساعته تغییر کرد - به‌جای چک‌کردن هر کندل ۱۵
    دقیقه، فقط یک‌بار در ساعت (لحظه‌ی بسته‌شدن هر کندل H1) چک می‌شه که آیا
    قیمت دوباره همون خط‌روند رو در جهت مخالف شکسته یا نه. خروجی: دیکشنری از
    check_time (لحظه‌ی بسته‌شدن کندل H1) -> {line_val_up, line_val_down}.
    """
    triggers = {}
    for _, row in h1_break_df.iterrows():
        check_time = row["timestamp"] + pd.Timedelta(hours=1)
        triggers[check_time] = {
            "line_val_up": row["line_val_up"],
            "line_val_down": row["line_val_down"],
        }
    return triggers


def compute_entry_triggers(h1_break_df, m15_df):
    """
    ورود دیگه منتظر بسته‌شدن کندل H1 نمی‌مونه. برای هر ساعت، خط‌روند از قبل
    (بر اساس سوینگ‌های H1 *قبل از* همین ساعت) مشخصه. کندل‌های ۱۵ دقیقه‌ای
    داخل همون ساعت (پول‌بک‌ها) یکی‌یکی به ترتیب زمان چک می‌شن - اولین کندل
    ۱۵ دقیقه‌ای که high/low ش از خط رد بشه، همون لحظه‌ی ورود می‌شه. قیمت
    ورود دقیقاً قیمت خودِ خط‌روند (نه قیمت کندل)، چون همون لحظه که قیمت به
    خط می‌رسه شکسته میشه - بدون نگاه به آینده (فقط از کندل‌های ۱۵ دقیقه‌ای
    که واقعاً قبل از این لحظه بسته شدن، برای تایید سوینگ‌ها استفاده می‌شه؛
    خودِ خط‌روند این ساعت از قبل، از سوینگ‌های H1 ثابت شده).
    خروجی: دیکشنری از trigger_time (تایم‌استمپ کندل ۱۵ دقیقه‌ای) -> {side, entry_price, breakout_price}
    """
    m15_df = m15_df.sort_values("timestamp").reset_index(drop=True)
    triggers = {}

    for _, h1_row in h1_break_df.iterrows():
        period_start = h1_row["timestamp"]
        period_end = period_start + pd.Timedelta(hours=1)
        line_up = h1_row["line_val_up"]
        line_down = h1_row["line_val_down"]

        window = m15_df[(m15_df["timestamp"] >= period_start) & (m15_df["timestamp"] < period_end)]
        for _, m15_row in window.iterrows():
            ts = m15_row["timestamp"]
            if not pd.isna(line_up) and m15_row["high"] >= line_up:
                triggers[ts] = {"side": "long", "entry_price": line_up, "breakout_price": line_up}
                break
            if not pd.isna(line_down) and m15_row["low"] <= line_down:
                triggers[ts] = {"side": "short", "entry_price": line_down, "breakout_price": line_down}
                break

    return triggers
    return triggers


def backtest(ltf_df, triggers, reversal_triggers, allowed_side):
    equity = INITIAL_CAPITAL
    position = None  # dict: side, entry, sl, tp1, tp3, qty1, qty2, half1_open, half2_open, mfe
    trades = []

    n = len(ltf_df)
    for i in range(max(PIVOT_LEN * 2, 20), n):
        row = ltf_df.iloc[i]
        high, low, close = row["high"], row["low"], row["close"]
        ts = row["timestamp"]

        # ---- مدیریت پوزیشن باز ----
        if position is not None:
            side = position["side"]

            # ردیابی بیشترین فاصله‌ای که قیمت به نفع معامله حرکت کرده (MFE)
            if side == "long":
                position["mfe"] = max(position["mfe"], high)
            else:
                position["mfe"] = min(position["mfe"], low)

            def close_remaining(exit_price, exit_reason, exit_ts):
                """هرچی از پوزیشن باز مونده (نیمه‌ی اول و/یا دوم) رو با یه قیمت می‌بنده."""
                nonlocal equity
                qty_remaining = 0.0
                if position["half1_open"]:
                    qty_remaining += position["qty1"]
                if position["half2_open"]:
                    qty_remaining += position["qty2"]
                if qty_remaining <= 0:
                    return
                if side == "long":
                    pnl = qty_remaining * (exit_price - position["entry"])
                    mfe_pct = (position["mfe"] - position["entry"]) / position["entry"] * 100
                else:
                    pnl = qty_remaining * (position["entry"] - exit_price)
                    mfe_pct = (position["entry"] - position["mfe"]) / position["entry"] * 100
                fee = (qty_remaining * position["entry"] + qty_remaining * exit_price) * (COMMISSION_PCT / 100)
                pnl -= fee
                equity += pnl
                trades.append({
                    "side": side, "entry_time": position["entry_time"], "entry": position["entry"],
                    "exit_time": exit_ts, "exit": exit_price, "reason": exit_reason,
                    "pnl": pnl, "equity_after": equity, "mfe_pct": mfe_pct,
                    "sl": position["sl"], "tp": position["tp1"],
                })
                position["half1_open"] = False
                position["half2_open"] = False

            # تارگت دوم: شکست معکوس خط‌روند - چک یک‌بار در ساعت. هرچی از
            # پوزیشن باز مونده (هر دو نیمه یا فقط نیمه‌ی دوم) رو می‌بنده.
            if ts in reversal_triggers:
                rev = reversal_triggers[ts]
                line_up = rev["line_val_up"]
                line_down = rev["line_val_down"]
                if side == "long" and not pd.isna(line_up) and close < line_up:
                    close_remaining(close, "trendline_reversal", ts)
                elif side == "short" and not pd.isna(line_down) and close > line_down:
                    close_remaining(close, "trendline_reversal", ts)

            # SL - هرچی از پوزیشن باز مونده رو می‌بنده
            if position is not None:
                if side == "long" and low <= position["sl"]:
                    close_remaining(position["sl"], "stop_loss", ts)
                elif side == "short" and high >= position["sl"]:
                    close_remaining(position["sl"], "stop_loss", ts)

            # TP1 - فقط نیمه‌ی اول
            if position is not None and position["half1_open"]:
                hit_tp1 = (side == "long" and high >= position["tp1"]) or (side == "short" and low <= position["tp1"])
                if hit_tp1:
                    qty1 = position["qty1"]
                    if side == "long":
                        pnl = qty1 * (position["tp1"] - position["entry"])
                        mfe_pct = (position["mfe"] - position["entry"]) / position["entry"] * 100
                    else:
                        pnl = qty1 * (position["entry"] - position["tp1"])
                        mfe_pct = (position["entry"] - position["mfe"]) / position["entry"] * 100
                    fee = (qty1 * position["entry"] + qty1 * position["tp1"]) * (COMMISSION_PCT / 100)
                    pnl -= fee
                    equity += pnl
                    trades.append({
                        "side": side, "entry_time": position["entry_time"], "entry": position["entry"],
                        "exit_time": ts, "exit": position["tp1"], "reason": "take_profit",
                        "pnl": pnl, "equity_after": equity, "mfe_pct": mfe_pct,
                        "sl": position["sl"], "tp": position["tp1"],
                    })
                    position["half1_open"] = False

            # TP3 - فقط نیمه‌ی دوم (تارگت۳، ضریب ATR بزرگ‌تر)
            if position is not None and position["half2_open"]:
                hit_tp3 = (side == "long" and high >= position["tp3"]) or (side == "short" and low <= position["tp3"])
                if hit_tp3:
                    qty2 = position["qty2"]
                    if side == "long":
                        pnl = qty2 * (position["tp3"] - position["entry"])
                        mfe_pct = (position["mfe"] - position["entry"]) / position["entry"] * 100
                    else:
                        pnl = qty2 * (position["entry"] - position["tp3"])
                        mfe_pct = (position["entry"] - position["mfe"]) / position["entry"] * 100
                    fee = (qty2 * position["entry"] + qty2 * position["tp3"]) * (COMMISSION_PCT / 100)
                    pnl -= fee
                    equity += pnl
                    trades.append({
                        "side": side, "entry_time": position["entry_time"], "entry": position["entry"],
                        "exit_time": ts, "exit": position["tp3"], "reason": "take_profit_tp3",
                        "pnl": pnl, "equity_after": equity, "mfe_pct": mfe_pct,
                        "sl": position["sl"], "tp": position["tp3"],
                    })
                    position["half2_open"] = False

            if position is not None and not position["half1_open"] and not position["half2_open"]:
                position = None

        if pd.isna(row.get("tp_atr", np.nan)) or pd.isna(row.get("adx", np.nan)):
            continue

        # ---- بررسی سیگنال ورود جدید - دقیقاً روی کندل تریگر (لحظه‌ی شکست H1) ----
        if position is None and ts in triggers:
            trig = triggers[ts]
            not_sideways = (not ADX_ENABLED) or (row["adx"] >= ADX_MIN)
            atr_val = row["tp_atr"]  # ATR از تایم‌فریم ATR_TF (۱ ساعته)، نه تایم ورود
            entry_price = trig["entry_price"]
            breakout_price = trig["breakout_price"]

            # سایز پوزیشن: مارجین ثابت × لوریج / قیمت ورود، تقسیم به دو نیمه‌ی مساوی
            qty_total = (POSITION_MARGIN * LEVERAGE) / entry_price
            qty_half = qty_total / 2

            if trig["side"] == "long" and allowed_side in ("long", "both") and not_sideways and not pd.isna(breakout_price):
                sl_price = breakout_price * (1 - SL_BREAKOUT_PCT / 100)
                if entry_price > sl_price:
                    position = {
                        "side": "long", "entry": entry_price, "sl": sl_price,
                        "tp1": entry_price + TP_ATR_MULT * atr_val,
                        "tp3": entry_price + TP3_ATR_MULT * atr_val,
                        "qty1": qty_half, "qty2": qty_half,
                        "half1_open": True, "half2_open": True,
                        "entry_time": ts, "mfe": entry_price,
                    }
            elif trig["side"] == "short" and allowed_side in ("short", "both") and not_sideways and not pd.isna(breakout_price):
                sl_price = breakout_price * (1 + SL_BREAKOUT_PCT / 100)
                if sl_price > entry_price:
                    position = {
                        "side": "short", "entry": entry_price, "sl": sl_price,
                        "tp1": entry_price - TP_ATR_MULT * atr_val,
                        "tp3": entry_price - TP3_ATR_MULT * atr_val,
                        "qty1": qty_half, "qty2": qty_half,
                        "half1_open": True, "half2_open": True,
                        "entry_time": ts, "mfe": entry_price,
                    }

    trades_df = pd.DataFrame(trades)
    return equity, trades_df


LTF_CANDIDATES = ["15m"]  # تایم مانیتورینگ/ورود - پول‌بک‌های ۱۵ دقیقه‌ای (برگشت برای مقایسه با ATR 15m + TP3)


def print_full_trades_table(trades_df, label):
    """
    جدول کامل همه‌ی معاملات - همه‌ی ردیف‌ها، همه‌ی ستون‌ها. برای همیشه بعد از هر
    بک‌تست چاپ می‌شه (طبق درخواست)، علاوه بر فایل CSV که همیشه هم ذخیره می‌شه.
    """
    print(f"\n\n===== جدول کامل معاملات: {label} ({len(trades_df)} معامله) =====")
    if len(trades_df) == 0:
        print("هیچ معامله‌ای ثبت نشد.")
        return
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                            "display.width", None, "display.float_format", "{:.4f}".format):
        print(trades_df.to_string())


def print_analysis_checklist(trades_df, label):
    """
    چک‌لیست استاندارد تحلیل بعد از هر بک‌تست:
    ۱) تعداد برد/باخت  ۲) تفکیک جهت  ۳) تفکیک سشن زمانی
    ۴) درصد حرکت قیمت بعد از برد/باخت  ۵) بررسی باخت‌های غیرعادی (بزرگ‌تر از SL)
    + یه بخش اضافه: مقایسه‌ی فاصله‌ی TP/SL و MFE (فضای باقی‌مونده بعد از TP)
    """
    print(f"\n\n===== چک‌لیست تحلیل: {label} =====")
    if len(trades_df) == 0:
        print("هیچ معامله‌ای ثبت نشد.")
        return

    df = trades_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["hour"] = df["entry_time"].dt.hour
    df["move_pct"] = (df["exit"] - df["entry"]) / df["entry"] * 100
    df.loc[df["side"] == "short", "move_pct"] = -df.loc[df["side"] == "short", "move_pct"]

    def session(h):
        if 0 <= h < 8:
            return "Asia"
        if 8 <= h < 16:
            return "London"
        return "NewYork"

    df["session"] = df["hour"].apply(session)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    print("---- 1) تعداد برد/باخت ----")
    print("برد:", len(wins), "| باخت:", len(losses), "| نرخ برد:", round(len(wins) / len(df) * 100, 2), "%")

    print("\n---- 2) تفکیک جهت ----")
    print("جهت - برد:")
    print(wins["side"].value_counts())
    print("جهت - باخت:")
    print(losses["side"].value_counts())

    print("\n---- 3) تفکیک سشن زمانی (UTC) ----")
    print("سشن - برد:")
    print(wins["session"].value_counts())
    print("سشن - باخت:")
    print(losses["session"].value_counts())

    print("\n---- 4) درصد حرکت قیمت بعد از برد/باخت ----")
    print("میانگین حرکت - برد:", round(wins["move_pct"].mean(), 3) if len(wins) else "-")
    print("میانگین حرکت - باخت:", round(losses["move_pct"].mean(), 3) if len(losses) else "-")
    print("بیشترین حرکت مثبت (برد):", round(wins["move_pct"].max(), 3) if len(wins) else "-")
    print("بیشترین حرکت منفی (باخت):", round(losses["move_pct"].min(), 3) if len(losses) else "-")

    print("\n---- 5) باخت‌های غیرعادی (بزرگ‌تر از ۱۰٪) ----")
    big = losses[losses["move_pct"].abs() > 10]
    print("تعداد:", len(big))
    if len(big):
        print(big[["side", "entry", "exit", "reason", "move_pct", "pnl"]])
    print("دلیل باخت‌ها:")
    print(losses["reason"].value_counts())

    print("\n---- گزارش تارگت‌ها: کدوم زودتر رسید؟ ----")
    print("تفکیک کل معاملات بر اساس دلیل خروج:")
    print(df["reason"].value_counts())
    print("درصد هرکدوم از کل:")
    print((df["reason"].value_counts(normalize=True) * 100).round(2))

    print("\n---- اضافه: فاصله‌ی TP/SL و فضای باقی‌مونده بعد از TP (MFE) ----")
    tp_hits = df[df["reason"] == "take_profit"]
    sl_hits = df[df["reason"] == "stop_loss"]
    if len(tp_hits):
        print("فاصله‌ی TP (میانگین):", round(tp_hits["move_pct"].mean(), 3), "%")
    if len(sl_hits):
        print("فاصله‌ی SL (میانگین):", round(sl_hits["move_pct"].abs().mean(), 3), "%")
    if len(tp_hits) and len(sl_hits) and sl_hits["move_pct"].abs().mean() != 0:
        print("نسبت TP به SL:", round(tp_hits["move_pct"].mean() / sl_hits["move_pct"].abs().mean(), 3))
    if "mfe_pct" in df.columns and len(tp_hits):
        room_after_tp = tp_hits["mfe_pct"] - tp_hits["move_pct"]
        print("فضای باقی‌مونده بعد از TP (فقط بردها - یعنی TP چقدر می‌تونست بزرگ‌تر باشه):")
        print(room_after_tp.describe())


def run_one_timeframe(exchange, symbol, ltf, h1_break_df, atr_df):
    print(f"\n--- تایم‌فریم ورود: {ltf} ({symbol}) ---")
    print(f"در حال دانلود دیتای {ltf} برای {symbol}...")
    ltf_df = fetch_ohlcv(exchange, symbol, ltf, MONTHS_BACK, end_date=OOS_END_DATE)
    print(f"⚠️ بازه‌ی واقعی دیتای دریافتی ({ltf}): از {ltf_df['timestamp'].min()} تا {ltf_df['timestamp'].max()} ({len(ltf_df)} کندل)")

    _, _, ltf_df["adx"] = dmi_adx(ltf_df, ADX_LEN)

    print(f"در حال محاسبه‌ی لحظه‌ی دقیق ورود (پول‌بک {ltf}، لحظه‌ی شکست خط H1)...")
    triggers = compute_entry_triggers(h1_break_df, ltf_df)
    print(f"تعداد تریگرهای ورود پیدا‌شده: {len(triggers)}")

    print(f"در حال هماهنگ‌سازی ATR (از {ATR_TF}) برای محاسبه‌ی TP...")
    ltf_df = merge_atr_to_ltf(ltf_df, atr_df)

    print("در حال محاسبه‌ی لحظه‌های چک تارگت دوم (یک‌بار در ساعت)...")
    reversal_triggers = compute_reversal_triggers(h1_break_df)

    print("در حال اجرای بک‌تست...")
    allowed_side = SYMBOL_DIRECTION.get(symbol, "both")
    final_equity, trades_df = backtest(ltf_df, triggers, reversal_triggers, allowed_side)

    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    safe_ltf = ltf.replace("/", "-")
    trades_df.to_csv(f"sync_scalper_trades_{safe_symbol}_{safe_ltf}.csv", index=False)
    ltf_df[["timestamp", "open", "high", "low", "close"]].to_csv(
        f"sync_scalper_prices_{safe_symbol}_{safe_ltf}.csv", index=False)

    print_full_trades_table(trades_df, f"{symbol} / {ltf}")
    print_analysis_checklist(trades_df, f"{symbol} / {ltf}")

    total_trades = len(trades_df)
    wins = (trades_df["pnl"] > 0).sum() if total_trades else 0
    win_rate = (wins / total_trades * 100) if total_trades else 0
    total_pnl = trades_df["pnl"].sum() if total_trades else 0

    return {
        "symbol": symbol,
        "ltf": ltf,
        "final_equity": final_equity,
        "pnl": total_pnl,
        "trades": total_trades,
        "win_rate": win_rate,
        "trades_df": trades_df,
    }


def run_one_symbol(exchange, symbol):
    print(f"\n\n########## نماد: {symbol} ##########")
    print(f"در حال دانلود دیتای {TREND_TF} (رسم خط‌روند) برای {symbol}...")
    h1_df = fetch_ohlcv(exchange, symbol, TREND_TF, MONTHS_BACK, end_date=OOS_END_DATE)
    print(f"⚠️ بازه‌ی واقعی دیتای دریافتی ({TREND_TF}): از {h1_df['timestamp'].min()} تا {h1_df['timestamp'].max()} ({len(h1_df)} کندل)")
    h1_df = add_htf_indicators(h1_df)

    print("در حال محاسبه‌ی شکست خط‌روند روی کندل‌های ۱ ساعته...")
    h1_break_df = compute_h1_break(h1_df)

    if ATR_TF == TREND_TF:
        atr_df = add_ltf_indicators(h1_df.copy())  # همون تایم‌فریم رسم خط‌روند، دوباره دانلود لازم نیست
    else:
        print(f"در حال دانلود دیتای {ATR_TF} (فقط برای محاسبه‌ی ATR) برای {symbol}...")
        atr_df = fetch_ohlcv(exchange, symbol, ATR_TF, MONTHS_BACK, end_date=OOS_END_DATE)
        print(f"⚠️ بازه‌ی واقعی دیتای دریافتی ({ATR_TF}): از {atr_df['timestamp'].min()} تا {atr_df['timestamp'].max()} ({len(atr_df)} کندل)")
        atr_df = add_ltf_indicators(atr_df)

    results = []
    for ltf in LTF_CANDIDATES:
        res = run_one_timeframe(exchange, symbol, ltf, h1_break_df, atr_df)
        results.append(res)
    return results


def print_combined_report(all_results, shared_capital=None):
    """
    چک‌لیست ترکیبی - همه‌ی نمادها با هم، به ترتیب زمانی واقعی:
    ۱) سرمایه‌ی مشترک (اگه همه‌ی نمادها از یه حساب واحد استفاده می‌کردن)
    ۲) بیشترین افت سرمایه (Max Drawdown)
    ۳) تعداد معامله در روز
    ۴) طول و مبلغ توالی‌های برد/باخت متوالی
    """
    if shared_capital is None:
        shared_capital = INITIAL_CAPITAL

    frames = []
    for r in all_results:
        df = r["trades_df"].copy()
        if len(df) == 0:
            continue
        df["symbol"] = r["symbol"]
        frames.append(df)

    if not frames:
        print("\n\n===== گزارش ترکیبی: هیچ معامله‌ای ثبت نشد =====")
        return

    both = pd.concat(frames, ignore_index=True)
    both["entry_time"] = pd.to_datetime(both["entry_time"])
    both["exit_time"] = pd.to_datetime(both["exit_time"])
    both = both.sort_values("exit_time").reset_index(drop=True)

    print(f"\n\n===== گزارش ترکیبی: همه‌ی نمادها با سرمایه‌ی مشترک ${shared_capital:.2f} =====")

    # ۱) سرمایه‌ی مشترک
    equity = shared_capital
    equity_curve = []
    for pnl in both["pnl"]:
        equity += pnl
        equity_curve.append(equity)
    both["shared_equity_after"] = equity_curve

    print(f"\n---- ۱) سرمایه‌ی مشترک ----")
    print(f"سرمایه‌ی نهایی (بعد از {len(both)} معامله‌ی همه‌ی نمادها با هم): ${equity:.2f}")
    print(f"سود/زیان کل: ${equity - shared_capital:.2f}")
    print(f"بازده: {(equity - shared_capital) / shared_capital * 100:.2f}%")
    print("تفکیک سهم هر نماد از سود/زیان کل:")
    print(both.groupby("symbol")["pnl"].sum())
    print(f"کمترین سرمایه‌ای که در طول مسیر لمس شده: ${min(equity_curve):.2f}")

    # ۲) بیشترین افت سرمایه (Max Drawdown) - از قله تا کف بعدش، نه فقط نسبت به سرمایه‌ی اولیه
    equity_series = pd.Series([shared_capital] + equity_curve)
    running_max = equity_series.cummax()
    drawdown = equity_series - running_max
    drawdown_pct = (drawdown / running_max) * 100
    max_dd_dollar = drawdown.min()
    max_dd_pct = drawdown_pct.min()
    max_dd_idx = drawdown.idxmin()
    peak_before_dd = running_max.iloc[max_dd_idx]

    print(f"\n---- ۲) بیشترین افت سرمایه (Max Drawdown) ----")
    print(f"مبلغ: ${max_dd_dollar:.2f} | درصد: {max_dd_pct:.2f}%")
    print(f"از قله‌ی ${peak_before_dd:.2f} تا کف ${peak_before_dd + max_dd_dollar:.2f}")
    print(f"بالاترین سرمایه‌ای که کل مسیر بهش رسیده: ${equity_series.max():.2f}")

    # ۳) تعداد معامله در روز
    both["date"] = both["entry_time"].dt.date
    daily_counts = both.groupby("date").size()
    total_days = (both["entry_time"].max() - both["entry_time"].min()).days or 1

    print(f"\n---- ۳) تعداد معامله در روز ----")
    print(f"روزهای فعال: {len(daily_counts)} از {total_days} روز تقویمی")
    print(f"میانگین در روز فعال: {daily_counts.mean():.2f} | میانه: {daily_counts.median():.1f}")
    print(f"بیشترین در یک روز: {daily_counts.max()} | کمترین در روز فعال: {daily_counts.min()}")
    print(f"میانگین روی کل بازه (شامل روزهای بی‌معامله): {len(both) / total_days:.2f}")

    # ۴) توالی‌های برد/باخت
    both["result"] = both["pnl"].apply(lambda x: "win" if x > 0 else "loss")
    streaks = []
    current_result, current_len, current_pnl = None, 0, 0.0
    for pnl, r in zip(both["pnl"], both["result"]):
        if r == current_result:
            current_len += 1
            current_pnl += pnl
        else:
            if current_result is not None:
                streaks.append((current_result, current_len, current_pnl))
            current_result, current_len, current_pnl = r, 1, pnl
    streaks.append((current_result, current_len, current_pnl))

    win_streaks = [(length, pnl) for res, length, pnl in streaks if res == "win"]
    loss_streaks = [(length, pnl) for res, length, pnl in streaks if res == "loss"]

    print(f"\n---- ۴) توالی‌های برد/باخت متوالی ----")
    if win_streaks:
        best_win = max(win_streaks, key=lambda x: x[1])
        longest_win = max(win_streaks, key=lambda x: x[0])
        print(f"بیشترین طول برد متوالی: {longest_win[0]} معامله")
        print(f"بهترین توالی برد (بیشترین سود): {best_win[0]} معامله -> ${best_win[1]:.2f}")
        print(f"میانگین طول توالی برد: {sum(l for l, _ in win_streaks)/len(win_streaks):.2f}")
        print(f"میانگین سود هر توالی برد: ${sum(p for _, p in win_streaks)/len(win_streaks):.2f}")
    if loss_streaks:
        worst_loss = min(loss_streaks, key=lambda x: x[1])
        longest_loss = max(loss_streaks, key=lambda x: x[0])
        print(f"بیشترین طول باخت متوالی: {longest_loss[0]} معامله")
        print(f"بدترین توالی باخت (بیشترین ضرر): {worst_loss[0]} معامله -> ${worst_loss[1]:.2f}")
        print(f"میانگین طول توالی باخت: {sum(l for l, _ in loss_streaks)/len(loss_streaks):.2f}")
        print(f"میانگین ضرر هر توالی باخت: ${sum(p for _, p in loss_streaks)/len(loss_streaks):.2f}")


def main():
    exchange = ccxt.okx({"enableRateLimit": True})
    print(f"=== تست Out-of-Sample ===")
    print(f"بازه‌ی درخواستی: ۶ ماه منتهی به {OOS_END_DATE}")

    all_results = []
    for symbol in SYMBOLS:
        all_results += run_one_symbol(exchange, symbol)

    print("\n\n=== مقایسه‌ی نمادها/تایم‌فریم‌های ورود (فقط فیلتر شکست خط‌روند) ===")
    print(f"بازه: {MONTHS_BACK} ماه | سرمایه اولیه: ${INITIAL_CAPITAL}")
    print(f"{'نماد':<16}{'تایم‌فریم':<10}{'سرمایه نهایی':<15}{'سود/زیان':<12}{'معاملات':<10}{'نرخ برد':<10}")
    for r in all_results:
        print(f"{r['symbol']:<16}{r['ltf']:<10}${r['final_equity']:<14.2f}${r['pnl']:<11.2f}{r['trades']:<10}{r['win_rate']:<9.2f}%")

    summary_df = pd.DataFrame([{
        "symbol": r["symbol"], "timeframe": r["ltf"], "final_equity": r["final_equity"], "pnl": r["pnl"],
        "trades": r["trades"], "win_rate": r["win_rate"],
    } for r in all_results])
    summary_df.to_csv("sync_scalper_symbol_comparison.csv", index=False)
    print("\nفایل مقایسه ذخیره شد: sync_scalper_symbol_comparison.csv")
    print("فایل معاملات هر نماد/تایم‌فریم هم جدا ذخیره شد: sync_scalper_trades_<نماد>_<تایم‌فریم>.csv")

    print_combined_report(all_results, shared_capital=INITIAL_CAPITAL)


if __name__ == "__main__":
    main()
