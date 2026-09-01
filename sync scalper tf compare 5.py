# -*- coding: utf-8 -*-
"""
Sync Scalper (M15 بایاس + H1 ترندلاین + ورود ۵m/۱۵m) - فقط شورت
------------------------------------------------------------------
تغییرات نسبت به نسخه‌ی قبلی:
- فیلتر پرایس‌اکشن (Engulfing) حذف شد
- فیلتر شکست خط‌روند برگشت، ولی این‌بار روی تایم‌فریم ۱ ساعته (H1) مستقل -
  بایاس M15 (EMA + ساختار سوینگ) دست‌نخورده مونده، فقط شکست خط‌روند اضافه شده
- SL دیگه از سوینگ‌پوینت M5 نیست - نسبت به قیمتی که شکست خط‌روند H1 در آن رخ
  داده، با فاصله‌ی ثابت SL_BREAKOUT_PCT (پیش‌فرض ۱٪) محاسبه میشه
- مقایسه‌ی دو تایم ورود: 5m و 15m (بایاس M15 و ترندلاین H1 برای هر دو ثابت)
- فقط شورت فعاله (لانگ غیرفعال، طبق نتیجه‌ی قبلی)
- فیلترهای ATR (فاصله از EMA، pullback) فعال و دست‌نخورده موندن

نحوه اجرا (روی Replit/Colab):
    pip install ccxt pandas numpy
    python sync_scalper_tf_compare.py
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ------------------ تنظیمات (دقیقاً مطابق ورودی‌های Pine Script) ------------------
SYMBOL = "XRP/USDT:USDT"  # فرمت یکپارچه‌ی ccxt برای فیوچرز پرپچوال OKX
HTF = "15m"     # Higher Timeframe (M15) - بایاس روند (EMA + ساختار سوینگ)
TREND_TF = "1h"  # تایم‌فریم مستقل برای فیلتر شکست خط‌روند
MONTHS_BACK = 6

# تست Out-of-Sample: به‌جای «۶ ماه اخیر»، «۶ ماه قبل‌تر از ۶ ماه اخیر» رو می‌گیریم -
# یعنی داده‌ای که موقع طراحی این ترکیب (H1 ترندلاین + SL ۱٪) اصلاً دیده نشده بود.
# برای رفتن به حالت عادی (۶ ماه اخیر)، این رو None کن.
OOS_END_DATE = (datetime.utcnow() - timedelta(days=30 * MONTHS_BACK)).strftime("%Y-%m-%dT%H:%M:%SZ")

INITIAL_CAPITAL = 10000.0
COMMISSION_PCT = 0.05  # درصد کارمزد هر طرف معامله

RISK_PCT = 0.75          # Risk % per trade
RR_MIN = 2.0              # Minimum Risk:Reward
MAX_SIGNALS_PER_DAY = 1

ALLOW_LONG = True    # هر دو جهت آزاد شد
ALLOW_SHORT = True

SL_BREAKOUT_PCT = 1.0  # SL نسبت به قیمت شکست خط‌روند (H1)، نه سوینگ‌پوینت - ۱٪ فاصله

EMA_LEN = 50
EMA_SLOPE_LOOKBACK = 3

RSI_LEN = 14
RSI_BUY_MIN = 50           # میونه شد (شل=45، سخت=55) - مومنتوم مثبت خفیف کافیه
RSI_SELL_MAX = 50          # میونه شد (شل=55، سخت=45)

PIVOT_LEN = 5             # Pivot Left/Right bars (ساختار M15 و شکست خط‌روند H1)

ADX_LEN = 14
ADX_MIN = 21               # میونه شد (شل=18، سخت=25)
MAX_ATR_DIST = 2.5        # حداکثر فاصله از EMA50 (ضریب ATR) - فعال، فیلتر "خیلی دور نشده"
PULLBACK_ATR_MAX = 0.8    # میونه شد (شل=1.0، سخت=0.6) - فعال

MIN_BODY_RATIO = 0.5      # میونه شد (شل=0.4، سخت=0.6)


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
    df["ema"] = ema(df["close"], EMA_LEN)
    df["pivot_high"], df["pivot_low"] = pivot_points(df, PIVOT_LEN, PIVOT_LEN)
    return df


def add_ltf_indicators(df):
    df["ema50"] = ema(df["close"], EMA_LEN)
    df["rsi14"] = rsi(df["close"], RSI_LEN)
    df["atr5"] = atr(df, 14)
    _, _, df["adx"] = dmi_adx(df, ADX_LEN)
    df["pivot_high"], df["pivot_low"] = pivot_points(df, PIVOT_LEN, PIVOT_LEN)
    return df


def merge_htf_bias(ltf_df, htf_df):
    """
    برای هر کندل M5، آخرین کندل M15 که *قبل از* بسته‌شدنش تمام شده را پیدا می‌کند
    (بدون نگاه به آینده - معادل request.security با lookahead_off).
    """
    htf_df = htf_df.set_index("timestamp").sort_index()
    merged_ema = []
    merged_close = []
    sw_high1, sw_high2, sw_low1, sw_low2 = np.nan, np.nan, np.nan, np.nan
    htf_ptr = -1
    htf_times = htf_df.index.values
    htf_records = htf_df.to_dict("records")

    bull_bias_list, bear_bias_list = [], []

    for ts in ltf_df["timestamp"]:
        # پیشروی نشانگر M15 تا آخرین کندلی که کاملاً قبل از این کندل M5 بسته شده
        while htf_ptr + 1 < len(htf_times) and htf_times[htf_ptr + 1] < np.datetime64(ts):
            htf_ptr += 1
            rec = htf_records[htf_ptr]
            if rec["pivot_high"]:
                sw_high2 = sw_high1
                sw_high1 = rec["high"]
            if rec["pivot_low"]:
                sw_low2 = sw_low1
                sw_low1 = rec["low"]

        if htf_ptr < EMA_SLOPE_LOOKBACK or htf_ptr < 0:
            merged_ema.append(np.nan)
            merged_close.append(np.nan)
            bull_bias_list.append(False)
            bear_bias_list.append(False)
            continue

        cur = htf_records[htf_ptr]
        prev_ema = htf_records[htf_ptr - EMA_SLOPE_LOOKBACK]["ema"]
        htf_ema_val = cur["ema"]
        htf_close_val = cur["close"]

        bull_structure = (not np.isnan(sw_high1) and not np.isnan(sw_high2)
                           and not np.isnan(sw_low1) and not np.isnan(sw_low2)
                           and sw_high1 > sw_high2 and sw_low1 > sw_low2)
        bear_structure = (not np.isnan(sw_high1) and not np.isnan(sw_high2)
                           and not np.isnan(sw_low1) and not np.isnan(sw_low2)
                           and sw_high1 < sw_high2 and sw_low1 < sw_low2)

        price_above = htf_close_val > htf_ema_val
        price_below = htf_close_val < htf_ema_val
        slope_up = htf_ema_val > prev_ema
        slope_down = htf_ema_val < prev_ema

        # برگردونده شد به OR (نسخه‌ی اصلی) - نسخه‌ی سخت‌گیرانه هم قیمت هم ساختار
        # سوینگ رو هم‌زمان می‌خواست که خیلی به‌ندرت با هم جور می‌شدن (علت اصلی افت
        # از ۱۶۷ به ۳ معامله). الان کافیه یکی از این دو (قیمت یا ساختار) به‌علاوه‌ی
        # شیب EMA تایید بشه - هنوز سخت‌گیرتر از نسخه‌ی خیلی شل، چون بقیه‌ی فیلترها
        # (RSI/ADX/pullback/body ratio) میونه موندن. فیلتر شکست خط‌روند حذف شد.
        bull_bias = (price_above or bull_structure) and slope_up
        bear_bias = (price_below or bear_structure) and slope_down

        merged_ema.append(htf_ema_val)
        merged_close.append(htf_close_val)
        bull_bias_list.append(bool(bull_bias))
        bear_bias_list.append(bool(bear_bias))

    ltf_df = ltf_df.copy()
    ltf_df["htf_bull_bias"] = bull_bias_list
    ltf_df["htf_bear_bias"] = bear_bias_list
    return ltf_df


def _line_value_at(x1, y1, x2, y2, x):
    """معادله‌ی خط بین دو نقطه‌ی سوینگ، مقدارش را در ایندکس x برمی‌گرداند."""
    if x2 == x1:
        return y2
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (x - x1)


def merge_trendline_h1(ltf_df, h1_df):
    """
    فیلتر مستقل شکست خط‌روند، روی تایم‌فریم ۱ ساعته (H1) - جدا از بایاس M15.
    خط نزولی از دو سقف سوینگ H1 آخر (برای شکست به بالا/لانگ)، خط صعودی از دو کف
    سوینگ H1 آخر (برای شکست به پایین/شورت). برای هر کندل ورود، آخرین کندل H1ای که
    *قبل از* بسته‌شدنش تمام شده استفاده می‌شود (بدون نگاه به آینده).

    خروجی، علاوه بر trendline_break_up/down، قیمت H1ای که شکست در آن رخ داده
    (breakout_price_up/down) را هم برمی‌گرداند - برای محاسبه‌ی SL بر پایه‌ی همین
    قیمت (به‌جای سوینگ‌پوینت خام).
    """
    h1_df = h1_df.set_index("timestamp").sort_index()
    sw_high1, sw_high2, sw_low1, sw_low2 = np.nan, np.nan, np.nan, np.nan
    sw_high1_idx, sw_high2_idx, sw_low1_idx, sw_low2_idx = None, None, None, None
    h1_ptr = -1
    h1_times = h1_df.index.values
    h1_records = h1_df.to_dict("records")

    break_up_list, break_down_list = [], []
    breakout_price_up_list, breakout_price_down_list = [], []

    for ts in ltf_df["timestamp"]:
        while h1_ptr + 1 < len(h1_times) and h1_times[h1_ptr + 1] < np.datetime64(ts):
            h1_ptr += 1
            rec = h1_records[h1_ptr]
            if rec["pivot_high"]:
                sw_high2, sw_high2_idx = sw_high1, sw_high1_idx
                sw_high1, sw_high1_idx = rec["high"], h1_ptr
            if rec["pivot_low"]:
                sw_low2, sw_low2_idx = sw_low1, sw_low1_idx
                sw_low1, sw_low1_idx = rec["low"], h1_ptr

        if h1_ptr < 0:
            break_up_list.append(True)
            break_down_list.append(True)
            breakout_price_up_list.append(np.nan)
            breakout_price_down_list.append(np.nan)
            continue

        h1_close_val = h1_records[h1_ptr]["close"]

        if sw_high1_idx is not None and sw_high2_idx is not None:
            break_up = h1_close_val > _line_value_at(
                sw_high2_idx, sw_high2, sw_high1_idx, sw_high1, h1_ptr)
        else:
            break_up = True  # داده‌ی کافی نیست - سخت‌گیر نباش

        if sw_low1_idx is not None and sw_low2_idx is not None:
            break_down = h1_close_val < _line_value_at(
                sw_low2_idx, sw_low2, sw_low1_idx, sw_low1, h1_ptr)
        else:
            break_down = True

        break_up_list.append(bool(break_up))
        break_down_list.append(bool(break_down))
        breakout_price_up_list.append(h1_close_val)
        breakout_price_down_list.append(h1_close_val)

    ltf_df = ltf_df.copy()
    ltf_df["h1_break_up"] = break_up_list
    ltf_df["h1_break_down"] = break_down_list
    ltf_df["h1_breakout_price_up"] = breakout_price_up_list
    ltf_df["h1_breakout_price_down"] = breakout_price_down_list
    return ltf_df


def backtest(ltf_df):
    equity = INITIAL_CAPITAL
    position = None  # dict: side, entry, sl, tp, qty
    trades = []

    signals_today = 0
    last_day = None

    n = len(ltf_df)
    for i in range(max(EMA_LEN, PIVOT_LEN * 2, ADX_LEN * 3), n):
        row = ltf_df.iloc[i]

        day = row["timestamp"].day
        if last_day is None or day != last_day:
            signals_today = 0
            last_day = day

        high, low, close, open_ = row["high"], row["low"], row["close"], row["open"]

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

        if pd.isna(row["adx"]) or pd.isna(row["atr5"]) or pd.isna(row["ema50"]) or pd.isna(row["rsi14"]):
            continue

        # ---- بررسی سیگنال ورود جدید ----
        if position is None and signals_today < MAX_SIGNALS_PER_DAY:
            dist_from_ema = abs(close - row["ema50"])
            not_too_far = dist_from_ema <= MAX_ATR_DIST * row["atr5"]
            pullback_ok = dist_from_ema <= PULLBACK_ATR_MAX * row["atr5"]
            not_sideways = row["adx"] >= ADX_MIN

            candle_range = (high - low) if (high - low) > 0 else 1e-9
            body_ratio = abs(close - open_) / candle_range
            bull_candle = close > open_ and body_ratio >= MIN_BODY_RATIO
            bear_candle = close < open_ and body_ratio >= MIN_BODY_RATIO

            # فیلتر مستقل شکست خط‌روند H1 - علاوه بر بایاس M15 لازمه
            h1_break_up = row.get("h1_break_up", True)
            h1_break_down = row.get("h1_break_down", True)

            buy_cond = (ALLOW_LONG and row["htf_bull_bias"] and h1_break_up
                        and close > row["ema50"] and pullback_ok and not_too_far
                        and not_sideways and bull_candle and row["rsi14"] > RSI_BUY_MIN)
            sell_cond = (ALLOW_SHORT and row["htf_bear_bias"] and h1_break_down
                         and close < row["ema50"] and pullback_ok and not_too_far
                         and not_sideways and bear_candle and row["rsi14"] < RSI_SELL_MAX)

            # SL دیگه از سوینگ‌پوینت خام نیست - نسبت به قیمتی که شکست خط‌روند H1
            # در آن رخ داده، به فاصله‌ی ثابت SL_BREAKOUT_PCT (پیش‌فرض ۱٪) محاسبه میشه.
            breakout_price_up = row.get("h1_breakout_price_up", np.nan)
            breakout_price_down = row.get("h1_breakout_price_down", np.nan)

            if buy_cond and not pd.isna(breakout_price_up):
                sl_price = breakout_price_up * (1 - SL_BREAKOUT_PCT / 100)
                risk = close - sl_price
                if risk > 0:
                    risk_amount = equity * (RISK_PCT / 100)
                    qty = risk_amount / risk
                    position = {
                        "side": "long", "entry": close, "sl": sl_price,
                        "tp": close + risk * RR_MIN, "qty": qty, "entry_time": row["timestamp"],
                    }
                    signals_today += 1
            elif sell_cond and not pd.isna(breakout_price_down):
                sl_price = breakout_price_down * (1 + SL_BREAKOUT_PCT / 100)
                risk = sl_price - close
                if risk > 0:
                    risk_amount = equity * (RISK_PCT / 100)
                    qty = risk_amount / risk
                    position = {
                        "side": "short", "entry": close, "sl": sl_price,
                        "tp": close - risk * RR_MIN, "qty": qty, "entry_time": row["timestamp"],
                    }
                    signals_today += 1

    trades_df = pd.DataFrame(trades)
    return equity, trades_df


LTF_CANDIDATES = ["5m", "15m"]  # مقایسه‌ی دو تایم ورود، طبق درخواست


def run_one_timeframe(exchange, ltf, htf_df, h1_df):
    print(f"\n--- تایم‌فریم ورود: {ltf} ---")
    print(f"در حال دانلود دیتای {ltf} برای {SYMBOL}...")
    ltf_df = fetch_ohlcv(exchange, SYMBOL, ltf, MONTHS_BACK, end_date=OOS_END_DATE)
    print(f"⚠️ بازه‌ی واقعی دیتای دریافتی ({ltf}): از {ltf_df['timestamp'].min()} تا {ltf_df['timestamp'].max()} ({len(ltf_df)} کندل)")
    ltf_df = add_ltf_indicators(ltf_df)

    print("در حال هماهنگ‌سازی بایاس M15 با این تایم‌فریم (بدون نگاه به آینده)...")
    ltf_df = merge_htf_bias(ltf_df, htf_df)

    print("در حال هماهنگ‌سازی فیلتر شکست خط‌روند H1 با این تایم‌فریم...")
    ltf_df = merge_trendline_h1(ltf_df, h1_df)

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
    print(f"در حال دانلود دیتای {HTF} (بایاس روند) برای {SYMBOL}...")
    htf_df = fetch_ohlcv(exchange, SYMBOL, HTF, MONTHS_BACK, end_date=OOS_END_DATE)
    print(f"⚠️ بازه‌ی واقعی دیتای دریافتی ({HTF}): از {htf_df['timestamp'].min()} تا {htf_df['timestamp'].max()} ({len(htf_df)} کندل)")
    htf_df = add_htf_indicators(htf_df)

    print(f"در حال دانلود دیتای {TREND_TF} (فیلتر شکست خط‌روند) برای {SYMBOL}...")
    h1_df = fetch_ohlcv(exchange, SYMBOL, TREND_TF, MONTHS_BACK, end_date=OOS_END_DATE)
    print(f"⚠️ بازه‌ی واقعی دیتای دریافتی ({TREND_TF}): از {h1_df['timestamp'].min()} تا {h1_df['timestamp'].max()} ({len(h1_df)} کندل)")
    h1_df = add_htf_indicators(h1_df)  # فقط pivot_high/pivot_low لازمه، همون تابع کافیه

    results = []
    for ltf in LTF_CANDIDATES:
        res = run_one_timeframe(exchange, ltf, htf_df, h1_df)
        results.append(res)

    print("\n\n=== مقایسه‌ی تایم‌فریم‌های ورود (M15 بایاس ثابت) ===")
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
