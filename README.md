SYMBOLS = ["ADA/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT"]

SYMBOL_DIRECTION = {
    "ADA/USDT:USDT": "both",
    "SOL/USDT:USDT": "both",
    "DOGE/USDT:USDT": "both",
}

OOS_END_DATE = "2026-09-01T00:00:00Z"  # پایان بازه - انتخابی
MONTHS_BACK = 7  # تقریبی، از تاریخ شروع تا پایان (2026-01-20 تا 2026-09-01)

INITIAL_CAPITAL = 10000.0
POSITION_MARGIN = 500.0
LEVERAGE = 3.0

# ---- بخش‌های زیر قفل‌ان و نباید تغییر کنن ----
# TREND_TF = "30m"
# ATR_TF = "15m"
# LTF_CANDIDATES = ["5m"]
# SL_BREAKOUT_PCT = 3.0
# TP_ATR_MULT = 3.04
# TP3_ATR_MULT = 5.04
# ADX_LEN = 14 | ADX_MIN = 21 | ADX_ENABLED = True
