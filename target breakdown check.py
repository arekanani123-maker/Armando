import pandas as pd

for tag in ["XRP-USDT-USDT", "ETH-USDT-USDT"]:
    df = pd.read_csv(f"sync_scalper_trades_{tag}_15m.csv")
    print(f"=== {tag} ===")
    counts = df["reason"].value_counts()
    print(f"تارگت ۱ (TP اصلی): {counts.get('take_profit', 0)}")
    print(f"تارگت ۲ (شکست معکوس): {counts.get('trendline_reversal', 0)}")
    print(f"تارگت ۳: {counts.get('take_profit_tp3', 0)}")
    print(f"استاپ‌لاس: {counts.get('stop_loss', 0)}")
    print(f"مجموع: {len(df)}")
    print()

# جمع کل هر دو نماد با هم
xrp = pd.read_csv("sync_scalper_trades_XRP-USDT-USDT_15m.csv")
eth = pd.read_csv("sync_scalper_trades_ETH-USDT-USDT_15m.csv")
both = pd.concat([xrp, eth], ignore_index=True)
counts = both["reason"].value_counts()
print("=== جمع کل (هر دو نماد) ===")
print(f"تارگت ۱ (TP اصلی): {counts.get('take_profit', 0)}")
print(f"تارگت ۲ (شکست معکوس): {counts.get('trendline_reversal', 0)}")
print(f"تارگت ۳: {counts.get('take_profit_tp3', 0)}")
print(f"استاپ‌لاس: {counts.get('stop_loss', 0)}")
print(f"مجموع: {len(both)}")
