import pandas as pd

xrp = pd.read_csv("sync_scalper_trades_XRP-USDT-USDT_15m.csv")
eth = pd.read_csv("sync_scalper_trades_ETH-USDT-USDT_15m.csv")
xrp["symbol"] = "XRP"
eth["symbol"] = "ETH"

both = pd.concat([xrp, eth], ignore_index=True)
t2 = both[both["reason"] == "trendline_reversal"]

print(f"تعداد معاملاتی که با تارگت۲ بسته شدن: {len(t2)}")
print(f"مجموع سود/زیان تارگت۲: ${t2['pnl'].sum():.2f}")
print()
wins = t2[t2["pnl"] > 0]
losses = t2[t2["pnl"] <= 0]
print(f"برد: {len(wins)} -> ${wins['pnl'].sum():.2f}")
print(f"باخت: {len(losses)} -> ${losses['pnl'].sum():.2f}")
print()
print("تفکیک بر اساس نماد:")
print(t2.groupby("symbol")["pnl"].agg(["count", "sum"]))
