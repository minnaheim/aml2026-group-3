import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
root = Path(__file__).parent.parent.parent / "out" / "holdout"
horizons = [3, 6, 12]
targets = ["CPI", "GDP", "UNRATE"]

rows = []
for h in horizons:
    macro = pd.read_csv(root / "final_holdout" / f"metrics_h{h}_macro.csv")
    emb   = pd.read_csv(root / "final_emb"     / f"metrics_h{h}_auto.csv")
    
    macro_tft = macro[macro["model"] == "TFT"].set_index("target")
    emb_tft   = emb[emb["model"] == "TFT"].set_index("target")
    
    for target in targets:
        rmse_macro = macro_tft.loc[target, "RMSE"]
        rmse_emb   = emb_tft.loc[target, "RMSE"]
        rows.append({
            "horizon": h,
            "target":  target,
            "RMSE_macro": round(rmse_macro, 5),
            "RMSE_emb":   round(rmse_emb, 5),
            "relRMSE":    round(rmse_emb / rmse_macro, 4),
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("out/holdout/relRMSE.csv", index=False)
print("\nSaved to out/holdout/relRMSE.csv")