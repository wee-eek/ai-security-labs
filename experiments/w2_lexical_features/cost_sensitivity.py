import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# ========== 路径定位 ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ========== 特征提取 ==========
def extract_features(url):
    try:
        num_subdomains = urlparse(url).netloc.count(".")
    except Exception:
        num_subdomains = 0
    return {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "has_at": "@" in url,
        "has_ip": bool(re.search(r'\d+\.\d+\.\d+\.\d+', url)),
        "num_subdomains": num_subdomains,
        "has_suspicious_word": any(w in url.lower() for w in
                                   ["login", "verify", "secure", "account", "update", "signin"]),
    }

# ========== 读数据 + 提特征 ==========
print("读取数据 + 提特征中...")
df = pd.read_csv(DATA_DIR / "URL dataset.csv")
feat_df = df["url"].apply(extract_features).apply(pd.Series)
feat_df["label"] = (df["type"] == "phishing").astype(int)

X = feat_df.drop(columns=["label"])
y = feat_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("训练随机森林...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
proba = rf.predict_proba(X_test)[:, 1]

# ========== 先扫一遍阈值，缓存 FP/FN ==========
thresholds = np.arange(0.05, 0.96, 0.05)
scan = []
for t in thresholds:
    pred = (proba >= t).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    scan.append({"threshold": round(t, 2), "FP": fp, "FN": fn})

scan_df = pd.DataFrame(scan)

# ========== 遍历成本比 ==========
C_FP = 5
ratios = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

print(f"\n固定 C_FP = {C_FP} 元，遍历成本比 C_FN / C_FP：")
print("-" * 50)

results = []
for ratio in ratios:
    C_FN = C_FP * ratio
    costs = scan_df["FN"] * C_FN + scan_df["FP"] * C_FP
    best_idx = costs.idxmin()
    best_t = scan_df.loc[best_idx, "threshold"]
    best_fn = scan_df.loc[best_idx, "FN"]
    best_fp = scan_df.loc[best_idx, "FP"]
    results.append({
        "ratio": ratio,
        "C_FN": C_FN,
        "best_threshold": best_t,
        "FP": best_fp,
        "FN": best_fn,
        "min_cost": int(costs.min()),
    })
    print(f"  比值 {ratio:>4}:1  →  最优阈值 {best_t:.2f}  "
          f"(FP={best_fp}, FN={best_fn}, 总成本={int(costs.min())}元)")

results_df = pd.DataFrame(results)

# ========== 画图 ==========
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(results_df["ratio"], results_df["best_threshold"],
        marker='o', color='tab:red', linewidth=2.2, markersize=8)
ax.set_xscale('log')
ax.set_xlabel('Cost Ratio  C_FN / C_FP  (log scale)', fontsize=11)
ax.set_ylabel('Optimal Threshold', fontsize=11)
ax.set_title('Sensitivity Analysis: Optimal Threshold vs Cost Ratio', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

for x, y in zip(results_df["ratio"], results_df["best_threshold"]):
    ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=9)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "cost_sensitivity.png", dpi=150)
print(f"\n图已保存: {RESULTS_DIR / 'cost_sensitivity.png'}")

results_df.to_csv(RESULTS_DIR / "cost_sensitivity.csv",
                  index=False, encoding="utf-8-sig")
print(f"表已保存: {RESULTS_DIR / 'cost_sensitivity.csv'}")