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

# ========== 特征提取函数 ==========
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

# ========== 训练模型 ==========
print("训练随机森林...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

proba = rf.predict_proba(X_test)[:, 1]

# ========== 成本参数 ==========
C_FN = 1000   # 一次漏报成本（元）
C_FP = 5      # 一次误报成本（元）

print(f"\n成本假设：C_FN = {C_FN} 元，C_FP = {C_FP} 元")

# ========== 扫描阈值 ==========
thresholds = np.arange(0.05, 0.96, 0.05)
records = []
for t in thresholds:
    pred = (proba >= t).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total_cost = fn * C_FN + fp * C_FP
    records.append({
        "threshold": round(t, 2),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "cost": total_cost,
    })

result_df = pd.DataFrame(records)
print("\n阈值扫描 + 成本计算：")
print(result_df.to_string(index=False))

# ========== 找成本最优阈值 ==========
best_idx = result_df["cost"].idxmin()
best_row = result_df.loc[best_idx]
print(f"\n成本最优阈值：{best_row['threshold']}")
print(f"  对应总成本：{best_row['cost']:.0f} 元")
print(f"  此时 recall = {best_row['recall']:.4f}，FN = {best_row['FN']}，FP = {best_row['FP']}")

# ========== 画图 ==========
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:red'
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Total Cost (CNY)', color=color1)
ax1.plot(result_df["threshold"], result_df["cost"], color=color1, marker='o', label='Total Cost')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.axvline(best_row["threshold"], color='gray', linestyle='--', alpha=0.7,
            label=f'Best threshold = {best_row["threshold"]}')

ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Recall', color=color2)
ax2.plot(result_df["threshold"], result_df["recall"], color=color2,
         marker='s', linestyle='--', label='Recall')
ax2.tick_params(axis='y', labelcolor=color2)

fig.suptitle('Cost-Optimal Threshold Analysis')
fig.tight_layout()
fig.savefig(RESULTS_DIR / "cost_curve.png", dpi=150)
print(f"\n成本曲线图已保存: {RESULTS_DIR / 'cost_curve.png'}")

# 保存结果表
result_df.to_csv(RESULTS_DIR / "cost_analysis.csv", index=False)
print(f"结果表已保存: {RESULTS_DIR / 'cost_analysis.csv'}")