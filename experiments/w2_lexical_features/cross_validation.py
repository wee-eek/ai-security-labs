import re
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, make_scorer

# 交叉验证实验

# ========== 路径定位 ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ========== 复用 W2 的特征提取函数 ==========
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
print(f"数据: {X.shape}, 钓鱼占比: {y.mean():.4f}")

# ========== 5 折分层交叉验证 ==========
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("\n运行 5 折交叉验证（45 万条，约 3-5 分钟）...")

# 指标 1：F1
f1_scores = cross_val_score(rf, X, y, cv=cv, scoring="f1", n_jobs=-1)
print(f"\nF1（钓鱼类）5 折结果: {f1_scores}")
print(f"F1 均值: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

# 指标 2：AUC
auc_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"\nAUC 5 折结果: {auc_scores}")
print(f"AUC 均值: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

# 指标 3：Recall
recall_scorer = make_scorer(recall_score, pos_label=1)
recall_scores = cross_val_score(rf, X, y, cv=cv, scoring=recall_scorer, n_jobs=-1)
print(f"\nRecall（钓鱼类）5 折结果: {recall_scores}")
print(f"Recall 均值: {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")

# ========== 结果解读 ==========
print("\n" + "="*50)
print("解读")
print("="*50)
print(f"5 折 F1 标准差: {f1_scores.std():.4f}")
if f1_scores.std() < 0.01:
    print("→ 标准差 < 0.01，模型非常稳定，单次结果可信")
elif f1_scores.std() < 0.03:
    print("→ 标准差 < 0.03，模型基本稳定")
else:
    print("→ 标准差较大，模型不稳定，需要检查")