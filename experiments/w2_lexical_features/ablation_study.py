import re
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

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
all_features = list(X.columns)
print(f"数据: {X.shape}, 特征: {all_features}")

# ========== 切分 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ========== 定义评估函数 ==========
def evaluate(X_tr, X_te, y_tr, y_te):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    proba = rf.predict_proba(X_te)[:, 1]
    return {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "auc": roc_auc_score(y_te, proba),
    }

# ========== 基准：全部特征 ==========
print("\n" + "="*60)
print("基准：全部 8 个特征")
print("="*60)
baseline = evaluate(X_train, X_test, y_train, y_test)
print(f"accuracy: {baseline['accuracy']:.4f}")
print(f"precision: {baseline['precision']:.4f}")
print(f"recall: {baseline['recall']:.4f}")
print(f"f1: {baseline['f1']:.4f}")
print(f"auc: {baseline['auc']:.4f}")

# ========== 消融：依次去掉每个特征 ==========
print("\n" + "="*60)
print("消融实验：依次去掉一个特征")
print("="*60)

results = {"baseline (全8个)": baseline}
for drop_col in all_features:
    print(f"\n去掉 {drop_col} ...")
    X_tr_drop = X_train.drop(columns=[drop_col])
    X_te_drop = X_test.drop(columns=[drop_col])
    res = evaluate(X_tr_drop, X_te_drop, y_train, y_test)
    results[f"去掉 {drop_col}"] = res
    print(f"  accuracy: {res['accuracy']:.4f}  recall: {res['recall']:.4f}  "
          f"f1: {res['f1']:.4f}  auc: {res['auc']:.4f}")

# ========== 汇总对比表 ==========
print("\n" + "="*60)
print("消融实验汇总表")
print("="*60)
summary = pd.DataFrame(results).T.round(4)
summary["ΔAUC"] = (summary["auc"] - baseline["auc"]).round(4)
summary["ΔF1"] = (summary["f1"] - baseline["f1"]).round(4)
print(summary.to_string())

# ========== 保存结果 ==========
summary.to_csv(RESULTS_DIR / "ablation_study.csv")
print(f"\n消融结果已保存: {RESULTS_DIR / 'ablation_study.csv'}")