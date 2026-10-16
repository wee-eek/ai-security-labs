import re
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, recall_score,
                             precision_score, accuracy_score)
from xgboost import XGBClassifier

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
print(f"数据: {X.shape}, 钓鱼占比: {y.mean():.4f}")

# ========== 切分 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"训练集: {len(X_train)}  测试集: {len(X_test)}")

# ========== 三个模型对比 ==========
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    ),
}

results = {}
print("\n" + "="*60)
print("模型对比实验")
print("="*60)

for name, model in models.items():
    print(f"\n训练 {name} ...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc": roc_auc_score(y_test, proba),
    }

    print(f"\n--- {name} ---")
    print(classification_report(y_test, pred, target_names=["合法", "钓鱼"], digits=4))
    cm = confusion_matrix(y_test, pred)
    print(f"混淆矩阵: TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")

# ========== 汇总对比表 ==========
print("\n" + "="*60)
print("模型对比汇总表")
print("="*60)
summary = pd.DataFrame(results).T.round(4)
print(summary.to_string())

summary.to_csv(RESULTS_DIR / "model_comparison.csv")
print(f"\n对比表已保存: {RESULTS_DIR / 'model_comparison.csv'}")