import re
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ========== 1. 特征提取函数（你刚写好的） ==========
def extract_features(url):
    # 单独处理 num_subdomains，防止畸形 URL 让整个脚本崩掉
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
        "num_subdomains": num_subdomains,      # ← 用 try 算出来的
        "has_suspicious_word": any(w in url.lower() for w in
                                   ["login", "verify", "secure", "account", "update", "signin"]),
    }

# ========== 2. 读数据 + 提特征 ==========
print("读取数据中...")
df = pd.read_csv("data/URL dataset.csv")
print(f"总样本: {len(df)}")

print("提取特征中（45万条，约1-2分钟）...")
feat_df = df["url"].apply(extract_features).apply(pd.Series)
feat_df["label"] = (df["type"] == "phishing").astype(int)

print("\n特征表前5行:")
print(feat_df.head())
print(f"\n标签分布:\n{feat_df['label'].value_counts()}")

# ========== 3. 切分（W1 那套） ==========
X = feat_df.drop(columns=["label"])
y = feat_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n训练集: {len(X_train)}  测试集: {len(X_test)}")

# ========== 4. 训练两个模型 ==========
print("\n训练 Logistic Regression...")
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("训练 Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ========== 5. 评估 ==========
def evaluate(name, y_true, y_pred, model, X_test):
    print(f"\n{'='*50}")
    print(f"{name}")
    print('='*50)
    print(classification_report(y_true, y_pred, target_names=["合法", "钓鱼"]))
    cm = confusion_matrix(y_true, y_pred)
    print(f"混淆矩阵:")
    print(f"  TN={cm[0][0]:6d}  FP={cm[0][1]:6d}")
    print(f"  FN={cm[1][0]:6d}  TP={cm[1][1]:6d}")
    proba = model.predict_proba(X_test)[:, 1]
    print(f"AUC: {roc_auc_score(y_true, proba):.4f}")

evaluate("Logistic Regression", y_test, lr_pred, lr, X_test)
evaluate("Random Forest", y_test, rf_pred, rf, X_test)

# ========== 6. 特征重要性 ==========
print(f"\n{'='*50}")
print("随机森林 - 特征重要性")
print('='*50)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances)

# 保存模型和图
joblib.dump(rf, "url_feature_rf.joblib")
importances.plot(kind="barh", figsize=(8, 5))
plt.title("Feature Importance (URL lexical features)")
plt.tight_layout()
plt.savefig("url_feature_importance.png")
print("\n模型和图已保存")