# -*- coding: utf-8 -*-
"""
最小实验：钓鱼网站二分类（数据 → 训练 → 评估 全流程）
数据集：UCI Phishing Websites Dataset (ID 327)
下载地址：https://archive.ics.uci.edu/dataset/327/phishing+websites+dataset
下载后把解压得到的 "Training Dataset.arff" 放到本脚本同一目录，直接运行本文件即可。

预期产出：
  1. 控制台：两个模型的 precision/recall/f1 + 混淆矩阵 + AUC 对比
  2. 当前目录：confusion_matrix_rf.png（混淆矩阵图）、phishing_model_rf.joblib（训练好的模型）
"""

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

# ---------- 1. 加载数据 ----------
raw, _ = arff.loadarff("Training Dataset.arff")
df = pd.DataFrame(raw)


def to_num(s: pd.Series) -> pd.Series:
    """arff 名义属性读进来是字节串(b'1'/b'-1')，统一转成数值。"""
    if s.dtype == object:
        return pd.to_numeric(s.map(lambda x: x.decode() if isinstance(x, bytes) else x))
    return s.astype(float)


df = df.apply(to_num)

# UCI 约定：Result = 1 合法站，-1 钓鱼站。转成 y：1=钓鱼（我们关心的正类），0=合法
y = (df["Result"] == -1).astype(int)
X = df.drop(columns=["Result"])

print("=== 数据概览 ===")
print(f"样本数: {len(df)}  特征数: {X.shape[1]}")
print(f"钓鱼站占比: {y.mean():.1%}   合法站占比: {(1 - y.mean()):.1%}")
print("缺失值总数:", int(df.isna().sum().sum()))

# ---------- 2. 划分训练/测试集 ----------
# stratify=y 保证训练集和测试集中钓鱼/合法比例一致，防止随机切偏
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 3. 训练两个模型做对比 ----------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]  # 预测为钓鱼的概率，算 AUC 用
    results[name] = (model, pred, proba)

    print(f"\n=== {name} ===")
    # labels=[1,0] 让报告先显示"钓鱼"这一类；钓鱼场景最该盯的是 recall（漏报代价大）
    print(classification_report(y_test, pred, labels=[1, 0],
                                target_names=["phishing(钓鱼)", "legitimate(合法)"],
                                digits=4))
    print("混淆矩阵 [[TN, FP], [FN, TP]]：")
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print(f"AUC: {roc_auc_score(y_test, proba):.4f}")

# ---------- 4. 混淆矩阵可视化（选 RF） ----------
_, pred_rf, _ = results["RandomForest"]
cm = confusion_matrix(y_test, pred_rf, labels=[0, 1])
fig, ax = plt.subplots(figsize=(4, 3.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["legitimate", "phishing"])
ax.set_yticklabels(["legitimate", "phishing"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("RandomForest Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig("confusion_matrix_rf.png", dpi=150)
print("\n混淆矩阵图已保存: confusion_matrix_rf.png")

# ---------- 5. 保存模型（为 W3-4 主项目预留接口） ----------
joblib.dump(results["RandomForest"][0], "phishing_model_rf.joblib")
print("模型已保存: phishing_model_rf.joblib")

# ---------- 6. 看看模型"学到了什么"（面试讲故事用） ----------
rf = results["RandomForest"][0]
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== Top10 重要特征（可解释为安全规则） ===")
print(importance.head(10).round(4))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

# 复现 W1 的数据（与之前同样的切分方式）
raw, _ = arff.loadarff("Training Dataset.arff")
df = pd.DataFrame(raw).apply(to_num)
y = (df["Result"] == -1).astype(int)
X = df.drop(columns=["Result"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

rf = joblib.load("phishing_model_rf.joblib")
proba = rf.predict_proba(X_test)[:, 1]   # 预测为钓鱼的概率

# 扫一遍阈值：0.1 到 0.9
print("阈值  钓鱼召回率  合法误报数(FP)  漏报数(FN)")
for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
    pred = (proba >= t).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    print(f"{t:.1f}   {recall_score(y_test, pred):.4f}      {cm[0][1]:4d}          {cm[1][0]:4d}")