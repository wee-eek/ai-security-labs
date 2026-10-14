import pandas as pd
from pathlib import Path

# ---------- 路径定位 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 读数据（注意文件名里有空格）
df = pd.read_csv(DATA_DIR / "URL dataset.csv")

# 1. 看数据规模：期望接近 (450176, 2)
print("shape:", df.shape)

# 2. 看列名：确认哪列是 URL、哪列是标签
print("columns:", df.columns.tolist())

# 3. 看前 5 行：亲眼看几条 URL 长什么样
print("\nhead:")
print(df.head())

# 4. 看标签分布：期望 phishing 约 10.4 万、legitimate 约 34.5 万
print("\nlabel distribution:")
print(df.iloc[:, 1].value_counts())   # 用第2列，避免列名猜错