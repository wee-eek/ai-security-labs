# 面向校园网络的钓鱼 URL 智能检测与研判系统

毕业设计项目：两阶段架构的钓鱼 URL 检测方法研究与原型实现。

---

## 一、项目简介

针对校园网场景下钓鱼 URL 检测面临的“大规模、实时性、可解释性”三重约束，设计两阶段检测系统：

- **第一阶段**：URL 词法特征 + 随机森林，零访问快速预筛
- **第二阶段**：LLM 深度研判，输出可读的研判报告（开发中）

---

## 二、目录结构

```
phishing-minlab/
├── README.md                  # 项目说明
├── requirements.txt           # 依赖清单
├── data/                      # 数据集
│   ├── Training Dataset.arff  # W1 数据（UCI）
│   └── URL dataset.csv        # W2 数据（Mendeley）
├── experiments/               # 各阶段实验
│   ├── w1_uci_baseline/       # W1：UCI 数据最小实验
│   └── w2_lexical_features/   # W2：词法特征 + 真实数据
├── results/                   # 实验产出
├── src/                       # 可复用核心代码（待填充）
└── paper/                     # 论文稿（待填充）
```

---

## 三、数据集

| 阶段 | 数据集 | 规模 | 来源 |
|---|---|---|---|
| W1 | UCI Phishing Websites | 11,055 条 | UCI ML Repository |
| W2 | Mendeley URL Dataset | 450,176 条 | Mendeley Data（原始来源 PhishTank、Majestic Million 等）|

---

## 四、实验进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| W1 | UCI 数据 ML 全流程 | ✅ 完成 |
| W2 | 词法特征 + 交叉验证 + 阈值扫描 | ✅ 完成 |
| W3-4 | LLM 研判层 | 🚧 进行中 |

---

## 五、运行方式

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行 W1：UCI 数据最小实验

```bash
cd experiments/w1_uci_baseline
python phishing_minlab.py
```

### 运行 W2：词法特征模型

```bash
cd experiments/w2_lexical_features
python url_feature_model.py
```

### 运行 W2：交叉验证

```bash
cd experiments/w2_lexical_features
python cross_validation.py
```

---

## 六、关键结果

### W1：UCI 数据（30 维现成特征）

- 逻辑回归：accuracy 92.76%、AUC 0.9784
- 随机森林：accuracy 97.65%、AUC 0.9967
- 最强特征：SSLfinal_State（32%）、URL_of_Anchor（26%）

### W2：Mendeley 数据（8 维自提词法特征）

- 随机森林单次结果：accuracy 0.93、AUC 0.9357
- 5 折交叉验证：AUC 0.9353 ± 0.0010、F1 0.8384 ± 0.0017
- 最强特征：num_subdomains（42%）、num_dots（21%）
- 阈值甜点区：0.2（召回率 75%→82%）

---

## 七、技术栈

- **编程语言**：Python 3.11
- **机器学习**：scikit-learn（LR、RandomForest）
- **数据处理**：pandas、numpy
- **特征提取**：Python 标准库（re、urllib.parse）
- **版本管理**：Git + GitHub

---

## 八、实验环境

```bash
Python 3.11
scikit-learn
pandas
numpy
matplotlib
scipy
joblib
```

---

## 九、说明

- 所有实验脚本均使用 `pathlib` 动态定位项目根目录，可在任意位置运行
- `data/` 和 `results/*.joblib` 已被 `.gitignore` 排除，不进入版本控制