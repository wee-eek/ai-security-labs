
```markdown
# 面向校园网络的钓鱼 URL 智能检测与研判系统

项目：基于“AI + 安全”理念，包含两阶段钓鱼检测、RAG代码审计与LLM越狱测试。

---

## 一、项目简介

针对校园网场景下钓鱼 URL 检测面临的“大规模、实时性、可解释性”三重约束，设计两阶段检测系统，并扩展了 LLM 自身安全性与代码审计的探索：

- **W1-W2（第一阶段）**：URL 词法特征 + 随机森林，实现零访问的快速预筛，并完成详尽的消融与成本建模。
- **W3（第二阶段）**：接入 DeepSeek 大模型，进行复核式深度研判，输出标准 JSON 格式的风险报告，解决传统模型可解释性差的痛点。
- **W4（安全扩展）**：构建 LLM 越狱与提示词注入（Prompt Injection）防御测试靶场。
- **W5（安全扩展）**：基于 LangChain 与 RAG（检索增强生成）技术，开发大模型代码安全审计助手。

---

## 二、目录结构

```text
phishing-minlab/
├── README.md                  # 项目说明
├── requirements.txt           # 依赖清单
├── app.py                     # Streamlit 两阶段研判演示系统
├── data/                      # 数据集（git 忽略）
├── experiments/               # 各阶段实验代码
│   ├── w1_uci_baseline/       # W1：UCI 数据最小实验
│   ├── w2_lexical_features/   # W2：词法特征 + 交叉验证 + 成本建模
│   ├── w3_llm_judge/          # W3：LLM 复核研判与批量评估
│   ├── w4_llm_security/       # W4：LLM 越狱与注入测试靶场
│   └── w5_code_audit/         # W5：基于 RAG 的代码审计助手
├── results/                   # 实验产出图表
└── src/                       # 预留模块
```

---

## 三、实验进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| W1 | UCI 数据 ML 全流程 | ✅ 完成 |
| W2 | 词法特征 + 交叉验证 + 阈值扫描 + 成本建模 | ✅ 完成 |
| W3 | LLM 研判层（DeepSeek 复核与批量评估） | ✅ 完成 |
| W4 | LLM 越狱与提示词注入防御测试靶场 | ✅ 完成 |
| W5 | 基于 RAG 的大模型代码安全审计助手 | ✅ 完成 |

---

## 四、运行方式

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行 Streamlit 系统界面（推荐演示）
```bash
streamlit run app.py
```

### 运行核心实验脚本
```bash
# W1：UCI 数据最小实验
python experiments/w1_uci_baseline/phishing_minlab.py

# W2：词法特征模型与交叉验证
python experiments/w2_lexical_features/url_feature_model.py
python experiments/w2_lexical_features/cross_validation.py

# W3：LLM 复核与批量评估
python experiments/w3_llm_judge/llm_judge.py
python experiments/w3_llm_judge/llm_batch_eval.py

# W4：LLM 越狱靶场测试
python experiments/w4_llm_security/llm_jailbreak_tester.py

# W5：RAG 代码审计助手
python experiments/w5_code_audit/ai_code_auditor.py
```
*(注：运行大模型相关脚本前，请在根目录创建 `.env` 文件并填入 `DEEPSEEK_API_KEY=你的密钥`)*

---

## 五、关键结果与技术栈

### 关键结果
- **W1（UCI 数据）**：随机森林 accuracy 97.65%，AUC 0.9967。
- **W2（Mendeley 45万数据）**：随机森林单次 AUC 0.9357；5折交叉验证 AUC 0.9353 ± 0.0010，模型高度稳定。
- **W3（LLM 复核）**：40条样本测试，LLM 与 ML 一致率 90%，成功识别并纠正了“私网 IP 误判”等分歧案例。
- **W4（越狱测试）**：模拟 5 类注入攻击，DeepSeek 防御成功率 100%。
- **W5（代码审计）**：成功利用 RAG 检索 CWE 知识库，引导大模型输出包含 CWE 编号及修复代码的 JSON 报告。

### 技术栈
- **编程语言**：Python 3.11
- **机器学习**：scikit-learn, XGBoost, Pandas
- **大模型与框架**：DeepSeek API, LangChain, OpenAI SDK, RAG (检索增强生成)
- **Web 框架**：Streamlit
- **版本管理**：Git + GitHub

---

## 六、说明
- 所有实验脚本均使用 `pathlib` 动态定位项目根目录，可在任意位置运行。
- `data/` 目录和 `*.joblib` 模型文件已被 `.gitignore` 排除，不进入版本控制。
```

---

