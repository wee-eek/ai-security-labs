import os
import re
import json
import time
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import OpenAI
import joblib

# ========== 路径定位 ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
load_dotenv(PROJECT_ROOT / ".env")

# ========== DeepSeek 客户端 ==========
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ========== 加载第一阶段模型 ==========
print("加载第一阶段随机森林模型...")
rf_model = joblib.load(RESULTS_DIR / "url_feature_rf.joblib")

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

# ========== Prompt 模板 ==========
SYSTEM_PROMPT = """你是一名资深的网络安全分析师，负责复核第一阶段机器学习模型的研判结果。

第一阶段模型（词法特征 + 随机森林）已经给出每条 URL 的钓鱼概率。
你的任务：复核第一阶段的判定，给出你自己的判断和理由。

输出要求：
- 必须输出合法的 JSON 格式，不含任何其他文字
- JSON 字段包括：
  - agrees_with_stage1: 布尔值，你是否同意第一阶段的判定
  - is_phishing: 布尔值，你的最终判断
  - risk_level: "高"/"中"/"低"
  - attack_type: 字符串
  - reasons: 字符串数组
  - suggestion: 字符串
  - conflict_explanation: 字符串（不一致时说明理由，一致时留空）
"""

USER_PROMPT_TEMPLATE = """请复核以下 URL：
URL: {url}
第一阶段钓鱼概率: {proba:.4f}（≥ 0.5 判钓鱼，< 0.5 判合法）
特征: 长度={url_length}, 子域名={num_subdomains}, 连字符={num_hyphens}, 数字={num_digits}, 含@={has_at}, IP直连={has_ip}, 含可疑词={has_suspicious_word}
请输出 JSON。"""

def llm_judge(url, proba):
    feats = extract_features(url)
    user_prompt = USER_PROMPT_TEMPLATE.format(url=url, proba=proba, **feats)
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

# ========== 抽样准备 ==========
print("读取数据...")
df = pd.read_csv(DATA_DIR / "URL dataset.csv")

# 各抽 20 条
phishing_df = df[df["type"] == "phishing"].sample(20, random_state=42)
legit_df = df[df["type"] == "legitimate"].sample(20, random_state=42)
sample_df = pd.concat([phishing_df, legit_df]).reset_index(drop=True)

print(f"抽样完成：{len(sample_df)} 条（20 钓鱼 + 20 合法）")

# ========== 批量评估 ==========
results = []
print("\n开始 LLM 复核（约 1-2 分钟）...\n")

for i, row in sample_df.iterrows():
    url = row["url"]
    true_label = 1 if row["type"] == "phishing" else 0
    
    # 第一阶段预测
    feats = extract_features(url)
    X_feat = pd.DataFrame([feats])
    proba = rf_model.predict_proba(X_feat)[0][1]
    stage1_pred = 1 if proba >= 0.5 else 0
    
    # LLM 复核
    print(f"[{i+1}/40] 研判中... {url[:50]}")
    llm_result = llm_judge(url, proba)
    time.sleep(0.5)  # 避免触发限流
    
    results.append({
        "url": url,
        "true_label": true_label,
        "stage1_proba": round(proba, 4),
        "stage1_pred": stage1_pred,
        "llm_agrees": llm_result.get("agrees_with_stage1"),
        "llm_pred": llm_result.get("is_phishing"),
        "llm_risk": llm_result.get("risk_level"),
        "llm_reasons": " | ".join(llm_result.get("reasons", [])),
        "conflict": llm_result.get("conflict_explanation", ""),
    })

# ========== 统计 ==========
result_df = pd.DataFrame(results)
result_df.to_csv(RESULTS_DIR / "llm_batch_eval.csv", index=False, encoding="utf-8-sig")

print("\n" + "="*60)
print("批量评估完成！")
print("="*60)

# 一致率
agree_count = result_df["llm_agrees"].sum()
total = len(result_df)
print(f"LLM 与第一阶段一致率: {agree_count} / {total} = {agree_count/total:.2%}")

# LLM 自己的准确率
llm_acc = (result_df["llm_pred"] == result_df["true_label"]).mean()
print(f"LLM 预测准确率: {llm_acc:.4f}")

# 第一阶段准确率
stage1_acc = (result_df["stage1_pred"] == result_df["true_label"]).mean()
print(f"第一阶段模型准确率: {stage1_acc:.4f}")

print(f"\n结果已保存: {RESULTS_DIR / 'llm_batch_eval.csv'}")