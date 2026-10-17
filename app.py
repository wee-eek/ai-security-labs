import os
import re
import json
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import OpenAI
import joblib
import streamlit as st

# ========== 页面配置 ==========
st.set_page_config(page_title="钓鱼 URL 检测系统", page_icon="🛡️", layout="wide")

# ========== 路径与模型加载 ==========
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
load_dotenv(PROJECT_ROOT / ".env")

@st.cache_resource
def load_rf_model():
    return joblib.load(RESULTS_DIR / "url_feature_rf.joblib")

rf_model = load_rf_model()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

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

# ========== LLM 研判 ==========
SYSTEM_PROMPT = """你是一名资深的网络安全分析师，负责复核第一阶段机器学习模型的研判结果。
输出必须是合法的 JSON 格式，包含字段：
- agrees_with_stage1: 布尔值
- is_phishing: 布尔值
- risk_level: "高"/"中"/"低"
- attack_type: 字符串
- reasons: 字符串数组
- suggestion: 字符串
- conflict_explanation: 字符串
只输出 JSON，不要其他文字。
"""

def llm_judge(url, proba):
    feats = extract_features(url)
    user_prompt = f"请复核以下 URL：\nURL: {url}\n第一阶段钓鱼概率: {proba:.4f}\n特征: {json.dumps(feats, ensure_ascii=False)}\n请输出 JSON。"
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

# ========== 页面布局 ==========
st.title("🛡️ 钓鱼 URL 智能检测与研判系统")
st.markdown("**两阶段架构：** 第一阶段（词法特征 + 随机森林）进行快速预筛 → 第二阶段（大语言模型）进行深度研判。")

url_input = st.text_input("请输入需要检测的 URL：", placeholder="例如：http://paypal-secure-login.xyz/account/update")

if st.button("开始检测", type="primary"):
    if not url_input:
        st.warning("请输入 URL")
    else:
        # 第一阶段
        with st.spinner("第一阶段：提取特征并预测概率..."):
            feats = extract_features(url_input)
            X_feat = [[feats[k] for k in feats.keys()]]
            proba = rf_model.predict_proba(X_feat)[0][1]
            stage1_pred = "钓鱼" if proba >= 0.5 else "合法"

        st.divider()
        
        # 展示第一阶段
        col1, col2 = st.columns(2)
        col1.metric("第一阶段钓鱼概率", f"{proba:.2%}")
        col2.metric("第一阶段判定结果", stage1_pred)

        # 第二阶段
        with st.spinner("第二阶段：大模型深度研判中..."):
            llm_result = llm_judge(url_input, proba)

        st.divider()
        st.subheader("第二阶段：大语言模型深度研判")
        
        if "error" in llm_result:
            st.error(f"LLM 研判失败：{llm_result['error']}")
        else:
            risk = llm_result.get("risk_level", "未知")
            is_phish = llm_result.get("is_phishing", False)
            attack_type = llm_result.get("attack_type", "未知")
            agrees = "✅ 同意第一阶段判定" if llm_result.get("agrees_with_stage1") else "⚠️ 不同意第一阶段判定"
            
            # 结果展示
            st.info(f"**最终结论：** {'🚨 判定为钓鱼' if is_phish else '✅ 判定为合法'} | **风险等级：** {risk} | **攻击类型：** {attack_type}")
            st.caption(f"复核状态：{agrees}")
            
            st.write("**判定理由：**")
            for reason in llm_result.get("reasons", []):
                st.write(f"- {reason}")
                
            st.write("**处置建议：**")
            st.success(llm_result.get("suggestion", "无"))
            
            if not llm_result.get("agrees_with_stage1") and llm_result.get("conflict_explanation"):
                st.warning(f"**冲突解释：** {llm_result['conflict_explanation']}")

st.sidebar.title("关于系统")
st.sidebar.info(
    "本项目为毕业设计原型，采用两阶段架构：\n"
    "1. 轻量级 ML 模型（RF）零访问快速预筛；\n"
    "2. LLM 深度研判与解释。\n"
    "支持输出人类可读的研判报告。"
)