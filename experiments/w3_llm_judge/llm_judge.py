import os
import re
import json
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import OpenAI

# ========== 路径定位 ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ========== DeepSeek 客户端 ==========
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ========== 特征提取（复用 W2） ==========
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
你的任务是：
1. 复核第一阶段的判定是否合理
2. 给出你独立的最终判断
3. 如果和第一阶段不一致，说明理由

输出要求：
- 必须输出合法的 JSON 格式，不含任何其他文字
- JSON 字段包括：
  - agrees_with_stage1: 布尔值，你是否同意第一阶段的判定（概率 ≥ 0.5 判钓鱼，< 0.5 判合法）
  - is_phishing: 布尔值，你的最终判断
  - risk_level: 字符串（"高"/"中"/"低"）
  - attack_type: 字符串（如"品牌仿冒"、"域名欺骗"、"诱导登录"、"无特定类型"）
  - reasons: 字符串数组（支持你判断的理由，至少 2 条）
  - suggestion: 字符串（处置建议）
  - conflict_explanation: 字符串。如果 agrees_with_stage1 为 false，说明你为什么不同意；如果为 true，填空字符串

严格要求：
- 理由必须基于 URL 本身的结构特征
- 不要编造未提供的信息
- 如果证据不足，倾向保守判断
- 只输出 JSON，不要输出任何解释性文字
"""

USER_PROMPT_TEMPLATE = """请复核以下 URL 的第一阶段研判结果：

URL: {url}

第一阶段模型（词法特征 + 随机森林）的分析结果：
- 钓鱼概率: {proba:.4f}（≥ 0.5 判定为钓鱼，< 0.5 判定为合法）
- URL 长度: {url_length}
- 子域名层级: {num_subdomains}
- 连字符数量: {num_hyphens}
- 数字字符数: {num_digits}
- 是否含 @ 符号: {has_at}
- 是否 IP 直连: {has_ip}
- 是否含可疑词(login/verify/secure等): {has_suspicious_word}

请输出 JSON 复核报告。"""

# ========== LLM 研判函数 ==========
def llm_judge(url, proba=0.5):
    feats = extract_features(url)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        url=url,
        proba=proba,
        url_length=feats["url_length"],
        num_subdomains=feats["num_subdomains"],
        num_hyphens=feats["num_hyphens"],
        num_digits=feats["num_digits"],
        has_at=feats["has_at"],
        has_ip=feats["has_ip"],
        has_suspicious_word=feats["has_suspicious_word"],
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"error": "JSON 解析失败", "raw": content}

    return result

# ========== 测试 ==========
if __name__ == "__main__":
    # 用不同 proba 测试：让 LLM 看到真实的概率
    test_cases = [
        ("https://www.google.com", 0.02),
        ("http://paypal-secure-login.xyz/account/update", 0.94),
        ("http://192.168.1.1/login/verify", 0.62),
        ("http://user@fake-bank.com/login", 0.87),
    ]

    for url, proba in test_cases:
        print(f"\n{'='*70}")
        print(f"URL: {url}")
        print(f"第一阶段概率: {proba}")
        print('='*70)
        result = llm_judge(url, proba=proba)
        print(json.dumps(result, ensure_ascii=False, indent=2))