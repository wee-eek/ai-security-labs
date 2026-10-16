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
SYSTEM_PROMPT = """你是一名资深的网络安全分析师，专门研判钓鱼 URL。

你的任务：判断给定 URL 是否为钓鱼网站，并输出结构化的研判报告。

输出要求：
- 必须输出合法的 JSON 格式，不含任何其他文字
- JSON 字段包括：
  - is_phishing: 布尔值（true/false）
  - risk_level: 字符串（"高"/"中"/"低"）
  - attack_type: 字符串（如"品牌仿冒"、"域名欺骗"、"诱导登录"、"无特定类型"）
  - reasons: 字符串数组（每条理由是一句话）
  - suggestion: 字符串（处置建议）

严格要求：
- 理由必须基于 URL 本身的结构特征，不要编造未提供的信息
- 如果证据不足以判定，is_phishing 设为 false，risk_level 设为"低"
- 只输出 JSON，不要输出任何解释性文字
"""

USER_PROMPT_TEMPLATE = """请研判以下 URL：

URL: {url}

第一阶段模型（词法特征 + 随机森林）的分析结果：
- 钓鱼概率: {proba:.4f}
- URL 长度: {url_length}
- 子域名层级: {num_subdomains}
- 连字符数量: {num_hyphens}
- 数字字符数: {num_digits}
- 是否含 @ 符号: {has_at}
- 是否 IP 直连: {has_ip}
- 是否含可疑词(login/verify/secure等): {has_suspicious_word}

请根据以上信息输出 JSON 研判报告。"""

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
    test_urls = [
        "https://www.google.com",
        "http://paypal-secure-login.xyz/account/update",
        "http://192.168.1.1/login/verify",
        "http://user@fake-bank.com/login",
    ]

    for url in test_urls:
        print(f"\n{'='*60}")
        print(f"URL: {url}")
        print('='*60)
        result = llm_judge(url, proba=0.5)
        print(json.dumps(result, ensure_ascii=False, indent=2))