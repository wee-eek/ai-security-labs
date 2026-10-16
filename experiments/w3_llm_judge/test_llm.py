import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 测试调用
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一名网络安全分析师。"},
        {"role": "user", "content": "请用一句话判断这个 URL 是否可疑：http://paypal-secure-login.xyz/account/update"}
    ],
    temperature=0.3,
)

print(response.choices[0].message.content)