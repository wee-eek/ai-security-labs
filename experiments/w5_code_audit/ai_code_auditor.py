import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# ========== 1. 路径与环境 ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ========== 2. 加载知识库并构建向量索引 ==========
print("正在加载 CWE 安全知识库...")
loader = TextLoader(str(Path(__file__).parent / "cwe_knowledge.txt"), encoding="utf-8")
docs = loader.load()

# 文本分块
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n")
splits = text_splitter.split_documents(docs)

# 使用 DeepSeek 的 Embedding 模型（或者兼容的本地模型）
# 注意：DeepSeek 官方目前没有直接的 Embedding 接口，这里为了演示，我们借用阿里通义或直接用 OpenAI 兼容的通用 Embedding
# 如果你没有其他 API，为了零成本跑通，这里我们用一种简单的“关键词检索”替代向量检索，重点在于体验 RAG 流程。
# 这里为了代码能直接跑，我们跳过真实向量库，写一个基于关键词的检索器。
def mock_retriever(query):
    # 简单的关键词匹配检索
    text = docs[0].page_content
    lines = text.split("\n")
    results = []
    for line in lines:
        # 如果代码中出现 sql、execute、os.system 等词，就召回相关漏洞知识
        if ("sql" in query.lower() and "CWE-89" in line) or \
           ("os.system" in query.lower() and "CWE-78" in line) or \
           ("<script>" in query.lower() and "CWE-79" in line):
            results.append(line)
    return "\n".join(results) if results else "未检索到相关漏洞知识。"

# ========== 3. 初始化大模型 ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.1
)

# ========== 4. 设计 Prompt 模板 ==========
prompt_template = """
你是一名资深的安全代码审计专家。请根据提供的【安全知识库参考】，审计以下【待审计代码】。

【安全知识库参考】
{context}

【待审计代码】
{code}

【任务要求】
1. 如果发现漏洞，请输出 JSON 格式报告，包含字段：
   - is_vulnerable: 布尔值
   - cwe_id: 对应的 CWE 编号
   - risk_level: 风险等级 (高/中/低)
   - vulnerability_type: 漏洞类型
   - reason: 漏洞原因分析
   - fixed_code: 修复后的安全代码片段
2. 如果代码安全，输出 {{"is_vulnerable": false}}。
3. 只输出 JSON，不要任何多余的解释。
"""
prompt = PromptTemplate.from_template(prompt_template)

# ========== 5. 组装 RAG 链条 ==========
def audit_code(code_snippet):
    # 1. 检索
    context = mock_retriever(code_snippet)
    # 2. 生成
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context, "code": code_snippet})
    return result

# ========== 6. 测试运行 ==========
if __name__ == "__main__":
    # 测试用例1：SQL注入漏洞
    test_code_sql = """
    def get_user(username):
        import sqlite3
        conn = sqlite3.connect('test.db')
        cursor = conn.cursor()
        # 拼接 SQL，存在注入
        query = "SELECT * FROM users WHERE name = '" + username + "'"
        cursor.execute(query)
        return cursor.fetchall()
    """

    # 测试用例2：安全代码
    test_code_safe = """
    def get_user_safe(username):
        import sqlite3
        conn = sqlite3.connect('test.db')
        cursor = conn.cursor()
        # 使用参数化查询，安全
        cursor.execute("SELECT * FROM users WHERE name = ?", (username,))
        return cursor.fetchall()
    """

    print("\n[1] 正在审计有漏洞的代码...")
    print(audit_code(test_code_sql))
    
    print("\n[2] 正在审计安全的代码...")
    print(audit_code(test_code_safe))