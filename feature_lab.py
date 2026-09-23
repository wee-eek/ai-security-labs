from urllib.parse import urlparse
import re
def extract_features(url):
    # 先把特征算出来，放到字典里
    features = {
        "url_length": len(url),                         #数字符串长度
        "num_dots": url.count("."),                     #数url中.的个数                    
        "num_hyphens": url.count("-"),                  
        "num_digits": sum(c.isdigit() for c in url),    #url中是数字的个数
        "has_at": "@" in url,   #"@" in url 是 Python 判断"这个字符串里有没有 @ 符号"，返回 True 或 False。
        "has_ip": bool(re.search(r'\d+\.\d+\.\d+\.\d+', url)),
                                                        # \d+ 意思是"一个或多个数字"
                                                        #\d+\.\d+\.\d+\.\d+ 就是"数字.数字.数字.数字"，也就是 IP 地址的形状
                                                        #re.search(...) 在 url 里找有没有这个模式，找到返回匹配对象，找不到返回 None
                                                        #bool(...) 把它转成 True/False
                                                        #为什么这是重要的安全特征：正规网站几乎不会用 http://192.168.1.1/login 这种 IP 直连形式，钓鱼站常用 IP 绕过域名黑名单。所以 has_ip = True 是强钓鱼信号。
        "num_subdomains": urlparse(url).netloc.count("."),
                                                        #urlparse("http://a.b.c.com/path") → 会拆成 scheme='http'、netloc='a.b.c.com'、path='/path'
                                                        #.netloc 取到域名部分 a.b.c.com
                                                        #.count(".") 数域名里点的个数 → 3
                                                        #为什么这有用：正规站通常是 www.xxx.com（2 个点），钓鱼站常用 www.secure.login.verify-bank.com 这种超多层级来伪装，num_subdomains 会显著偏高
        "has_suspicious_word": any(w in url.lower() for w in
                           ["login", "verify", "secure", "account", "update", "signin"]),
                                                        #url.lower() 把 URL 转小写（避免大小写漏判）
                                                        #for w in [...] 遍历可疑词列表
                                                        #any(...) 只要有任意一个词出现就返回 True
                                                        #注意：这 6 个词是我凭经验给的，你可以自己补。但别加太多——词太多会让所有 URL 都命中，特征就失去区分力了。6-8 个是合理范围。
    }
    return features

# 测试
test_urls = [
    "https://www.google.com",
    "http://192.168.1.1/login/verify",
    "http://paypal-secure-login.xyz/account/update",
    "http://user@fake-bank.com/login",
]

for u in test_urls:
    print(u)
    print(extract_features(u))
    print("-" * 50)