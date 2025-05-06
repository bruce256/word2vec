
import jieba
import xml.etree.ElementTree as ET
from gensim.models import Word2Vec

# 读取XML文件并解析
file_path = 'data/weibo.xml'
tree = ET.parse(file_path)
root = tree.getroot()

# 获取所有<article>标签的内容
texts = [record.find('article').text for record in root.findall('RECORD')]
print(len(texts))

# 停用词列表，实际应用中需要根据实际情况扩展
stop_words = set(["的", "了", "在", "是", "我", "有", "和", "就"])

# 分词和去除停用词
processed_texts = []
for text in texts:
    if text is not None:
        words = jieba.cut(text)
        processed_text = [word for word in words if word not in stop_words]
        processed_texts.append(processed_text)

# 打印预处理后的文本
# for text in processed_texts:
#     print(text)

print(len(processed_texts))

# 训练Word2Vec模型
model = Word2Vec(sentences=processed_texts, vector_size=100, window=5, min_count=1, workers=10, sg=1)
# 保存模型
model.save("model/word2vec.model")