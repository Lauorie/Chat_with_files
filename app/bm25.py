import jieba
jieba.setLogLevel("ERROR")
from rank_bm25 import BM25Okapi



class BM25Model:
    def __init__(self, data_list):
        tokenized_documents = [jieba.lcut(doc) for doc in data_list]
        self.bm25 = BM25Okapi(tokenized_documents)
        self.data_list = data_list

    def bm25_similarity(self, query, k = 10):
        query = jieba.lcut(query) 
        res = self.bm25.get_top_n(query, self.data_list, n=k)
        return res


if __name__ == '__main__':

    data_list = ["我爱自然语言处理", "我不喜欢人工智能", "我非常喜欢机器学习"]
    BM25 = BM25Model(data_list)
    query = "我喜欢自然语言处理"
    print(BM25.bm25_similarity(query, k = 2))
