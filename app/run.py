import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from LLMSetting import LLMPredictor
from embeddings import HuggingFaceEmbedding, Reranker
from langchain_community.vectorstores.faiss import FAISS
from pdfparser import extract_page_text
from bm25 import BM25Model
import torch
import jieba
jieba.setLogLevel("ERROR") # 只显示ERROR日志
from tqdm import tqdm
from loguru import logger

filepath = r'/root/web_demo/HybirdSearch/data/CH-302-Rotary-Manual-v5.pdf'   

def main():
    num_input_docs = 3 # 最终选中输入进LLM的文档数
    
    # Define the model and tokenizer
    model_name_or_path = '/root/web_demo/HybirdSearch/models/models--Qwen--Qwen1.5-7B-Chat'
    stella_large_embedding = '/root/web_demo/HybirdSearch/models/models--infgrad--stella-large-zh-v3-1792d'  # 512
    gte_large_embedding = '/root/web_demo/HybirdSearch/models/models--thenlper--gte-large-zh' # 512
    bge_large_embedding = '/root/web_demo/HybirdSearch/models/models--BAAI--bge-large-zh'  # 512
    reranker_model_path = '/root/web_demo/HybirdSearch/models/models--BAAI--bge-reranker-large' # 512
    llm = LLMPredictor(model_name_or_path)
    reranker = Reranker(reranker_model_path)
    # Data processing
    # 解析两次，一次是长度300的块，前后重叠100长度；另一次是长度500的块，前后重叠200；两次合并起来构成总的解析数据集，用于后续召回
    texts = extract_page_text(filepath=filepath, max_len=300, overlap_len=100) + extract_page_text(filepath=filepath, max_len=500, overlap_len=200)
    corpus = [item.page_content for item in texts]

    # Embedding       
    db_stella = FAISS.from_documents(texts, HuggingFaceEmbedding(model_path=stella_large_embedding))
    BM25 = BM25Model(corpus) # BM25召回模型，此时它是存储在内存中的
    
    while True:
        input_query = input("请输入问题：")
        query_group = llm.generate_queries(input_query) # rewrite the query
        
        # recall 双数vectorsearch，单数bm25
        recall_results = [
            db_stella.similarity_search(query*3, k=5) if i % 2 == 0 else BM25.bm25_similarity(query*3, k=3)
            for i, query in enumerate(query_group)
            ]
        
        # rerank
        rerank_results = reranker.rerank(input_query, recall_results, k=num_input_docs)
        logger.info(f"rerank_results: {rerank_results}")
        # query + rerank + llm
        res = llm.get_answer_from_pdf(rerank_results, input_query)
        logger.info(f"用户的问题是: {input_query}")
        print("\n")
        print(res)
    
    
if __name__ == "__main__":
    main()