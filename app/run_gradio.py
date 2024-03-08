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
import gradio as gr

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
    
    # Embedding database
    # 数据经过两种不同的embedding模型存入两个不同的FAISS向量库
    # embedding_model1 = HuggingFaceEmbedding(model_path=gte_large_embedding)
    # db1 = FAISS.from_documents(texts, embedding_model1)
    # embedding_model2 = HuggingFaceEmbedding(model_path=bge_large_embedding)
    # db2 = FAISS.from_documents(texts, embedding_model2)
       
    db_stella = FAISS.from_documents(texts, HuggingFaceEmbedding(model_path=stella_large_embedding))
    BM25 = BM25Model(corpus) # BM25召回模型，此时它是存储在内存中的
    
    # while True:
    #     input_query = input("请输入问题：")
    #     query_group = llm.generate_queries(input_query) # rewrite the query
        
    #     # recall TODO 没有标记分不清是哪个模型召回的
    #     recall_results = []
    #     for query in tqdm(query_group):
    #         # 第一个query使用db_stella召回，第二个query使用BM25召回，第三个query使用db_stella召回，以此类推
    #         if query_group.index(query) % 2 == 0:
    #             recall_results.append(db_stella.similarity_search(query*3, k=5))
    #         else:
    #             recall_results.append(BM25.bm25_similarity(query*3, k=3))
        
    #     # rerank
    #     rerank_results = reranker.rerank(input_query, recall_results, k=num_input_docs)
    #     logger.info(f"rerank_results: {rerank_results}")
    #     # query + rerank + llm
    #     res = llm.get_answer_from_pdf(rerank_results, input_query)
    #     print("\n")
    #     print(res)
    def query_llm(input_query):
        query_group = llm.generate_queries(input_query)
        recall_results = []
        for query in query_group:
            if query_group.index(query) % 2 == 0:
                recall_results.append(db_stella.similarity_search(query * 3, k=5))
            else:
                recall_results.append(BM25.bm25_similarity(query * 3, k=3))

        rerank_results = reranker.rerank(input_query, recall_results, k=num_input_docs)
        logger.info(f"rerank_results: {rerank_results}")
        return llm.get_answer_from_pdf(rerank_results, input_query)

    interface = gr.Interface(fn=query_llm, inputs="text", outputs="text", title="Hybrid Search with Reranking",
                             description="Ask a question about the provided PDF document.")
    interface.launch(server_port=5066, share=True,debug=True)

    
    
if __name__ == "__main__":
    main()