import os
import uvicorn
from fastapi import FastAPI, Query
from loguru import logger

from LLMSetting import LLMPredictor
from embeddings import HuggingFaceEmbedding, Reranker
from langchain_community.vectorstores.faiss import FAISS
from pdfparser import extract_page_text
from bm25 import BM25Model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

filepath = r'/root/web_demo/HybirdSearch/data/CH-302-Rotary-Manual-v5.pdf'


class HybridSearchAPI:
    def __init__(self):
        # Define the model and tokenizer
        self.model_name_or_path = '/root/web_demo/HybirdSearch/models/models--Qwen--Qwen1.5-7B-Chat'
        self.stella_large_embedding = '/root/web_demo/HybirdSearch/models/models--infgrad--stella-large-zh-v3-1792d'  # 512
        self.reranker_model_path = '/root/web_demo/HybirdSearch/models/models--BAAI--bge-reranker-large'  # 512

        self.llm = LLMPredictor(self.model_name_or_path)
        self.reranker = Reranker(self.reranker_model_path)

        self.texts = extract_page_text(filepath=filepath, max_len=300, overlap_len=100) + \
                     extract_page_text(filepath=filepath, max_len=500, overlap_len=200)
        self.corpus = [item.page_content for item in self.texts]

        self.db_stella = FAISS.from_documents(self.texts, HuggingFaceEmbedding(model_path=self.stella_large_embedding))
        self.BM25 = BM25Model(self.corpus)

        self.num_input_docs = 3  # Number of documents to send to the LLM

    async def search(self, query: str):
        """
        Searches the PDF and retrieves answers using the hybrid search model.

        Args:
            query (str): The user's query.

        Returns:
            str: The answer retrieved from the PDF.
        """
        query_group = self.llm.generate_queries(query)
        recall_results = [
            self.db_stella.similarity_search(query*3, k=5) if i % 2 == 0 else self.BM25.bm25_similarity(query*3, k=3)
            for i, query in enumerate(query_group)
        ]

        # Rerank
        rerank_results = self.reranker.rerank(query, recall_results, k=self.num_input_docs)
        logger.info(f"rerank_results: {rerank_results}")

        # Query + rerank + LLM
        res = self.llm.get_answer_from_pdf(rerank_results, query)
        return res


app = FastAPI()
hybrid_search_api = HybridSearchAPI()

@app.get("/search")
async def search(query: str = Query(...)):
    return await hybrid_search_api.search(query)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5066)