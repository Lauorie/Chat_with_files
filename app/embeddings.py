from transformers import AutoTokenizer,  AutoModel, AutoModelForSequenceClassification
import torch
from langchain.schema.embeddings import Embeddings
from typing import List
import numpy as np
from loguru import logger

class HuggingFaceEmbedding(Embeddings):
    def __init__(self, model_path, lora_path=None, batch_size=4, **kwargs):
        super().__init__(**kwargs)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer= AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # if lora_path is not None:
        #     self.model = PeftModel.from_pretrained(self.model, lora_path).eval()
        #     print('merged embedding model')
        self.device = torch.device('cuda:0')
        self.model.half()
        self.model.to(self.device)
        self.batch_size = batch_size
        if 'bge' in model_path:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："
        else:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = ""
        self.model_path = model_path
        logger.info(f"HuggingFaceEmbedding model loaded from {model_path}")

    def compute_kernel_bias(self, vecs, n_components=384):
        """计算kernel和bias
        vecs.shape = [num_samples, embedding_size]，
        最后的变换：y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(axis=0, keepdims=True) # 计算均值
        cov = np.cov(vecs.T)  # 计算协方差矩阵
        u, s, vh = np.linalg.svd(cov) # svd奇异值分解
        """ 
        使用奇异值分解的结果计算变换矩阵 W。
        这里，np.diag(1 / np.sqrt(s)) 创建一个对角矩阵，其对角线上的元素是 s 中元素的倒数平方根。
        这个变换的目的是对数据进行白化（whitening），即去除数据之间的相关性，并使所有特征的方差相等。
        """
        W = np.dot(u, np.diag(1 / np.sqrt(s))) 
        return W[:, :n_components], -mu  # 选择前n_components个奇异向量,embedding_size==>n_components 

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [t.replace("\n", " ") for t in texts]
        num_texts = len(texts)

        sentence_embeddings = []

        for start in range(0, num_texts, self.batch_size):
            end = min(start + self.batch_size, num_texts)
            batch_texts = texts[start:end]
            encoded_input = self.tokenizer(batch_texts, 
                                           max_length=512, # 输入长度只有512，注意切分时的chunk_size
                                           padding=True, 
                                           truncation=True,
                                           return_tensors='pt')
            encoded_input.to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                if 'gte' in self.model_path:
                    batch_embeddings = model_output.last_hidden_state[:, 0] #[batch_size, embedding_size]
                else:
                    batch_embeddings = model_output[0][:, 0]

                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1) # 归一化，p=2是L2范数
                sentence_embeddings.extend(batch_embeddings.tolist())

        # sentence_embeddings = np.array(sentence_embeddings)
        # self.W, self.mu = self.compute_kernel_bias(sentence_embeddings)
        # sentence_embeddings = (sentence_embeddings+self.mu) @ self.W
        # self.W, self.mu = torch.from_numpy(self.W).cuda(), torch.from_numpy(self.mu).cuda()
        return sentence_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        if 'bge' in self.model_path:
            encoded_input = self.tokenizer([self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH + text], # bge模型需要加上提示语
                                           padding=True,
                                           truncation=True, 
                                           return_tensors='pt')
        else:
            encoded_input = self.tokenizer([text], 
                                           padding=True,
                                           truncation=True, 
                                           return_tensors='pt')
        encoded_input.to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        # sentence_embeddings = (sentence_embeddings + self.mu) @ self.W
        return sentence_embeddings[0].tolist()
   
class Reranker:
    def __init__(self, model_path, device='cuda:0'):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.model.half()
        logger.info(f"Loaded rerank model from {model_path}")

    def rerank(self, query:str, docs:List, k=10):
        """Rerank a list of documents based on a query using a reranking model.
        Args:
            docs: The list of documents to rerank.
        Returns:
            The reranked list of documents.   
        """
        docs_ = []
        for item in docs[0]:
            if isinstance(item, str):
                docs_.append(item)
            else:
                docs_.append(item.page_content)               
        docs = list(set(docs_))
        pairs = []
        for d in docs:
            pairs.append([query, d])
        with torch.no_grad():
            inputs = self.tokenizer(pairs, 
                                      padding=True, 
                                      truncation=True, 
                                      return_tensors='pt', 
                                      max_length=512).to('cuda') # 注意此时是将query和doc拼接为512长度的输入
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().tolist()
        
        docs = [(docs[i], scores[i]) for i in range(len(docs))]
        docs = sorted(docs, key = lambda x: x[1], reverse = True)
        return [item[0] for item in docs[:k]]