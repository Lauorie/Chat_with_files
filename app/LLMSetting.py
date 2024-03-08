from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationConfig
import torch
from loguru import logger

def build_query_rewrite():
    query_rewrite_prompt = (
        """用户输入的指令如下：\n
        {}\n
        请根据用户输入的指令生成3个语义相似的指令，每行一个。
        生成的3个语义相似的指令：\n"""
    )
    return query_rewrite_prompt

def build_template():
    prompt_template = """请你基于以下材料回答用户问题。回答要清晰准确，包含正确关键词。
                        不要胡编乱造。如果所给材料与用户问题无关，只输出：无答案。\n
                        以下是材料：\n---
                        {}\n
                        用户问题：\n
                        {}\n
                        务必注意，如果所给材料无法回答用户问题，只输出无答案，不要自己回答。"""
    return prompt_template

def build_summary_template():
    prompt_template = """请你将给定的杂乱文本重新整理，使其不丢失任何信息且有较强的可读性，同时要求不丢失关键词。\n
                      以下是杂乱文本：\n---
                      {}\n"""

    return prompt_template

def build_query_rewrite_from_history():
    rewrite_query_from_history = """你是一名阅读理解专家。
    根据以下对话历史和后续问题，请你将后续问题重新表述为一个独立的问题。
    如果后续问题与对话历史无关，请直接返回后续问题。
    对话历史：\n
    {}
    后续问题：\n
    {}
    重新表述的问题：\n
    """
    return rewrite_query_from_history

class LLMPredictor(object):
    def __init__(self, model_name_or_path, adapter_path=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            # device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token='<|extra_0|>'
        self.tokenizer.eos_token='<|endoftext|>'
        self.tokenizer.padding_side='left'
        self.max_token = 4096
        self.query_rewrite_prompt = build_query_rewrite()
        self.query_rewrite_prompt_from_history = build_query_rewrite_from_history()
        self.prompt_template = build_template()
        self.summary_template = build_summary_template()
        self.history = []
        self.device = "cuda:0"
        self.model.to(self.device)
        logger.info(f"Loaded model from {model_name_or_path}")


    def generate_queries(self, query):
        """Generate 3 queries based on the input query"""
        fmt_prompt = self.query_rewrite_prompt.format(query)
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": fmt_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        with torch.no_grad():  
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        query_group = ['0.' + query] + responses
        return query_group
    
    def get_answer_from_pdf(self, context, query):
        """Generate an answer based on the input context and query"""
        content = "\n".join(context)
        content = self.prompt_template.format(content, query)
        messages = [{"role": "user", "content": content}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        with torch.no_grad():  
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return "".join(response)
    
    # clear history
    def clear_history(self):
        return "",[]
    

    # 实现多轮对话
    def chat_with_pdf(self, context, query):
        pass