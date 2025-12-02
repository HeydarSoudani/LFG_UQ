import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import transformers

from utils.general_utils import passages2string
from modules.longform_response_generation.src.llm_generator import LLMGenerator_api, LLMGenerator_hf_local
from modules.longform_response_generation.src.retrievers_local import BM25Retriever, RerankRetriever, DenseRetriever


class BasicRAG:
    def __init__(self, device, args):
        self.args = args
        
        # --- Generators
        if args.model_source == 'api':
            self.generator = LLMGenerator_api(args.model_name_or_path)
        elif args.model_source == 'hf_local':
            backbone_model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=torch.bfloat16).to(device) # attn_implementation="eager"
            backbone_tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
            self.generator = LLMGenerator_hf_local(backbone_model, backbone_tokenizer, device, args)
        else:
            raise NotImplementedError
        
        # --- Retrievers 
        if args.retriever_name == 'bm25':
            self.retriever = BM25Retriever(args)
        elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
            self.retriever = RerankRetriever(args)
        elif args.retriever_name in ['contriever', 'dpr', 'e5', 'bge']:
            self.retriever = DenseRetriever(args)

        # --- Prompt
        self.system_prompt = ''
        self.user_prompt_with_context = "{documents}\n\nQ: {question}\nA:"
     
       
class NoRetrieval(BasicRAG):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.user_prompt_template = "Question: {user_query}\nAnswer:"
    
    def inference(self, question, generation_temp=0.7):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_template.format(user_query=question)}
        ]
        prediction, _ = self.generator.generate(messages, temperature=generation_temp)
        
        return prediction


class SingleRetrieval(BasicRAG):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.user_prompt_template = "<information>{documents}</information>\n\nQuestion: {user_query}\nAnswer:"
    
    def inference(self, question, generation_temp=0.7):
        search_docs = self.retriever.search(question)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_template.format(
                documents=passages2string(search_docs),
                user_query=question
            )}
        ]
        prediction, _ = self.generator.generate(messages, temperature=generation_temp)
        
        return prediction
