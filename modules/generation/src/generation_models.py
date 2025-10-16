import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.generation.src.llm_generator import LLMGenerator, StopOnSequence
from modules.generation.src.retrievers_local import BM25Retriever, RerankRetriever, DenseRetriever

class BasicRAG:
    def __init__(self, generation_model, generation_tokenizer, device, args):
        self.args = args
        self.generator = LLMGenerator(generation_model, generation_tokenizer, device, args)
        
        # === Retrievers ============================= 
        if args.retriever_name == 'bm25':
            self.retriever = BM25Retriever(args)
        elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
            self.retriever = RerankRetriever(args)
        elif args.retriever_name in ['contriever', 'dpr', 'e5', 'bge']:
            self.retriever = DenseRetriever(args)

        # === Prompt =================================
        self.system_prompt = ''
        self.user_prompt_wo_context = "Q: {question}\nA:"
        self.user_prompt_with_context = "{documents}\n\nQ: {question}\nA:"
        
class NoRetrieval(BasicRAG):
    def __init__(self, generation_model, generation_tokenizer, device, args):
        super().__init__(generation_model, generation_tokenizer, device, args)
    
    def inference(self, question, generation_temp=0.7):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_wo_context.format(question=question)}
        ]
        _, output_text = self.generator.generate(messages, temperature=generation_temp)
        
        return output_text

class SingleRetrieval(BasicRAG):
    def __init__(self, generation_model, generation_tokenizer, device, args):
        super().__init__(generation_model, generation_tokenizer, device, args)
    
    def inference(self, question, generation_temp=0.7):
        search_docs = self.retriever.search(question)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_with_context.format(
                documents=search_docs,
                question=question
            )}
        ]
        _, output_text = self.generator.generate(messages, temperature=generation_temp)
        
        return output_text