import re
import dataclasses
from abc import ABC
from typing import Union, Any
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

import utils.safe_utils as utils
from modules.generation.src.retrievers_local import BM25Retriever, RerankRetriever, DenseRetriever

from .safe import (
    SUPPORTED_LABEL,
    NOT_SUPPORTED_LABEL,
    _STATEMENT_PLACEHOLDER,
    _KNOWLEDGE_PLACEHOLDER,
    _NEXT_SEARCH_FORMAT,
    _FINAL_ANSWER_FORMAT,
    FinalAnswer,
    _generate
)

@dataclasses.dataclass()
class RetrievalResult:
    query: str
    result: str


class ClaimEvaluator(ABC):
    def __init__(
        self,
        rater: Union[PreTrainedModel, str],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        args = None,
        max_steps: int = 5,
        max_retries: int = 10,
        num_searches: int = 3,
    ):
        self.rater = rater
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.num_searches = num_searches
        self.args = args
        
        # === Retrievers ============================= 
        if args.retriever_name == 'bm25':
            self.retriever = BM25Retriever(args)
        elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
            self.retriever = RerankRetriever(args)
        elif args.retriever_name in ['contriever', 'dpr', 'e5', 'bge']:
            self.retriever = DenseRetriever(args)
    
    def __call__(self, atomic_fact: str) -> dict:
        return check_atomic_fact(
            atomic_fact,
            self.retriever,
            self.rater,
            self.tokenizer,
            self.max_steps,
            self.max_retries,
            self.num_searches,
        )
    
    def __str__(self):
        return (
            "SAFE claim evaluator with rater: "
            + str(self.rater)
            + " max_steps: "
            + str(self.max_steps)
            + " max_retries: "
            + str(self.max_retries)
            + " num_searches: "
            + str(self.num_searches)
        )


def call_retriever(retriever, search_query: str, num_searches:int = 3, search_postamble: str = "") -> str: # ex: 'site:https://en.wikipedia.org'
    """Call Google Search to get the search result."""
    search_query += f" {search_postamble}" if search_postamble else ""
    search_docs = retriever.search(search_query, num=num_searches)
    search_docs_txt = '. '.join([doc['contents'] for doc in search_docs])
    
    return search_docs_txt


def maybe_get_next_search(
    atomic_fact: str,
    past_searches: list[RetrievalResult],
    retriever: None,
    model: Union[PreTrainedModel, str],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    num_searches: int = 3,
    search_postamble: str = "",  # ex: 'site:https://en.wikipedia.org'
    **kwargs,
) -> Union[RetrievalResult, None]:
    
    """Get the next query from the model."""
    knowledge = "\n".join([s.result for s in past_searches])
    knowledge = "N/A" if not knowledge else knowledge
    full_prompt = _NEXT_SEARCH_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
    full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
    full_prompt = utils.strip_string(full_prompt)
    model_response = _generate(full_prompt, model, tokenizer, **kwargs)
    query = utils.extract_first_code_block(model_response, ignore_language=True)
    # print(f'Search query: {query}')

    if model_response and query:
        return RetrievalResult(
            query=query,
            result=call_retriever(
                retriever=retriever,
                search_query=query,
                num_searches=num_searches,
                search_postamble=search_postamble,
            ),
        )

    return None

def maybe_get_final_answer(
    atomic_fact: str,
    searches: list[RetrievalResult],
    model: Union[PreTrainedModel, str],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    **kwargs,
) -> Union[FinalAnswer, None]:
    """Get the final answer from the model."""
    knowledge = "\n".join([search.result for search in searches])
    full_prompt = _FINAL_ANSWER_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
    full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
    full_prompt = utils.strip_string(full_prompt)
    model_response = _generate(full_prompt, model, tokenizer, **kwargs)
    answer = utils.extract_first_square_brackets(model_response)
    answer = re.sub(r"[^\w\s]", "", answer).strip()

    if model_response and answer in [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL]:
        return FinalAnswer(response=model_response, answer=answer)

    return None


def check_atomic_fact(
    atomic_fact: str,
    retriever: None,
    rater: Union[PreTrainedModel, str],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    max_steps: int = 5,
    max_retries: int = 10,
    num_searches: int = 3,
    search_postamble: str = "",  # ex: 'site:https://en.wikipedia.org'
    **kwargs,
) -> tuple[Union[FinalAnswer, None], dict[str, Any]]:
    """Check if the given atomic fact is supported."""
    search_results = []

    for i in range(max_steps):
        next_search, num_tries = None, 0

        while not next_search and num_tries <= max_retries:
            # print(f'Step {i} Search trial #{num_tries}')s
            
            next_search = maybe_get_next_search(
                atomic_fact=atomic_fact,
                past_searches=search_results,
                retriever=retriever,
                model=rater,
                tokenizer=tokenizer,
                num_searches=num_searches,
                search_postamble=search_postamble,
                **kwargs,
            )
            num_tries += 1

        if next_search is None:
            utils.maybe_print_error("Unsuccessful parsing for `next_search`")
            break
        else:
            search_results.append(next_search)

    search_dicts = {"google_searches": [
        dataclasses.asdict(s) for s in search_results]}
    final_answer, num_tries = None, 0

    while not final_answer and num_tries <= max_retries:
        num_tries += 1
        final_answer = maybe_get_final_answer(
            atomic_fact=atomic_fact,
            searches=search_results,
            model=rater,
            tokenizer=tokenizer,
            **kwargs,
        )

    if final_answer is None:
        utils.maybe_print_error("Unsuccessful parsing for `final_answer`")
        return {"answer": None, "response": None, "search_details": None}

    return {
        "answer": final_answer.answer,
        "response": final_answer.response,
        "search_details": search_dicts,
    }

