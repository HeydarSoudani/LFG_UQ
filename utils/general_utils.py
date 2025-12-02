import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import copy
import torch
import random
import numpy as np
from typing import Union
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def passages2string(retrieval_result):
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        if 'title' in doc_item:
            text = doc_item['contents']
            title = doc_item['title']
        else:
            content = doc_item['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
        
        format_reference += f"Doc {idx+1} (Title: {title}) {text}\n"
    return format_reference    

    
def check_system_prompt_support(tokenizer):
    chat = [{"role": "system", "content": "Test"}]
    try:
        tokenizer.apply_chat_template(chat, tokenize=False)
        return True
    except:
        return False

def fix_tokenizer_chat(tokenizer, chat):
    # tokenizer = copy.deepcopy(tokenizer)
    chat = copy.deepcopy(chat)
    if tokenizer.chat_template == None:
        tokenizer.chat_template = """{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{ message['content'].strip() + '\n' }}
    {%- elif message['role'] == 'system' %}
        {{ message['content'].strip() + '\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{ message['content'].strip() + '\n' }}
    {%- endif %}
{%- endfor %}""".strip()
    else:
        if check_system_prompt_support(tokenizer) == False:
            # replace system prompt with the next user prompt
            for i in range(len(chat)):
                if chat[i]["role"] == "system":
                    try:
                        if chat[i + 1]["role"] == "user":
                            chat[i]["role"] = "user"
                            chat[i]["content"] = (
                                chat[i]["content"] + " " +
                                chat[i + 1]["content"]
                            )
                            chat[i + 1]["role"] = "popped"
                        else:
                            chat[i]["role"] = "user"

                    except:
                        chat[i]["role"] = "user"
            # remove popped elements
            chat = [chat[i]
                    for i in range(len(chat)) if chat[i]["role"] != "popped"]

    return tokenizer, chat


def generate(
    text: str,
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    **kwargs
) -> dict:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        model_output = model.generate(**inputs, **kwargs, max_new_tokens=2048)
        tokens = model_output[0][len(inputs["input_ids"][0]):]
        generated_text = tokenizer.decode(tokens, skip_special_tokens=False)
        generated_text_return = tokenizer.decode(tokens, skip_special_tokens=True)

    return {
        "generated_text_skip_specials": generated_text_return,
        "generated_text": generated_text,
        "tokens": tokens,
        "all_ids": model_output,
    }

