import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import glob
import torch
import argparse
import transformers
from tqdm import tqdm
from datasets import load_dataset
from accelerate import Accelerator


from utils.general_utils import set_seed
from modules.generation.src.generation_models import *
from modules.generation.src.decomposition_methods import *
from modules.generation.src.claim_evaluation_methods.safe import ClaimEvaluator as SafeClaimEvaluator
from modules.generation.src.claim_evaluation_methods.rafe import ClaimEvaluator as RafeClaimEvaluator

import os
os.environ["OPENAI_API_KEY"] = 'your_open_ai_key' #to use openai models
os.environ['SERPER_API_KEY'] = 'your_serper_key' #for long form generation evaluation: https://serper.dev/

def generation(args):
    # === MultiGPU setup ========================
    accelerator = Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        print("\n== RAG Generation ...")
        print(f"""
            Model name:  {args.model_name_or_path}
            Dataset:     {args.dataset} ({args.fraction_of_data_to_use})
            Gen. Model:  {args.generation_model}
            Retriever:   {args.retriever_name}
            Seed:        {args.seed}
            Run:         {args.run}
        """.replace('        ', ''))
        # --- Define CUDA device
        if torch.cuda.is_available():
            print(f"Number of available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. No GPUs detected.")
        print('\n')

    # === Dataset ===============================
    data_path = f"data/processed_testset/{args.dataset}.jsonl"
    test_dataset_ = load_dataset("json", data_files=data_path, split="train")
    
    # ====== Subsampling the dataset ============
    if args.fraction_of_data_to_use < 1.0:
        shuffled_dataset = test_dataset_.shuffle(seed=args.seed)
        num_samples = int(args.fraction_of_data_to_use * len(shuffled_dataset))
        test_dataset = shuffled_dataset.select(range(num_samples))
    elif args.fraction_of_data_to_use > 1.0:
        shuffled_dataset = test_dataset_.shuffle(seed=args.seed)
        test_dataset = shuffled_dataset.select(range(int(args.fraction_of_data_to_use)))
    else:
        test_dataset = test_dataset_

    print(f"Final dataset size: {len(test_dataset)}")
    print(test_dataset[0])
    
    # === Read existing data ===================
    generated_qids = []
    if os.path.exists(args.generation_results_file):
        with open(args.generation_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    generated_qids.append(data['qid'])
    generated_qids = set(generated_qids)
    filtered_dataset = test_dataset.filter(lambda example: example['qid'] not in generated_qids)
    
    
    # === Generation Model ======================
    backbone_model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=torch.bfloat16).to(device) # attn_implementation="eager"
    backbone_tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.generation_model == 'no_retrieval':
        generation_model = NoRetrieval(backbone_model, backbone_tokenizer, device, args)
    elif args.generation_model == 'single_retrieval':
        generation_model = SingleRetrieval(backbone_model, backbone_tokenizer, device, args)
    else:
        raise NotImplementedError

    # === Evaluation Model ======================
    decomposition_method = StructuredDecompositionLocal(backbone_model, backbone_tokenizer, decomposition_depth=1) #Utilize HF models to decompose text
    if args.claim_evaluation_method == "SAFE":
        # claim_evaluator = ClaimEvaluator(rater='gpt-4o-mini', tokenizer = None, max_steps = 5, max_retries = 10, num_searches = 3)
        claim_evaluator = SafeClaimEvaluator(rater=backbone_model, tokenizer = backbone_tokenizer, max_steps = 5, max_retries = 10, num_searches = 3)
    elif args.claim_evaluation_method == "RAFE":
        claim_evaluator = RafeClaimEvaluator(rater=backbone_model, tokenizer = backbone_tokenizer, args=args, max_steps = 5, max_retries = 10, num_searches = 3)
    else:
        raise NotImplementedError

    # === Inference =============================
    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(filtered_dataset) as test_dataset_shard:
        generation_results_file_ranked = f"{args.output_dir}/generation_results_rank{accelerator.process_index}.jsonl"
        with open(generation_results_file_ranked, 'w') as res_f:
            for i, sample in enumerate(tqdm(test_dataset_shard, desc=f"[Rank {accelerator.process_index}]")):
                if i == 1:
                    break
                qid, prompt = sample['qid'], sample['prompt']
                prompt = prompt.strip()
                
                # Step 1: Generate response
                generated_text = generation_model.inference(prompt)
                
                # Step 2: Get claims, Factual Decomposition
                claims = decomposition_method(generated_text)
                
                # Step 3: Assign label to each claim
                # claims_label = [claim_evaluator(atomic_fact=claim) for claim in claims]
                # print(claims_label)
                
                print(len(claims))
                claims_label = []
                for idx, claim in enumerate(claims):
                    if idx == 5:
                        break
                    print(f"{idx} -> {claim}")
                    claim_evaluation_obj = claim_evaluator(atomic_fact=claim)
                    claims_label.append(claim_evaluation_obj['answer']) 
                    print(f'Label: {claim_evaluation_obj['answer']}')
                
                item = {
                    "qid": qid,
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "claims": claims,
                    "claims_label": claims_label
                }
                res_f.write(json.dumps(item) + '\n')


def merge_result_files(args):
    results_shard_files = f"{args.output_dir}/generation_results_rank*.jsonl"
    results_shard_files = sorted(glob.glob(results_shard_files))
    
    with open(args.generation_results_file, "a") as fout:
        for shard_file in results_shard_files:
            if shard_file == args.generation_results_file:
                continue
            with open(shard_file, "r") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(shard_file)
            print(f"Deleted shard file: {shard_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--decomposition_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--dataset', type=str, default='factscore_bio', choices=[
        'factscore_bio', 'longfact_concepts', 'longfact_objects'
    ])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=50.0)
    
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='bge', choices=[
        'bm25', 'rerank_l6', 'rerank_l12', 'contriever', 'dpr', 'e5', 'bge'
    ])
    parser.add_argument('--data_dir', type=str, default='data/corpus')
    parser.add_argument('--corpus_path', type=str, default='data/corpus/wiki-18_100.jsonl')
    parser.add_argument('--retrieval_topk', type=int, default=3)
    parser.add_argument('--faiss_gpu', action='store_false', help='Use GPU for computation')
    parser.add_argument('--retrieval_query_max_length', type=int, default=64)
    parser.add_argument('--retrieval_use_fp16', action='store_false', help='')
    parser.add_argument('--retrieval_batch_size', type=int, default=512)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    
    parser.add_argument('--generation_model', type=str, default='single_retrieval', choices=[
        'no_retrieval', 'single_retrieval'
    ])
    parser.add_argument('--claim_evaluation_method', type=str, default='RAFE', choices=[
        'SAFE', 'RAFE', 'FactScore', 'ICAT'
    ])
    
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_1')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    args = parser.parse_args()
    
    
    # === Files ====================
    model_ = args.model_name_or_path.split('/')[-1]
    if args.generation_model in ['no_retrieval']:
        args.output_dir = f"run_output/{args.run}/{model_}/{args.dataset}/{args.generation_model}"
    else:
        args.output_dir = f"run_output/{args.run}/{model_}/{args.dataset}/{args.generation_model}_{args.retriever_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    args.generation_results_file = f"{args.output_dir}/generation_results.jsonl"
    
    set_seed(args.seed)
    generation(args)
    # merge_result_files(args)
    
    # python modules/generation/run_module.py
    # accelerate launch --multi_gpu modules/generation/run_module.py