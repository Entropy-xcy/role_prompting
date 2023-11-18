import argparse
import hashlib
import json
import logging
import os
import torch
from lm_eval import tasks, evaluator, utils
import transformers
from typing import List, Any, Tuple
from role_prompt_task import *
import argparse

parser = argparse.ArgumentParser()

# Role Related
parser.add_argument("--occupation", type=str, default="lawyer")
parser.add_argument("--education", type=str, default="Law Degree")
parser.add_argument("--gender", type=str, default="male")
parser.add_argument("--age", type=str, default="mid-aged")
parser.add_argument("--nationality", type=str, default="American")

# Model Related
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--model", type=str, default="huggyllama/llama-7b")
args = parser.parse_args()

model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
role = {
    "occupation": args.occupation,
    "education": args.education,
    "gender": args.gender,
    "age": args.age,
    "nationality": args.nationality 
}

task = TruthfulQAMultipleChoiceRole(role)

results = evaluator.simple_evaluate(
    model=model,
    batch_size=32,
    tasks=[task],
    limit=args.limit
)

# update the results dict with the role
results["role"] = role
# dump the results to a json file
data_md5 = hashlib.md5(json.dumps(results).encode()).hexdigest()

with open(f"results/{data_md5}.json", "w") as f:
    json.dump(results, f, indent=4)
