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

# Model Related
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--model", type=str, default="huggyllama/llama-7b")
args = parser.parse_args()

model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")

task = TruthfulQAMultipleChoice()

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

with open(f"baseline_results/{data_md5}.json", "w") as f:
    json.dump(results, f, indent=4)
