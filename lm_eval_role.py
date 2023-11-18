import argparse
import json
import logging
import os
import torch
from lm_eval import tasks, evaluator, utils
import transformers
from typing import List, Any, Tuple
from role_prompt_task import *


model = transformers.AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype=torch.float16, device_map="cuda:0")

results = evaluator.simple_evaluate(
    model=model,
    batch_size=8,
    tasks=["truthfulqa_mc", "truthfulqa_gen"],
)

print(results)
