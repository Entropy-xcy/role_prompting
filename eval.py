from lm_eval import tasks, evaluator, utils
from lm_eval.base import BaseLM
from typing import List, Any, Tuple

"""
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag
"""


class RolePromptingLM(BaseLM):
    def __init__(self):
        super().__init__()
        self.model = "role_prompting"

    def generate_until(self, requests) -> List[str]:
        print(type(requests))
        print(requests[0])
        raise NotImplementedError("No support for generation.")

    def loglikelihood(self, requests):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("No support for logits.")

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()

    @property
    def max_length(self):
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        raise NotImplementedError("No idea about anthropic tokenization.")

    def tok_decode(self, tokens):
        raise NotImplementedError("No idea about anthropic tokenization.")

    @property
    def eot_token_id(self):
        raise NotImplementedError("No idea about anthropic tokenization.")


model = RolePromptingLM()

results = evaluator.simple_evaluate(
    model=model,
    model_args="engine=davinci",
    tasks="coqa",
)
