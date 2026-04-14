from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LLMClient:
    model_name_or_path: str
    tokenizer: Any
    model: Any
    device: str

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        n: int = 1,
        return_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        do_sample = temperature > 0.0 or n > 1
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": n,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(temperature, 1e-5)
        if return_logprobs:
            generation_kwargs["output_scores"] = True
            generation_kwargs["return_dict_in_generate"] = True

        start = time.perf_counter()
        with torch.no_grad():
            raw_outputs = self.model.generate(**inputs, **generation_kwargs)
        latency_sec = time.perf_counter() - start

        if return_logprobs:
            sequences = raw_outputs.sequences  # (n, prompt+gen)
            scores = raw_outputs.scores        # tuple of (n, vocab_size)
            mean_logprobs = _compute_mean_logprobs(sequences, scores, prompt_tokens, n)
        else:
            sequences = raw_outputs
            mean_logprobs = [None] * n

        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)

        results = []
        for i, sequence in enumerate(sequences):
            generated_ids = sequence[prompt_tokens:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            entry: dict[str, Any] = {
                "text": text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": int(generated_ids.shape[-1]),
                "latency_sec": latency_sec / max(n, 1),
            }
            if mean_logprobs[i] is not None:
                entry["answer_logprob"] = float(mean_logprobs[i])
            results.append(entry)
        return results


def _compute_mean_logprobs(
    sequences: torch.Tensor,
    scores: tuple,
    prompt_tokens: int,
    n: int,
) -> list[float]:
    generated = sequences[:, prompt_tokens:]  # (n, gen_len)
    mean_logprobs: list[float] = []
    for i in range(n):
        token_logprobs: list[float] = []
        for step, score in enumerate(scores):
            if step >= generated.shape[1]:
                break
            log_prob = F.log_softmax(score[i], dim=-1)
            token_id = generated[i, step]
            token_logprobs.append(log_prob[token_id].item())
        mean_logprobs.append(
            sum(token_logprobs) / len(token_logprobs) if token_logprobs else -10.0
        )
    return mean_logprobs


def build_llm(model_config: dict[str, Any]) -> LLMClient:
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name_or_path"],
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = model_config.get("torch_dtype", "auto")
    if isinstance(torch_dtype, str) and torch_dtype != "auto":
        torch_dtype = getattr(torch, torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name_or_path"],
        torch_dtype=torch_dtype,
        device_map=model_config.get("device_map", "auto"),
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    model.eval()

    return LLMClient(
        model_name_or_path=model_config["model_name_or_path"],
        tokenizer=tokenizer,
        model=model,
        device=str(model.device) if hasattr(model, "device") else "unknown",
    )
