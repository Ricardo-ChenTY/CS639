from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
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
    ) -> list[dict[str, Any]]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        do_sample = temperature > 0.0 or n > 1
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": max(temperature, 1e-5) if do_sample else None,
            "num_return_sequences": n,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        latency_sec = time.perf_counter() - start

        if n == 1:
            outputs = outputs.unsqueeze(0) if outputs.dim() == 1 else outputs

        results = []
        for sequence in outputs:
            generated_ids = sequence[prompt_tokens:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(
                {
                    "text": text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": int(generated_ids.shape[-1]),
                    "latency_sec": latency_sec / max(n, 1),
                }
            )
        return results


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
