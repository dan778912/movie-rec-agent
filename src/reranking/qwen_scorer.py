"""Qwen LoRA scorer used for official P1 inference."""

from __future__ import annotations

from pathlib import Path

from .prompts import build_explanation_prompt, build_ranking_prompt


class QwenLoRAScorer:
    """Score candidates and generate explanations with a LoRA adapter."""

    def __init__(
        self,
        adapter_path: str | Path,
        max_length: int = 1024,
    ) -> None:
        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Reranking dependencies are missing. Install "
                "requirements-reranking.txt."
            ) from error

        self.torch = torch
        self.max_length = max_length
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        adapter_path = Path(adapter_path)
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model_name = peft_config.base_model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            else torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
        )
        self.model.to(self.device)
        self.model.eval()

    def _label_log_probability(
        self,
        prompt: str,
        label: str,
    ) -> float:
        torch = self.torch

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
        )["input_ids"]
        label_ids = self.tokenizer(
            label,
            add_special_tokens=False,
        )["input_ids"]

        available_prompt = self.max_length - len(label_ids)
        if available_prompt <= 0:
            raise ValueError("Label exceeds the configured maximum length")

        prompt_ids = prompt_ids[-available_prompt:]
        full_ids = prompt_ids + label_ids

        input_tensor = torch.tensor(
            [full_ids],
            dtype=torch.long,
            device=self.device,
        )

        with torch.inference_mode():
            logits = self.model(input_ids=input_tensor).logits

        log_probabilities = torch.log_softmax(logits, dim=-1)
        start = len(prompt_ids) - 1
        token_scores = []

        for offset, token_id in enumerate(label_ids):
            token_scores.append(
                log_probabilities[0, start + offset, token_id]
            )

        return torch.stack(token_scores).mean().item()

    def score(self, profile: str, candidate_document: str) -> float:
        """Return the relevant-versus-not-relevant log-probability margin."""
        prompt = build_ranking_prompt(profile, candidate_document)

        relevant_score = self._label_log_probability(
            prompt,
            "relevant",
        )
        not_relevant_score = self._label_log_probability(
            prompt,
            "not_relevant",
        )

        return relevant_score - not_relevant_score

    def explain(self, profile: str, candidate_document: str) -> str:
        """Generate a short explanation from only supplied evidence."""
        torch = self.torch
        prompt = build_explanation_prompt(profile, candidate_document)

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        explanation = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        if not explanation:
            return (
                "This candidate was selected based on the supplied user "
                "profile and movie metadata."
            )

        return explanation