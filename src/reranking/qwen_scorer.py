"""Qwen LoRA scorer used for official P1 inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .prompts import build_explanation_prompt, build_ranking_prompt


class QwenLoRAScorer:
    """Score candidates in GPU batches with a trained LoRA adapter."""

    def __init__(
        self,
        adapter_path: str | Path,
        max_length: int = 1024,
        batch_size: int = 2,
        generate_explanations: bool = False,
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

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.torch = torch
        self.max_length = max_length
        self.batch_size = batch_size
        self.generate_explanations = generate_explanations
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
            dtype=dtype,
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
        )
        self.model.to(self.device)
        self.model.eval()

    def _prepare_tasks(
        self,
        profile: str,
        candidate_documents: Sequence[str],
    ) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []

        for candidate_index, candidate_document in enumerate(
            candidate_documents
        ):
            prompt = build_ranking_prompt(profile, candidate_document)
            prompt_ids = self.tokenizer(
                prompt,
                add_special_tokens=True,
            )["input_ids"]

            for label, sign in (
                ("relevant", 1.0),
                ("not_relevant", -1.0),
            ):
                label_ids = self.tokenizer(
                    label,
                    add_special_tokens=False,
                )["input_ids"]

                available_prompt = self.max_length - len(label_ids)
                if available_prompt <= 0:
                    raise ValueError(
                        "Label exceeds the configured maximum length"
                    )

                truncated_prompt_ids = prompt_ids[-available_prompt:]
                tasks.append(
                    {
                        "candidate_index": candidate_index,
                        "sign": sign,
                        "prompt_length": len(truncated_prompt_ids),
                        "label_ids": label_ids,
                        "input_ids": truncated_prompt_ids + label_ids,
                    }
                )

        return tasks

    def _run_task_batch(
        self,
        tasks: Sequence[dict[str, Any]],
    ) -> list[tuple[int, float, float]]:
        torch = self.torch
        pad_token_id = self.tokenizer.pad_token_id
        maximum_length = max(len(task["input_ids"]) for task in tasks)

        padded_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []

        for task in tasks:
            input_ids = task["input_ids"]
            padding_length = maximum_length - len(input_ids)
            padded_ids.append(
                input_ids + ([pad_token_id] * padding_length)
            )
            attention_masks.append(
                ([1] * len(input_ids)) + ([0] * padding_length)
            )

        input_tensor = torch.tensor(
            padded_ids,
            dtype=torch.long,
            device=self.device,
        )
        attention_tensor = torch.tensor(
            attention_masks,
            dtype=torch.long,
            device=self.device,
        )

        with torch.inference_mode():
            logits = self.model(
                input_ids=input_tensor,
                attention_mask=attention_tensor,
                use_cache=False,
            ).logits

        results: list[tuple[int, float, float]] = []

        for row_index, task in enumerate(tasks):
            start = task["prompt_length"] - 1
            token_scores = []

            for offset, token_id in enumerate(task["label_ids"]):
                position_logits = logits[row_index, start + offset]
                token_log_probability = torch.log_softmax(
                    position_logits,
                    dim=-1,
                )[token_id]
                token_scores.append(token_log_probability)

            mean_score = torch.stack(token_scores).mean().item()
            results.append(
                (
                    task["candidate_index"],
                    task["sign"],
                    mean_score,
                )
            )

        return results

    def score_many(
        self,
        profile: str,
        candidate_documents: Sequence[str],
    ) -> list[float]:
        """Score all candidates for one user in GPU batches."""
        if not candidate_documents:
            return []

        tasks = self._prepare_tasks(profile, candidate_documents)
        margins = [0.0] * len(candidate_documents)

        for start in range(0, len(tasks), self.batch_size):
            batch = tasks[start : start + self.batch_size]

            for candidate_index, sign, score in self._run_task_batch(batch):
                margins[candidate_index] += sign * score

        return margins

    def score(self, profile: str, candidate_document: str) -> float:
        """Score one candidate while preserving the original interface."""
        return self.score_many(profile, [candidate_document])[0]

    def explain(self, profile: str, candidate_document: str) -> str:
        """Generate an optional short explanation from supplied evidence."""
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