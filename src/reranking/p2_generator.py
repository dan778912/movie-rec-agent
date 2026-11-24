"""Load a P2 LoRA adapter and generate one structured agent action."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .p2_training_data import (
    P2_TASKS,
    P2TrainingDataError,
    parse_completion,
)


class P2GenerationError(RuntimeError):
    """Raised when P2 inference cannot safely produce an action."""


def validate_generated_action(
    text: str,
    *,
    expected_action: str | None = None,
) -> dict[str, Any]:
    """Strictly validate a raw generated JSON action."""
    try:
        action = parse_completion(text)
    except P2TrainingDataError as error:
        raise P2GenerationError(
            f"Generated output is not a valid P2 action: {error}"
        ) from error

    if (
        expected_action is not None
        and expected_action not in P2_TASKS
    ):
        raise ValueError(
            f"Unknown expected action: {expected_action!r}"
        )

    if (
        expected_action is not None
        and action["action"] != expected_action
    ):
        raise P2GenerationError(
            f"Expected action {expected_action!r}, got "
            f"{action['action']!r}"
        )

    return action


class P2ActionGenerator:
    """Service-B-compatible P2 model client.

    The public ``generate(prompt) -> str`` method preserves the raw model
    output. Validation is separate so callers can record invalid attempts
    before applying the frozen fallback policy.
    """

    def __init__(
        self,
        adapter_path: str | Path,
        *,
        base_model: str,
        model_revision: str,
        max_input_tokens: int = 6144,
        max_new_tokens: int = 1024,
        allow_cpu: bool = False,
    ) -> None:
        if not isinstance(base_model, str) or not base_model:
            raise ValueError("base_model must be nonempty")

        if (
            not isinstance(model_revision, str)
            or not model_revision
        ):
            raise ValueError("model_revision must be nonempty")

        if (
            type(max_input_tokens) is not int
            or max_input_tokens <= 0
        ):
            raise ValueError(
                "max_input_tokens must be a positive integer"
            )

        if (
            type(max_new_tokens) is not int
            or max_new_tokens <= 0
        ):
            raise ValueError(
                "max_new_tokens must be a positive integer"
            )

        adapter_path = Path(adapter_path)

        if not adapter_path.is_dir():
            raise FileNotFoundError(adapter_path)

        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as error:
            raise P2GenerationError(
                "P2 dependencies are missing. Install "
                "requirements-reranking.txt."
            ) from error

        if not torch.cuda.is_available() and not allow_cpu:
            raise P2GenerationError(
                "CUDA GPU not detected. P2 inference should run on "
                "GPU hardware. Use allow_cpu only for an intentional "
                "slow diagnostic."
            )

        peft_config = PeftConfig.from_pretrained(adapter_path)

        if peft_config.base_model_name_or_path != base_model:
            raise P2GenerationError(
                "Adapter base model does not match the requested "
                f"base model: "
                f"{peft_config.base_model_name_or_path!r} != "
                f"{base_model!r}"
            )

        self.torch = torch
        self.adapter_path = adapter_path
        self.base_model = base_model
        self.model_revision = model_revision
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_bfloat16 = (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
        model_dtype = (
            torch.bfloat16
            if use_bfloat16
            else torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )
        self.model_dtype = model_dtype

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            revision=model_revision,
            dtype=model_dtype,
        )
        self.model = PeftModel.from_pretrained(
            base,
            adapter_path,
        )
        self.model.to(self.device)
        self.model.eval()

        # The upstream generation configuration contains sampling
        # defaults. Clear them because P2 uses deterministic greedy
        # decoding.
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.model.generation_config.num_beams = 1

    def count_input_tokens(self, prompt: str) -> int:
        """Count the exact model input without truncation."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")

        input_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        return len(input_ids)

    def generate(self, prompt: str) -> str:
        """Generate one raw JSON action using greedy decoding."""
        input_tokens = self.count_input_tokens(prompt)

        if input_tokens > self.max_input_tokens:
            raise P2GenerationError(
                f"Prompt has {input_tokens} tokens, exceeding "
                f"max_input_tokens={self.max_input_tokens}. "
                "The prompt was not truncated."
            )

        encoded = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=False,
        )
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = encoded["input_ids"].shape[1]
        new_tokens = generated[0, prompt_length:]

        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

    def settings(self) -> dict[str, Any]:
        """Return immutable loading and decoding settings."""
        return {
            "adapter_path": str(self.adapter_path),
            "base_model": self.base_model,
            "model_revision": self.model_revision,
            "device": str(self.device),
            "dtype": str(self.model_dtype),
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
            "decoding": "greedy",
            "chat_template_applied": False,
            "truncation_allowed": False,
        }