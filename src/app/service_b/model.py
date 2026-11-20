"""A persistent inference subprocess, terminated on timeout rather than abandoned GPU work."""
from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import re
import threading
import time
from pathlib import Path

from src.retrieval.online import canonical, file_hash


def checkpoint_identity(path):
    path = Path(path)
    required = ("adapter_config.json", "tokenizer_config.json")
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        raise ValueError("Checkpoint needs adapter_config.json, tokenizer_config.json, adapter weights, and tokenizer files")
    if not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError("Missing LoRA weights")
    files = {p.relative_to(path).as_posix(): file_hash(p) for p in sorted(path.rglob("*")) if p.is_file()}
    return {"sha256": hashlib.sha256(canonical(files)).hexdigest(), "files": files}


def _worker(connection, settings):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel, PeftConfig
        device = settings.get("device", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use CPU only for an explicitly configured smoke run")
        torch.manual_seed(settings["seed"])
        tokenizer = AutoTokenizer.from_pretrained(settings["adapter_path"], trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        adapter = PeftConfig.from_pretrained(settings["adapter_path"])
        if adapter.base_model_name_or_path != settings["base_model"]:
            raise ValueError("Adapter base model disagrees with configured model")
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if device == "cuda" else torch.float32)
        base = AutoModelForCausalLM.from_pretrained(settings["base_model"],
            revision=settings["base_revision"], dtype=dtype, trust_remote_code=False)
        model = PeftModel.from_pretrained(base, settings["adapter_path"]).to(device).eval()
        context_size = getattr(model.config, "max_position_embeddings", tokenizer.model_max_length)
        if settings["max_input_tokens"] + settings["max_new_tokens"] > context_size:
            raise ValueError("Configured input + output exceeds model context window")
        connection.send(("ready", {"device": str(device), "dtype": str(dtype)}))
        while True:
            prompt = connection.recv()
            if prompt is None:
                break
            try:
                encoded = tokenizer(prompt, return_tensors="pt", truncation=False, return_token_type_ids=False)
                if encoded["input_ids"].shape[1] > settings["max_input_tokens"]:
                    raise ValueError("Prompt exceeds configured token budget; truncation is disabled")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                with torch.inference_mode():
                    output = model.generate(**encoded, do_sample=False,
                        max_new_tokens=settings["max_new_tokens"],
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                text = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
                connection.send(("ok", text))
            except Exception as error:
                connection.send(("error", f"{type(error).__name__}: {error}"))
    except BaseException as error:
        try:
            connection.send(("error", f"{type(error).__name__}: {error}"))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class ProcessModel:
    def __init__(self, settings, *, worker=_worker):
        self.settings = dict(settings)
        if not re.fullmatch(r"[0-9a-f]{40}", settings.get("base_revision", "")):
            raise ValueError("Base model revision must be pinned to a full Git ID")
        for key in ("max_input_tokens", "max_new_tokens", "call_timeout_seconds", "startup_timeout_seconds"):
            if (type(settings.get(key)) not in (int, float) or not math.isfinite(settings[key]) or settings[key] <= 0):
                raise ValueError(f"{key} must be positive")
        for key in ("max_input_tokens", "max_new_tokens", "seed"):
            if type(settings.get(key)) is not int or settings[key] < (0 if key == "seed" else 1):
                raise ValueError(f"{key} must be a valid integer")
        self.identity = checkpoint_identity(settings["adapter_path"])
        self.deadline = None
        self.lock = threading.Lock()
        self._worker = worker
        self._loaded = False
        self._closed = False
        self.restart_count = 0
        self.max_restarts = settings.get("max_worker_restarts", 2)
        if type(self.max_restarts) is not int or self.max_restarts < 0:
            raise ValueError("max_worker_restarts must be a nonnegative integer")
        self._start()

    def _start(self):
        self._loaded = False
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=self._worker, args=(child, self.settings), daemon=True)
        self.process.start()
        child.close()
        try:
            if not self.connection.poll(self.settings["startup_timeout_seconds"]):
                raise TimeoutError("Checkpoint startup timed out")
            status, info = self.connection.recv()
            if status != "ready":
                raise RuntimeError(info)
            self.runtime = info
            self._loaded = True
        except BaseException:
            self._stop()
            raise

    @property
    def ready(self):
        return self._loaded and not self._closed and self.process.is_alive()

    def recover(self):
        """Restore only future requests; never retry a failed action invisibly."""
        with self.lock:
            if self._closed or self.ready:
                return False
            if self.restart_count >= self.max_restarts:
                raise RuntimeError("Inference restart budget exhausted; operator restart required")
            if checkpoint_identity(self.settings["adapter_path"]) != self.identity:
                raise RuntimeError("Checkpoint changed; refusing recovery with different weights")
            self.restart_count += 1
            self._stop()
            self._start()
            return True

    def generate(self, prompt):
        with self.lock:
            if not self.ready:
                raise RuntimeError("Inference worker unavailable")
            timeout = self.settings["call_timeout_seconds"]
            if self.deadline is not None:
                timeout = min(timeout, self.deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("Request deadline exhausted")
            self.connection.send(prompt)
            if not self.connection.poll(timeout):
                self._stop()
                raise TimeoutError("Inference timeout; worker terminated")
            try:
                status, value = self.connection.recv()
            except EOFError as error:
                raise RuntimeError("Inference worker exited") from error
            if status != "ok":
                raise RuntimeError(value)
            return value

    def _stop(self):
        self._loaded = False
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
        self.connection.close()

    def close(self):
        with self.lock:
            self._closed = True
            self._stop()
