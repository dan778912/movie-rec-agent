import tempfile
import time
import unittest
from pathlib import Path

from src.app.service_b.model import ProcessModel


def stalled_worker(connection, settings):
    connection.send(("ready", {"device": "synthetic"}))
    connection.recv()
    time.sleep(20)


def recoverable_worker(connection, settings):
    connection.send(("ready", {"device": "synthetic"}))
    connection.recv()
    marker = Path(settings["marker"])
    if not marker.exists():
        marker.touch()
        time.sleep(20)
    else:
        connection.send(("ok", "recovered"))
        connection.recv()


class ModelLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from peft import LoraConfig, get_peft_model
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        base_path, adapter_path = root / "base", root / "adapter"
        torch.manual_seed(0)
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=5, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0))
        base.save_pretrained(base_path)
        base.name_or_path = str(base_path)
        adapter = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", target_modules=["q_proj"], r=2))
        adapter.save_pretrained(adapter_path)
        tok = Tokenizer(WordLevel({"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "hello": 4}, unk_token="<unk>"))
        tok.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<bos>",
                                eos_token="<eos>", unk_token="<unk>").save_pretrained(adapter_path)
        cls.settings = {"adapter_path": str(adapter_path), "base_model": str(base_path), "base_revision": "a" * 40,
            "seed": 0, "device": "cpu", "max_input_tokens": 8, "max_new_tokens": 4,
            "call_timeout_seconds": 10, "startup_timeout_seconds": 60}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_tiny_qwen_lora_load_generate_and_reject_overlong_prompt(self):
        model = ProcessModel(self.settings)
        self.addCleanup(model.close)
        self.assertTrue(model.ready)
        self.assertIsInstance(model.generate("hello"), str)
        with self.assertRaisesRegex(RuntimeError, "truncation is disabled"):
            model.generate("hello " * 12)
        self.assertTrue(model.ready)
        self.assertEqual(len(model.identity["sha256"]), 64)

    def test_timeout_terminates_worker_and_marks_unavailable(self):
        model = ProcessModel({**self.settings, "call_timeout_seconds": .1}, worker=stalled_worker)
        self.addCleanup(model.close)
        with self.assertRaises(TimeoutError):
            model.generate("hello")
        self.assertFalse(model.ready)

    def test_timeout_recovery_reloads_same_checkpoint_with_bounded_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = ProcessModel({**self.settings, "call_timeout_seconds": .2,
                                  "marker": str(Path(tmp) / "failed"), "max_worker_restarts": 1}, worker=recoverable_worker)
            self.addCleanup(model.close)
            with self.assertRaises(TimeoutError):
                model.generate("first")
            self.assertTrue(model.recover())
            self.assertTrue(model.ready)
            self.assertEqual(model.generate("second"), "recovered")
            with model.lock:
                model._stop()
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                model.recover()


if __name__ == "__main__":
    unittest.main()
