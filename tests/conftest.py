"""Tiny local model fixtures; no downloads and no claims about task performance."""
import copy
import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
                          Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM)
from workshop.common import DEFAULT_CONFIG, load_config, seed_all

torch.set_num_threads(1)

@pytest.fixture
def tiny_assets(tmp_path):
    seed_all(42)
    vocabulary = {t: i for i, t in enumerate(["<pad>", "<unk>", "<s>", "</s>",
                  "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "user", "assistant", "system", "math", "answer", "good", "bad"])}
    raw = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>", bos_token="<s>", eos_token="</s>")
    tok.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' </s> ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    q2 = tmp_path / "qwen2"
    q3 = tmp_path / "qwen3"
    qm = tmp_path / "qwen3_moe"
    for folder, cls, cfg in ((q2, Qwen2ForCausalLM, Qwen2Config), (q3, Qwen3ForCausalLM, Qwen3Config), (qm, Qwen3MoeForCausalLM, Qwen3MoeConfig)):
        kw = dict(vocab_size=len(vocabulary), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                  bos_token_id=2, eos_token_id=3, pad_token_id=0, attention_dropout=0.0)
        if cfg in (Qwen3Config, Qwen3MoeConfig):
            kw["head_dim"] = 8
        if cfg is Qwen3MoeConfig:
            kw.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16)
        cls(cfg(**kw)).save_pretrained(folder)
        tok.save_pretrained(folder)
    config = copy.deepcopy(load_config(DEFAULT_CONFIG))
    config["models"] = {"policy": str(q2), "proxy": str(q2), "judge": str(q3), "judge30b": str(qm)}
    config["runtime"].update(device="cpu", dtype="float32", attention="eager")
    config["generation"].update(max_new_tokens=8, batch_size=2)
    config["scoring"].update(mode="expected_digit", batch_size=2)
    config["ppo"].update(lora_r=2, lora_alpha=4, learning_rate=.001, value_learning_rate=.001,
                         minibatch_size=2, epochs=2, target_update_kl=1)
    return config, {"policy": "main", "proxy": "main", "judge": "main", "judge30b": "main"}


def items():
    return [{"id": "a", "question": "1 2", "reference": "#### 3", "response": "3", "prompt_ids": [2, 4, 5], "response_ids": [6, 3]},
            {"id": "b", "question": "4 5", "reference": "#### 9", "response": "2", "prompt_ids": [2, 7], "response_ids": [5, 7, 3]}]
