"""A real, download-free OpenRLHF/Qwen/LoRA CUDA smoke and resume check."""
from pathlib import Path
from tempfile import TemporaryDirectory


def check():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from .assets import check_runtime
    from .common import DEFAULT_CONFIG, ENGINE_ID, load_config, seed_all
    from .models import Policy
    from .ppo import PPOTrainer, load_checkpoint, save_checkpoint

    config = load_config(DEFAULT_CONFIG)
    runtime = check_runtime(config)
    seed_all(42)
    config['ppo'].update(lora_r=2, lora_alpha=4, minibatch_size=2,
                         learning_rate=.001, value_learning_rate=.001,
                         epochs=2, target_update_kl=1.)
    config['runtime']['ppo_microbatch_size'] = 1
    with TemporaryDirectory(prefix='workshop-openrlhf-check-') as directory:
        root = Path(directory)
        model_path = root/'model'
        vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3,
                 'one': 4, 'two': 5, 'three': 6, 'four': 7}
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token='<unk>')),
            pad_token='<pad>', unk_token='<unk>', bos_token='<s>', eos_token='</s>')
        tokenizer.save_pretrained(model_path)
        Qwen2ForCausalLM(Qwen2Config(vocab_size=8, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, bos_token_id=2, eos_token_id=3, pad_token_id=0,
            attention_dropout=0.)).save_pretrained(model_path)
        config['models']['policy'] = str(model_path)
        policy = Policy(config, {'policy': 'main'})
        initial = policy.trainable_state()
        trainer = PPOTrainer(policy, config)
        rows = [{'prompt_ids': [2, 4, 5], 'response_ids': [6, 3]},
                {'prompt_ids': [2, 7], 'response_ids': [5, 7, 3]}]
        result = trainer.update(rows, [1., -1.], 0)
        after = policy.trainable_state()
        if not any(not torch.equal(v, after['adapter'][k]) for k, v in initial['adapter'].items()):
            raise RuntimeError('The OpenRLHF smoke update did not change the actor.')
        if torch.equal(initial['value']['weight'], after['value']['weight']):
            raise RuntimeError('The OpenRLHF smoke update did not change the value head.')
        path = root/'checkpoint.pt'
        save_checkpoint(path, policy, trainer.optimizer, 1, 'backend-check', 'ridge')
        trainer.update(rows, [-.5, .8], 1)
        uninterrupted = policy.trainable_state()
        load_checkpoint(path, policy, trainer.optimizer, 'backend-check', 'ridge')
        trainer.update(rows, [-.5, .8], 1)
        for part, values in policy.trainable_state().items():
            for name, value in values.items():
                torch.testing.assert_close(value, uninterrupted[part][name], rtol=0, atol=0)
    return {'ok': True, 'engine': ENGINE_ID, 'runtime': runtime,
            'optimizer_steps': result['optimizer_steps'], 'exact_resume': True,
            'pretrained_models_downloaded': 0}
