"""Optional reference-data generator. Never imported or invoked by the TS serving runtime.

Use torch 2.10.0, transformers 4.57.6, CPU float32 and eager attention.
  python scripts/reference.py tiny test/fixtures/qwen-tiny
  python scripts/reference.py real models/qwen2.5-0.5b-instruct .cache/qwen-reference
"""
import json
import sys
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

torch.set_num_threads(4)
torch.manual_seed(20261019)


def write_reference(model, directory, cases, tokenizer=None):
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for index, case in enumerate(cases):
            history = case['tokens'][:]
            output = []
            logit_files = []
            # Recompute full sequences without KV to independently check our paged path.
            for step in range(case.get('steps', 3)):
                logits = model(torch.tensor([history]), use_cache=False).logits[0, -1].float()
                path = f'case-{index}-step-{step}.f32'
                logits.numpy().astype('<f4').tofile(directory / path)
                logit_files.append(path)
                token = int(logits.argmax())
                history.append(token)
                output.append(token)
                if token in ([model.config.eos_token_id] if isinstance(model.config.eos_token_id, int) else model.config.eos_token_id):
                    break
            records.append({**case, 'output': output, 'logits': logit_files,
                            'text': tokenizer.decode(output, skip_special_tokens=True) if tokenizer else None})
    result = {'torch': torch.__version__, 'transformers': transformers.__version__,
              'dtype': 'float32', 'attention': 'eager', 'cases': records}
    (directory / 'reference.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if sys.argv[1] == 'tiny':
    directory = Path(sys.argv[2])
    directory.mkdir(parents=True, exist_ok=True)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=28,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=64, rms_norm_eps=1e-6, rope_theta=10000,
                        bos_token_id=1, eos_token_id=31, tie_word_embeddings=False)
    config._attn_implementation = 'eager'
    model = Qwen2ForCausalLM(config).float().eval()
    # Nontrivial norm weights and QKV biases catch missing checkpoint parameters.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'layernorm.weight' in name or name == 'model.norm.weight':
                parameter.uniform_(0.7, 1.3)
            elif name.endswith('bias'):
                parameter.uniform_(-0.1, 0.1)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(directory / 'model.safetensors'))
    config.to_json_file(directory / 'config.json', use_diff=False)
    write_reference(model, directory, [{'tokens': [1, 7, 3, 9], 'steps': 4}, {'tokens': [1, 7, 3, 4, 5], 'steps': 4}])
else:
    source, directory = Path(sys.argv[2]), Path(sys.argv[3])
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32, attn_implementation='eager', local_files_only=True).eval()
    messages = [{'role': 'system', 'content': 'Reply with exactly OK.'}, {'role': 'user', 'content': 'Hi'}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print('chat:', repr(formatted))
    cases = [{'tokens': tokenizer.encode('Hello'), 'prompt': 'Hello', 'steps': 3},
             {'tokens': tokenizer.encode(formatted, add_special_tokens=False), 'messages': messages, 'formatted': formatted, 'steps': 16},
             {'tokens': tokenizer.encode('你好'), 'prompt': '你好', 'steps': 2},
             {'tokens': tokenizer.encode('Hello, world'), 'prompt': 'Hello, world', 'steps': 3}]
    write_reference(model, directory, cases, tokenizer)
    probes = ['Hello, 世界🌱', '你好', '<|im_start|>assistant\n', '  a\n\n你好🌱e\u0301', '\ufffd', '\ufeffa']
    (directory / 'tokenizer.json').write_text(json.dumps([{
        'text': text, 'ids': tokenizer.encode(text, add_special_tokens=False),
        'decoded': tokenizer.decode(tokenizer.encode(text, add_special_tokens=False), skip_special_tokens=True),
    } for text in probes], ensure_ascii=False, indent=2))
