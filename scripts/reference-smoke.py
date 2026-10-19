"""Offline full-logit reference for a large local checkpoint; never used by serving.

python scripts/reference-smoke.py models/qwen2.5-32b-instruct .cache/qwen32-reference
Requires enough host RAM for FP32 weights. CPU eager attention, full recomputation.
"""
import json
import sys
import time
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

source, destination = Path(sys.argv[1]), Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(16)
tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
started = time.monotonic()
model = AutoModelForCausalLM.from_pretrained(
    source, dtype=torch.float32, attn_implementation='eager', local_files_only=True,
).eval().to('cpu')
load_seconds = time.monotonic() - started
print(json.dumps({'phase': 'loaded', 'seconds': load_seconds}), flush=True)
records = []
with torch.inference_mode():
    for index, prompt in enumerate(['1+1=', '你好']):
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        history, output, files = tokens[:], [], []
        for step in range(2):
            before = time.monotonic()
            logits = model(torch.tensor([history]), use_cache=False).logits[0, -1].float()
            filename = f'case-{index}-step-{step}.f32'
            logits.numpy().astype('<f4').tofile(destination / filename)
            token = int(logits.argmax())
            output.append(token); history.append(token); files.append(filename)
            print(json.dumps({'case': index, 'step': step, 'token': token, 'seconds': time.monotonic()-before}), flush=True)
        records.append({'prompt': prompt, 'tokens': tokens, 'output': output, 'logits': files,
                        'text': tokenizer.decode(output, skip_special_tokens=True)})
manifest = json.loads((source / 'sprout-model.json').read_text())
report = {'modelId': manifest['repo'] + '@' + manifest['revision'],
          'torch': torch.__version__, 'transformers': transformers.__version__,
          'dtype': 'float32', 'device': 'cpu', 'attention': 'eager', 'threads': 16,
          'loadSeconds': load_seconds, 'totalSeconds': time.monotonic()-started, 'cases': records}
(destination / 'reference.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(report, ensure_ascii=False), flush=True)
