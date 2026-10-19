# Qwen2 correctness fixture

Generated locally with `scripts/reference.py tiny test/fixtures/qwen-tiny`, PyTorch 2.10.0 and Transformers 4.57.6, seed 20261019. These are random test weights, not a downloaded trained checkpoint.

The configuration uses 2 layers, hidden size 16, 4 query heads, 2 KV heads, untied output weights, nontrivial learned norm weights and QKV biases. `reference.json` records inputs and greedy output IDs; each `.f32` file contains the complete final-position logits in little-endian FP32, computed by Transformers with eager attention and no KV cache. The checkpoint is about 24 KiB.

The native implementation must match these independently computed logits within absolute error `2e-6`, including after paging, prefix reuse and preemption/recomputation. Tests load the files directly; no Python installation is needed for `npm test`.
