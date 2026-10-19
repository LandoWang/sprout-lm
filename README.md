# SproutLM

用纯 TypeScript 拆解 LLM 推理引擎的核心机制：连续批处理、分块 prefill、分页 KV cache、前缀复用、内存压力下的抢占与重算。

模型是一个 **2 层、2 个 attention head、hidden size 16 的固定随机权重 Transformer**，包含因果 attention、RMSNorm、SiLU FFN 和正弦位置编码。权重没有训练，输出 token 没有语言质量。这里验证的是推理结果与内存、调度行为。

这是 CPU 教学实现。Attention 确实通过页表读取实际 K/V 数值；每轮 batch 中的计算用 JS 循环依次执行，没有 GPU 并行、融合算子，也不代表生产吞吐。

## 运行

需要 Node.js 24+，直接运行 TypeScript，无运行时依赖：

```sh
git clone https://github.com/LandoWang/sprout-lm.git
cd sprout-lm
npm run demo
npm run demo -- --pressure
npm test
```

普通演示中，A/B 先入队，C 在两轮后加入，复用 A 的共享前缀。输出逐轮展示 prefill/decode、计算 token 数和逻辑页到物理页的映射。

`--pressure` 把 KV 池从 18 页缩到 5 页，观察 `preempted`、重新入队与 `recompute`。`node src/demo.ts --json` 输出可供分析的完整 JSON；也可以同时传入 `--pressure`。

类型检查需要开发依赖：

```sh
npm ci
npm run check
```

## 从哪里读

| 文件 | 关注的问题 |
| --- | --- |
| [src/engine.ts](src/engine.ts) | 每轮谁执行、执行多少 token、何时让出内存？ |
| [src/kv-cache.ts](src/kv-cache.ts) | 逻辑位置怎样映射到物理页？谁还持有这页？ |
| [src/model.ts](src/model.ts) | 新 token 怎样使用历史 K/V 算 attention？ |
| [src/demo.ts](src/demo.ts) | 请求动态加入后，每轮调度发生什么？ |
| [test/engine.test.ts](test/engine.test.ts) | 如何证明换了调度和存储布局，计算仍然正确？ |

建议按上表顺序读。`Engine.step()` 是一次完整的调度迭代，调用方可以在相邻两次 step 之间加入或取消请求。

## 一轮的工作

1. 从等待队列接纳请求，最多同时运行 `maxSequences` 个；接纳时匹配已计算的完整前缀页。
2. 优先安排 decode，每个请求本轮至多生成一个 token；剩余 `maxBatchTokens` 预算分给 prefill/recompute，每个请求最多计算 `prefillChunkSize` 个输入 token。
3. 为即将追加的 token 确保页空间；空间不足时先驱逐无活跃使用者的 LRU 缓存页，再抢占较新的请求。
4. 前向计算新 token，将每层 K/V 写入其物理页，attention 通过页表读取从位置 0 到当前位置的 K/V。
5. 如果当前输入处理完毕，从 logits 取 argmax 生成 token。完成或取消的请求立即释放其活跃页引用。

新的请求可以在旧请求还在 decode 时加入。这里的“连续批处理”指每轮成员与工作量重新决定，不需要等待整批所有请求完成。

## Prefill 和 decode 为什么不同

对 prompt `P0 P1 P2`：

```text
prefill: 输入 P0、P1、P2，写入三个位置的 KV → 采样 G0
decode:  输入 G0，复用历史 KV             → 采样 G1
decode:  输入 G1，复用历史 KV             → 采样 G2，结束
```

最后输出的 G2 不会再作为输入，不需要它的 KV。因此，请求最坏需要的槽位为 `prompt.length + maxNewTokens - 1`。超过整个 KV 池容量的请求会在入队时被拒绝。

prefill 和 decode 都使用同一个逐 token 前向计算入口。生产实现通常会让 prefill 使用并行计算多个 prompt token 的算子；本项目保留清晰的因果数据流。

## 分页 KV cache

KV 池为每层预分配固定大小的 K/V 数组。每个请求只有一张页表，例如：

```text
blockSize = 4
request A: logical page [0, 1, 2] → physical page [3, 0, 7]
position 5 → logical page 1, offset 1 → physical page 0, offset 1
```

`forwardToken()` 调用 `kv.get(layer, position)` 完成地址转换，没有先把整段 KV 拼回连续数组。不同请求的物理页可以交错放置；请求增长到页边界时才需要新页。页大小以每个请求的 token 数计，一页在各层都保留对应的 K/V 槽位。

这里用常规 softmax 和多次逐位置读取，让分页寻址容易看清。没有复刻 vLLM 的 CUDA PagedAttention kernel、在线 softmax 或并行归约。

## 前缀共享与引用计数

- 只发布和共享已经算完的完整页；活跃请求持有的共享页不可追加写入。部分页不共享，因此本版本不需要 copy-on-write。
- 前缀键包含从第一个 token 开始的完整上下文。同一 token 块处在不同历史之后，K/V 也可能不同，不能只对当前块做匹配。
- 为便于读源码，键直接使用完整 token 序列字符串。实际系统通常采用链式哈希等更紧凑的索引。
- 每个活跃请求对其每个物理页持有一个引用，前缀索引另外持有一个缓存引用。共享页在所有活跃使用者释放前都不会被驱逐。
- 前缀只缓存 K/V，不缓存 logits；匹配时保留至少一个输入 token 重新计算，才能生成下一 token。prompt 恰好整页对齐时也遵守此规则。
- 请求结束后，缓存引用可能保留，`freePages` 不一定恢复为总页数。`activeRefs` 应变为 0；调用 `cache.clearPrefixes()` 才清空剩余缓存引用。

每个 Engine 持有一个固定模型和自己的 KV 池；前缀缓存不跨模型共享。

## 抢占与重算

当没有可驱逐的缓存页时，较老请求可让较新请求释放 KV、重新等待；较新请求不能反过来抢占较老请求，空间不够时自行等待。这样在每个请求单独都能放下、请求集合有限的前提下，较老请求能持续推进。

被抢占请求保留 prompt 和已经输出的 token。重新接纳后，复用仍存在的前缀页，重建缺失的历史 KV，然后继续 decode；重算期间不会重复输出 token。这是教学用的重算策略，没有磁盘/CPU swap。

## 正确性检查

`forwardDense()` 是独立的参考计算路径：每次从头按层计算整个序列，不使用 KV cache。测试覆盖：

- 所有位置的分页 attention logits 与完整重算在 `1e-12` 容差内一致，包括不连续物理页。
- 混合请求、动态加入、分块 prefill 的生成序列与独立运行一致。
- 共享前缀复用相同物理页，整页对齐仍保留生成 logits 所需计算。
- 不同前文下的相同后缀不会误命中缓存。
- 有/无前缀缓存的内存压力场景都能完成，抢占重算不改变输出。
- 驱逐不伤害活跃引用，取消请求不伤害共享同一页的其他请求。
- 每轮不超过 token 预算，结束后可以回收全部物理页。

目前按 `maxNewTokens` 结束，采用 greedy argmax；没有 EOS、采样参数、beam search、真实模型权重加载、HTTP 服务或 GPU 后端。下一步适合在现有对照测试约束下替换计算后端。
