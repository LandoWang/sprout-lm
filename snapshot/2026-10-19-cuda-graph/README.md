# CUDA Graph：Qwen2.5-32B 单 token 前向重放

日期：2026-10-19（America/Los_Angeles）。本次在 A100 主机的 GPU 1 上运行独立实验，GPU 0 的 `sprout-lm-32b.service` 保持原版本。代码工作目录为 `/mnt/nvme/sprout-lm/runs/graph-experiment`，权重和依赖引用原项目已有文件。

## 实现范围

- `CUDA_GRAPH=1` 显式启用，默认关闭。TypeScript Runner 仍决定模型的层循环与算子顺序；原生层只提供捕获、实例化、重放和释放。
- 第一次需要输出的前向先按普通路径执行，以预热 cuBLAS 并得到本次正确结果；随后捕获 Graph、实例化、上传。捕获过程不再执行一次模型。
- Graph 包含 Embedding、Qwen 的 64 层、最终 Norm / LM head，以及采样路径的 argmax。读回完整 logits 的验证路径有单独的 Graph。
- 每次重放通过稳定的 pinned host 缓冲更新 token ID、位置和页表，GPU kernel 读取这些动态参数。页号不必连续，位置可以随新请求、前缀复用和重算而变化。
- 工作区地址稳定。每个 Runner 为当前 KV 池最多缓存 logits / sample 两张图；切换池会关闭旧图，关闭 backend 先销毁 Graph，再释放 tensor / 权重。
- 非输出 prefill 仍逐算子执行。采样结果读回、完成等待、KV 提交仍同步；没有异步 Runner、算子融合、JIT、Tensor Core GEMM、batch prefill 或多请求 tensor batch。
- Graph 路径把输入元数据使用 pinned memory，并将结果读回与结束等待集中到一次原生重放调用。对照测量的是整个 Graph 模式的变化，不是孤立的 `cudaGraphLaunch` 函数成本。

## 主对照方法与结果

同一张 A100 80GB PCIe、相同 Qwen2.5-32B-Instruct revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`、相同 BF16 权重 / FP32 SGEMV 算法。每次只运行一个实验进程。

输入为对话“从1开始逐行数到100，每行只写一个数字。”，应用同一个聊天模板后为 47 个 prompt token；生成 32 个 token。关闭 prefix cache，KV 池容量 256，`maxSequences=1`，每轮预算 / prefill chunk 均为 4。prompt 内部仍逐 token 前向。

每种模式记录一个冷请求，再记录 3 个稳定请求；每个稳定请求有 31 次 decode，共 93 个 decode 样本。模型加载单列，所有请求输出的 32 个 token 完全相同。每轮之后 `setImmediate` 让出事件循环；此处测量 Engine 路径，不包含 HTTP、网络和客户端读取。

| 指标 | 普通逐算子提交 | CUDA Graph |
| --- | ---: | ---: |
| 稳定 decode 均值 | 264.650 ms/token | 259.467 ms/token |
| 稳定 decode 中位数 | 264.696 ms/token | 259.475 ms/token |
| 稳定 decode p95 | 265.410 ms/token | 260.377 ms/token |
| 单请求 decode 速度 | 3.779 token/s | 3.854 token/s |
| 无前缀复用的首 token 均值 | 12.046 s | 11.999 s |
| 首次捕获 / 实例化 / 上传 | 无 | 38.441 ms |

Graph 模式 decode 速度提升约 **2.00%**，每步延迟减少约 **1.96% / 5.18 ms**。首 token 基本保持不变，大部分 prefill 没有使用 Graph。此次 sample Graph 有 3443 个节点，后续输出前向只需一次 Graph launch，但内部数值工作没有减少。

每个稳定请求仍执行 78 次前向；Graph 模式其中 32 次输出前向重放，46 次非输出 prefill 普通执行。D2H 都是 128 字节采样 ID；Graph 比普通路径多上传 256 字节，即 32 次重放各 8 字节 token / position。两条路径均未重复上传权重。

原始数据：`eager-32b-benchmark.json`、`graph-32b-benchmark.json` 及对应日志。主对照代码和二进制指纹见 `benchmark-source-sha256.txt`。随后加入了捕获期间非法操作 / 缺少 argmax 的拒绝检查，再运行反向次序复测；实际 kernel 和重放计算路径保持一致。

## 反向复测与 Nsight 时间线

完成主对照后，按 Graph → 普通模式的反向次序各加载一次模型，各记录一个冷请求和一个稳定请求。稳定 decode 为 **259.493 vs 264.380 ms/token**，速度提升 **1.88%**，输出仍完全相同。没有锁定 GPU 时钟；此结果支持约 2% 的小幅收益，不是普适的吞吐承诺。

另用 Nsight Systems 2025.3.2 捕获短请求 `1+1=`（4 输入 / 8 输出），只分析后面的 7 次 decode。配置 CUDA node-level trace，关闭 CPU sampling / context-switch tracing。profiling 会改变延迟，下面的数据用于解释执行结构，提速比例使用上面的无 profiler 实测。

| 每次 decode 的平均值 | 普通模式 | Graph 模式 |
| --- | ---: | ---: |
| 实际 GPU kernel 数 | 3315 | 3315 |
| 与这些 kernel 关联的 CPU runtime 提交 | 3251 次 `cudaLaunchKernel` + 64 次 `cudaLaunchKernelExC` | 1 次 `cudaGraphLaunch` |
| 首 kernel 开始至末 kernel 结束 | 275.498 ms | 270.506 ms |
| kernel 时间并集 | 271.681 ms | 270.438 ms |
| 该区间内的 kernel 间隔总量 | 3.817 ms | 0.067 ms |

Graph trace 中，类型转换 kernel 约 **154.473 ms/token**，cuBLAS kernel 约 **91.137 ms/token**，合计约 **90.8% 的 kernel 时间**；Norm 约 **22.304 ms/token**。因此提交间隔确实缩小了，但数值 kernel 的工作量基本保留，整体收益有限。这里的时间占比不能直接解释成显存带宽利用率或 SM 利用率；没有采集这些硬件计数器。

分析脚本另将同进程 / 同设备上下文的 memcpy 和 memset 区间加入并集，以区分非 kernel 设备活动与没有记录到设备活动的间隔。统计只覆盖每次 forward 的首末 kernel 之间，不包括 token 之间、HTTP、网络或其他进程。

原始 `.nsys-rep`、smoke 报告和 `profile-summary.json` 随此快照保存；SQLite 导出保留在远端实验的 `runs/`。复算方式：

```sh
nsys export --type=sqlite --output=eager-32b-profile.sqlite eager-32b-profile.nsys-rep
nsys export --type=sqlite --output=graph-32b-profile.sqlite graph-32b-profile.nsys-rep
python3 analyze-trace.py eager-32b-profile graph-32b-profile
```

采样命令的核心为 `nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node`，启动 `CUDA_EXECUTION=full CUDA_GRAPH=0|1 CUDA_DEVICE=1 PROMPT=1+1= MAX_NEW_TOKENS=8 node scripts/smoke-cuda.ts models/qwen2.5-32b-instruct`。GPU 驱动版本 580.159.03；没有修改功率或时钟设置。

## 正确性与回归

- TypeScript 检查、32 项轻量测试通过。
- 普通完整 GPU 路径 4 项通过；Graph 模式 5 项通过，新增捕获中止、跨 backend 句柄、释放后重放、非法页表、非有限 logits 等检查。
- 原有混合 CUDA 路径 5 项通过。
- Graph 模式真实 0.5B：独立完整 logits、分页调度、HTTP JSON / SSE 和 EOS 验证通过。
- Graph 模式真实 32B：2 个 prompt、4 个输出位置，每个位置完整 152064 logits；最大绝对误差 `0.000865936279296875`，与已有普通 GPU 基线相同。容差保持 `1e-3 + abs(reference) * 1e-4`，token 序列相同。
- 非连续页、前缀共享、抢占重算、取消、改变位置及跨 256 位置至 384-token KV 边界均通过微型模型验证。上述范围不代表长上下文或吞吐验收。

## 复现

在有空闲 A100 的 Linux 主机执行，不能在已装载 32B 的同一张卡再起第二份权重。

```sh
npm run build:cuda
CUDA_DEVICE=1 CUDA_GRAPH=1 npm run test:cuda-device
CUDA_DEVICE=1 CUDA_GRAPH=1 MATRIX_BACKEND=cuda-device npm run test:model
CUDA_DEVICE=1 CUDA_GRAPH=1 node scripts/check-cuda-model.ts \
  models/qwen2.5-32b-instruct .cache/qwen32-reference
CUDA_DEVICE=1 CUDA_GRAPH=0 REPORT_PATH=runs/eager-32b-benchmark.json \
  npm run bench:cuda-graph -- models/qwen2.5-32b-instruct
CUDA_DEVICE=1 CUDA_GRAPH=1 REPORT_PATH=runs/graph-32b-benchmark.json \
  npm run bench:cuda-graph -- models/qwen2.5-32b-instruct
```

Graph 保持为可选模式；本次实验没有把现有 GPU 0 系统服务切到 Graph。
