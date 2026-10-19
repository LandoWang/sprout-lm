# BF16 权重直接读取：Qwen2.5-32B 单卡矩阵向量乘

日期：2026-10-19（America/Los_Angeles）。在 `yuawang-a100x2` 的 GPU 1（A100 80GB PCIe）运行，代码目录为 `/mnt/nvme/sprout-lm/runs/bf16-experiment`。权重、依赖和离线参考复用原项目；GPU 0 的 `sprout-lm-32b.service` 保持原版本。

## 改了什么

- `native/cuda-matvec.inc` 提供通用 BF16 / F16 GEMV。权重直接从显存读取，在寄存器中转换并与 FP32 激活相乘；使用 FP32 FMA、四个局部累加器和 warp shuffle 归约。不生成整块 FP32 权重矩阵。
- 每个 warp 计算一行，四行一个 CTA。宽度是 4 的倍数时使用 64-bit 权重 / 128-bit 激活读取，其他宽度走带边界检查的标量循环；尾部不满四行的 warp 整体退出。
- 全设备常驻路径每个半精度矩阵一次 kernel launch；混合 offload 路径仍按原 FP32 等效 tile 容量分块上传，但同样直接计算半精度权重。F32 矩阵继续使用 cuBLAS SGEMV。
- `CUDA_MATVEC=direct` 默认启用；`CUDA_MATVEC=convert` 保留原来的分块转换 + SGEMV 对照。也可用 backend 的 `matvec` 选项显式设置，创建后固定。原生构造接口增加模式参数，更新后需要重新构建扩展。
- direct 模式的 Graph 不再预分配 FP32 权重转换工作区。TS Runner 继续负责层循环，Graph 的动态 token / position / 页表协议、KV 引用和同步完成语义保留。

这里的收益同时来自取消转换缓冲读写、更换 GEMV 实现以及减少分块提交；不是把某一个转换 kernel 单独删掉的消融实验。激活和 KV 保持 FP32，权重没有重新量化。归约顺序改变，因此完整 logits 按既有误差标准验证，不要求逐位一致。

## 对照方法

模型固定为 `Qwen/Qwen2.5-32B-Instruct@5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`，64 层、hidden 5120、FFN 27648、40 Q / 8 KV heads、词表 152064。常驻矩阵载荷 65,525,514,240 字节。

同一张 GPU、同一份原生构建，**两种矩阵模式都开启 CUDA Graph**。按 convert → direct 顺序，每种模式一个冷请求加三个稳定请求；之后按 direct → convert 顺序，每种模式一个冷请求加一个稳定请求复测。进程顺序运行，没有并发加载两份实验权重，也没有锁 GPU 时钟。

输入：“从1开始逐行数到100，每行只写一个数字。”，使用相同聊天模板后为 47 token，生成 32 token。KV 池 256 token、页大小 16、单活跃请求、关闭前缀缓存，每轮预算 / prefill chunk 为 4，每轮结束 `setImmediate`。非输出 prefill 仍逐 token 普通提交，输出前向使用 Graph。这是 Engine 基准，不包含 HTTP、网络与客户端耗时。

## 结果

| 指标 | convert + Graph | direct + Graph |
| --- | ---: | ---: |
| 稳定 decode 样本数 | 93 | 93 |
| decode 均值 | 259.471 ms/token | 66.459 ms/token |
| decode 中位数 | 259.487 ms | 66.435 ms |
| decode p95 | 260.391 ms | 67.234 ms |
| 单请求 decode 速度 | 3.854 token/s | 15.047 token/s |
| 无前缀复用首 token 均值 | 11.999 s | 3.033 s |
| 首次捕获 / 实例化 / 上传 | 43.840 ms | 7.990 ms |
| sample Graph 节点数 | 3443 | 1220 |
| FP32 权重转换工作区 | 128 MiB | 0 |

decode 速度提升 **3.904×**，每步延迟下降 **74.39%**。反向次序复测为 **259.480 → 66.470 ms/token，3.904×**。两个顺序的所有请求输出 token 完全一致；每种模式的稳定请求仍只有 128 字节 D2H 采样结果，没有生成期间的权重上传。

首 token 的提升来自逐 token prefill 同样使用新矩阵算子；还没有实现 batch prefill。FP32 权重转换工作区为零，不代表所有临时设备内存为零：激活、Attention scores、Graph 元数据等仍存在，分别记录在 tensor 和 Graph 相关资源中。

完整逐轮数据见 `convert-32b-benchmark.json`、`direct-32b-benchmark.json`，复测见 `*-recheck.json`。`comparison.json` 汇总两次对照并确认输入 / 配置与输出一致。

## 正确性与内存检查

| 验证 | 结果 |
| --- | --- |
| TypeScript / 轻量测试 | 通过，32 项 |
| 混合 offload / resident 路径 | 5 项通过 |
| 完整 GPU 普通 / Graph 模式 | 普通 4 项通过、Graph 专属项跳过；Graph 5 项通过 |
| 新矩阵算子 | BF16 / F16 共 22 组 dtype / shape；direct、convert 和 offload 与 JS FP64 点积对照，最大绝对误差 0.0000616561 |
| 算子边界 | 奇数宽度、尾行、文件偏移未对齐、零 / 次正规数、混合符号，以及重复 Graph 重放通过 |
| CUDA Compute Sanitizer memcheck | 对上述矩阵测试检查，0 errors |
| 真实 0.5B | 完整 logits、调度、JSON / SSE / EOS 通过，最大误差 0.0002478361 |
| 真实 32B | 2 个 prompt、4 个输出位置、每位置完整 152064 logits，最大误差 0.0009183884 |
| 32B 原参考容差 | 保持 `1e-3 + abs(reference) * 1e-4`；最大误差 / 对应容差比 0.44239，greedy token 一致 |

微型完整模型继续覆盖非连续页、共享前缀、抢占重算、取消和 384-position 边界。32B 的独立数值范围限于上述 4 个位置，未完成长上下文或 agent 质量评测。

## Nsight 观察

另取短续写 `1+1=`（4 输入 / 8 输出）做 CUDA node-level trace，分析其中 7 个 decode。**profile 延迟不用于计算上述加速比**。

- 每次 decode 为 1220 个 kernel；没有独立 `convert` kernel 或 cuBLAS kernel，矩阵部分全部为 `directMatvecKernel`。
- kernel 时间并集约 63.656 ms，首末 kernel 区间内的空隙约 0.032 ms。
- 直接矩阵计算约 **38.564 ms/token**；Norm 仍约 **22.340 ms/token**；Attention 约 1.107 ms/token。
- Graph 去除提交空隙的作用保留。下一处明显成本是现有串行归约的 Norm，本次没有修改它。

这些是执行区间，不是 SM occupancy 或显存带宽利用率。原始报告 `direct-32b-profile.nsys-rep`、`profile-summary.json` 和分析脚本一并保存；SQLite 导出留在远端实验目录。

## 复现

以下命令只在有足够空闲显存的实验 GPU 上执行：

```sh
npm run build:cuda
CUDA_DEVICE=1 npm run test:cuda-matvec
CUDA_DEVICE=1 CUDA_MATVEC=direct CUDA_GRAPH=1 npm run test:cuda-device
CUDA_DEVICE=1 CUDA_MATVEC=direct CUDA_GRAPH=1 MATRIX_BACKEND=cuda-device npm run test:model
CUDA_DEVICE=1 CUDA_MATVEC=direct CUDA_GRAPH=1 REPORT_PATH=runs/32b-validation.json \
  node scripts/check-cuda-model.ts models/qwen2.5-32b-instruct .cache/qwen32-reference
CUDA_DEVICE=1 CUDA_MATVEC=convert CUDA_GRAPH=1 REPORT_PATH=runs/convert-32b-benchmark.json \
  npm run bench:cuda-graph -- models/qwen2.5-32b-instruct
CUDA_DEVICE=1 CUDA_MATVEC=direct CUDA_GRAPH=1 REPORT_PATH=runs/direct-32b-benchmark.json \
  npm run bench:cuda-graph -- models/qwen2.5-32b-instruct
CUDA_DEVICE=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  --target-processes all node test/cuda-matvec.integration.ts
```

复测使用 `REPETITIONS=1` 并交换两种模式的执行顺序。profile 命令：

```sh
nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --output=runs/direct-32b-profile env CUDA_DEVICE=1 CUDA_GRAPH=1 CUDA_MATVEC=direct \
  CUDA_EXECUTION=full PROMPT=1+1= MAX_NEW_TOKENS=8 REPORT_PATH=runs/direct-32b-profile-smoke.json \
  node scripts/smoke-cuda.ts models/qwen2.5-32b-instruct
nsys export --type=sqlite --output=runs/direct-32b-profile.sqlite runs/direct-32b-profile.nsys-rep
python3 analyze-trace.py runs/direct-32b-profile
```

驱动 580.159.03、CUDA Toolkit 13.0、Node 24.16.0、Nsight Systems 2025.3.2、Compute Sanitizer 2025.3.1。源码 / 原生二进制指纹见 `source-sha256.txt`、`binary-sha256.txt`；远端校验见 `source-verification.log`。最终设备和服务状态见 `final-state.json`。

实验结束后 GPU 1 释放。GPU 0 服务未重启或切换实现；本次未提交或推送 Git。
