# 14B：单卡矩阵权重常驻

日期：2026-10-19。承接 [72B CPU + GPU 协同实验](../2026-10-19-qwen72b-cpu-gpu/README.md)，用户提出试一个当前单卡显存能容纳的较小模型。

**Qwen2.5-14B-Instruct 的矩阵权重已常驻一张 A100。相同模型和输入的短测中，decode 从平均 1.754 秒降至 0.166 秒，生成期间权重上传为零；另一个中文对话样本自然 EOS 结束。**

## 设置与机制

- 模型：`Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`，8 个分片已校验哈希。
- 48 层、hidden 5120、40 Q head / 8 KV head、headDim 128、FFN 13824、词表 152064；普通 Qwen2 结构，无滑动窗口或 scaled RoPE。
- 权重总载荷：29,540,067,328 字节；常驻 GPU 的矩阵：29,538,385,920 字节，约 27.51 GiB。
- A100 80GB PCIe GPU 0，初始空闲 47,319 MiB；Node.js 24.16.0，CUDA Toolkit 13.0。其他服务继续运行。
- `CUDA_WEIGHT_MODE=resident`，矩阵预算 32 GiB，每次分配至少保留 4 GiB 空闲；默认 128 MiB FP32 转换 tile。
- 矩阵加载时一次上传并保留 BF16 存储；每次 matvec 从设备权重转换当前块，再调用 FP32 SGEMV。不是 BF16 Tensor Core 计算。
- TypeScript QwenRunner、Attention、norm、bias、激活、KV 与采样仍在 CPU；embedding 行读取使用主机映射。输入 / 输出向量仍往返 CPU 和 GPU。
- 所有调用同步完成；单请求，prompt + 输出最多 256 token，greedy。

## 相同模型的对照

原始输入 `1+1=`，4 个 prompt token，8 个输出 token。resident 后 offload，顺序各运行一次，7 次 decode 的均值如下：

| 指标 | offload | resident |
| --- | ---: | ---: |
| 加载 / 初始化 | 0.712 s | 4.508 s，包含上传 |
| 首 token，加载之后 | 8.369 s | 0.744 s |
| 总生成 | 20.656 s | 1.911 s |
| decode 均值 | 1.754 s | 0.166 s |
| decode 步 / 秒 | 0.570 | 6.009 |
| 生成期间权重 H2D | 303.122 GB | 0 |
| 生成期间总 H2D | 303.217 GB | 94.241 MB |
| 进程显存采样最大值 | 620 MiB | 28,872 MiB |

两个路径输出完全相同的 token：`[17, 374, 264, 259, 2717, 2449, 13, 2160]`，文本为 `2 is a tautology. Is`，因长度上限结束。这里是原始文本续写和执行路径对照，不用这段续写评估聊天能力。

约 10.5 倍的 decode 改善来自同模型的这组对照；不能用不同模型的 72B / 14B 延迟差推导常驻优化的收益。各模式只有一次短测，受共享设备负载、初始化和缓存影响，尚不是严格吞吐基准。

## 中文聊天

```text
用户：用一句话解释为什么天空是蓝色的，不超过40个汉字。
模型：天空散射太阳光中的蓝光，故呈现蓝色。
```

使用模型原有 Jinja 模板后输入 44 token，输出 14 token（含 EOS），结束原因 `eos`。加载 4.139 s，首 token 7.931 s，总生成 10.811 s；13 次 decode 平均 221 ms，约 4.52 步 / 秒，包含 EOS 步。当前 prefill 仍逐 token 执行，所以更长 prompt 的首 token 延迟明显增加。

常驻矩阵约 27.51 GiB，FP32 工作 buffer 高水位约 128.07 MiB。500 ms 间隔的进程显存采样最大值为 28,872 MiB，约 28.20 GiB，包含 context / 库开销，不是严格峰值。主机保留只读文件映射，结束 RSS 约 28.17 GiB。生成期间权重上传为零，但输入向量 H2D 仍有约 487.8 MB、输出向量 D2H 约 501.6 MB。

全部 7 个 KV 页回收。测试进程退出后两卡空闲显存恢复到 47,319 / 21,584 MiB。

## 验证边界

本地 32 项轻量测试和 TypeScript 检查通过。A100 上 5 项 CUDA 集成测试覆盖两种模式的 F32 / BF16 / F16、非方阵、跨 tile、重复 matvec 无权重重传、非连续页、预算拒绝和关闭清理。

真实 0.5B 常驻路径与独立 Transformers FP32 eager 对照，18 个输出位置的完整词表最大误差仍为 `0.00024497509002685547`，原容差未变；JSON / SSE 与自然 EOS 通过。这不是 14B 的独立全 logits 对照。14B 当前验证了两种执行模式的短输出一致和一条中文问答，长上下文、多请求、完整 GPU 前向及 GPU HTTP 服务仍待验证。

## 复现

权重下载与构建命令见 [README](../../README.md)。在 A100 项目目录执行：

```sh
CUDA_DEVICE=0 CUDA_WEIGHT_MODE=resident CUDA_WEIGHT_LIMIT_MIB=32768 CUDA_RESERVE_MIB=4096 \
PROMPT='1+1=' MAX_NEW_TOKENS=8 REPORT_PATH=runs/cuda-resident-14b-smoke.json \
  npm run smoke:cuda -- models/qwen2.5-14b-instruct

CUDA_DEVICE=0 CUDA_WEIGHT_MODE=offload PROMPT='1+1=' MAX_NEW_TOKENS=8 \
REPORT_PATH=runs/cuda-offload-14b-smoke.json \
  npm run smoke:cuda -- models/qwen2.5-14b-instruct

CUDA_DEVICE=0 CUDA_WEIGHT_MODE=resident CUDA_WEIGHT_LIMIT_MIB=32768 CUDA_RESERVE_MIB=4096 \
CHAT=1 PROMPT='用一句话解释为什么天空是蓝色的，不超过40个汉字。' MAX_NEW_TOKENS=64 \
REPORT_PATH=runs/cuda-resident-14b-chat.json \
  npm run smoke:cuda -- models/qwen2.5-14b-instruct
```

原始证据：

- 常驻短样本：[JSON](cuda-resident-14b-smoke.json) / [日志](cuda-resident-14b-smoke.log)。
- offload 对照：[JSON](cuda-offload-14b-smoke.json) / [日志](cuda-offload-14b-smoke.log)。
- 中文聊天：[JSON](cuda-resident-14b-chat.json) / [日志](cuda-resident-14b-chat.log)。
- [显存采样](cuda-14b-memory.log)：`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -lms 500`；各次实验 PID 记录在相应日志首行，可筛出本进程。
- [0.5B 独立参考验证](cuda-resident-0.5b-validation.log)。
- [相关源码指纹](source-sha256.txt)：当时本地源文件 SHA-256；尚未提交，不构成完整历史源码副本。

本次保留了 CPU / GPU 混合执行结构，消除了重复权重上传。下一步可围绕逐 token prefill、CPU Attention / KV 和向量往返继续定位开销；这里不将完整 GPU 执行或 M3 标记为完成。
