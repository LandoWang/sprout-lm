# 32B：完整 GPU 前向与 API 部署

日期：2026-10-19。用户提出部署约 35B 的完整 GPU 模型，并确认选择 **Qwen2.5-32B-Instruct**，复用 Qwen2 结构。

**SproutLM 已在一张 A100 上运行完整模型前向、设备分页 KV 和 GPU greedy 采样，并通过原生 HTTP API 提供服务。模型结构仍由 TypeScript Runner 编排。**

## 这次跨过的边界

前一次 14B 实验让矩阵常驻 GPU，但 Attention、norm、激活、KV 和采样仍在 CPU。本次增加了通用设备算子，将这些数值工作迁到 GPU；Engine 继续管理调度、逻辑页、共享前缀和引用。

| 部分 | 本次位置 |
| --- | --- |
| Tokenizer、模板、HTTP、调度、页所有权 | TypeScript / CPU |
| Qwen 层循环、算子调用顺序 | TypeScript CudaQwenRunner |
| Embedding、norm、bias、RoPE、GQA Attention、SwiGLU、残差 | CUDA |
| 矩阵投影 | cuBLAS FP32 SGEMV；BF16 权重在设备内按块转换 |
| 权重、KV、激活、logits、greedy argmax | GPU；生成只回传 token ID |

激活与 KV 存储为 FP32；部分归约 / 位置运算用 FP64，保持教学参考的数值语义。权重主机映射仍保留，加载后 RSS 可能很大；实际生成不再从主机读取这些权重。原生层没有嵌入完整 Qwen 模型或 Python 推理服务。

## 固定模型与容量

- `Qwen/Qwen2.5-32B-Instruct@5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`。
- 64 层，hidden 5120，FFN 27648，40 Q head / 8 KV head，headDim 128，词表 152064；普通 Qwen2。
- 17 个 safetensors 分片逐文件校验，权重载荷 65,527,752,704 字节，约 61.03 GiB。
- 用户授权停止旧 GPU 服务后，每张 A100 空闲 81,152 MiB。模型部署在 GPU 0。
- 常驻矩阵 61.025 GiB；服务的 2048-token FP32 KV 池为 1 GiB；转换工作区约 128 MiB，另有设备向量、CUDA context 和库开销。
- API 请求后进程显存采样 64,076 MiB，约 62.57 GiB。GPU 1 没有部署模型。

## 独立数值验证

32B 参考由 torch `2.13.0+cu130` / transformers `5.17.0` 在 CPU 上以 FP32 eager 生成，16 线程，每个位置重算完整历史，关闭 KV cache。参考仅用于离线验证。

| 输入 | 两个 greedy 输出 token | 文本续写 |
| --- | --- | --- |
| `1+1=` | `[17,11]` | `2,` |
| `你好` | `[3837,35946]` | `，我` |

4 个输出位置各比较 152064 个 logits，最大绝对误差 **0.000865936279296875**。保持原容差 `1e-3 + abs(reference) * 1e-4`，最大误差与对应容差之比约 0.416；所有 token 一致。这些是短续写的数值样本，不是通用能力评测。

另有 32 项轻量回归、5 项原有 CUDA 混合测试、4 项完整 GPU 集成测试通过。新测试覆盖非连续页、前缀共享、抢占重算、取消、GPU argmax、关闭资源，以及跨 256 位置至 384-token 边界的 CPU / GPU 对照。真实 0.5B 的 18 个输出位置完整词表最大误差仍为 0.00024497509002685547，JSON / SSE 与自然 EOS 通过。

## 中文生成与传输证据

```text
用户：用一句话解释为什么天空是蓝色的，不超过40个汉字。
模型：天空蓝色因大气散射太阳光中蓝光较多。
```

输入含模板共 44 token，输出 14 token（含 EOS），自然结束。加载 15.266 s，首 token 11.305 s，总生成 14.732 s；13 次 decode 平均 **263.46 ms，约 3.80 步 / 秒**，包含 EOS 步。

这 57 次前向期间：

- H2D **528 字节**，全部是页表。
- D2H **56 字节**，即 14 个 int32 token ID。
- 权重重复上传 **0**；Attention、KV 和中间张量没有回传 CPU。
- 全部 7 个 KV 页回收，引用归零。

该 CLI 使用 112-token KV 容量，500 ms 采样的进程显存最大值为 63,108 MiB；它比服务的 2048-token 池小，不能混为同一容量数据。显存采样不是严格峰值。CPU 保留文件映射，结束 RSS 约 61.575 GiB。

prefill 仍逐 token 执行，因此 44 个输入 token 的首 token 延迟约 11 秒。当前没有 batch prefill、Tensor Core GEMM、FlashAttention、异步 Runner 或多卡。GPU 操作在同一 stream 上有序执行，每个 token 同步完成后才发布 KV / 结果，Node 在轮次之间处理 I/O 和取消。2048-token 是部署容量，长上下文质量和吞吐尚未验收。

## 常驻 API

- A100 上 `http://127.0.0.1:3000`，`GET /health`、`POST /generate`、`DELETE /requests/:id`。
- 系统服务 `sprout-lm-32b`，运行用户 `azureuser`，开机启动，失败有限重试。
- GPU 0，`BACKEND=cuda`，`KV_TOKENS=2048`；最多 4 个活跃请求，所有请求及缓存共享 KV 池，按调度预算顺序计算。
- 服务配置：[deploy/sprout-lm-32b.service](../../deploy/sprout-lm-32b.service)。

同一中文请求的 JSON 和 SSE 输出与 CLI 完全一致；JSON 总耗时 14.747 s。后续 SSE 复用了共享前缀，首 token 3.089 s、总计 6.517 s，不能当作冷 prefill 测量。取消一个流式请求时，另一个请求仍完成。超出容量的请求返回 400，结束后健康状态正常、pending 为零。

停止系统服务后 GPU 0 显存占用恢复到 0 MiB、空闲 81,152 MiB；重新启动后健康检查正常，算术请求输出 `[17,11]`，与独立参考一致。服务最后保留运行状态。

本机访问时可先建立隧道：

```sh
ssh -N -L 3000:127.0.0.1:3000 yuawang-a100x2
```

构建、下载、启动、管理与验证命令见 [README](../../README.md)。

## 原始记录

- [中文 smoke JSON](cuda-device-32b-chat.json) / [日志](cuda-device-32b-chat.log) / [显存采样](cuda-device-32b-memory.log)。
- [32B logits 对照报告](cuda-device-32b-validation.json) / [日志](cuda-device-32b-validation.log)。
- [独立参考元数据](reference-metadata.json)；原始 `.f32` logits 位于 A100 的 `.cache/qwen32-reference`，本地留存于 `.cache/a100-validation/qwen32-reference`。
- [API 检查](cuda-device-32b-api.json) / [日志](cuda-device-32b-api.log)。
- [服务停止、重启与状态](cuda-device-32b-service.json)。
- [完整 GPU 集成测试](cuda-device-tests.log)、[0.5B 独立对照](cuda-device-0.5b-validation.log)、[原混合路径回归](cuda-hybrid-regression.log)。
- [源码 SHA-256](source-sha256.txt)；对应当时本地工作树，改动尚未提交。

本阶段完成单卡完整 GPU 数值链路与 API 部署。M3 的第二架构、异步执行、低精度 KV、真正 batch 和长期稳定性验证继续保留为后续工作。
