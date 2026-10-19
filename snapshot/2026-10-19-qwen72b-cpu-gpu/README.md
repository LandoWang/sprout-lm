# 72B：CPU + GPU 协同推理

日期：2026-10-19。项目：SproutLM。

**我们让自己的 TypeScript 推理框架，协同 CPU 和 A100，实际运行了 Qwen2.5-72B-Instruct。虽然很慢，但这证明模型编排、设备算子和存储可以分开实现，并组成一条能生成文本的链路。**

这是项目的阶段性里程碑。用户为这次实验留下的判断是：“虽然很慢但是是有标志意义的。”

## 当时的目标与选择

目标是在已有 Qwen Runner 上尝试 70B 级模型。使用用户选定的 `Qwen/Qwen2.5-72B-Instruct`，固定 revision 为 `495f39366efef23836d0cfae4fbe635880d2be31`。

机器有两张 A100 80GB，但已有任务占用显存，合计空闲约 67.3 GiB；模型 BF16 权重约 135.4 GiB。主机有约 432 GiB RAM，当时可用约 413 GiB。因此使用主机文件映射承载权重，只在 GPU 0 上分块做矩阵计算。原有 GPU 服务继续运行。

## 实际分工

| 部分 | 实现与位置 |
| --- | --- |
| 调度、prefill / decode、采样 | TypeScript Engine，CPU |
| 模型层循环及前向结构 | TypeScript QwenRunner，CPU |
| 大矩阵权重 | BF16 / F16 / F32 safetensors 的只读主机文件映射 |
| 矩阵运算 | Node-API → C++ / CUDA → cuBLAS SGEMV，A100 GPU 0 |
| 精度转换 | 当前权重块上传后在 GPU 转为 FP32，再进行 SGEMV |
| Embedding 行读取、RMSNorm、RoPE、Attention、SwiGLU 激活 | CPU |
| 分页 KV 与逻辑页管理 | CPU；沿用已有非连续物理页接口 |
| Tokenizer / 对话模板 | 独立插件；此次算术样本使用原始文本 |

一次矩阵计算的路径：

```text
TypeScript Runner 发起 matvec
  → 从主机映射权重中取一块
  → 上传到 A100
  → 转 FP32、调用 cuBLAS SGEMV
  → 将结果向量返回 CPU
  → Runner 执行后续步骤
```

这里的“协同”是 CPU 与 GPU 分工执行；原生调用同步等待 GPU 完成，尚未实现 CPU / GPU 计算重叠或流水线。模型结构保留在 TypeScript 中；原生层只承担矩阵映射、存储、转换、计算和计时。实际推理由本项目执行，没有代理其他推理引擎或 Python 服务。

## 固定环境与模型

- Linux，Node.js 24.16.0，CUDA Toolkit 13.0，A100 80GB PCIe；构建目标 `sm_80`。
- GPU 0 / 1 的初始空闲显存：47,319 / 21,584 MiB；本次仅使用 GPU 0。
- 模型目录：A100 主机上的 `/mnt/nvme/sprout-lm/models/qwen2.5-72b-instruct`。
- 80 层，hidden size 8192，64 个 Q head / 8 个 KV head，head dimension 128，词表 152064。
- 37 个 safetensors 分片，逐文件通过 Hub manifest 的 SHA-256 校验。
- 权重载荷 145,412,407,296 字节，约 135.4 GiB；独立输出投影。
- 配置上下文上限 32768；此实验只验证极短输入，不以配置值代替长上下文验证。

## 真正得到的结果

```text
输入文本：1+1=
输入 token：[16, 10, 16, 28]
输出 token：[17, 13]
输出文本：2.
结束原因：length（生成上限 2 token）
```

| 指标 | 实测值 |
| --- | ---: |
| 映射与文本插件准备 | 539 ms |
| 首 token 延迟 | 42.875 s |
| 两个输出 token 的总生成时间 | 51.631 s |
| 唯一一次 decode | 8.750 s |
| 前向次数 | 4 次 prefill + 1 次 decode |
| GEMV / tile 数量 | 2,802 / 12,076 |
| 累计 CPU → GPU 流量 | 707,235,340,288 字节，约 707.2 GB |
| 上传与同步耗时 | 47.443 s |
| 转换与 GEMV 的 CUDA event 区间累计 | 2.679 s |
| 累计 GPU → CPU 流量 | 138,432,512 字节 |
| 扩展工作 buffer 高水位 | 201,461,248 字节，约 192.1 MiB |
| `nvidia-smi` 采样到的本进程显存 | 约 620 MiB，包含 context / 库开销 |
| 结束时主机 RSS | 143,580,717,056 字节，约 133.7 GiB |
| 结束后 KV 空闲页 | 1 / 1，所有引用归零 |

539 ms 只是建立映射与插件，不代表全部权重已搬入 GPU。显存采样不是严格峰值；CUDA event 区间也不代表 GPU 利用率。仅有一次 decode，不能当作稳定吞吐基准。

## 为什么慢，以及这次证明了什么

每个 token 都重新上传所需矩阵。4 个 prompt token 逐个前向，再执行 1 次 decode，累计搬运约 707 GB。上传约占总生成时间的 92%；这组数据直接揭示了权重搬运的成本。

这次验证了：

1. 分片权重可以由自己的加载器读入，并接入已有的 Qwen 前向结构。
2. TypeScript 可以继续承担模型编排，底层 CUDA 库执行矩阵计算。
3. 模型大于当前空闲显存时，可以用主机内存换取可运行性。
4. 同一 Runner、Engine 与分页 KV 链路能扩展到真实 72B 的短样本推理。
5. 可以分别测量搬运和计算，确定下一步应优先消除重复权重上传。

同一 CUDA 矩阵后端先通过了微型 Qwen 和真实 0.5B 的独立 Transformers logits 对照。0.5B 覆盖 4 组输入、18 个输出位置的完整词表，最大绝对误差 `0.00024497509002685547`，保持原来的容差；生成、分页调度、HTTP JSON / SSE 与自然 EOS 通过。

72B 本身尚未做独立全 logits 对照，也未验证长文本、对话、多请求吞吐或 agent。两卡协同、权重与 KV 完整常驻设备、GPU Attention、真正的张量 batch 和异步执行均是后续工作。本快照不将 M3 标记为完成。

## 复现与原始证据

在已准备好依赖和权重的 A100 项目目录中：

```sh
cd /mnt/nvme/sprout-lm
npm run build:cuda
CUDA_DEVICE=0 CUDA_WEIGHT_MODE=offload PROMPT='1+1=' MAX_NEW_TOKENS=2 \
REPORT_PATH=runs/cuda-72b-smoke.json \
  npm run smoke:cuda -- models/qwen2.5-72b-instruct
```

当时 offload 是唯一模式；这里显式指定 `CUDA_WEIGHT_MODE=offload`，以便后续加入常驻模式后仍能选择原路径。运行结束释放本实验的 GPU 资源，原有服务继续运行。

- [原始运行报告](cuda-72b-smoke.json)：参数、token、逐轮时间、内存与传输统计。
- [原始运行日志](cuda-72b-smoke.log)：加载与逐步输出。
- [0.5B 数值验证日志](cuda-0.5b-validation.log)：相同后端的独立参考对照。
- [当时核心源码 SHA-256](source-sha256.txt)：五个关键源文件的指纹。该版本尚未提交；本快照保存实验材料与指纹，不包含完整历史源码副本。

下一步对照：选择当前单卡空闲显存能容纳的 14B，将矩阵权重一次上传并常驻 GPU，对比传输量与生成延迟。结果另行记录，保留本次 72B 原始证据。
