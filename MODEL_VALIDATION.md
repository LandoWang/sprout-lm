# Qwen 模型接入验证

验证日期：2026-10-19。前半部分记录本地 CPU 基线；后面的 A100 实验单独列出执行方式、结果和边界。

## 固定条件

| 项目 | 本次配置 |
| --- | --- |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| Revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| 权重 SHA-256 | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| 原始参数 | 494,032,768 个，BF16，单个 safetensors 文件 |
| SproutLM | Node.js 24.12.0，FP32 权重 / 激活 / KV，JS number 乘加与归约，greedy argmax |
| 参考实现 | PyTorch 2.10.0、Transformers 4.57.6，CPU float32，eager attention，`use_cache=False` |
| 环境 | Apple M3，16 GiB 统一内存，未使用 GPU |
| 文本插件 | HF tokenizer 资源与同 revision 的 Jinja 模板；生成文本隐藏特殊 token |

参考程序逐步重算完整序列，不复用 SproutLM 的分页 KV 或 attention 实现。两边均使用原始 logits 的 greedy argmax，不应用模型生成配置中的随机采样和 repetition penalty。EOS 列表参与生成结束判断。

## 已验证的行为

- 实际 checkpoint 的 4 组输入、共 18 个生成位置，逐一比较整个 151936 维词表 logits；生成 token 序列全部一致。
- 最大绝对 logit 误差：`0.00024259090423583984`。测试阈值为每元素 `abs(error) <= 1e-3 + 1e-4 * abs(reference)`。
- tokenizer 对齐普通中英文、空白、组合字符、特殊 token、emoji 和 BOM；对话模板展开与参考一致。
- 流式文本与参考完整解码一致。输入中的 Unicode 可能被 tokenizer 正规化，例如组合字符转成 NFC；验证对比参考解码结果，不强求还原原始输入字节。
- Qwen 多请求在完整页共享、分块 prefill 和内存压力抢占重算之后，仍与参考输出一致。
- HTTP JSON 与 SSE 的原始 token、文本、usage 和结束原因一致，聊天样本自然遇到 EOS。
- 轻量测试单独覆盖 EOS 计数、共享页释放、配置拒绝、权重数据转换，以及插件与请求间 decoder 隔离；已有取消和模型故障回归继续通过。

| 输入 | 输入 token 数 | 输出 token 数 | 参考与 SproutLM 输出 |
| --- | --- | --- | --- |
| 原始文本 `Hello` | 1 | 3 | `\n\nI am` |
| system: `Reply with exactly OK.`；user: `Hi` | 19 | 10（含 EOS） | `Hello! How can I assist you today?` |
| 原始文本 `你好` | 1 | 2 | `，我` |
| 原始文本 `Hello, world` | 3 | 3 | `! I'm` |

聊天样本没有遵从“只输出 OK”的指令，但与参考模型输出一致；这区分了模型自身的指令能力和推理实现是否正确。以上样本验证短上下文路径，不证明所有长上下文或其他 checkpoint 都已支持。

## 一次运行的时间记录

这些数字用于后续比较，没有做严格吞吐基准测试。加载时间受文件缓存影响，首 token 时间从模型已加载后开始计，包含 prompt 计算；总时间还包含参考数据读取和断言。

| 样本 | 首 token | 总时间 |
| --- | --- | --- |
| `Hello`，生成 3 token | 681 ms | 1668 ms |
| 19-token 聊天，生成 10 token | 5963 ms | 10116 ms |
| `你好`，生成 2 token | 400 ms | 880 ms |
| `Hello, world`，生成 3 token | 1022 ms | 1933 ms |

这次加载耗时 2474 ms；加载完成时 Node RSS 采样为 768 MiB，**不是峰值内存或全部权重占用**。全部 FP32 权重约 1.98 GB，操作系统的驻留 / 压缩情况会影响 RSS。聊天样本首 token 之后，9 个输出约花 4.15 s，约 2.2 token/s，包含测试额外开销。

## 复现

正常运行服务和轻量测试只需要 Node.js：

```sh
npm ci
npm run check
npm test
npm run model:download
MODEL_DIR=models/qwen2.5-0.5b-instruct npm run serve
```

实际权重对照测试需先生成参考数据。下面的 Python 环境仅用于离线验证，服务和模型执行都不调用它：

```sh
python3 -m venv .venv-reference
.venv-reference/bin/pip install -r scripts/reference-requirements.txt
.venv-reference/bin/python scripts/reference.py real \
  models/qwen2.5-0.5b-instruct .cache/qwen-reference
npm run test:model
```

`MODEL_DIR`、`REFERENCE_DIR` 可以修改实际模型与参考数据目录；默认使用上面的路径。`test:model` 不自动下载模型或运行 Python，缺少数据会失败。

仓库内的小 Qwen2 fixture 用同一参考工具生成，包含不同于默认值的 norm 权重与 QKV bias，以及不共享的输出投影，用于检查真实 0.5B 模型未覆盖的分支：

```sh
.venv-reference/bin/python scripts/reference.py tiny test/fixtures/qwen-tiny
```

该 fixture 的 seed 是 `20261019`，2 层、宽度 16、4 个 Q head / 2 个 KV head，权重约 24 KiB；logits 对照阈值为绝对误差 `2e-6`。TinyTransformer 原有 Float64 路径仍保留 `1e-12` 数值验证。

## A100 CUDA 矩阵 offload 验证

这是独立的实验后端：QwenRunner 的 TypeScript 模型结构和 Engine / 分页 KV 保留，C++ 通过 Node-API 提供通用矩阵算子。BF16 / F16 权重以只读文件映射留在主机，分块上传后转为 FP32，用 cuBLAS SGEMV 计算；Attention、RoPE、norm、激活与 KV 仍在 CPU。该路径没有调用 vLLM、Ollama 或 Python 推理服务。

环境为 `yuawang-a100x2`，Linux、Node.js 24.16.0、CUDA Toolkit 13.0、A100 80GB PCIe。机器有两张卡，但本实验只使用 GPU 0；初始可用显存分别为 47,319 MiB 和 21,584 MiB，已有服务保持运行。`nvidia-smi topo -m` 显示卡间 NV12，本实验没有使用多卡通信。权重和实验文件位于独立目录 `/mnt/nvme/sprout-lm`。

默认 128 MiB 的 FP32 tile 配合最多约 64 MiB 的 BF16 / F16 输入缓冲，向量空间另算。统计的 `peakScratchBytes` 只覆盖本扩展分配的工作 buffer，不包含 CUDA context、cuBLAS 内部空间或其他进程的显存。`mappedWeightBytes` 是映射的逻辑权重字节数，不是额外复制或峰值物理内存；主机 RSS 会受到文件页缓存和映射影响。

原生调用同步返回，只有调度轮次之间让出事件循环。每个 token 会再次读取和上传所需矩阵，GPU event 区间覆盖转换与 GEMV，并可能包含提交间隙；不能据此声称测得了 GPU 利用率或纯计算活跃时间。这是一条验证大模型可运行的容量折中路径，不是完整 M3 或生产性能结果。

### 从算子到真实 0.5B 的数值检查

| 检查 | 实际结果 |
| --- | --- |
| 本地轻量回归 | 32 项通过；新增索引分片与单文件生成对照、shape / 缺失 tensor / 路径拒绝 |
| A100 分块 GEMV | F32 / BF16 / F16、非方阵、跨 tile、非法输入和关闭资源检查通过 |
| A100 微型 Qwen | 非连续页及多步 logits 对照通过；最大绝对误差 `2.2351741790771484e-8`，token 一致 |
| A100 真实 0.5B | 4 组输入、18 个生成位置完整词表对照通过；最大绝对误差 `0.00024497509002685547`，token 全部一致 |
| 分页调度与服务 | 共享前缀、压力重算、JSON / SSE 与自然 EOS 通过；完整模型测试约 9.53 s |
| 独立 smoke CLI | 原始 prompt `1+1=`，输出 `[17, 3837]`，文本 `2，`，长度上限结束；清空缓存后全部页释放 |

真实 0.5B 的参考仍为前文固定的 Transformers CPU FP32 eager 结果，阈值仍是 `1e-3 + 1e-4 * abs(reference)`，没有为 CUDA 路径放宽。微型 CUDA fixture 阈值为 `1e-5`。这些数值对照不等于对 72B 全模型 logits 已完成独立验证。

在准备好相同的 0.5B 权重和 `.cache/qwen-reference` 后运行：

```sh
npm ci
npm run build:cuda
npm run test:cuda
MATRIX_BACKEND=cuda-offload npm run test:model
```

0.5B 完整测试进程累计上传约 75.0 GB，原生统计上传耗时约 5.65 s，转换 / GEMV event 区间合计约 0.48 s；这些包含多个参考与 HTTP 场景，不是单请求指标。这个差异说明该实现的权重搬运成本值得单独记录，不能只观察矩阵计算时间。

### Qwen2.5-72B-Instruct 短样本实际结果

固定 checkpoint：`Qwen/Qwen2.5-72B-Instruct@495f39366efef23836d0cfae4fbe635880d2be31`。配置为 80 层、hidden size 8192、64 个 Q head / 8 个 KV head、headDim 128、独立输出投影。37 个 safetensors 分片逐文件通过下载 manifest 的 SHA-256 校验；索引记录权重载荷为 145,412,407,296 字节，约 135.4 GiB。

两张 A100 当时合计空闲显存约 67.3 GiB，不足以常驻这个 BF16 checkpoint，所以本次使用上面的主机权重 offload 路径，只在 GPU 0 做矩阵计算。主机有约 413 GiB 可用内存，NVMe 在准备模型前有约 704 GiB 空闲；实验未停止已有服务。

| 项目 | 结果 |
| --- | --- |
| 输入方式 | 原始文本续写，未应用聊天模板 |
| Prompt / token | `1+1=` / `[16, 10, 16, 28]` |
| 输出 / token | `2.` / `[17, 13]`，拼接得到 `1+1=2.` |
| 结束原因 | `length`，输出上限 2 token，未验证该样本自然 EOS |
| 建立模型映射与文本插件 | 539 ms；这是建立映射，不能解释成全部权重已搬到 GPU |
| 首 token 延迟 | 42.875 s，包含 4 个 prompt token 的前向 |
| 总生成时间 | 51.631 s，共 5 次前向：4 次 prefill + 1 次 decode |
| 单次 decode | 8.750 s；只有一个 decode 样本，不作为吞吐基准 |
| CUDA 矩阵调用 / 工作块 | 2,802 次 GEMV / 12,076 个 tile |
| 主机 → GPU 总量 / 耗时 | 707,235,340,288 字节 / 47.443 s，包含重复搬运与同步开销 |
| 转换 + GEMV event 区间 | 累计 2.679 s；不是 GPU 利用率 |
| GPU → 主机结果 | 138,432,512 字节 |
| 扩展工作 buffer 高水位 | 201,461,248 字节，约 192.1 MiB |
| `nvidia-smi` 运行中采样 | 本实验 Node 进程约 620 MiB 显存，包含 context / 库开销；不是连续采集的严格峰值 |
| 结束时主机 RSS | 143,580,717,056 字节，约 133.7 GiB，主要为只读映射权重的驻留文件页 |
| KV 回收 | 1 / 1 页空闲，活跃引用和缓存引用归零 |

权重没有变成 707 GB；这个数字是五次前向重复上传同一批矩阵的累计流量。矩阵主要在主机内存，约 620 MiB 的 GPU 进程采样不代表能将整个 72B 模型装进这么小的显存。第一次前向约 16.67 s，后续约 8.64–8.90 s，仍受映射首次访问、CUDA 初始化和缓存状态影响。

进程结束后 GPU 0 / 1 空闲显存回到 47,319 / 21,584 MiB，已有进程仍在，没有留下推理服务或占用 GPU 的测试进程。权重与代码保留在 A100 的 `/mnt/nvme/sprout-lm`，以便复用。

在该主机的项目目录复现：

```sh
CUDA_DEVICE=0 PROMPT='1+1=' MAX_NEW_TOKENS=2 \
REPORT_PATH=runs/cuda-72b-smoke.json \
  npm run smoke:cuda -- models/qwen2.5-72b-instruct
```

原始报告在远端 `runs/cuda-72b-smoke.json`，本地也保存到被 Git 忽略的 `.cache/a100-validation/cuda-72b-smoke.json`。完整下载 / 构建命令见 README 的 A100 实验部分。

**验证范围：**这次证明指定 72B 权重经过 SproutLM 自己的 Runner、调度与分页，能够完成短文本生成，并实际使用 A100 执行矩阵乘法。72B 全模型 logits 尚未与独立参考逐元素对照，也没有验证它的聊天、长上下文、多请求性能、BF16 设备常驻、GPU attention / KV 或双卡执行。M3 后续需将更多算子和状态迁到设备并消除重复搬运；单次 smoke 不计为 M3 完成。

本次历史记录及原始证据已固定保存到 [72B CPU + GPU 实验快照](snapshot/2026-10-19-qwen72b-cpu-gpu/README.md)。

### Qwen2.5-14B-Instruct：单卡矩阵权重常驻

日期：2026-10-19。固定 `Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`，48 层、hidden 5120、40 个 Q head / 8 个 KV head、headDim 128、FFN 13824、词表 152064。8 个分片通过下载 manifest 哈希校验，总权重载荷 29,540,067,328 字节。GPU 0 初始空闲 47,319 MiB，选择 14B 以保留运行余量；已有其他服务继续运行。

新增 `CUDA_WEIGHT_MODE=resident`：矩阵在加载时按原始 dtype 一次上传到 GPU。BF16 矩阵仍分块转换为 FP32 做 SGEMV；CPU 上的 QwenRunner、Attention、norm、bias、激活、采样和 KV 保持原路径。Embedding 矩阵有设备副本，行读取仍使用主机映射。

同一原始 prompt `1+1=`（4 token），greedy 输出 8 token，两种模式得到相同 token 序列和文本 `2 is a tautology. Is`，均因长度结束。该续写用于执行路径对照，不作为回答质量结论。两次顺序运行，各只有一个样本：

| 指标 | 主机 offload | GPU 常驻矩阵 |
| --- | ---: | ---: |
| 加载 / 初始化时间 | 0.712 s，仅映射与初始化 | 4.508 s，包含一次权重上传 |
| 首 token 延迟，加载之后 | 8.369 s | 0.744 s |
| 8 个输出 token 总生成时间 | 20.656 s | 1.911 s |
| 7 次 decode 平均 | 1.754 s | 0.166 s |
| decode 步数 / 秒 | 0.570 | 6.009 |
| 生成期间权重上传 | 303.122 GB | 0 |
| 生成期间总 H2D | 303.217 GB | 94.241 MB，输入向量 |
| 显存常驻矩阵 | 0 | 29,538,385,920 字节，约 27.51 GiB |
| 扩展工作 buffer 高水位 | 192.06 MiB | 128.07 MiB |
| 每 500 ms 采样进程显存的最大值 | 620 MiB | 28,872 MiB，约 28.20 GiB |

同模型短测的 decode 均值改善约 10.5 倍。常驻路径仍累计回传约 100 MB 输出向量；权重传输归零不等于所有 CPU / GPU 传输归零。显存是离散采样最大值，不是严格峰值；两模式保留主机文件映射，常驻测试结束 RSS 约 28.16 GiB。

独立中文聊天样本启用模型原有 Jinja 模板：

```text
用户：用一句话解释为什么天空是蓝色的，不超过40个汉字。
模型：天空散射太阳光中的蓝光，故呈现蓝色。
```

应用模板后输入 44 token；输出 14 token（包含 EOS），自然结束。加载 4.139 s，首 token 7.931 s，总生成 10.811 s；13 次 decode 平均 221 ms，约 4.52 步 / 秒，包含 EOS 步。首 token 仍需逐个处理 44 个 prompt token。全部 7 个 KV 页回收，生成期间权重上传为零。GPU 进程退出后两卡空闲恢复到 47,319 / 21,584 MiB。

验证：本地 32 项轻量测试、TypeScript 检查、A100 5 项 CUDA 集成测试通过。CUDA 覆盖两种模式的 F32 / BF16 / F16 非方阵和跨 tile、重复 matvec 不重复上传、非连续页 logits、预算失败保留已有矩阵和资源关闭。真实 0.5B 的 18 个位置完整 logits 对照最大误差仍为 `0.00024497509002685547`，原容差未改变，JSON / SSE 与自然 EOS 通过。14B 尚未与独立参考逐元素比较全 logits，未验证长上下文、多请求或 GPU HTTP 服务。

完整命令、原始 JSON、日志及显存采样见 [14B 常驻实验快照](snapshot/2026-10-19-qwen14b-resident/README.md)。这一步证明单卡矩阵常驻可消除重复上传；完整 GPU 前向、设备 KV、batch prefill 和异步 Runner 继续列为后续工作。

### Qwen2.5-32B-Instruct：完整 GPU 前向与常驻 API

日期：2026-10-19。用户确认选用 `Qwen/Qwen2.5-32B-Instruct@5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`。经用户授权停止旧 GPU 服务后，两卡分别空闲 81,152 MiB。本次模型只使用 GPU 0。

配置：64 层、hidden 5120、FFN 27648、40 个 Q head / 8 个 KV head、headDim 128、词表 152064、未启用 sliding window / scaled RoPE，EOS `[151645,151643]`。17 个分片通过下载器 manifest 哈希校验，BF16 权重载荷为 65,527,752,704 字节，约 61.03 GiB。

`CudaQwenRunner` 的层循环留在 TypeScript，原生后端暴露通用设备算子；embedding、norm、bias、RoPE、Attention、SwiGLU、残差、矩阵投影和 greedy argmax 都在 GPU，KV 是设备 FP32 池。矩阵保持源 dtype，并在设备内按块转换为 FP32 做 SGEMV；激活为 FP32，部分归约 / 位置运算使用 FP64，保持现有教学数值语义。没有 CPU 前向算子或推理代理。Tokenizer、模板、调度和页引用仍由 CPU 管理。

#### 独立数值与状态验证

| 检查 | 结果 |
| --- | --- |
| 本地 TypeScript / 轻量测试 | 检查通过，32 项通过，CPU 路径保留 |
| 旧 CUDA 混合路径 | 5 项通过，offload / resident 均保留 |
| 新完整 GPU 路径 | 4 项通过：独立微型 logits、非连续页、共享前缀 / 抢占 / 取消、GPU argmax / 生命周期，以及跨 256 位置至 384-token 边界与 CPU 对照 |
| 真实 0.5B | 18 个位置完整词表对照，最大绝对误差 `0.00024497509002685547`；JSON / SSE / EOS 通过 |
| 真实 32B | 2 个 prompt、4 个输出位置、每位置 152064 logits；最大绝对误差 `0.000865936279296875`，greedy token 一致 |
| 容差 | 保持 `1e-3 + abs(reference) * 1e-4`；32B 最大误差 / 对应容差之比约 0.416 |

32B 独立参考使用已有的 torch `2.13.0+cu130` / transformers `5.17.0`，CPU FP32 eager、16 线程、每个输出位置完整重算历史且关闭 KV cache。离线权重加载约 16.62 s，参考生成含加载约 36.61 s。输入 `1+1=` 得 `[17,11]`（`2,`），输入 `你好` 得 `[3837,35946]`（`，我`）；这些是两 token 的原始文本续写数值样本。参考版本与旧 0.5B 基线不同，分别记录，未覆盖旧参考文件。Python 不参与服务。

#### 32B 全设备中文 smoke

```text
用户：用一句话解释为什么天空是蓝色的，不超过40个汉字。
模型：天空蓝色因大气散射太阳光中蓝光较多。
```

| 指标 | 实测值 |
| --- | ---: |
| 输入 / 输出 | 44 / 14 token，输出包含 EOS，自然结束 |
| 加载与初始化 | 15.266 s，其中矩阵一次上传约 14.281 s |
| 首 token，加载之后 | 11.305 s，44 个 prompt token 逐个前向 |
| 总生成 | 14.732 s |
| 13 次 decode 均值 | 263.46 ms，约 3.80 步 / 秒（含 EOS 步） |
| 常驻矩阵 | 65,525,514,240 字节，61.025 GiB |
| 设备 tensor（norm / bias、工作向量、KV 等） | 61.205 MiB；此 smoke 只申请了 112-token KV 容量 |
| FP32 转换工作区高水位 | 127.988 MiB |
| 500 ms 间隔的本进程显存采样最大值 | 63,108 MiB，约 61.63 GiB，不是严格峰值 |
| 结束时主机 RSS | 61.575 GiB，保留了被加载访问过的只读文件映射 |
| 生成期间 H2D | 528 字节，仅页表 |
| 生成期间 D2H | 56 字节，仅 14 个 int32 token ID |
| 生成期间权重上传 | 0 |
| 回收 | 7 / 7 个 KV 页空闲，引用归零 |

57 次前向经过 65,749 次设备算子调用。设备 event 区间约 14.716 s，包含 GPU 执行及提交间隙，不能当作 GPU 利用率。不同模型之间不直接用延迟差估计优化收益；本次仍为单个短问答的功能与开销记录。

#### 常驻部署

系统服务 `sprout-lm-32b` 使用 [部署配置](deploy/sprout-lm-32b.service)，以 `azureuser` 运行，GPU 0，监听 A100 上的 `127.0.0.1:3000`。配置 `KV_TOKENS=2048`，是所有请求和前缀缓存共用的 1 GiB FP32 KV 池；最多 4 个活跃请求，依次执行。服务 GPU 进程在请求后采样为 64,076 MiB（约 62.57 GiB），GPU 1 留空。

部署验证：同一中文样本 JSON 与 SSE 的输出 token / 文本 / EOS 与 CLI 一致。JSON 总耗时约 14.747 s；后续 SSE 复用前缀，首 token 约 3.089 s、总计 6.517 s，不能作为冷 prefill 指标。取消一个流式请求时，另一个请求仍完成并输出预期 token。超过容量的请求返回 400，完成后 `/health` 为 `ok` 且 pending 为零。系统服务停止 / 重启和重启后算术推理另有状态记录。

“完整 GPU”指模型数值前向、KV 和采样均在 GPU；单 token 仍同步完成，Node 只能在轮次之间响应 I/O / 取消。仍无 batch prefill、张量 batch、多卡、Tensor Core GEMM 或 FlashAttention，2048-token 是部署容量配置，不代表已完成该长度的质量 / 吞吐验收。32B 的独立数值对照范围是上述 4 个位置，不能代替长上下文或 agent 评测。

原始报告、API 检查、服务状态、数值参考元数据和代码指纹保存在 [32B GPU 实验快照](snapshot/2026-10-19-qwen32b-gpu/README.md)。


### 可选 CUDA Graph（2026-10-19）

`CUDA_GRAPH=1` 捕获有输出的单 token GPU 前向，首次预热 / 捕获后重放；非输出 prefill 仍为普通路径。页表、token、position 动态更新，调度与 KV 引用管理保持原语义。Graph 仍同步等待完成，没有新增算子融合或低精度算法。

Graph 模式通过微型 Qwen 的非连续页、共享前缀、抢占 / 重算、取消、384-position 边界和 Graph 生命周期测试，真实 0.5B 的完整 logits / HTTP / EOS 测试通过。真实 32B 的 4 个输出位置完整 logits 最大误差仍为 `0.000865936279296875`，容差与原参考不变。

单张 A100 GPU 1、47 输入 / 32 输出、关闭前缀缓存；每模式一个冷请求加三个稳定请求，每模式 93 次稳定 decode，全部输出相同：普通模式 **264.650 ms/token / 3.779 token/s**，Graph **259.467 ms/token / 3.854 token/s**，速度提升约 **2.00%**。首次捕获 / 实例化 / 上传 **38.441 ms**。反向次序复测提升 **1.88%**。

Nsight 短样本中，每个 decode 的 3315 个 GPU kernel 保留，CPU 提交合成一次 Graph launch；首末 kernel 内部的间隔从约 3.817 ms 降至 0.067 ms。约 90.8% kernel 时间仍花在 BF16 → FP32 转换与 cuBLAS 上。profile 延迟不用于计算实际提速，event / kernel 时间均不等于硬件利用率。

详细方法、完整数据、原始 Nsight 报告和限制见 [CUDA Graph 快照](snapshot/2026-10-19-cuda-graph/README.md)。实验进程结束后 GPU 1 释放；GPU 0 的既有服务未切换版本。


### BF16 / F16 权重直接读取（2026-10-19）

新增默认 `CUDA_MATVEC=direct`：半精度权重在寄存器中解码后与 FP32 激活相乘，用 FP32 FMA 和 warp 归约累加，不物化 FP32 权重矩阵。`convert` 保留原 SGEMV 对照。F32 权重路径保留；FP32 KV / 激活、TS 模型编排和同步完成语义未改。

- 新矩阵算子测试覆盖 BF16 / F16、22 组 dtype / shape、奇数宽度、尾行、未对齐 checkpoint 文件偏移、零 / 次正规数和混合符号，以及 Graph 重放；与 JS FP64 点积比较的最大绝对误差为 `0.00006165605773844618`（含 direct、convert、offload 三条结果）。全设备 direct 的 FP32 权重转换缓冲为零。
- 轻量测试 32 项通过；混合路径 5 项通过；全设备普通模式 4 项通过、Graph 专属项跳过，Graph 模式 5 项通过。
- 真实 0.5B 完整 logits、调度、JSON / SSE / EOS 通过，最大绝对误差 `0.0002478361129760742`。
- 真实 32B 两个 prompt、4 个输出位置、每位置 152064 logits，最大绝对误差 `0.0009183883666992188`，最大误差 / 对应容差比 `0.4423885391121997`；容差仍为 `1e-3 + abs(reference) * 1e-4`，greedy token 一致。

两种矩阵模式都开启 Graph，沿用前次 47 输入 / 32 输出、关闭前缀缓存的单请求基准：每模式一个冷请求加三个稳定请求，每模式 93 次稳定 decode；旧模式 `259.471 ms/token / 3.854 token/s`，新模式 `66.459 ms/token / 15.047 token/s`，加速约 `3.90×`。无前缀复用的首 token 从 `11.999 s` 降到 `3.033 s`。没有量化激活或 KV，也没有新增 batch prefill。

详细复测、CUDA 内存检查、profile 与原始报告见 [BF16 直接读取快照](snapshot/2026-10-19-bf16-matvec/README.md)。GPU 0 的既有服务保持原版本。


### 三进程同机 PD（2026-10-19）

`BACKEND=cuda-pd` 将 API / GenerationService / Scheduler、Prefill Worker、Decode Worker 分为三个 Node.js 进程。每个 Worker 一张 A100、一份完整 32B 权重，继续由 TypeScript 编排；CPU 协调进程只维护 token、请求和页引用。启动时 CUDA IPC 映射 P 的 KV 池，D 通过 P2P 复制有效 token 的 KV，完成 ACK 后释放 P 页。

- 本地 TypeScript 检查和 39 项轻量测试通过，新增异步调度、目的容量不足排队、在途取消、Worker 故障、关闭、EOS / 单 token 免传输及过期完成消息拒绝测试。
- 双 GPU 的 3 项集成测试通过：非连续源 / 目标页、完整独立微型 logits（最大误差 `2.2351741790771484e-8`）、HTTP JSON / SSE 并发，以及 Worker 被终止后回收整组进程。
- 0.5B 跨进程 P → D：18 个参考位置、每位置完整 151936 logits，最大误差 `0.0002478361129760742`。
- 32B 跨进程 P → D：4 个参考位置、每位置完整 152064 logits，最大误差 `0.0009183883666992188`，最大误差 / 对应容差比 `0.4423885391121997`。两种模型均保持 `1e-3 + abs(reference) * 1e-4` 容差和 greedy token 一致。
- 32B HTTP/SSE 的 47 输入 / 32 输出样本与单卡 BF16 基线全部 token 一致；三个稳定请求、93 个 token 间隔平均 `66.996 ms`（`14.926 token/s`），首 token 平均 `3024.462 ms`。这是 HTTP/SSE 本机回环计时，不与前次 Engine-only 基准直接计算加速比。
- 后到请求的 prefill 与先到请求的全部 31 次 decode 有 Worker 执行区间重叠；并发时该 decode 流的 token 间隔平均 `66.908 ms`。在 1313 次健康检查中 p95 为 `0.675 ms`。Worker 区间使用同机墙钟毫秒，不代表 GPU occupancy。
- 每次 47-token KV 复制有效载荷为 `24,641,536` 字节（23.5 MiB），7 次设备复制 CUDA event 均值 `0.192169 ms`；不包括全部控制 IPC 与排队成本，不是 NVLink 线路带宽测试。
- 取消在途 prefill 的客户端流约 `4.514 ms` 结束，实际页回收等 Worker 完成；并行请求输出正确。最终双端 16 / 16 页空闲。

已部署 `sprout-lm-pd.service`，每卡 2048-token KV 池，旧单卡服务停止并取消开机启动。验证中终止 D Worker，旧协调 / P / D 全部退出，systemd 启动新三进程组，重启前后 `1+1=` 输出 `[17,11]`，最终健康且无待处理请求。协调进程不出现在 CUDA compute-process 列表中。

当前 PD 路径提前预留整个请求的 D 容量，无前缀缓存 / 抢占重算 / tensor batch，也没有跨机器 RDMA 或无损恢复。单卡旧后端保留。详细记录见 [PD snapshot](snapshot/2026-10-19-pd-three-process/README.md)。
