# SproutLM 实验快照

按日期保存项目的重要实验：当时的设计、运行环境、真实结果、原始记录，以及尚未验证的边界。后续优化使用新的实验记录，保留旧数据作为对照。

| 日期 | 实验 | 意义 |
| --- | --- | --- |
| 2026-10-19 | [72B：CPU + GPU 协同推理](2026-10-19-qwen72b-cpu-gpu/README.md) | TypeScript 编排真实 72B 模型，主机承载权重和 KV，原生 CUDA 执行矩阵计算，完成短文本生成 |
| 2026-10-19 | [14B：单卡矩阵权重常驻](2026-10-19-qwen14b-resident/README.md) | 同一模型对比反复上传与一次上传，消除生成期间的权重传输，并完成中文问答 |
| 2026-10-19 | [32B：完整 GPU 前向与 API 部署](2026-10-19-qwen32b-gpu/README.md) | TypeScript 编排模型，数值前向、KV 和采样全部在单张 A100；独立 32B logits 对照及常驻服务验证通过 |
| 2026-10-19 | [CUDA Graph：单 token 前向重放](2026-10-19-cuda-graph/README.md) | 同一张 A100、同一 32B 模型对比逐算子提交与 Graph；记录正确性、捕获成本及稳定 decode 性能 |
| 2026-10-19 | [BF16：直接读取权重](2026-10-19-bf16-matvec/README.md) | 通用 CUDA GEMV 用 FP32 累加，取消权重转换缓冲；32B 在相同 Graph 配置下 decode 约加速 3.90 倍 |
| 2026-10-19 | [三进程 PD：同机双卡交接 KV](2026-10-19-pd-three-process/README.md) | API / Scheduler、P、D 三个进程；CUDA IPC / NVLink 复制 KV，32B 并行执行、取消回收与服务重启验证 |


这里保存实验记录及小型证据文件。模型权重保存在 A100 主机，代码的后续开发进度见 [MILESTONES.md](../MILESTONES.md)。
