# 三进程 PD：API / Scheduler + Prefill + Decode

日期：2026-10-19（America/Los_Angeles）。目标是把 API 和调度从同步 GPU 调用中拆开，让 Prefill 与 Decode 在两张 A100 上独立推进，并安全交接分页 KV。

## 最终部署

`yuawang-a100x2` 当前运行 `sprout-lm-pd.service`，启用开机启动与失败重启。监听原地址 `127.0.0.1:3000`，代码目录 `/mnt/nvme/sprout-lm/deployments/pd`。旧 `sprout-lm-32b.service` 停止并取消开机启动，原目录 `/mnt/nvme/sprout-lm` 的单卡代码和原生构建保留。

| 进程 | 设备与职责 |
| --- | --- |
| API + GenerationService + PDScheduler | 接请求、文本插件、排队、两端页引用、输出缓冲和取消；不创建 CUDA context |
| Prefill Worker | GPU 0，完整 32B 模型与自己的 KV 池，逐 token / 分块 prefill |
| Decode Worker | GPU 1，完整 32B 模型与自己的 KV 池，KV 接收、decode 和 GPU greedy 采样 |

两张卡各配置 2048-token FP32 KV（各 1 GiB），各占约 63,948 MiB 进程显存。每张卡各有一份完整模型；这是 PD 角色分离，不是把一份模型做 TP / PP 切分。

检查到 GPU 0 ↔ GPU 1 为 `NV12`，P2P 读写均支持。计算保持上一版的 BF16 权重直接读取、FP32 激活 / KV / 矩阵累加和 CUDA Graph；TS 编排模型层次。无 NCCL、网卡 RDMA 或主机 KV staging。

## 协议与生命周期

1. 协调进程通过 `child_process.fork` 启动 P / D；Linux 的 Node IPC socket pair 只传小型计划、token 和完成消息。
2. P 用 CUDA IPC 导出 KV 大池；D 在启动时映射一次。校验模型 spec、池布局及设备 P2P 支持；不支持就明确失败，不回退主机搬运。
3. Scheduler 在准入时为 P 预留 prompt 页，为 D 预留整个请求的最大所需容量。容量不足就排队，防止算完 prompt 后等待 D 空间形成堵塞。
4. 每个 Worker 最多一个在途计划。P 按 chunk 轮转，D 在就绪的交接 / decode 计划间轮转；协调进程等待任一完成消息，而不是等待 P、D 同时结束。
5. P 完成 prompt 时返回首 token，GPU 计算已完成。若已到 EOS / 长度上限就释放双端预留，不传 KV；否则 D 开始交接。
6. D 按源 / 目标页映射，用 `cudaMemcpy2DAsync` 复制所有层 K/V 的有效 token。页号不要求相同或连续，尾页只复制有效部分。目标 stream 上 CUDA event 完成后才返回 ACK；D 并未重新计算 prompt。
7. Scheduler 收到复制 ACK 后发布 D 的 KV 长度、释放 P 引用，并把首 token 作为 D 的下一次输入。D 与 P 的后续请求可以并行工作。

RPC 有每个 Worker 通道内单调递增的编号，完成消息还核对请求 ID。连接不重用、不自动重试旧计划。取消立即结束客户端流，在途页保留到 GPU 操作完成；迟到结果被丢弃后才回收页。超时会终止 Worker，不能把超时本身当作可回收证明。

任一 Worker 失败时整个引擎停止发布结果，先退出 D importer、再退出 P exporter。进程退出确认后回收协调端账本。服务以失败状态退出，systemd 重启整组；没有单 Worker 无损恢复或在途请求重试。正常关闭也遵守 importer → exporter 的顺序。

## 验证

| 检查 | 结果 |
| --- | --- |
| TypeScript 与轻量测试 | 39 项通过，包括旧 CPU / HTTP / Service 测试 |
| 新调度状态测试 | 容量不足排队、P / D 独立推进、复制期间取消、关闭 / 故障延迟回收、终止首 token 免传输、过期完成消息拒绝 |
| 双 GPU 集成 | 3 项通过：非连续 KV 重映射和独立 logits、JSON / SSE 并发、Worker 退出与 peer 清理 |
| 微型独立 logits | 最大绝对误差 `2.2351741790771484e-8`，含完整页和部分页 |
| 真实 0.5B | 18 个参考位置 × 151936 logits，最大绝对误差 `0.0002478361129760742` |
| 真实 32B | 4 个参考位置 × 152064 logits，最大绝对误差 `0.0009183883666992188` |
| 参考容差 | 保持 `1e-3 + abs(reference) * 1e-4`，32B 最大误差 / 对应容差比 0.44239，greedy 输出一致 |
| 原 CUDA 路径回归 | Graph 完整 GPU 5 项通过，BF16 / F16 算子测试通过 |
| 常驻部署与监督 | 重启前后 JSON 生成正确；主动终止 D Worker 后整组三个旧进程退出，新三进程组自动启动 |

32B 固定为 `Qwen/Qwen2.5-32B-Instruct@5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`。0.5B 使用已有 revision `7ae557604adf67be50417f59c2c2f167def9a775`。独立参考版本等元数据保存在 `pd-*-validation.json`。

## 32B API 实测

启动三个进程，在本机 HTTP/SSE 回环接口发送“从1开始逐行数到100，每行只写一个数字。”。聊天模板后为 47 输入 token，生成 32 token。KV 池每卡 256 token、页大小 16、P chunk 4、最多 4 个活跃请求、无前缀缓存。启动加载单列，先一个冷请求，再三个稳定请求。

| 指标 | 结果 |
| --- | ---: |
| 双模型与文本资源启动 | 14.937 s |
| 稳定请求首 token 均值 | 3.024 s |
| 93 个稳定 token 间隔均值 | 66.996 ms |
| 单请求 decode 速度 | 14.926 token/s |
| 另一个请求 prefill 时已有流的 token 间隔均值 | 66.908 ms |
| 与后到 prefill 重叠的 decode 执行计划 | 31 / 31 |
| 1313 次健康检查 p95 | 0.675 ms |
| 健康检查最大值（含冷请求） | 25.177 ms |
| 47-token KV 有效载荷 | 24,641,536 字节，23.5 MiB |
| 7 次复制 CUDA event 均值 / 最大值 | 0.192169 / 0.216960 ms |

所有完整生成结果与上一版单卡 BF16 基线的 32 个 token 一致。另测试在途 prefill 取消：客户端流约 4.514 ms 结束，没有输出 token；实际引用继续保留到 Worker 完成。并行的另一请求正确完成，最终两端 16 / 16 页空闲。

这里的 token 间隔包含 HTTP、IPC 和客户端读流开销，不能直接与之前 Engine-only 数据计算加速比。复制 event 区间不包含全部控制 IPC / 排队时间，也不是 NVLink 线路带宽。P / D 重叠来自同机 Worker 的执行开始 / 结束墙钟记录，不是 GPU occupancy 或完整 Nsight 时间线。

本次重点是角色并行和 API 响应性，未做批量吞吐验收。原始数据见 `pd-32b-api.json`，汇总见 `performance-summary.json`；0.5B 也执行了同样的 API / 并行 / 取消流程。

## 部署与回退

本次先在 `/mnt/nvme/sprout-lm/runs/pd-experiment` 构建和测试，32B 双卡实验期间临时停止旧服务，结束后恢复；全部验证通过后，将同一份源码和原生二进制部署到独立目录并切换系统服务。

```sh
# A100 主机上查看当前三进程服务
sudo systemctl status sprout-lm-pd
sudo journalctl -u sprout-lm-pd -n 30 --no-pager
curl http://127.0.0.1:3000/health

# 如需回到旧单卡实现
sudo systemctl disable --now sprout-lm-pd
sudo systemctl enable --now sprout-lm-32b
```

部署文件为仓库的 `deploy/sprout-lm-pd.service`，`KillMode=mixed` 先让协调进程执行有序退出，超时才由 systemd 回收整个组。`deployment.json` 记录被终止的 Worker、三个旧 PID / 新 PID、GPU 占用和重启前后 `[17,11]` 的生成结果；协调进程不在 CUDA compute-process 列表中。最终服务状态见 `final-state.json`。

## 复现开发验证

```sh
npm run check
npm test
npm run build:cuda
npm run test:cuda-pd
REPORT_PATH=runs/pd-validation.json node scripts/check-pd-model.ts \
  models/qwen2.5-32b-instruct .cache/qwen32-reference
REPORT_PATH=runs/pd-api.json npm run bench:pd -- models/qwen2.5-32b-instruct
```

这些命令需要两张卡各能容纳完整模型；不要在已装载两份 32B 的卡上再起两份大模型。`PREFILL_DEVICE` / `DECODE_DEVICE` 可指定设备。当前原生构建目标是 Linux / A100 `sm_80`，Node 24.16.0，CUDA 13.0，驱动 580.159.03。

源码、部署配置和测试指纹见 `source-sha256.txt`，实验 / 部署目录校验见 `source-verification.log`，原生二进制指纹见 `binary-sha256.txt`。Python 仅用于离线部署验证，不参与推理服务。本次未提交或推送 Git。

## 当前边界

PD 路径尚未实现前缀缓存、抢占重算、按需增长 D 容量、多流复制与 D 计算重叠、tensor batch、跨机器 RDMA、无损重连或自动重试。Worker 内部仍同步执行；通过进程隔离让 API 和另一张卡继续工作。32B 独立数值只验证上述短样本，部署 2048-token 池容量不代表已完成长上下文质量 / 吞吐验收。
