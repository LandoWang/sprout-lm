# 第四个进程：按 requestId 收集 Serving 事件

日期：2026-10-20（America/Los_Angeles）。

目标：让独立 Observer 接收 API / Service / Scheduler 与 Prefill / Decode Worker 的事件，能按一个 requestId 还原请求生命周期，并保持观测故障与推理故障分离。

## 实现与部署

- 三个推理进程各自通过 Unix stream socket 直连 Observer；不经过 Worker 的控制 IPC，不传 KV 数值。
- Observer 只做校验、有界排队和滚动 NDJSON 写入，不持有 CUDA context。
- 服务仍叫 sprout-lm-pd，部署目录改为 /mnt/nvme/sprout-lm/deployments/observer，开启 OBSERVABILITY=1。
- 原模型、GPU 分工、BF16 直接读取、CUDA Graph 与每卡 2048-token FP32 KV 保持不变。原生源文件没有修改，沿用上一版已验证的 CUDA 二进制。
- 上一版部署目录 /mnt/nvme/sprout-lm/deployments/pd 保留。previous.service 保存切换前的 unit。

验证时进程：协调 1405136，Observer 1405148，P 1405165，D 1405166。GPU 进程列表只有 P / D，各 63958 MiB。服务 NRestarts=0、健康、无待处理请求；这些 PID 是记录时状态，后续操作需重新检查。

## 真实 32B 请求

重放“什么是世界上最美的语言”，模型为固定 revision 的 Qwen2.5-32B-Instruct：

- requestId：f6f4a789-6dbf-4691-8d41-69f54dbc41aa。
- prompt 33 token，生成 136 token，以 EOS 结束，输出文字与此前问答一致。
- 收集到 1005 条请求事件，来源为 coordinator / prefill / decode。
- 共 145 个计划：9 个 prefill chunk、1 次 KV 交接、135 次 decode。
- 每个 plan 都有一次 sent / started / completed / completion_received。
- token.ready、token.published、http.token_written 各 136 条。
- 已核对首 token 早于交接 ACK，ACK 不晚于 P 页释放；最终资源释放一次，HTTP 正常结束。
- 本样例无丢失报告，查询未发现损坏行；不据此承诺任意负载下无损。

当前运行目录：

~~~text
/mnt/nvme/sprout-lm/runs/events/run-113ecaf8-8005-4359-b682-9f73d629eb29
~~~

在 A100 的新部署目录中查询：

~~~sh
node scripts/request-events.ts \
  /mnt/nvme/sprout-lm/runs/events/run-113ecaf8-8005-4359-b682-9f73d629eb29 \
  f6f4a789-6dbf-4691-8d41-69f54dbc41aa
~~~

本地保存的 raw-run 是仍在运行的 Collector 的截取，因此 manifest 的 cleanShutdown=false 是预期值；也可将 raw-run 目录传给同一查询工具。

## 验证与证据

- validation.log：A100 类型检查、45 项轻量测试、3 项真实双 GPU 集成测试通过。
- 双 GPU 微型模型完整 logits 最大绝对误差仍为 2.2351741790771484e-8。
- 两条并发集成请求分别记录 53 / 65 条事件，计划可跨三种来源逐一关联。
- 轻量故障测试覆盖 SIGSTOP / SIGKILL Collector 后推理可用、生产者缓冲有界、丢失标记、文件轮转、异常埋点以及取消后的延迟回收。未在常驻 32B 服务上故意终止 Collector。
- 32b-response.sse：真实流式响应。
- 32b-request-trace.json：按 requestId 查询的结果。
- summary.json：事件覆盖统计；deployment.log：进程、GPU 占用和健康检查。
- source-sha256.txt / source-verification.log：本地与部署源码对应关系；binary-sha256.txt：沿用二进制的校验。

这次验证事件覆盖和故障隔离，没有进行开 / 关观测的严格性能对照。已有 Worker 执行区间不是 GPU occupancy；KV 的 CUDA event 也不包含全部控制开销。

## 边界

事件是 best-effort。生产者默认 64 KiB 在途缓冲、单条 8 KiB；Collector 队列 1 MiB；每个运行目录最多 4 × 8 MiB 事件文件。取消后继续记录在途完成及回收。写入 barrier 仅用于关闭 / 显式诊断，有超时，不在推理热路径等待。

没有实时 UI、Perfetto 导出、跨机时钟对齐、Collector 自动重连或跨运行目录自动清理。Observer 失效时记录不完整，推理继续；重启服务才启动新的 Observer。详细协议见 ../../OBSERVABILITY.md。

## 回退

只在需要回退时，在 A100 执行：

~~~sh
sudo install -m 644 /mnt/nvme/sprout-lm/deployments/pd/deploy/sprout-lm-pd.service /etc/systemd/system/sprout-lm-pd.service
sudo systemctl daemon-reload
sudo systemctl restart sprout-lm-pd
~~~

这会使用原三进程部署。此次修改未提交或推送 Git。
