# Request events：独立观测进程

开启后，Serving 有三个推理进程，加一个独立 Observer：

~~~text
HTTP → API / GenerationService / PDScheduler
                  │ 控制 IPC       │ 控制 IPC
             Prefill Worker    Decode Worker
                  └── CUDA IPC / P2P KV ──→

协调进程 ───┐
P Worker ───┼── 独立 Unix stream socket ──→ Observer → 滚动 NDJSON
D Worker ───┘
~~~

Observer 不调度请求、不持有 CUDA context、不参与 KV 交接 ACK。三个生产者分别直连它，事件不经过推理 Worker 的控制 IPC。现有 CUDA 算子和同步点保持不变。

## 启动与查询

~~~sh
# 默认 Tiny CPU 模型也能观察 API / Service 事件
OBSERVABILITY=1 EVENT_DIR=runs/events npm run serve

# 已准备两张卡及模型时
OBSERVABILITY=1 EVENT_DIR=runs/events \
BACKEND=cuda-pd MODEL_DIR=models/qwen2.5-32b-instruct \
PREFILL_DEVICE=0 DECODE_DEVICE=1 CUDA_GRAPH=1 npm run serve

# observer_ready 启动日志给出运行目录和 Collector PID。
# request ID 来自 X-Request-Id 响应头、JSON id 或 SSE queued 事件。
npm run trace:request -- runs/events/run-UUID REQUEST_ID
~~~

OBSERVABILITY 默认关闭，设为 1 后在加载模型前启动 Collector；无法启动时打印诊断，服务继续。Collector 运行中退出不触发 GPU Worker 失败，也不自动重启 Collector；重新启动服务会建立新的运行目录与连接。

查询在独立 CLI 中读取保留文件，按同机单调时间排序，返回请求事件和全局丢失 / 进程生命周期诊断。查不到请求可能是尚未写入、事件丢失或超出保留范围。

## 事件协议

- version、runId：协议与服务运行标识。
- requestId：API 在读取请求体前生成；全链路复用，客户端不能指定。
- planId：Scheduler 为执行计划生成，贯穿发送、Worker 执行与完成回执。
- source、pid、seq：事件来源与进程内序号。
- timeNs：产生事件时的 process.hrtime.bigint()，用十进制字符串保留精度，只在同机时钟约定下比较。
- wallTimeMs：展示用途，不能用墙钟调整后的差值计算可靠耗时。
- name、data：事件及结构化字段。

不记录 prompt、输出文字、token 内容、权重或 KV 数值。计划事件包含 token 数、位置、源 / 目的页号。

| 范围 | 主要事件 |
| --- | --- |
| 请求与文本 | request.received、http.body_read、text.template_started/completed、text.encode_started、text.encoded |
| Service | service.queued、token.published（含缓冲深度）、service.buffer_full、generation.finished、request.cancelled/failed |
| 准入与等待 | scheduler.queued/admitted、request.waiting / wait_ended；区分名额、P 页、D 页、FIFO 前序请求、Worker 忙 |
| 执行计划 | plan.sent、plan.started、plan.completed、plan.completion_received、plan.execution_failed / failed |
| KV 交接 | kv.transfer_queued、kv.transfer_ack_received、kv.source_pages_released、decode.ready |
| 输出与资源 | token.ready、http.token_written、http.backpressure_started/ended、http.response_finished/closed、request.resources_released |
| 故障与进程 | worker.starting/ready/stopped/failed、scheduler.failed/stopping、workers.exited、serving.ready/stopped |
| 观测自身 | observer.started/stopped、observer.events_dropped、observer.producer_disconnected |

完整 Scheduler / Worker 事件目前覆盖 PD 后端。原 CPU / 单卡 Engine 路径只接入 API / Service 事件。

## 怎么读

- plan.sent → plan.started 包含控制 IPC 和 Worker 等待；plan.started → plan.completed 是 Worker 主机执行区间；plan.completed → plan.completion_received 包含完成消息返回。Worker 区间不等于 GPU kernel 时间。
- KV 的 cudaEventMs 复用已有 CUDA Event。没有新增逐算子事件或 GPU 同步；它不包含全部控制 / 排队成本，也不代表 GPU 利用率。
- 首 token 由 P 产生，可能先于 KV 交接返回。分别看 TTFT、首 token 到第二 token 的间隔、后续 decode 间隔。
- http.token_written 表示调用服务端 socket write，将数据交给 Node / socket 缓冲，不代表客户端收到。JSON 模式只在整份输出完成后写响应。
- 取消后继续记录在途完成（discarded=true）和资源释放。客户端关闭与 GPU 页回收是不同事件。
- Worker 内同步计算期间可能不能及时冲刷 socket；时间戳仍是产生时间。到达日志的先后不一定等于跨进程因果顺序，要结合 planId。
- 等待只记录原因变化，不在每轮重复刷相同状态。

## 有界与失败语义

每个生产者默认最多 64 KiB 在途事件，单条最多 8 KiB。Collector 最多接受 16 个连接，写盘队列默认 1 MiB。推理热路径不等待 Collector 回执；编码 / 提交仍有 CPU 成本，尚未给出 32B 性能开销结论。

队列或单条大小超限会丢弃观测事件；恢复可写后发出带数量 / 序号范围的丢失标记。Collector 队列丢失也写入标记。进程突然退出或磁盘失败可能使末尾记录与丢失标记本身来不及保存；查询始终按 best-effort 描述完整性。

每个运行目录保留最多 4 个 8 MiB 文件（事件文件合计最多 32 MiB）。轮转只移除该 Collector 自己创建的旧事件文件，manifest 记录轮转数量。每次服务启动使用独立目录；跨运行目录的保留策略尚未实现，应按实验周期归档 / 清理。

正常关闭先结束 GPU Workers、再冲刷协调端事件并结束 Collector。只有关闭或显式诊断使用有超时的写入 barrier；ACK 表示之前已接受的记录写入文件系统，不是逐条 fsync。cleanShutdown 表示 Collector 正常收尾，不保证每个生产者未丢事件。

## 验证

~~~sh
npm run check
npm test
# 两张 CUDA 卡：微型模型，不加载两份 32B
npm run test:cuda-pd
~~~

轻量测试覆盖真实 Collector 子进程与 HTTP、不同来源关联、生产者溢出、暂停 / 杀掉 Collector、磁盘轮转、异常埋点，以及取消后页回收顺序。双 GPU 测试核对每个 plan 的发送 / 开始 / 完成 / 回执与数值结果。

## Observatory 界面

~~~sh
npm run observe
# http://127.0.0.1:3001

# 一个或多个 event 根目录 / 单次运行目录
OBSERVE_PORT=3001 npm run observe -- runs/events /path/to/archived-run
~~~

默认读取 `EVENT_DIR`（未设置时为 `runs/events`），并提供 `snapshot/2026-10-20-request-events/raw-run` 的真实 A100 32B 实验。快照始终标为保存的实验；`cleanShutdown=false` 只表示没有正常收尾标记，不能作为服务存活判断。

界面在独立 Node 进程运行，只读日志，不连接执行 RPC 或 Collector、不持有 GPU、不启动推理。默认绑定 `127.0.0.1:3001`，没有修改 / 取消请求接口。静态资源本地提供，无 CDN 或上传。

- 左侧按 request ID 搜索和选择请求；顶部选择不同运行。
- 时间线可切到全请求、首 token 和 P → D 交接；支持拖选、Ctrl / ⌘ + 滚轮缩放和滑块平移。普通滚轮保留页面滚动。
- 按 API、准入、等待、下发 IPC、P、D / KV、回执 IPC、HTTP 背压以及 token 发布 / 写出分轨。
- 点击执行区间下钻 plan 的耗时、页号与原始字段，并联动筛选事件表；原始事件可搜索、分页、导出请求 JSON。
- 淡色区间显示同一运行里其他请求的 Worker 执行。Worker 间隙按所有保留请求的相邻计划计算，不把其他请求的执行误当成空闲。
- 每 3 秒刷新可选，默认关闭；这是滚动文件轮询，有写盘与传输延迟，不是同步实时 profiler。

### 指标口径

- Service TTFT：`request.received → token.published(index=0)`。HTTP TTFT：`request.received → http.token_written(index=0)`。两者均不是客户端观测值。JSON 响应没有逐 token HTTP 写出事件，显示 `—`。
- 首 → 第二 token 和 ITL：同一观测点的相邻 token index 差值。缺少 index 0 不用后续 token 冒充 TTFT；缺号不跨缺口计算相邻间隔。P50 / P95 使用 nearest-rank。
- 生成总时长到 generation / cancellation / error 终态；取消后仍展示在途完成与资源释放。
- 各类耗时是完整区间的并集：准入队列与等待原因重叠部分不重复计数；不同类别可能重叠，所以不能直接相加当作端到端耗时。首 token 拆分裁剪到其发布时间。
- 未闭合 Worker / 等待区间用透明虚线显示到最后一条请求记录，不计入完整耗时统计。缺起点或终点时指标显示缺失，不能解释为零开销。
- KV `cudaEventMs` 仅展示已有拷贝测量；尚无 GPU kernel / stream / utilization 或锁持有者的测量。

### 读取与资源边界

每次读取固定当时文件长度，忽略末尾未写完行，下次刷新重读；兼容轮转时文件消失。校验协议、runId 和纳秒字符串，对来源 / PID / seq 去重、报告内部序号缺口。跨源乱序按单调时钟排序。序号缺口也可能是尚未送达，刷新后可能消失。

Viewer 每次最多扫描 64 MiB，并用环形窗口保留最近 25,000 条有效事件；更老记录省略会明确提示。Collector 原有单次运行保留上限仍是 32 MiB。最多列出 100 个运行，缓存两个分析结果 2 秒，最多同时读取两个运行。查询字符串不能指定任意文件路径，只能选择启动时配置目录中发现的运行。UI 不回压推理，但磁盘读取与 JSON 分析仍有 CPU / I/O 成本；大负载下可复制日志到另一台机器离线分析。

在 A100 使用时，将相关 Viewer 文件放入部署目录，再在该机器启动 `npm run observe -- /mnt/nvme/sprout-lm/runs/events`；本地通过 `ssh -N -L 3001:127.0.0.1:3001 yuawang-a100x2` 访问。无需重启推理服务。当前界面的本地开发不等于已经部署到 A100。

尚无 Perfetto 导出、Prometheus 指标或跨主机时钟对齐。
