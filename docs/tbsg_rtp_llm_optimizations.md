# 淘宝闪购 RTP-LLM 引擎自定义优化集合

记录 `dev/tbsg_rtp_llm_master` 分支上淘宝闪购 RTP-LLM 推理引擎的自定义优化（不含上游 RTP-LLM 已有的官方优化），后续新增在对应分类追加。

## 1. 框架优化

### 1.1 批内前缀复用（intra-batch prefix reuse）

- **优化功能简介**：允许**同一调度轮（同 `batch_epoch`）内**的多个独立请求复用尚未完成计算的公共前缀 KV Block，采用"树外私有注册表 + 全局树叠加匹配"设计。先准入的 stream 在进入 `RUNNING` 前，把自己已分配的完整块前缀按 epoch 注册为"批内可见"；同 epoch 的兄弟 stream 在 `malloc` 阶段匹配到"全局 device-ready 前缀 + 私有尾段"，跳过该段的 prefill 计算与 KV 写入，并与生产者**共享同一批物理块**（同时省下显存）。私有块从不进入 `TreeNode`，也不进入发布面、`BlockTreeStorer`、`BlockTreeEvictor`；请求正常完成时按原有 `epoch=0` 路径插入全局树，注册表条目随 owner 退出即注销。三项收益：prefill 计算省 `(兄弟数−1) × 前缀块数` 个 token、复用块不重写 KV、device 显存按兄弟数去重。
  - **匹配语义**：复用长度 = `max(全局 device-ready 前缀 M, 私有全量前缀 N)`。私有段只作为**被复用前缀的尾段**拼接（复用区间始终从 token 0 开始，不是后缀匹配），遇 key 缺失、group 不完整或 `valid_prefix` 截断即停止（截断到空洞前）。"全局优先"而非旧实现的"先私有再 fallback 全局"是有意选择：两者复用长度等价，但全局优先把"同 forward 写先于读"这一假设的暴露面限制在增量尾段，同时保持全局节点的 LRU 热度与 `is_resident` 语义不被遮蔽。
  - **关键实现点**：`MultiNodeResource` 新增 `start_index` 显式携带段起点（原 `matched_device_blocks - node_blocks.size()` 反推公式只支持单段），`prepareKVCache` 绑定处加不变量断言；`epoch > 0` 时**无条件 device-only**——不构造低层加载描述符、不构造 `StorageRequest`、不产生 `async_context`；注册表是按 canonical key 的扁平索引（`key → Entry{epoch, blocks, owners}` + `(epoch, owner) → keys` 反向索引），同 epoch 多 owner 共享条目、跨 epoch 同 key 跳过，连续性由匹配循环保证而非表结构。
  - **正确性前提**：依赖同一次 forward 内"生产者逐层 KV store 先于兄弟 Attention read"（Lazy Filling）。Epoch 只约束 CPU 侧缓存可见性，**不提供 CUDA 同步**。
- **适用范围**：
  - **拓扑**：所有 GroupSet 均为 `FULL`。含 SWA/Linear 的混合拓扑下自动失效（`match` 把 epoch 降级为 0、注册分支 warn-once 跳过），因为任意尾段无法叠加到非全前缀组上。
  - **调度器**：仅 `FIFOScheduler` 产生正 epoch（`nextBatchEpoch()` 返回 `++batch_epoch_counter_`）；`FIFOSchedulerBase` 默认返回 0，故 `BatchDecodeScheduler`、`PDFusionRatioScheduler` 等不生效。epoch 只在**首次准入**时下发（`!hasEvent(CanRun)`），已准入流保留原 epoch。
  - **请求**：`reuseCache()` 为真、存在 cache manager、`curBlocksNum() > 0`、`batchEpoch() > 0`；`epoch <= 0` 时注册被拒绝并告警，防止批内数据按 epoch 0 提升进全局。
  - **负载前提（决定是否有收益）**：同一调度轮内准入 ≥2 个请求，**且**公共前缀 token 数 ≥ `SEQ_SIZE_PER_BLOCK`（默认 64）。典型满足：RAG 同文档多问、共享长 system prompt 的多路采样、会话树状分叉、批量评测同一 prompt 变体。典型不满足：同主题不同实体的在线流量（共享前缀常不足一个块，命中为 0）。
  - **明确不支持（v1 边界）**：跨调度轮/跨 batch 复用；私有条目向 HOST/DISK/Remote 导出；`epoch>0` 时从低层加载全局段；原子提交/回滚接口（`commitEpoch`）；生产者失败时的跨请求数值回滚。
- **代码目录**：
  - 私有注册表（新增）：[BatchPrivateBlockRegistry.h](../rtp_llm/cpp/cache/block_tree_cache/batch_private/BatchPrivateBlockRegistry.h)、[BatchPrivateBlockRegistry.cc](../rtp_llm/cpp/cache/block_tree_cache/batch_private/BatchPrivateBlockRegistry.cc)
  - 匹配叠加：[BlockTreeLoader.cc](../rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.cc)（`createMatchResult` 的全局段 + 私有尾段拼接、device-only 短路）、[BlockTreeLoader.h](../rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h)、[GroupSet.h](../rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h)（`MultiNodeResource::start_index`）
  - 缓存门面与统计：[BlockTreeCache.cc](../rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.cc)、[BlockTreeCache.h](../rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h)（`registerBatchPrivate` / `dropBatchPrivate` / `dumpBatchPrivate` / `supportsBatchPrivate`、`batch_private_max_entries`、`CacheStats::batch_private_entries`）
  - 分配与插入路径：[CoordinatorCacheManager.cc](../rtp_llm/cpp/cache/CoordinatorCacheManager.cc)（`prepareKVCache` 段绑定与断言、`insertIntoCache` 的 `epoch>0` 分支、`batchPrivateTopologySupported`）、[KVCacheManager.cc](../rtp_llm/cpp/cache/KVCacheManager.cc)（注销/诊断转发）
  - 资源类型：[Types.h](../rtp_llm/cpp/cache/Types.h)（`MallocInfo::epoch`、`InsertInfo::epoch` / `owner_stream_id`）、[KVCacheResource.cc](../rtp_llm/cpp/cache/KVCacheResource.cc) 与 [BatchKVCacheResource.h](../rtp_llm/cpp/cache/BatchKVCacheResource.h)（`prefixCopy` 轻量前缀切片）
  - 引擎侧：[StreamCacheResource.cc](../rtp_llm/cpp/engine_base/stream/StreamCacheResource.cc)（`insertBatchPrivateBlocks` / `releaseBatchPrivateBlocks`、epoch 下发到 `malloc_info`）、[GenerateStateMachine.cc](../rtp_llm/cpp/engine_base/stream/GenerateStateMachine.cc)（三处注册时机，均紧邻 `transitionToRunning()` 之前）、[GenerateStream.cc](../rtp_llm/cpp/engine_base/stream/GenerateStream.cc)（`batch_epoch_`）
  - 调度侧：[FIFOScheduler.cc](../rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.cc)、[FIFOSchedulerBase.cc](../rtp_llm/cpp/engine_base/schedulers/FIFOSchedulerBase.cc)（`nextBatchEpoch()`）、[NormalEngine.cc](../rtp_llm/cpp/normal_engine/NormalEngine.cc)（构造传参，修复上游接线回归）
  - 配置与指标：[ConfigModules.h](../rtp_llm/cpp/config/ConfigModules.h)、[ConfigInit.cc](../rtp_llm/cpp/pybind/ConfigInit.cc)（pybind + pickle v9）、[kv_cache_group_args.py](../rtp_llm/server/server_args/kv_cache_group_args.py)、[ResourceContext.cc](../rtp_llm/cpp/engine_base/stream/ResourceContext.cc)、[RtpLLMMetrics.cc](../rtp_llm/cpp/metrics/RtpLLMMetrics.cc)、[BlockTreeCacheMetricsReporter.cc](../rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.cc)
  - 单测：[BatchPrivateBlockRegistryTest.cc](../rtp_llm/cpp/cache/block_tree_cache/test/BatchPrivateBlockRegistryTest.cc)、[BlockTreeCacheTestUtils.cc](../rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.cc)（`makeStructuralDevicePool` 增加 `preallocate_all_blocks` 开关）、对应 `BUILD`
  - 改动规模：40 个文件（新增 3 / 修改 37），约 +1214 / −55 行
- **执行与启用**：开关 `--enable_reuse_cache_in_batch`（env `ENABLE_REUSE_CACHE_IN_BATCH`，`type=str2bool`，**默认 `False`**）。贯通链路：CLI/env → `KVCacheConfig` → pybind 与 pickle（版本 v8→v9，71 元组，`__setstate__` 识别 v9 后切片降级复用 v8 路径，旧 pickle 可加载且字段缺省 false）→ `ResourceContext` → `NormalEngine::initScheduler` → `FIFOScheduler`。关闭时 `nextBatchEpoch()` 恒返回 0，全链路短路，行为与上游 HEAD 完全一致。回滚只需关配置并重启：无持久化数据结构，pickle v9 对旧版本向后兼容，降级也安全。
  - **这是本优化唯一暴露的可调参数**。其余为写死的 v1 边界：容量兜底 `BlockTreeCacheConfig::batch_private_max_entries{4096}`（常量，且 `BlockTreeCacheFactory` 未接线到 `KVCacheConfig`，是已知待补项）、FULL-only 拓扑门槛、`epoch>0` 无条件 device-only、空洞截断、注册生产者全量前缀、epoch 仅首次准入下发且单调递增、回收搭在注册路径同步执行（无后台线程）、拓扑告警 warn-once。
- **可观测性与判停**：`RtpLLMCacheReuseMetrics` 组新增 5 个 gauge——`batch_private_registered_blocks`、`batch_private_matched_blocks`、`batch_private_reclaimed_blocks`、`batch_private_active_epochs`、`batch_private_match_cost_us`；另有 `CacheStats::batch_private_entries` 与调试接口 `dumpBatchPrivate()`（仅日志输出 epoch/owner/条目数）。判停标准（任一成立即关闭并复盘）：`matched_blocks` 长期为 0 → 负载不满足前提，功能无效；`reclaimed_blocks` 持续非零 → 容量兜底被触发，4096 上限偏小或注销不及时；`active_epochs` 不随批次结束归零 → 注销路径泄漏；输出一致性抽检出现差异 → 立即关闭并排查 backend 写读序。
- **验证**：
  - **单测（已交付）**：`BatchPrivateBlockRegistryTest` 10 例，CPU 可跑，覆盖注册与同 epoch 命中、空/尺寸不符/块不完整的拒绝、同 owner 重注册幂等、第二 owner 共享条目至最后一次 drop、跨 epoch key 跳过、未知 owner 返回 0、`reapStaleEpochs` 回收、`reclaimToLimit` 淘汰最旧且**最新 epoch 永不淘汰**、`dump` 对空表/活表安全；并验证单块引用轨迹 `malloc → 请求 incRef(1) → 注册(2) → 兄弟匹配(3) → 生产者释放(2) → 注销(1) → 兄弟释放(0)`。执行：`bazel test //rtp_llm/cpp/cache/block_tree_cache/test:batch_private_block_registry_test`。
  - **实测（已完成）**：数值一致性——黄金集开关开/关输出逐 token 一致；轻载压测——两构建成功率无差异，TTFT 与整请求 cost 差异在噪声内，**无回归**；真实 batch 形态——用线上日志重建调度批冷启动回放，**批内私有命中为 0**，原因是同批共享前缀不足一个块；调小 `SEQ_SIZE_PER_BLOCK` 后命中率上升，但整体复用率与 RT 无可测收益。附带结论：较小的块对全局复用更友好，收益不随块继续变小而增长。
  - **验证缺口（待补）**：匹配叠加层单测（全局+私有拼接、`start_index`、device-only 不建 `async_context`、CP 分片）、`CoordinatorCacheManager` 插入分支单测、`FIFOScheduler` epoch 接线回归单测、`GenerateStateMachine` 三处注册时机单测、`StreamCacheResource`/`prefixCopy` 单测均未交付；Lazy Filling 写读序**未按 Attention backend 建立允许清单**（一致性测试通过属"未观察到失败"，非"已证明安全"）；生产者 forward 中途失败的注入测试未做；长时间浸泡未做（现有压测时长较短，注册表长期稳定性未验证）。
- **已知边界与风险**：
  - **收益强依赖负载形态**：对纯全局树是弱占优（复用长度只增不减），但公共前缀已在全局树 device-ready、或不足一个 block、或批内仅单请求时收益为 0。当前线上流量多属此类，**建议保持默认关闭**，待切换到长公共前缀 + 同轮高并发的流量再评估。
  - **数值正确性边界**：生产者中途失败时，兄弟已绑定的块因持有独立请求引用不会 use-after-free，但读到的可能是未写完的 KV。当前只保证不崩溃，不做跨请求数值回滚（与旧实现同一边界）。
  - **decode 流也会注册**：三处注册点中第三处（`incrKVBlock` 之后）不做 role 门控，故 decode 流（含 PD fallback 后的 PREFILL 流）同样注册。影响有限——epoch 停在准入轮，只对同 epoch 兄弟可见，而这些兄弟通常已结束；未做专项验证。
  - **回收存活 epoch 是安全的**：`reapStaleEpochs` 可能删掉 owner 仍存活的旧 epoch 条目（如长寿命 decode 流）。注册表只释放自己那一份引用，owner 的请求引用仍在，块不会被回收；owner 后续注销找不到条目返回 0（幂等），损失的只是本已无意义的批内可见性。
  - **热路径开销**：开启后每次 `malloc` 多一段 O(私有复用长度) 的哈希查找（在既有 cache 锁内），注册路径多一次 O(条目数) 的回收扫描；`batch_private_match_cost_us` 可直接观测。关闭时零开销。
- **设计文档**：《批内前缀复用迁移可行性分析》（三代 key 演进、cherry-pick 不可行实证、方案对比与风险评估）与《批内前缀复用落地实施方案》（as-built：总体设计、分阶段实施、接口清单、测试矩阵、风险、灰度判停、设计取舍 FAQ、实测报告索引）。两份为内部文档，当前未随本分支提交。

### 1.2 chat 前端模板编译缓存（chat template compile cache）

- **优化功能简介**：chat completions 前端渲染 prompt 时，原实现**每个请求**都新建 Jinja2 `Environment` 并从模板源编译一次 chat template。大模板（Qwen 系等）单次编译开销明显高于渲染本身，且在前端事件循环内串行执行，成为高并发下的吞吐上限与 RT 拐点来源。优化把编译产物缓存在 renderer 实例内：首次请求懒编译，之后只执行 `template.render(...)`，渲染语义与输出均不变。
- **实现要点**：
  - `_build_prompt()` 改为调用新增的 `_get_compiled_template()` 获取模板后再渲染；异常包装范围与原实现一致。
  - `_get_compiled_template()` 以 `self.chat_template` 源串为 key（`_compiled_template_src`）：命中则直接返回 `_compiled_template`；未命中时创建 `Environment`、应用 `_customize_jinja_env`、`from_string` 编译后缓存。
  - `Environment` 参数、子类环境定制钩子、渲染上下文均未改变；`chat_template` 若被替换会按源串自动重编译。
- **正确性边界**：
  - 只缓存编译产物，渲染输出与逐请求编译一致；
  - 编译后模板的 `render` 为只读操作，跨请求复用不改变线程模型；
  - 仅作用于经过 renderer 的入口（如 OpenAI chat completions），不经过 renderer 的入口不受影响；
  - 首个请求仍支付一次编译成本；模板越大、QPS 越高，收益越明显。
- **代码目录**：[reasoning_tool_base_renderer.py](../rtp_llm/openai/renderers/reasoning_tool_base_renderer.py)（`_build_prompt` / `_get_compiled_template`）；commit `a1c084099`。
- **执行与启用**：无开关，默认生效；回滚直接 revert 对应 commit，无配置与数据结构变更。
- **验证**：已通过压测与线上验证——高并发下前端 RT 不再出现阶跃，吞吐随并发近线性增长，进程稳定无异常重启；渲染结果与修复前一致。

## 2. 模型优化

暂无。

## 3. 部署与运维优化

### 3.1 /batch_infer 的 chunk 级 FE 指标（batch_infer chunk-level metrics）

- **功能简介**：为前端 `/batch_infer` 入口补充 chunk 级可观测性，直接回答“一个 batch_infer chunk 的端到端 RT 是多少、前处理占多少、每 chunk 带多少 prompt、成功/失败速率”。指标均为 FE 侧采集、经监控系统上报，与引擎侧 `py_rtp_*` / `ft_*` 指标并列，不改变推理行为。
- **指标清单**（定义见 [kmonitor_metric_reporter.py](../rtp_llm/metrics/kmonitor_metric_reporter.py)，`AccMetrics` / `GaugeMetrics`）：
  - `ft_batch_infer_e2e_rt`（Gauge，ms）：**batch RT 本体**。`/batch_infer` handler 入口 → `ORJSONResponse` 成功构造完成，**不含 socket 写回**；仅统计成功的 chunk。
  - `ft_batch_infer_pre_rt`（Gauge，ms）：其中“前处理段”——`Pipeline.batch_infer` 入口 → `batch_enqueue` 之前（build config + tokenize + 组装 inputs）。恒有 `pre_rt ≤ e2e_rt`。
  - `ft_batch_infer_prompt_count`（Gauge）：每个成功 chunk 携带的 prompt 数。
  - `ft_batch_infer_qps`（Acc）：每个成功 chunk 计 1，tags `rank_id` / `server_id`。
  - `ft_batch_infer_error_qps`（Acc）：`except BaseException` 中每个失败 chunk 计 1，tags 同上；失败 chunk 不计入 RT / prompt_count / qps。
- **代码目录**：
  - 指标定义：[kmonitor_metric_reporter.py](../rtp_llm/metrics/kmonitor_metric_reporter.py)（`BATCH_INFER_QPS_METRIC`、`BATCH_INFER_ERROR_QPS_METRIC`、`BATCH_INFER_PRE_RT_METRIC`、`BATCH_INFER_E2E_RT_METRIC`、`BATCH_INFER_PROMPT_COUNT_METRIC`）
  - 计时与上报：[pipeline.py](../rtp_llm/pipeline/pipeline.py)（`batch_infer` 入口取 `begin_time`，`batch_enqueue` 前上报 `pre_rt`）、[frontend_server.py](../rtp_llm/frontend/frontend_server.py)（`_report_batch_chunk` / `_report_batch_chunk_error`，成功路径构造 response 后、异常路径 re-raise 前上报）
  - 改动规模：3 个文件，约 +40 / −1 行（commit `709762085`）
- **查询方式与口径注意**：
  - 监控侧 PromQL 可按服务维度的 app 标签过滤。
  - 平台指标查询工具的指标目录带服务标签白名单，暂不包含这 5 个名字；需要走监控 API/大盘直接查。
  - Gauge 有约 1 分钟的上报/落库延迟，发完请求立刻查可能为空。
  - `ft_batch_infer_qps` 用 1s 步长的 `avg` 会被零值摊薄（低频流量下显示 0），看瞬时速率用 `max`/`sum`。
- **验证（已完成）**：直连 `POST /batch_infer` 发送批量请求，均返回 HTTP 200；监控侧五个指标正常上报，`ft_batch_infer_e2e_rt` 不小于 `ft_batch_infer_pre_rt`（满足 `pre_rt ≤ e2e_rt`），`prompt_count` 与请求批量一致，`error_qps` 为 0。

