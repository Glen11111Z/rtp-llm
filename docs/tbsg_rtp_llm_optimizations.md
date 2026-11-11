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
  - **负载前提（决定是否有收益）**：同一调度轮内准入 ≥2 个请求，**且**公共前缀 token 数 ≥ `SEQ_SIZE_PER_BLOCK`（默认 64）。典型满足：RAG 同文档多问、共享长 system prompt 的多路采样、会话树状分叉、批量评测同一 prompt 变体。典型不满足：同主题不同实体的在线流量（实测共享前缀仅 4–17 token，凑不满一个块，命中恒为 0）。
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
  - **GPU 实测（已完成，`catering_retail_qwen3_06B`，base `7cdc403` vs test1b 开启本优化）**：数值一致性——黄金集开关开/关输出逐 token 一致；轻载压测——QPS=5 × 60s × 301 请求，两构建成功率均 100%，TTFT 13.79ms vs 13.59ms、整请求 cost 75.37ms vs 75.52ms、复用率均 0.818，差异全在噪声内（<0.3ms），**无回归**；真实 batch 形态——用线上日志重建 dispatcher batch 冷启动回放，**批内私有命中全部为 0**，原因是同批共享前缀仅 5–23 字符（约 4–17 token）不足一个 64-token 块；调小块到 `SEQ_SIZE_PER_BLOCK=16` 后命中条数 2.9%→26.5%，但整体复用率仅 2.5%→3.5%，RT 无可测收益。附带结论：块=16 是全局复用的甜点（复用率 0.863、TTFT 最低 12.97ms），块=8 收益不再涨而 TTFT 反弹。
  - **验证缺口（待补）**：匹配叠加层单测（全局+私有拼接、`start_index`、device-only 不建 `async_context`、CP 分片）、`CoordinatorCacheManager` 插入分支单测、`FIFOScheduler` epoch 接线回归单测、`GenerateStateMachine` 三处注册时机单测、`StreamCacheResource`/`prefixCopy` 单测均未交付；Lazy Filling 写读序**未按 Attention backend 建立允许清单**（一致性测试通过属"未观察到失败"，非"已证明安全"）；生产者 forward 中途失败的注入测试未做；24h 浸泡未做（现有压测仅 60s，注册表长期稳定性未验证）。
- **已知边界与风险**：
  - **收益强依赖负载形态**：对纯全局树是弱占优（复用长度只增不减），但公共前缀已在全局树 device-ready、或不足一个 block、或批内仅单请求时收益为 0。当前餐饮零售在线流量属此类，**建议保持默认关闭**，待切换到长公共前缀 + 同轮高并发的流量再评估。
  - **数值正确性边界**：生产者中途失败时，兄弟已绑定的块因持有独立请求引用不会 use-after-free，但读到的可能是未写完的 KV。当前只保证不崩溃，不做跨请求数值回滚（与旧实现同一边界）。
  - **decode 流也会注册**：三处注册点中第三处（`incrKVBlock` 之后）不做 role 门控，故 decode 流（含 PD fallback 后的 PREFILL 流）同样注册。影响有限——epoch 停在准入轮，只对同 epoch 兄弟可见，而这些兄弟通常已结束；未做专项验证。
  - **回收存活 epoch 是安全的**：`reapStaleEpochs` 可能删掉 owner 仍存活的旧 epoch 条目（如长寿命 decode 流）。注册表只释放自己那一份引用，owner 的请求引用仍在，块不会被回收；owner 后续注销找不到条目返回 0（幂等），损失的只是本已无意义的批内可见性。
  - **热路径开销**：开启后每次 `malloc` 多一段 O(私有复用长度) 的哈希查找（在既有 cache 锁内），注册路径多一次 O(条目数) 的回收扫描；`batch_private_match_cost_us` 可直接观测。关闭时零开销。
- **设计文档**：《批内前缀复用迁移可行性分析》（三代 key 演进、cherry-pick 不可行实证、方案对比与风险评估）与《批内前缀复用落地实施方案》（as-built：总体设计、分阶段实施、接口清单、测试矩阵、风险、灰度判停、设计取舍 FAQ、实测报告索引）。两份为内部文档，当前未随本分支提交。

## 2. 模型优化

暂无。

## 3. 部署与运维优化

暂无。
