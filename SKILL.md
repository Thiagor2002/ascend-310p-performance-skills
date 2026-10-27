---
name: ascend-310p-performance
description: >-
  定位、拆分并验证 Ascend 310P 上 vLLM-Ascend 的推理性能与优化。覆盖 MRV2 MTP 元数据、ACLGraph、MoE/RMSNorm/L2 算子、ais_bench 分项和 torch_npu profiler。面向 Qwen3/3.5/3.8 与 Gemma。在讨论 310P 吞吐、TTFT、TPOT、profiler、图模式或算子性能时使用。不用于纯精度定位。
---

# Ascend 310P 性能看护与优化

目标是把 310P 上的性能问题收成可复核的证据：哪一段 step 慢、慢在 host 同步还是算子、改动前后主指标和精度护栏各是什么。代码路径、服务指标、trace、数值/生成护栏是不同层证据，不能互相替代。PR 已合入不等于在当前 CANN/PTA、模型 shape 和配置上已经取得固定收益。

本技能不绑定特定 agent。用当前环境实际具备的文件读取和命令执行能力。没有 310P 或没有 profiler 产物时，只做代码路径分析并标明未实测。

## 按任务读取

- 判断热点属于哪类已有优化、检查代码有没有回退：读 [references/optimization-map.md](references/optimization-map.md)。
- 设计 ais_bench 或 curl 的分项性能实验：读 [references/aisbench-verification.md](references/aisbench-verification.md)。
- 采集或解读 profiler，并把图转成改动方案：读 [references/profiler-analysis.md](references/profiler-analysis.md)。
- 搜索并替换融合算子：读 [references/operator-substitution.md](references/operator-substitution.md)。
- 图模式、MTP、Qwen/Gemma 路径差异：读 [references/graph-mtp-models.md](references/graph-mtp-models.md)。

一次只读当前分支需要的资料。跨阶段时再打开下一份。

## 先分型

1. **回归**：以前更快，或 MRV1 快于 MRV2。先对照 optimization-map 的不变量，再做单变量复测。
2. **读图**：已有 profiler 或用户要求采集。先读 profiler-analysis，不要直接改算子。
3. **算子替换**：热点是小 vector 链、Python 路由或每步 layout 转换。即使用户没点名算子，也主动搜索当前仓库与当前 torch_npu/CANN 可用的候选；找不到受支持 kernel 就保持原实现，并交付候选淘汰理由和验证计划。
4. **服务调参**：图、MTP、并发、batch。一次只改一个变量，用 aisbench-verification 的层级。

精度回归交给 `ascend-310p-accuracy-troubleshooting`。用户同时要性能和精度时，性能候选仍要带本技能的精度护栏，但不能把性能实验当成精度根因定位。

## 工作流

### 1. 固定口径

能从命令、日志或源码得到的字段自己查，不重复问。至少记下：

- vLLM 与 vLLM-Ascend 的路径、branch、commit；CANN、PTA、驱动；
- 模型、量化、TP、物理卡、`ASCEND_RT_VISIBLE_DEVICES`；
- 图模式、`cudagraph_capture_sizes`、MTP 步数、prefix cache、max length / max seqs / batched tokens；
- 主指标（TTFT、TPOT、output throughput、request throughput 之一）和对比基线；
- 已有 trace、ais_bench 命令和用户明确不让改的开关。

当前树的 310P MRV2 不支持 MLA、sleep、DP、PP、prefill/decode context parallel、EP、非 MTP 投机和 KV transfer。LoRA、prefix caching、Qwen3-VL/MRoPE 与 Qwen3.5 hybrid 已在当前校验入口放行，不能沿用旧限制表误报。配置踩中真实限制时，先报告启动失败原因，不把它写成性能数字。

### 2. 建立可比较运行

沿用用户已有服务则先核对运行时 import 路径和 commit。新起服务时从确认过的 vLLM-Ascend 源码目录启动，记录 `pwd` 和 HEAD。每个候选重启进程，不复用 KV cache 或 prefix cache。不抢占已有 NPU 任务。

`enable_npugraph_ex=false` 保持为 310P 约束，除非用户给出该版本已经验证过的其他设置。

### 3. 取证

按成本递增，在前一层已经能分开两个候选时停：

1. 读代码，标出 step 路径上的同步、pageable 拷贝、小算子链、每步 format cast、图 replay 条件。
2. 单请求、temperature=0、eager、MTP 关，拉长输出，看稳态 TPOT。
3. 只打开可疑开关（图、MTP、某个算子）再测。
4. 短窗口 profiler，分开 prefill、图捕获和稳态 decode。
5. 目标并发的 ais_bench。只在前面几层已经解释得通时作为发布口径。

重复至少 3 次，报中位数和极差。warmup 不计入。OOM、捕获失败、启动失败记 blocked，不记成变慢。

### 4. 改动

用户要求修改时才改代码。一个补丁只动一个热点。方案里写明 op/同步名、shape、dtype/format、self time 或 host gap 占比、替换后的 kernel 或删掉的同步、等价条件、精度护栏和性能层级。

合入前的最低验证：数值或固定 prompt 的 token 对比，加上对应性能层级的前后中位数。图、MTP 或自定义算子还要跑覆盖该路径的单测。用户要求全量数据集时再跑全量，不把小样本说成榜单恢复。

## 判断原则

- 310P decode 经常先死在 host 同步和发射开销，而不是 cube 没融合。pipe 利用率低且时间线有气泡时，先查拷贝和同步。
- 记账留在 CPU，计算留在 NPU。greedy rejection 是计算；请求长度、slot、Mamba 计划是记账。
- pageable tensor 上的 `non_blocking=True` 不是真异步。元数据用预分配 pinned buffer。
- 小向量用一次 bulk H2D，不用逐 index `index_copy_`。二维拷贝保持连续，避免 310P 走 strided copy。
- `.item()`、`.tolist()`、全流 `synchronize()` 不放进每个请求的循环。允许的 MTP 同步点是结果 event，而且只在 CPU 即将消费时等待。
- 融合算子必须满足 dtype、对齐、layout 和归约定义。不满足的分支留在原路径。
- 加载时能做完的事（NZ、transpose、常量 weight）不要放进 token 循环。
- 只给真正的消费者付钱：KV zero、RecoverSSM 物化、概率采样的 CPU 路径都是窄门闩。
- 开图后仍看到完整 eager op 链，就是没有 replay，先查 capture size 和 attention state，不先换 kernel。
- MTP 报告必须带平均接受长度。接受率下降时先查正确性。
- rejection 的路径选择是 batch 级的；batch 中只要有一个非 greedy 请求就走 CPU 概率实现。greedy 性能实验使用同质 temperature=0 batch。
- `ignore_eos=True` 的性能结果不能当作精度证据。

## 交付

- 口径：commit、模型、量化、图、MTP、并发、主指标；
- 证据：代码位置或 trace 中的 op / shape / self time；未采到的层写未测；
- 证据级别：静态代码推断、单算子/单请求实测、profiler 证据、目标并发服务实测分别标注；
- 对比：单变量两侧的中位数和极差，以及精度护栏的结果；
- 结论对应到哪一层实验，不把单请求 TPOT 说成并发吞吐；
- 未改或未验证的风险，以及下一层最有区分度的实验。

## 边界

- 不升级、降级或卸载驱动、固件、CANN、PTA。只有证据指向该组件且用户明确同意时才动。
- 不终止其他任务，不批量删除数据集、trace 或构建目录。
- 不发明 `torch_npu` API。仓库和当前环境都找不到的算子保持原实现。
- 不把 service profiler 里缺失的 v1 span 解释成 MRV2 host 侧没有开销。
