# Profiler 采集与读图

只在采集 trace 或把已有 trace 写成优化方案时读取。没有 trace 时不要编造 op 时间。

## 采集

实现是 `vllm_ascend/profiler/torch_npu_profiler.py` 的 `TorchNPUProfilerWrapper`。

- `--profiler-config.profiler=torch`
- `--profiler-config.torch_profiler_dir=<dir>`
- 同时记录 CPU 与 NPU activity，Text 导出，`ProfilerLevel.Level1`，AIC 指标 `PipeUtilization`
- `additional_config.msmonitor_use_daemon` 与 torch profiler 不能同时开
- 先关 `with_stack`。需要 Python 栈时另采一次短窗口

窗口：warmup 之后采集 5–20 个稳态 decode step。prefill、图捕获、稳态 decode 分成不同 trace。捕获期的编译会淹没 TPOT。

`vllm_ascend/profiling_config.py` 的 service symbol 主要打在 v1 `NPUModelRunner.execute_model`、`_prepare_inputs`、`propose_draft_token_ids`、`rejection_sample`。它们不覆盖 `NPUModelRunner310V2`。缺 span 不等于 host 侧没有开销。MRV2 以 op timeline 为准。

需要自己的 span 时，使用当前 PyTorch 实际存在的 `torch.profiler.record_function`（旧版可用 `torch.autograd.profiler.record_function`）包住这些区间；不要假设存在 `torch.npu.profiler.record_function`。若扩展 service symbol，再确认路径是 `_310p.worker.v2`：

- prepare inputs
- 模型前向（target 与 draft 分开）
- `RejectionSampler310V2.__call__`
- `synchronize_cpu`
- `postprocess_sampled`
- draft propose

测量区间内部不要再调用 `torch.npu.synchronize()`，除非同步本身就是研究对象。看 side stream 是否与下一步前向重叠时，CPU 和 NPU activity 都要开。

## 读图

1. 对齐 CPU 与 NPU 时间线。NPU 空隙若落在 CPU 同步或小拷贝上，就是 host-bound。此时换一个更大的融合算子没有帮助。
2. 用 op summary 的 self time 排序，不用 inclusive time 给模型前向排名。
3. 分成四类再排：融合大算子、可融合小 vector 链、拷贝与格式转换、同步与 host 调用。
4. 读实际导出的表头。关注 shape、dtype、format。同步 token-id D2H、每步 ND↔NZ、行数等于 MRoPE 维数的非连续拷贝，都比算子大类更有用。若出现 `tokens × vocab` D2H，它是严重的额外回退，但不是 #16489 直接前序的正常基线。
5. PipeUtilization 低且时间线有气泡：先查 launch 和同步。pipe 低但 kernel 很长、没有气泡：再考虑算子替换或 layout。

## 现象对照

| 观察到 | 假设 | 下一步 |
|---|---|---|
| 同步两侧 NPU 空，CPU 在 prepare/postprocess | 记账回到 device，或过早全流同步 | 对照 optimization-map 的 #16489 不变量 |
| 同步 D2H draft/token id，随后 CPU Python 循环 | #16489 前序 greedy 路径被恢复 | 确认 NPU rejection kernel 与 side-stream 小结果 D2H |
| D2H shape 接近 `tokens × vocab` | logits 被非预期搬回 CPU | 查 rejection 输入附近的 `.cpu()` / `.numpy()`；不要把它误写成 #16489 的直接基线 |
| 大量微秒级 IndexCopy / Gather | 小向量没有 bulk H2D | `StagedWriteTensor`、采样参数、idx mapping |
| strided Memcpy，行数为 MRoPE dim | position 列切片 | 改回整块连续 copy 后重采同一步 |
| Pow、Mean、Rsqrt、Cast 成串 | norm 或 Q/K L2 没接 `npu_rms_norm` | 先做数值等价，再确认该串消失 |
| MoE 前有 Sort / Bincount | 没走 gating 或 init_routing kernel | 核对 softmax、FP16、非 group、routing-v2 与 group-list 消费契约 |
| 每步权重 shape 的 format cast | NZ 没留在 parameter 上 | 加载后 dump format |
| 稳态 decode 仍是完整 eager 链 | 没 replay | capture size 与 SpecDecoding override |
| replay 前后全流 sync | 空 handshake 被加回 | 先看小模型 TPOT |
| hybrid 每步大块 Zero 或额外 H2D | KV zero 或 RecoverSSM 过宽 | 用 dense 对照；关掉对应特性再采 |

## 写成方案

没有 op 名、shape 和 self time 占比，就不提出代码改动。方案固定四项：

1. 热点是哪一个 op/同步，shape、dtype、format 是什么，self time 或 CPU gap 占采集窗口的多少；
2. 准备删掉的同步 / 拷贝，或准备换上的已存在 kernel；
3. 等价条件：dtype、对齐、eps、layout、group、接受率；
4. 精度护栏和性能层级（见 aisbench-verification）分别是哪一层。

target 与 draft 的时间分开报。视觉 encoder 与 decode 分开报。图捕获不计入稳态 TPOT。
