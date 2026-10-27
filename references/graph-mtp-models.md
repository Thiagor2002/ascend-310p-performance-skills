# 图模式、MTP 与模型差异

在实验涉及 ACLGraph、MTP、Qwen 系列或 Gemma 时读取。

## 图模式

310P MRV2 使用 ACLGraph，对外配置名仍是 cudagraph。

- `FULL_DECODE_ONLY`：mixed mode 为 NONE，prefill 和 prefix-cache hit 保持 eager。这是默认应对比的图模式。
- 若配置里仍有 mixed FULL，`PrefillCacheHit` 和 `ChunkedPrefill` 必须强制 eager，不能 replay decode 图。
- capture size 的单位是 token，不是请求数。纯 decode 的 batch 4 是 4 token；MTP1 的 batch 4 是 8 token。尺寸对不上就整步 eager。
- `ModelAclGraphManager310` 在 MTP capture 时把 `q_len = 1+K` 的 dummy batch 标成 `AscendAttentionState.SpecDecoding`。上游 dummy 默认 DecodeOnly，抓错之后 replay 与运行时不是同一条 attention。
- 310P 没有 FIA/PA `graph_task`。`resolved_tasks` 为空时跳过 update↔compute handshake。不要为了“保险”加 `torch.npu.synchronize()`，小模型 MTP 的 TPOT 会先退。
- seq_lens、slot mapping 在 compute stream 上刷新。同流顺序足够，不需要额外的全流同步。
- `enable_npugraph_ex=false` 保持，除非该版本有单独的验证记录。
- 性能数字把 TTFT、捕获阶段和稳态 TPOT 分开。开图的第一次长 prompt 含捕获成本。

稳态 decode 的 trace 若仍是一长串 attention 小算子而不是 graph replay，先修 capture，不换融合算子。

## MTP

310P MRV2 的投机解码只接受 `method="mtp"`。

- K=1 与 K>1 用不同的 greedy kernel。变长 draft 走通用 kernel。不要用 K=1 的结果代表 K>1。
- greedy 在 NPU 上做 argmax 和接受判定；D2H 只搬 token id 和两个计数，并且推迟到 CPU 记账。
- 概率采样保持 CPU。性能实验用 temperature=0，避免两条路径混在一个 TPOT 里。
- sampler 用 batch 级 `any_random` 分叉：任一请求 `temperature >= 1e-5`，整批进入 CPU 概率实现。greedy 与随机请求分批测。
- chunked prefill 行的 `num_sampled` 必须是 0，否则会把未完成的 prefill 写进请求状态。
- draft propose 与 target verify 分开计时。优化 target 时 draft 模型会稀释端到端收益，反之亦然。
- 报告平均接受长度。收益约等于少做的 target step，减去 draft 成本和 rejection 成本。接受率掉了，先查 sampler，不查 matmul。
- KV zero 只在 Mamba + eagle block drop + K>1 时开启。dense MTP1 不应出现每步清零。
- RecoverSSM 的 GPU 物化不是普通 MTP step 的一部分。

## 模型

| 模型 | 性能上真正要看的路径 | 不要误用 |
|---|---|---|
| Qwen3 dense | attention NZ KV、`npu_rms_norm` / `npu_add_rms_norm`、图 replay、host 记账 | 不要开 EP；没有 MoE routing |
| Qwen3 MoE | 上述项，加上 FP16 gating、routing v2、加载期 NZ、hidden 对齐 16 | 不要开 EP；group 门控不要塞进 softmax kernel |
| Qwen3-VL | dense 路径 + MRoPE 整块 position 拷贝 + 视觉 encoder | 视觉时间不要算进 decode TPOT |
| Qwen3.5 / Qwen3.8 | 基于 Qwen3.5 架构的 hybrid：GDN 的 `l2norm_310p`、gated RMSNorm、conv/ssm 的 CPU 计划、MTP | 先按 checkpoint `architectures` / config 确认具体变体；chunk 与 recurrent 都要测；eps 必须除 dim |
| Gemma | 310P norm 参照为 `npu_rms_norm(x, 1+weight, eps)`；Gemma4 MTP 在通用 V2 有专门实现 | 不要自动推断该通用 V2 模块已被 310P MRV2 放行；先过 `_validate_config` 和真实启动 |

Qwen3.5 / 3.8 的线性注意力在 prefill 走 chunk、decode 走 recurrent。只压测其中一个会漏掉另一条 L2Norm 或 state 路径。

Gemma 的 RMSNorm 定义与 Qwen 不同。仓库里的 `AscendGemmaRMSNorm310` 是 norm 数学的正确参照。通用 V2 的 Gemma4 MTP 支持不能自动外推为 `_310p` 端到端支持；先检查模型注册、310P runner gate、attention/KV layout 和真实启动，再决定可测范围。

## 不支持就不要测性能

当前 `_validate_config` 会拒绝 MLA、sleep、PP、DP、prefill/decode context parallel、EP、非 MTP 投机和 KV transfer；它已放行 LoRA、prefix caching、Qwen3-VL/MRoPE 与 Qwen3.5 hybrid。这些边界会随代码演进，执行时以当前校验函数和测试为准。真实失败要记录异常原文，不要关掉检查来得到一个“性能数”。
