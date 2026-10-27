# 310P 性能优化地图

只在需要判断“慢在哪类代码”或“这段优化有没有被回退”时读取。行号会变，以符号和数据流为准。实现在 `vllm_ascend/_310p/`。

## 总原则

每个 decode step 只做必须在 NPU 上做的计算。请求记账、切片计划和加载期 layout 留在 CPU 或权重加载阶段。310P 上 pageable H2D、逐元素同步和 strided copy 的成本经常大于一次小 kernel。

## 元数据与 MTP（#16489）

合入 `9d14ad3d`。MRV2 把上游 GPU runner 的 device 记账带回了 310P，本 PR 把 MRV1 的 CPU owner 模型接回去，同时把 greedy rejection 留在 NPU。

保持这些不变量：

1. `NPUModelRunner310V2` 的 seq lens、input ids、idx mapping、cu_num_logits、logits indices、sampled/rejected 使用预分配 pinned buffer 或 `CpuGpuBuffer`。step 内不要 `async_copy_to_gpu` 新 numpy，也不要 `torch.tensor` 临时上卡。
2. `_post_update_cpu` 用 CPU 上的 sampled / num_sampled / num_rejected 更新 `all_token_ids`、`last_sampled_tokens_cpu`、`total_len`、`num_computed_tokens`。`total_len` 与 computed token 通过 `stage_write_elem` 写入。直接改 `.np` 而不登记 dirty，NPU 会看到旧值。
3. `computed` 增量是 `query_len - num_rejected`。chunked prefill 的采样计数保持 0。
4. temperature≈0 走 `rejection_greedy_sample_spec_len_1_pytorch`（每请求恰好 1 个 draft）或 `rejection_greedy_sample_pytorch`。不要回到 `greedy_rejection_sample_cpu`：#16489 的直接前序已在 NPU argmax，但会同步 token id 并在 Python 逐 token 接受。`tokens × vocab` D2H 是更严重的额外回退，不是该直接前序的正常行为。
5. greedy 结果经 `_copy_stream` 异步 D2H。`synchronize_cpu` 仅在 `_copy_pending` 时等待，调用点在 `postprocess_sampled` 消费结果之前。propose 复用这份 host `num_rejected`。
6. temperature>0 保持 CPU Leviathan / IS_NGRAM 和 CPU generator。这条路径返回前填好 CPU buffer，并把 `_copy_pending` 置假。路径选择按 batch 内 `any_random` 决定：只要一条请求非 greedy，整批走 CPU 概率实现；不要用混合 temperature batch 衡量 greedy 优化。
7. `Ascend310PStagedWriteTensor.apply_write` 对小请求向量做 `gpu.copy_(cpu)`，不做逐 index `index_copy_`。UVA 路径不额外拷贝。
8. 采样参数写在 CPU，`apply_staged_writes` 只在 dirty 时 bulk H2D。
9. RoPE：decode position 直接写入持久 `positions_np`。H2D 拷贝整块 `positions_cpu`。不要改成 `positions[:, :num_tokens]`，storage 宽度是 `max_num_tokens + 1`，列切片在 310P 上是昂贵的 strided copy。
10. Mamba：`_num_accepted_tokens_cpu` 与 `_mamba_state_idx_cpu` 是计划数据。`_copy_mamba_state_from_cpu_plan` 只对跨 block 的 state 下发 NPU copy。conv 维序遵守 `is_conv_state_dim_first()`。循环里对 GPU index 调 `.item()` 即回退。
11. RecoverSSM 才把 accepted token 和 state index 拷上 NPU，然后把 GPU 结果读回 CPU。普通 hybrid step 不做这次物化。
12. KV zero 仅当同时满足：有 Mamba layer、eagle block drop、`num_speculative_tokens > 1`。见 `_needs_kv_cache_zeroing_310p`。

调度器可能在 step 之间改写 `num_computed_tokens`。post-rejection 的 CPU 值是 owner；发现被覆盖时按 CPU 恢复，不要用刚被 scheduler 写过的 GPU 值覆盖 CPU。

## MoE（#9105）

合入 `2ce60baa`。当前入口是 `AscendGroupedTopKRouter310` 和 `TokenDispatcherWithAllGather310`，不再是已删除的 `experts_selector.py`。

- softmax、非 grouped、无 `e_score_correction_bias`、无 custom routing：`npu_moe_gating_top_k_softmax`。logits 必须是 FP16 ND。token 数大于 1024 时按 1024 切块。expert id 转 int32。其他门控走父类。
- dispatch：`npu_moe_init_routing_v2`，`quant_mode=-1`，`row_idx_type=0`，`drop_pad_mode=0`。hidden 最后一维 `% 16 == 0`。当前合入组合把 `expert_tokens` 直接交给 `group_list_type=0` 的 grouped matmul。producer/consumer 语义必须成对验证；版本升级时用已知多 expert assignment 检查边界，不凭变量名增加或删除 `cumsum`。
- `w13` / `w2` 的 NZ 在 `process_weights_after_loading` 做一次。前向里再次 format cast 说明格式没保持住。
- EP 在 `NPUModelRunner310V2._validate_config` 拒绝。不要在 310P 上打开 `--enable-expert-parallel` 来“优化” MoE。

## Norm（#9489、#10408，以及相邻融合）

- 无 residual 的 RMSNorm：`npu_rms_norm`。有 residual 的 Qwen 路径：`npu_add_rms_norm`。
- Gemma：`npu_rms_norm(x, 1 + weight, eps)`。不要套用 Qwen 的 add+rms 融合或裸 weight。
- `AscendRMSNormGated310`：`group_size is None` 时按 `norm_before_gate` 决定 mul 在 rms 前还是后。activation 只有 sigmoid 与 silu/swish。`group_size` 不为空时必须留在 `forward_native`。
- GDN 的 Q/K L2：`l2norm_310p`。`weight = 1/sqrt(dim)` 且 `eps_rms = eps / dim`，默认 eps `1e-6`。漏除 dim 会在大 head 上偏离 L2。`use_qk_l2norm_in_kernel=False` 时不能多归一化一次。chunk 与 recurrent 两条路径都要接上。

## Attention 与图

- 310P attention 使用私有 ACL NZ 的 K/V。`supports_standardized_shared_kv_backing = False`。head size 不超过 256，`block_size * head_size <= 128 * 128`。
- prefill、chunked prefill、prefix-cache hit 不 replay FULL decode 图。
- MTP 均匀 batch 的 `q_len = 1 + K` 在 capture 时标成 `SpecDecoding`，不是 DecodeOnly。见 `ModelAclGraphManager310`。
- 没有 FIA/PA graph task 时跳过 update stream handshake，也不要补全流 `synchronize()`。这对小模型 MTP 的 TPOT 是直接回归。

## 检索时的起点

| 症状 | 先看 |
|---|---|
| MRV2 比 MRV1 慢，尤其 MTP | `rejection_sampler.py`、`_post_update_cpu`、`synchronize_cpu` |
| 开图不加速 | capture sizes、`_force_eager_*`、`ModelAclGraphManager310.make_dummy` |
| MoE 慢 | router 是否走到 softmax kernel，dispatcher 的 `group_list_type`，权重 format |
| hybrid 慢 | `l2norm_310p`、gated RMSNorm、`_copy_mamba_state_from_cpu_plan`、KV zero 门闩 |
| 多模态 TTFT 差、decode 正常 | MRoPE position 拷贝与视觉 encoder，不要和 decode TPOT 混结论 |
