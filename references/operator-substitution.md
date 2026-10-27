# 算子搜索与替换

在 profiler 或代码审查已经把热点收成“小 vector 链、Python 路由、每步 layout 转换”时读取。用户没有点名算子，也可以做这一步。用户只要求解释时，给出候选和验证计划，不改代码。

## 什么时候搜

搜：当前 step 里重复出现、且数学是固定归约或固定重排的 op 链。已有样例是 `npu_rms_norm`、`npu_add_rms_norm`、`npu_moe_gating_top_k_softmax`、`npu_moe_init_routing_v2`、`maybe_trans_nz`。

不搜：attention 主 kernel 的内部实现、通信算法、调度策略。这些不是换一个 `torch_npu` 调用就能结束的。也不要为了“还有更快算子”替换已经融合且 profiler 占比很低的 op。

## 怎么搜

1. 在本仓库搜同名数学的现有 310P 或 Ascend 调用，优先复用已落地的封装。
2. 在当前环境的 `torch_npu` 里确认符号存在，并记录 torch_npu/CANN 版本。不存在就淘汰该候选、保持原实现。不要编造函数名或参数。
3. 读该调用在仓库中的已落地样例和测试，再核对当前版本文档：dtype、连续性、ACL format、维度对齐、输出语义、图捕获支持。
4. 先做独立 microbench，覆盖目标 shape、warmup 和同步边界。只有新 kernel 的数值、耗时和 workspace 都可接受，才接服务路径。
5. 对照原式，写出必须改写的参数。已知坑：
   - L2 经由 RMSNorm：`weight = 1/sqrt(dim)`，`eps_rms = eps / dim`。
   - Gemma RMSNorm：缩放是 `1 + weight`。
   - Qwen residual norm 才能用 `npu_add_rms_norm`。
   - MoE gating：FP16 ND，单次调用 token 上限 1024，非 group 且无 correction bias。
   - MoE routing v2：hidden `% 16 == 0`；当前代码把 routing-v2 的 `expert_tokens` 直接交给 `group_list_type=0` 的消费者。用已知多 expert assignment 验证成对语义，不凭名字自行 `cumsum`。
   - NZ 放在 `process_weights_after_loading`，不放进前向。

## 必须留下的回退

- grouped RMSNorm；
- grouped topk、custom routing、带 score correction 的门控；
- 不支持的 activation 按当前接口契约显式失败；只有上游要求支持且数值验证通过时才增加 native fallback；
- `temperature > 0` 的 rejection；
- `use_qk_l2norm_in_kernel=False`；
- 输入不满足对齐或 dtype 时的明确失败，或已有的 native 路径。不要静默 pad 出每步都在的临时 buffer，除非证明它比拒绝更便宜且数值一致。

图捕获区域内不能新增数据相关的 Python 分支或 host 同步。捕获期形状不合法时，把该算子留在 eager 路径，或改成 capture-stable 的写法后再抓图。

## 验证

按顺序做，前一步失败就停：

1. 数值：与替换前的实现或 CPU 参考比 max abs 和余弦。覆盖 rank、dtype、group 开闭、`norm_before_gate` 两种顺序、chunk 与 recurrent。CPU UT 若 mock 了 `npu_rms_norm`，mock 必须实现同一公式，不能用空 MagicMock。
2. 单算子性能：目标 shape 至少 3 轮，报中位数；同步边界一致，不能让旧实现多一次 synchronize。
3. 服务：aisbench-verification 的 B、F 或 G。主指标用中位数。
4. profiler：目标小算子消失，新 kernel 出现，且没有把同步、format cast 或 workspace 开销挪到别的 step。
5. 精度护栏：temperature=0 的固定 prompt token 对比。用户要求数据集时再跑数据集。
6. 并发吞吐只在用户要发布口径时跑。

报告里分开写数值误差、TPOT 或吞吐、trace 中的 op 变化。三项缺一，不说“优化已验证”。
