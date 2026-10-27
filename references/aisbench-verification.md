# ais_bench 与分项性能验证

只在需要设计或执行性能对比时读取。精度数据集命令和 L0–L5 定义以精度技能的 `accuracy-testing.md` 为准；本文只规定性能实验怎么拆。

## 口径

对比两侧除目标变量外保持一致，并写进记录：

- 源码目录、commit、CANN、PTA、驱动；
- 权重、量化、TP、物理卡、`ASCEND_RT_VISIBLE_DEVICES`；
- `max-model-len`、`max-num-seqs`、`max-num-batched-tokens`、`gpu-memory-utilization`；
- `cudagraph_mode`、`cudagraph_capture_sizes`、`enable_npugraph_ex`、MTP 步数、prefix cache；
- chat template 与 `enable_thinking`。

`enable_npugraph_ex=false` 是 310P 约束。`gpu-memory-utilization=0.6` 只是 RC 设备上的经验起点，改动必须记入对比。服务从确认过的 vLLM-Ascend 源码目录启动。每个候选重启，不复用 KV 或 prefix cache。不抢占其他任务占用的卡。

## 指标

实验开始前指定一个主指标。其余只做护栏，不临时更换主指标来宣布胜利。

| 指标 | 敏感项 |
|---|---|
| TTFT | prefill 长度、视觉 encoder、图捕获是否混进首包 |
| TPOT | host 气泡、图 replay、MTP 接受率、norm/attention |
| output throughput | 稳态 decode 与并发 |
| request throughput | 调度、并发、短请求 |

至少 3 次，报中位数和极差。warmup 不计入。单次数字不作为结论。OOM、图捕获失败、启动失败记 blocked。

MTP 必须同时报平均接受长度。接受率下降时先停性能结论，检查 rejection。

## 层级

前一层已经能分开两个候选就停。不要用并发吞吐定位一个算子。

| 层级 | 配置 | 证明什么 |
|---|---|---|
| A | 单请求，temperature=0，短输出，eager，MTP 关 | 服务可跑 |
| B | 同 A，输出 256–1024 token | 稳态 TPOT，暴露 host 与 norm/attention |
| C | 只开 FULL_DECODE_ONLY，capture sizes 覆盖实际 batch | 图是否快于 eager decode |
| D | 只开 MTP，先 K=1 再 K>1，eager | draft + rejection 的收益 |
| E | 图与 MTP 同时开；token 数覆盖 `num_reqs * (1+K)` | 叠加是否掉回 eager |
| F | MoE 上重复 B | routing、dispatch、NZ 要能在 profiler 里分开 |
| G | hybrid 上重复 B 与 D | L2、gated norm、Mamba align |
| H | 目标并发的 stream 性能 | 发布口径，不是定位口径 |

独立变量，不要合成一个开关：eager 与图、MTP 关与 K=1 与 K>1、prefix cache、量化、单个算子、capture sizes、`max_num_seqs`。

## ais_bench

在 benchmark 仓库根目录执行，先用当前版本的 `--help` / `--search` 找到模型配置与字段，不能照抄其他版本命令。性能用 `vllm_api_stream_chat`：`ignore_eos=True`，固定输入输出长度，`request_rate` 和 `batch_size` 按本层实验设定；`request_rate` 是到达速率，不等同于实际并发，二者分别记录。精度护栏用 `vllm_api_general_chat`：`temperature=0`，`ignore_eos=False`。两套结果不能互相引用。

算子或元数据补丁的最低服务证据是对应的 B/F/G 中位数，加上 profiler 中目标小算子消失、融合算子出现。H 只在用户要发布口径时跑。

精度护栏至少是固定 prompt 的 token 对比，或用户指定数据集的小子集。L2Norm、gating、NZ、rejection 任一改动都要这条护栏。全量榜仅在用户要求时跑。

## 命令模板

先用当前版本的 `vllm serve --help` 核对参数，不要照抄过期 flag。

```bash
cd <vllm-ascend-source>
export ASCEND_RT_VISIBLE_DEVICES=<physical-ids>
vllm serve <weight-path> \
  --host 127.0.0.1 --port 1025 \
  --served-model-name <model> \
  --trust-remote-code \
  --max-model-len <len> \
  --gpu-memory-utilization 0.6 \
  --max-num-seqs <n> \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}'
```

MTP、量化、`--dtype float16`、`-tp` 仅在该实验的目标变量或用户基线里出现。capture sizes 在 MTP 实验中改成实际 token 数，而不是只写并发请求数。

单请求 TPOT 可以用流式 curl，记录首 token 时间与后续 token 间隔。curl 的 host、port、model 必须与服务一致。保存完整命令、日志路径和三次原始结果。
