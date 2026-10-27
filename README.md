# Ascend 310P 性能 Skill

给 Cursor、Codex、Claude Code 使用的同一份技能。它指导 agent 在 Ascend 310P 上维护 vLLM-Ascend 的推理性能：对照已有优化不变量、用 ais_bench 拆开验证、解读 profiler，并在证据足够时替换融合算子。

面向 Qwen3、Qwen3-VL、Qwen3.5、Qwen3.8，以及 Gemma 的 RMSNorm 差异。不用于纯精度定位。

## 安装

保留整个目录。三个平台都读取 `SKILL.md`，Codex 另外读取 `agents/openai.yaml`。

本仓库中的源目录是 `ascend-310p-performance-skills/`；安装时目标文件夹名保持为 frontmatter 中的 `ascend-310p-performance`。

复制到对应技能目录后重新打开会话：

| 平台 | 目录 |
|---|---|
| Cursor | `~/.cursor/skills/ascend-310p-performance` |
| Codex | `~/.codex/skills/ascend-310p-performance` |
| Claude Code | `~/.claude/skills/ascend-310p-performance` |

Windows 上 `~` 是 `%USERPROFILE%`。也可以放进项目的 `.cursor/skills/`、`.codex/skills/` 或 `.claude/skills/`。不要只复制 `SKILL.md`。

目录遵循开放的 Agent Skills 结构：`SKILL.md` 只放触发条件和主流程，按需细节放在一级 `references/`，避免每次调用都加载全部背景。参考：[OpenAI Skills 指南](https://developers.openai.com/api/docs/guides/tools-skills)、[Cursor Skills 文档](https://cursor.com/docs/skills)、[Claude Agent Skills 概览](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) 与 [Claude 最佳实践](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)。

## 使用

显式调用：

```text
$ascend-310p-performance
```

描述里包含 310P、TTFT、TPOT、吞吐、profiler、ACLGraph、MTP 或算子性能时，agent 也应载入本技能。

建议一次提供：

- vLLM-Ascend 源码路径、commit、模型与量化；
- 服务命令，尤其是图模式、MTP、TP 和 batch；
- 主指标和现有基线；
- profiler 目录或 ais_bench 命令；
- 本轮是否允许改代码、是否允许占空闲卡。

示例：

```text
使用 $ascend-310p-performance 分析 310P 上 Qwen3.5 的 MTP TPOT。
源码在 <path>，服务命令为 <command>。主指标是稳态 TPOT。
先不要改代码。若 profiler 显示 host 同步或 L2Norm 分解，给出单变量验证方案。
```

## 目录

```text
ascend-310p-performance/
├── SKILL.md
├── README.md
├── agents/openai.yaml
└── references/
    ├── optimization-map.md
    ├── aisbench-verification.md
    ├── profiler-analysis.md
    ├── operator-substitution.md
    └── graph-mtp-models.md
```

背景分析见 `work-docs/310p-perf/`。技能正文不依赖该目录也能执行。
