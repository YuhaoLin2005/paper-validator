# Comment Analysis & Verification Plan — 2026-07-13

> 全量评论分析 + René Zander 深度研读 + 双池专家交叉验证 + 优先行动计划
> 触发: 用户指令 "我们一起认真分析评论提取出有用的以及从那个认亲的看他的文章提取出有用的对我们有意义的作为我们接下的目标去验证跑通"

## 一、评论全景 Inventory

### 人物画像

| 评论者 | 身份 | 评论数 | 所在文章 |
|--------|------|--------|---------|
| **René Zander** | Enterprise AI Architect, Berlin. 36篇DEV.to文章, 独立构建了skillgate（与我们的机械门相同模式） | 2评论+1reaction | 150 Tasks, Follow-Up |
| **Mike Czerwinski** (jugeni) | 深度评论者，术语发明者（"receipt-of-action vs receipt-of-diligence"） | 3评论+2reaction | 150 Tasks, Follow-Up |
| **Dipankar Sarkar** | 最严谨评论者，决策token概念提出者 | 4评论+1reaction | 150 Tasks, Neural Gate, Meta-Cog, Follow-Up |
| **Max Quimby** | 务实派，"ceiling effect is the finding" | 1评论 | 150 Tasks |
| **Edu Peralta** | 日常agent使用者，关注diff-level验证 | 1评论(最新!) | Neural Gate |
| **Ponsubash Raj R** | "granular staleness"概念提出者 | 1评论+1reaction | Self-Referential |
| **Luis** (topstar_ai) | 生产AI系统实践者 | 1评论 | File-Timestamp |
| **CodeKitHub** | 一般正面反馈 | 1评论 | Follow-Up |

---

## 二、René Zander 深度研读 —— "认亲"分析

### 2.1 身份与项目

- Enterprise AI Architect @ Berlin, 36篇DEV.to文章
- 开源: **skillgate** (npm @reneza/skillgate v0.5.0), **pi-safe** 生态(8个仓库)
- 核心理念: "The model requests, the harness owns the boundaries"

### 2.2 已读三篇文章核心提取

**文章 "Your AI agent says it's done" (Jun 16)**:
- 基于 Shin 2026 "The Compliance Gap" (arXiv:2605.01771): 6模型/2000+session/**0%合规率**
- Theorem 2 (DPI): 偏差从输出文本中**不可检测**
- 解决方案: "Move the finish line out of the model's reach"
- **与我们的Prose Barrier + L1机械门完全一致**

**文章 "Sandboxing an AI Coding Agent" (Jul 3)**:
- 架构: 模型请求→路由→上下文控制→Agent(沙箱)→补丁评估器→真实仓库
- "A sandbox is not an evaluator" — 两个不同的工作
- "The best part is no part" — 当底层可以拥有边界时删除自定义层

**skillgate README (GitHub)**:
- Gate类型: file-exists, file-contains, absent, command, trivy, evidence, not-empty, instruction-sync
- **instruction-sync**: 跟踪CLAUDE.md/AGENTS.md/.cursor/rules之间的漂移（天才设计）
- "Gate, not loop" + "Loop + gate" 组合模式

### 2.3 架构对齐表

| 我们的概念 | René的概念 | 对齐度 | 差异 |
|-----------|-----------|--------|------|
| Prose Barrier | Compliance Gap Theorem 2 | **完全相同** | — |
| L1 机械门 | skillgate deterministic evaluator | **完全相同** | — |
| quality-gate.py→再生 | Loop + Gate pattern | **完全相同** | — |
| Gate outside control loop | "harness owns boundaries" | **完全相同** | — |
| L2 神经门(logprob) | ❌ 不存在 | **分歧** | René: "No model in the loop" |
| L3 因果编码 | ❌ 不存在 | **分歧** | René缩小残差而非评判残差 |
| L4 漂移预测器 | instruction-sync(简化版) | **部分对齐** | — |
| 奇异环(自再生) | ❌ 不存在 | **我们的独特贡献** | skillgate是静态的 |

---

## 三、逐条评论深度分析

### 3.1 Mike Czerwinski — 战略级

**Mike-1**: Ceiling effect + 两个追问
- ~0.7%残差是否聚集在gate未覆盖的任务类型上？→ **可验证: 用现有数据做聚类分析**
- "format mattered for reasoning depth"在GateGuard-off下是否仍成立？→ **可验证: 跑GateGuard-off格式实验**

**Mike-2**: Receipt-of-action vs Receipt-of-diligence
- "The gate can now check that a file exists, not that its contents reflect a real review."
- **Prose Barrier just moves from chat into the file.**
- 解决方案: artifact需要携带只有真实审查才能产生的东西(diff caught, exit code, specific value)

**Mike-3**: 决策token注释时机 + mechanizability-scanner
- **方法论红线**: 决策token边界是re-scoring前固定的还是看到结果后划的？
- 五层架构是"a good map drawn by hand, not yet a thing the system can verify about itself"
- 需要mechanizability-scanner: 从规则结构推断层归属

**Mike-4**: "Syllogism only buys you anything in exactly the world you're arguing nobody should run in"
- 逼我们选择: 格式不重要(gate覆盖一切) OR 格式是gaps的fallback(gate不能无处不在)
- **我们的答案: 立场B**

### 3.2 Dipankar Sarkar — 方法论专家

**Dipankar-1**: 语义only实验 — 固定机械门，只看语义决策的格式效应
**Dipankar-2**: 决策token delta — 单delta平均掩盖信号，penetration活在决策token
**Dipankar-3**: Catastrophic forgetting — all-four-dimensions decline是forgetting而非meta-cognition失败
**Dipankar-4**: P1 scorer风险 — LLM judge自带格式敏感性，"you would be measuring the oracle's bias, not the gate"

### 3.3 其他评论者

**Max**: 55.9%→0.7%是完整故事; "gate it" vs "can only nudge it"的边界在哪里？
**Edu** (最新!): logprob differential在agent重写推理链时是否仍有效？
**Ponsubash**: 为self-model添加provenance链接; granular staleness而非全量重建

---

## 四、优先行动计划

### 🔴 Priority 0 — 文档诚实性修正（今天）

| # | 事项 | 来源 |
|---|------|------|
| P0-1 | 声明决策token注释时机（re-scoring前固定 vs 过程中调整） | Mike-3 |
| P0-2 | Follow-Up文章: 区分reframe vs result（P1还没跑） | Mike-3 |
| P0-3 | 选定立场: 格式是gaps的fallback（立场B） | Mike-4 |
| P0-4 | Meta-Cog文章: all-four-dimensions decline→forgetting | Dipankar-3 |

### 🟡 Priority 1 — 核心实验（1-2天）

| # | 实验 | 来源 |
|---|------|------|
| P1-1 | 残差违规聚类分析（~0.7%在gate未覆盖类型上？） | Mike-1 |
| P1-2 | GateGuard-off下格式效应复测 | Mike-1 |
| P1-3 | 语义only条件实验（固定gate，只看语义决策） | Dipankar-1 |
| P1-4 | 决策token-only delta scoring实现 | Dipankar-2 |
| P1-5 | 自重写推理链logprob测试 | Edu |

### 🟢 Priority 2 — 基础设施（3-5天）

| # | 构建 | 来源 |
|---|------|------|
| P2-1 | Mechanizability-scanner（规则文本→推断层归属） | Mike-3 + Max |
| P2-2 | 非LLM P1 scorer（离散rubric + strip格式） | Dipankar-4 |
| P2-3 | Per-layer残差探针v3 | Dipankar-2 |
| P2-4 | Receipt-of-diligence artifact设计 | Mike-2 |

### 🔵 Priority 3 — 长期方向

| # | 方向 | 来源 |
|---|------|------|
| P3-1 | instruction-sync adoption | René |
| P3-2 | Self-model provenance links | Ponsubash |
| P3-3 | Granular staleness | Ponsubash |
| P3-4 | 指令遵循退化探针（forgetting验证） | Dipankar-3 |
