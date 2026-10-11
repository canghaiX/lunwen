# RevImpact / RICA-Select 论文可行性讨论稿

> 用途：与导师讨论研究问题、算法创新、实验结果和投稿可行性。
>
> 当前状态：800 条 pilot 已完成；模型标签和受控标签仍属于 silver/pilot 标签；300 条自然 VitaminC 样本的真人复核尚未完成。因此，本文档中的结果用于判断方向和实验设计，不作为最终论文结果。

## 一分钟结论

本研究不研究“文档只要发生改动，就修改大模型参数”。研究的问题是：

> 给定专业文档的旧版、新版、查询及查询条件，判断当前查询的结论是否真正受到修订影响，并选择更新、保持或人工复核。

这可以定义为一个 NLP/LLM 任务：Revision Impact Prediction for Selective Knowledge Maintenance。

当前 pilot 已经观察到可测信号：RICA-Select 在保持 Changed Recall = 1.0 的同时，整体 Invariant Accuracy 约为 0.295，高于 text-diff 的 0，也高于原始 RICA 的约 0.093。但是，这个收益主要来自 controlled 样本；natural VitaminC 中的不变样本仍然很难识别。

当前判断：

- 研究问题值得继续；
- 当前算法已形成可实验的方法雏形；
- 还不能声称算法在真实文档上已经有效；
- 下一阶段关键工作是自然样本真人审核和数据扩充，而不是继续堆叠网络结构；
- ACL Findings 或 CCF-B 有现实可能，CCF-A 目前不应作为主要预期。

## 研究问题

### 背景

专业知识库通常由版本化文档支撑。文档修订后，维护人员需要判断哪些知识单元实际受影响，而不是简单地重建整个知识库。

例如：

    旧版：A 型设备在高温工况下压力上限为 10 MPa。
    新版：A 型设备在高温工况下压力上限为 8 MPa。

不同查询的正确动作不同：

| 查询 | 正确判断 | 动作 |
|---|---|---|
| A 型、高温、9 MPa | 旧版允许，新版不允许 | 更新 |
| A 型、高温、7 MPa | 两版结论都允许 | 保持或验证 |
| B 型、高温 | 修订条件不适用 | 保持 |
| A 型、常温 | 修订条件不适用 | 保持 |
| 型号或证据不明 | 无法可靠判断 | 人工复核 |
| 指定历史版本 | 使用历史版本结论 | 不被新版覆盖 |

关键区别是：

    文档文本变化 != 当前查询结论变化 != 知识条目必须更新

### 形式化任务

每条样本包含：

    D_old       旧版文档片段
    D_new       新版文档片段
    q           自然语言查询
    c           查询条件（型号、时间、地区、工况、数值等）
    y_old       旧版结论：entailed / contradicted / unknown
    y_new       新版结论：entailed / contradicted / unknown
    e_old/e_new 旧版和新版证据片段

模型输出：

    should_change  当前结论是否改变
    decision       change / keep / abstain
    evidence       支持判断的证据
    action         update / keep / verify / human_review

含义：

- change：新旧结论不同，或修订使当前条件下的答案必须改变；
- keep：修订不适用，或新旧结论仍然相同；
- abstain：证据不足、条件冲突或置信度不足，交给人工复核；
- verify：底层证据发生变化，但当前结论暂未改变，可作为辅助动作。

## 主要创新点

### 1. Revision Impact Prediction 任务

已有工作分别研究检索最新知识、回答新文档问题、编辑模型参数、验证证据或拒答。本文研究一次文档修订对当前查询的条件化影响：

> 修订是否影响该查询，而不是文档是否有字符串差异。

### 2. 条件化影响范围

修订条件和查询条件进行逻辑匹配：

    condition_compatibility(c_query, c_revision)
        -> match / overlap / disjoint / unknown

数值条件还判断：

    before / overlap / after / unknown

这使系统能够区分同一规则对不同型号、工况和数值区间的影响。

### 3. Change / Keep / Abstain 选择性维护

模型不被迫把所有实例判为更新或保持。证据不足和新旧证据冲突时输出 abstain，将自动维护风险交给人工。

研究目标是：

    尽量发现需要更新的知识，
    尽量保留不受影响的知识，
    在证据不足时减少错误自动决策。

### 4. 修订边界评测

同一修订家族同时构造：

- 边界内、应该改变的查询；
- 边界外、应该保持的查询；
- 证据改变但结论不变的查询；
- 证据不足或冲突的查询。

这比只报告新版文档问答准确率更能测量模型是否理解修订影响范围。

## 算法：RICA-Select

算法名称暂定为 RICA-Select：Revision- and Invariance-Conditioned Adapter for Selective Knowledge Maintenance。

### 编码和修订表示

冻结本地 Qwen3-0.6B 编码器，只训练轻量 adapter/head：

    h_old = Encoder(D_old, q, c)
    h_new = Encoder(D_new, q, c)

    delta = h_new - h_old
    diff  = abs(h_new - h_old)

    r = [h_old, h_new, delta, diff, condition_embedding]

条件特征包括型号、工况、地区、端点、数值和时间有效性等结构化信息。

### 预测头

1. old entailment head：预测旧版结论；
2. new entailment head：预测新版结论；
3. impact head：预测修订是否影响当前查询；
4. abstention head：判断是否需要人工复核；
5. evidence selector：选择旧版/新版支持证据；
6. boundary relation head：预测条件关系和边界。

显式三分类 decision head 已作为实验分支实现，但在当前类别不平衡 pilot 上明显偏向 change。因此当前主路线使用 impact + abstention 两个分数，并在 dev 集校准阈值。三分类 head 保留为消融实验。

### 损失函数

    L = L_old + L_new
      + λ1 L_impact
      + λ2 L_conditioned_consistency
      + λ3 L_boundary_rank
      + λ4 L_evidence
      + λ5 L_abstain

主要约束：

- L_old / L_new：旧版和新版结论；
- L_impact：Change/Keep 影响判断；
- L_conditioned_consistency：不受影响条件下保持新旧结论稳定；
- L_boundary_rank：边界内风险高于边界外风险；
- L_evidence：支持证据选择；
- L_abstain：unsupported/conflict 样本学习转人工。

当前实验说明，过强的一致性损失会产生不稳定结果。因此后续应使用较弱、条件化的一致性约束，而不是对所有样本强行要求新旧分布相同。

### 推理决策

    impact_score >= τ_change 且证据充分 -> change / update
    impact_score <  τ_change 且证据充分 -> keep
    证据冲突或 abstain_score >= τ_review -> abstain / human_review

阈值只在 dev 集选择，test 集只使用固定阈值评测。

### 算法创新程度的诚实判断

当前算法不是全新的基础网络结构。双路编码、差分表示、一致性学习和拒答机制分别已有相关思想。可主张的算法贡献是：

> 将条件兼容性、版本差分、边界排序和人工复核决策统一到版本修订影响预测任务中，并使模型同时学习 Change、Keep 和 Abstain。

因此：

| 部分 | 当前创新性 |
|---|---|
| 新任务定义 | 中等偏强 |
| 条件化评测协议 | 中等偏强 |
| 数据集构造 | 有潜力，取决于真人质量和真实修订规模 |
| RICA-Select 组合方法 | 中等 |
| 单个网络结构 | 偏弱 |

论文应强调任务、数据和选择性维护机制的联合贡献，不应夸大为全新 Transformer 结构。

## 与相关工作的区别

下表是投稿前需要重点核对和引用的相关方向。具体版本、会议和引用信息在投稿前应以 ACL Anthology、正式会议论文或 arXiv 最新版本为准。

| 方向/代表工作 | 主要研究问题 | 与本文区别 |
|---|---|---|
| Self-RAG（2023/2024） | 何时检索、生成和自我反思 | 本文关注文档修订是否影响特定查询，以及是否更新、保持或转人工 |
| Corrective RAG（2024） | 检索证据质量不足时纠错 | 本文把证据变化放在旧版/新版对比和条件影响范围中 |
| FreshLLMs（2024） | 最新知识和时效性问答 | 本文不仅选择最新版本，还判断旧结论在当前条件下是否仍有效 |
| KnowEdit（2024） | 评测模型知识编辑、局部性和泛化 | 本文不直接编辑模型参数，维护对象是外部版本化知识 |
| WISE（2024） | 模型参数的持续知识编辑和知识保留 | 本文研究外部知识库的选择性维护，参数更新不是主任务 |
| RAGTruth（2024） | 生成内容和证据支持关系 | 本文将证据支持用于修订影响和人工复核决策 |
| LongMemEval（2025） | 长期交互记忆和历史信息保留 | 本文以文档版本和修订边界为核心，不是一般对话记忆评测 |
| 时间感知 RAG/知识冲突工作 | 时间有效性和新旧信息冲突 | 本文进一步判断当前查询是否受影响，而非仅按时间选择最新证据 |

最重要的区分句：

> 知识编辑工作通常问“如何把新事实写入模型”；RAG 工作通常问“如何检索并回答”；本文问“文档发生修订后，哪些查询结论需要更新，哪些应保持，哪些必须人工复核”。

## 数据集

### 当前 RevImpact-800

| 项目 | 数量/状态 |
|---|---:|
| 查询实例 | 800 |
| 自然 VitaminC | 300 |
| 受控难例 | 500 |
| train/dev/test | 560 / 119 / 121 |
| family split 泄漏 | 0 |
| natural changed/invariant | 250 / 50，总体目标；各 split 存在偏斜 |
| controlled changed | 140 |
| controlled invariant | 140 |
| controlled boundary | 120 |
| controlled unsupported/conflict | 100 |
| 真人复核 | 尚未完成，300 条自然样本均 pending |

受控样本覆盖数值上限、范围边界、型号、地区、工况、时间、无关事实、否定、例外、证据替换和证据冲突。

### 数据质量限制

1. natural VitaminC 中 changed 比例较高；
2. test 只有 121 条，natural invariant 只有 8 条；
3. 300 条自然样本尚未真人确认；
4. controlled 样本用于补足类别，不能替代自然修订；
5. unsupported 不能把 unknown 占位标签当作普通结论金标准。

### 建议最终规模

    至少 2,000 条查询实例
    至少 400 个文档修订家族
    至少 3 个领域
    至少 1,000 条自然/真实修订查询
    至少 2,000 条人工确认记录

train/dev/test 必须按文档家族切分，另设 unseen-family 和 cross-domain test。

## 实验流程图

    旧版文档 D_old ─┐
                    ├─> 修订对齐与差分 ─> 结构化条件和证据 ─┐
    新版文档 D_new ─┘                                      │
                                                            v
    查询 q + 条件 c ───────────────────────────────> 冻结 Qwen 双路编码
                                                            |
                                                            v
                                                  修订差分表示
                                                            |
                                                            v
                                          Impact / Old-New / Evidence / Abstain
                                                            |
                                                            v
                                            Dev 阈值校准和 Test 固定评测
                                             /             |             \
                                      Change/更新     Keep/保持     Abstain/人工复核

## 建议准备的图表

当前文档已经包含任务流程图和结果表。正式论文或组会汇报建议准备以下图表：

| 图表 | 内容 | 当前状态 |
|---|---|---|
| Figure 1 | RICA-Select 架构：旧/新版双路编码、修订差分、条件门控、Change/Keep/Abstain 输出 | 可根据本节算法直接绘制 |
| Figure 2 | 10 MPa 改为 8 MPa 的边界案例，展示 9 MPa、7 MPa、B 型和常温查询的不同动作 | 已有案例，可制作成示意图 |
| Figure 3 | Text-diff、普通双路模型、RICA-Select 的 Impact F1、Changed Recall、Invariant Accuracy 柱状图 | 可用当前 pilot 结果生成；需标注 silver/pilot |
| Figure 4 | impact threshold 与 Changed Recall、Invariant Accuracy、误更新率之间的曲线 | 可从 dev threshold sweep 生成 |
| Figure 5 | natural VitaminC 与 controlled 样本的分组结果 | 可用当前结果生成；重点展示自然样本失败 |
| Figure 6 | Change/Keep/Abstain 混淆矩阵和风险-覆盖率曲线 | 可用当前 RICA-Select 预测生成 |
| Table 1 | 数据来源、领域、revision type、family 数量和 split 统计 | 当前已有部分，扩充数据后更新 |
| Table 2 | 主基线、RICA 消融、三随机种子均值和标准差 | 当前已有部分，需统一补齐所有 baseline |
| Table 3 | 真人标注一致性、80B/35B/真人字段级一致率 | 等待人工复核后生成 |

最终论文中 Figure 3--6 和 Table 2--3 必须同时报告 natural/controlled 分组，不能只报告混合总体分数。

## 实验设计

### 基线

1. Majority；
2. Text-diff；
3. Evidence-overlap；
4. BM25/向量检索 + NLI；
5. 新旧文档分别询问 LLM；
6. 普通双路编码器；
7. RICA 去掉 consistency；
8. RICA 去掉 boundary ranking；
9. RICA 去掉 condition gate；
10. RICA-Select。

condition_rule_oracle 只能作为结构化规则上界，不能作为 learned method 结果。

### 主指标

- Impact Precision/Recall/F1；
- Changed Recall；
- Keep/Invariant Accuracy；
- Change/Keep/Abstain Macro-F1；
- Boundary F1；
- Abstention risk-coverage；
- natural/controlled 分组结果；
- unseen-family 和 cross-domain 结果。

### 辅助指标

- old/new entailment Macro-F1；
- evidence precision/recall/F1；
- false update rate；
- missed update rate；
- 人工复核比例；
- bootstrap 95% 置信区间。

### 评测约束

- 预测阶段不能读取 metadata.category、gold label 或 should_change；
- unsupported 不进入主 Impact F1；
- evidence-overlap 返回全部候选证据的结果只能作为诊断；
- dev 选择阈值，test 只评测一次；
- 保存模型、阈值、seed、数据 hash 和原始预测。

## 当前实验结果

### 总体结果

| 方法 | Impact F1 | Changed Recall | Invariant Accuracy | 其他 |
|---|---:|---:|---:|---|
| Text-diff | 0.762 | 1.000 | 0.000 | 文本变化全部倾向更新 |
| 原始完整 RICA | 0.780 | 1.000 | 0.093 | 一致性约束收益弱 |
| RICA-Select seed 7 | 0.826 | 1.000 | 0.326 | Selective Macro-F1 0.773 |
| RICA-Select seed 17 | 0.817 | 1.000 | 0.279 | Selective Macro-F1 0.751 |
| RICA-Select seed 27 | 0.817 | 1.000 | 0.279 | Selective Macro-F1 0.751 |
| RICA-Select 三次平均 | 0.820 ± 0.006 | 1.000 ± 0.000 | 0.295 ± 0.027 | Macro-F1 0.758 ± 0.012 |

三次 RICA-Select 平均：

    coverage       = 0.926
    selective risk = 0.271
    abstain F1     = 1.000

### natural/controlled 分组

当前阈值型 RICA-Select seed 7：

| 数据来源 | Impact F1 | Changed Recall | Invariant Accuracy | 解释 |
|---|---:|---:|---:|---|
| natural VitaminC | 0.905 | 1.000 | 0.000 | test 中 8 条 invariant 全部误更新 |
| controlled | 0.747 | 1.000 | 0.400 | 受控边界样本出现 Keep 信号 |

这个结果非常关键：当前总体提升不能解释为已经解决真实文档维护。它说明模型在受控条件边界上学到了一些模式，但 natural 不变性泛化仍不足。

### 消融结论

- 完整 RICA 的强一致性约束没有稳定提高 Invariant Accuracy；
- 去掉 consistency 的单次结果约为 Invariant Accuracy = 0.302；
- 去掉 condition gate 的三个 seed 波动较大，约为 0.093 / 0.186 / 0.488，不稳定；
- RICA-Select 三次 seed 结果比上述单次消融更稳定，但 natural 分组仍失败；
- 显式三分类 decision head 在当前不平衡 pilot 上倾向全部预测 Change，测试 Keep Accuracy 为 0 到 0.093，因此只保留为 ablation。

### 结果可信度

当前结果可以支持：

- 任务有可测信号；
- text-diff 不能处理 Keep；
- 轻量条件化模型在 controlled pilot 上有初步收益；
- Abstain 可以识别当前 unsupported 样本。

当前结果不能支持：

- 已经解决真实知识库自动维护；
- 可以直接更新大模型参数；
- natural 文档上已有稳定泛化；
- 当前数字可以作为最终论文主结果。

## 后续两个月计划

### 第 1-2 周：真人审核和数据修正

- 完成 300 条 natural VitaminC 全部人工复核；
- 至少 100 条由第二名标注者独立复核；
- 报告 Cohen kappa 或 Krippendorff alpha，目标 >= 0.80；
- 优先检查 invariant、boundary 和 conflict；
- 保留 uncertain，不强迫进入 Change/Keep。

### 第 3-4 周：自然数据扩充

- 增加真实版本化软件文档、指南、标准或百科修订；
- 补足 natural invariant、boundary 和 evidence replacement；
- 控制 changed/keep/abstain 比例；
- 维护 family-disjoint split；
- 扩展到至少 2,000 条查询。

### 第 5 周：固定基线和主模型

- 固定 text-diff、overlap、NLI/RAG、普通双路模型；
- 固定 RICA-Select 结构和训练参数；
- 不根据 test 调阈值；
- 跑至少三个随机种子；
- 输出 bootstrap 置信区间。

### 第 6 周：泛化与消融

- unseen-family；
- cross-domain；
- natural/controlled 分组；
- changed/invariant/boundary/conflict 分组；
- consistency、condition gate、boundary rank、abstain head 消融；
- 错误案例按误更新、漏更新、错误保持、应人工未人工分类。

### 第 7 周：铁路应用和数字人接口

- 接入可获得的铁路专业文档作为应用案例；
- 输出版本、条件、证据、结论变化原因和人工复核建议；
- 数字人只展示决策和证据，不改变核心任务定义；
- 做最小可运行 demo。

### 第 8 周：投稿决策

| 结果 | 论文定位 |
|---|---|
| natural 和 controlled 都有效，unseen-family 仍提升 | ACL Findings / CCF-B |
| 主要 controlled 有效，natural 弱 | Revision-aware selective prediction 任务/数据集论文 |
| Keep 提升但 Changed Recall 明显下降 | 风险控制和 abstention 论文 |
| 所有方法都无法处理 natural invariant | 重新设计数据和任务 |

## 继续标准

    真人复核覆盖率 = 100%
    双人标注 kappa >= 0.80
    Changed Recall >= 0.95
    Keep Accuracy 比 text-diff 提高 >= 10-15 个百分点
    RICA-Select 稳定优于普通双路模型
    至少两个领域方向一致
    unseen-family 仍有收益

如果只满足 controlled 样本收益，而 natural 样本没有收益，研究仍可继续，但论文主张应收缩为：

> 面向版本修订的选择性预测任务、数据集和初步方法。

## 发表可行性判断

### ACL/ACL Findings

有机会，但关键不在当前 800 条 pilot，而在后续真人数据、跨域实验和任务定义清晰度。ACL Findings 比 ACL 主会现实。

### CCF-B

有合理基础。需要至少 2,000 条以上高质量数据、family-disjoint 和 unseen-family 测试、完整基线和消融，并且 natural 数据上不能完全失败。

### CCF-A

目前机会较低。需要更强算法原创性、更大规模真实数据、明显跨域泛化，以及理论分析或更强通用性证据。

## 需要和导师讨论的问题

1. 是否同意把论文主任务固定为 Change / Keep / Abstain，而不是自动更新模型参数？
2. 是否可以获得至少 2,000 条真实或人工严格复核的版本化文档查询？
3. 是否可以安排第二名标注者完成至少 100 条独立标注？
4. 是优先冲 ACL Findings/CCF-B，还是把铁路数字人作为应用论文主线？
5. 是否接受 controlled hard cases 作为补充数据，而不是把它们当作自然数据？
6. 铁路文档能否提供可公开引用的版本、证据和授权信息？
7. 是否把数字人定位为 demo/application，而将论文主贡献放在版本修订影响预测？

## 当前可复现实验

在 Docker lunwen 环境中：

    python scripts/train_rica.py \
      --data data/revimpact_800 \
      --model /models/Qwen/Qwen3-0.6B \
      --output outputs/rica_selective_threshold_seed7.pt \
      --epochs 3 --batch-size 16 --max-length 384 --seed 7 \
      --selective --no-decision-head

    python scripts/evaluate_rica_selective.py \
      --data data/revimpact_800 \
      --checkpoint outputs/rica_selective_threshold_seed7.pt \
      --model /models/Qwen/Qwen3-0.6B \
      --threshold-mode threshold \
      --output outputs/rica_selective_threshold_dev.json \
      --test-output outputs/rica_selective_threshold_test.json

    python scripts/aggregate_rica_selective.py \
      --files outputs/rica_selective_test.json \
              outputs/rica_selective_seed17_test.json \
              outputs/rica_selective_seed27_test.json \
      --output outputs/rica_selective_summary.json

    PYTHONPATH=.:scripts pytest -q

当前回归测试：27 passed。

## 文件索引

- 数据摘要：data/revimpact_800/summary.json
- 当前 pilot 报告：outputs/rica_feasibility_report.md
- RICA-Select 三 seed 汇总：outputs/rica_selective_summary.json
- RICA-Select 分组结果：outputs/rica_selective_threshold_test.json
- 算法实现：src/revimpact/models/rica.py
- 训练脚本：scripts/train_rica.py
- 选择性评测：scripts/evaluate_rica_selective.py
- 策略校准：src/revimpact/evaluation/selective.py

## 最终建议

当前不建议放弃方向，也不建议继续无止境增加 loss 或更换更大模型。最合理的下一步是：

    真人审核 300 条自然样本
    -> 补充真实 invariant/boundary/conflict
    -> 扩展到 2,000 条以上
    -> 固定 RICA-Select 和基线
    -> 重新做 natural/cross-domain/unseen-family 实验
    -> 根据结果决定 ACL Findings 或 CCF-B

当前最可信的论文主张应是：

> 对版本化专业文档，修订影响不应由文本差异直接决定。通过条件化版本差分和选择性人工复核，可以在保持较高变更召回率的同时，减少一部分不必要的知识更新；该能力在自然文档上的泛化仍需要更大规模真人数据验证。
