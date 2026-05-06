# 架构设计

## 产品目标

这个系统不是简单的信息流，而是一个研究情报系统。核心问题是：

- 今天用户应该看什么？
- 为什么值得看？
- 和用户研究方向有什么关系？
- 哪些内容应该跳过？
- 哪些趋势可能变成未来 1-2 个月的选题机会？
- 哪些 repo 可以作为 baseline 或二次开发起点？

## 流水线

```text
ProfileAgent
    ↓
DiscoveryAgent
    ↓
FilteringAgent
    ↓
ValueAnalysisAgent
    ↓
EvidenceAgent
    ↓
TrendAgent
    ↓
RecommendationAgent
    ↓
Daily Report
```

按需能力：

```text
RepoQAAgent
```

## 网页版本

当前 Web MVP 使用 Python 标准库 `http.server` 提供 API 和静态页面，不引入 FastAPI、React 或 Node 构建链。这样可以先验证产品信息架构和推荐流程。

页面结构：

- `Brief`：日报总览、行动建议、信号分布。
- 全局 `Assistant` 抽屉：从任意页面打开，默认以当前报告为上下文；点击 item 的 `Ask` 后自动切换到该 item。
- `Papers`：论文价值分析。
- `Repos`：开源项目分析；项目问答通过全局 Assistant 或 item-level Ask 触发。
- `Trends`：趋势雷达。
- `Filtered`：过滤审计，展示 reject/low priority/candidate/high priority 和原因。
- `Saved`：用户反馈队列。
- `Profile`：用户画像编辑。

API：

- `GET /api/profile`
- `POST /api/profile`
- `GET /api/report`
- `POST /api/run`
- `GET /api/candidates`
- `POST /api/repo-qa`
- `POST /api/assistant`
- `GET /api/feedback`
- `POST /api/feedback`

## 数据交接原则

Agent 之间不直接传自然语言长文，而是传结构化对象：

- `UserProfile`
- `ContentItem`
- `FilterDecision`
- `ValueAnalysis`
- `TrendInsight`
- `DailyReport`

这样方便测试、回溯、替换 LLM、接入前端和做离线评估。

## 价值判断维度

`ValueAnalysisAgent` 当前使用 8 个评分维度：

- `relevance`：和用户画像相关度。
- `novelty`：是否有新方法、新任务或新视角。
- `technical_depth`：技术细节是否足够。
- `evidence_strength`：实验、benchmark、leaderboard、引用证据是否扎实。
- `reproducibility`：代码、示例、测试、数据、license 是否支持复现。
- `practical_utility`：能否用于 baseline、项目或科研流程。
- `trend_signal`：是否代表方向变化。
- `research_opportunity`：是否能引出选题或项目机会。

## 过滤策略

过滤不是只做关键词匹配，而是结合：

- 用户画像匹配。
- 内容类型。
- 技术信号。
- 实验证据。
- README 和 repo 元数据。
- prompt collection、营销文章、过时项目等负面信号。

过滤结果分为：

- `reject`
- `low_priority`
- `candidate`
- `high_priority`

MVP 只把 `candidate` 和 `high_priority` 进入日报分析。

## 生产化方向

## LLM 增强

当前 `ValueAnalysisAgent` 支持可选 LLM 增强：

```text
规则分析所有候选
    ↓
选出 top N
    ↓
DashScope/Qwen Chat Completions JSON 分析
    ↓
失败则回退规则分析
```

默认关闭。配置：

```text
ENABLE_LLM_ANALYSIS=true
DASHSCOPE_API_KEY=...
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
LLM_ANALYSIS_LIMIT=10
```

LLM 只分析高价值候选，不负责 discovery 和 filtering，以控制成本和降低不可解释风险。

## Assistant RAG

全局 Assistant 不应直接用模板回答。当前实现是一个轻量 RAG：

```text
用户问题
    ↓
构建上下文：当前日报、top papers、top repos、trends、actions、候选内容、选中 item
    ↓
本地词法检索 top chunks
    ↓
如果 ENABLE_LLM_ANALYSIS=true 且配置 DASHSCOPE_API_KEY：
        交给 Qwen 结合检索上下文生成答案
      否则：
        使用本地检索结果和少量意图规则给有限回答
```

因此，未开启千问时，Assistant 不会具备完整自然语言推理能力，只会给出基于当前上下文的有限回答。开启千问后，也要求模型只使用检索上下文，不允许编造论文、repo、实验或指标。

## 反馈闭环

前端反馈事件会写入：

```text
data/feedback/{profile_id}.json
```

反馈类型包括：

- `relevant`
- `not_relevant`
- `save`
- `deeper`
- `baseline`
- `skip`

当前版本会把反馈轻量更新到 `UserProfile.feedback_weights`。下一步应将这些权重用于 Discovery query expansion、Filtering 阈值和 Recommendation 排序。

### 数据源

优先接入：

- arXiv，已接入
- Semantic Scholar，已接入
- Papers with Code，已接入
- GitHub Search API，已接入
- OpenReview
- Hugging Face Models / Spaces / Datasets
- 重点公司 research blog

### 存储

MVP 用 JSON 文件。生产版建议：

- PostgreSQL：用户、内容、日报、反馈。
- pgvector 或 Qdrant：语义检索。
- Redis：任务缓存。
- Elasticsearch 或 Meilisearch：关键词检索。

### Agent 编排

MVP 用显式 Python 调用链。生产版可升级为：

- LangGraph 状态机。
- Temporal / Celery 定时任务。
- 每个 Agent 输出强 schema JSON。

### 用户反馈

必须记录：

- 点击。
- 收藏。
- 忽略。
- 标记不相关。
- 追问。
- 实际运行 repo。
- 用户手动修改兴趣方向。

反馈用于更新：

- 搜索 query。
- 召回源权重。
- 过滤阈值。
- 日报长度。
- 推荐排序。
