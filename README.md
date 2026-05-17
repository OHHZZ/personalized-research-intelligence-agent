<p align="center">
  <h1 align="center">个性化研究情报智能体</h1>
  <p align="center">
    面向研究人员的每日论文、代码仓库与趋势情报系统，支持 RAG 有据问答。
  </p>
  <p align="center">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white">
    <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-1.1%2B-0f766e?style=flat-square">
    <img alt="asyncio" src="https://img.shields.io/badge/async-并行流水线-6d28d9?style=flat-square">
    <img alt="Storage" src="https://img.shields.io/badge/存储-JSON%20%7C%20pgvector-2563eb?style=flat-square">
  </p>
</p>

---

## 项目概览

个性化研究情报智能体将分散的研究信号整理成每日决策简报。系统从 5 个学术/代码平台并发抓取候选论文和仓库，按照用户画像进行相关性过滤，运行 8 维价值分析（可选 LLM 增强），检测趋势主题，并以 RAG 检索证据回答问题，每个回答附带幻觉风险评分。

![每日简报首页](docs/images/home_page.png)

---

## 功能

| 模块 | 能力 |
|------|------|
| 发现 | 5 个并发 Connector（arXiv、Semantic Scholar、OpenAlex、PapersWithCode、GitHub），在线失败时自动切换 sample 兜底 |
| 过滤 | 4 级相关性 × 质量评分；候选数不足时自动放宽阈值 |
| 工具补全 | 分析前自动填充缺失摘要（arXiv）、引用数（S2）、star 速度（GitHub）|
| 价值分析 | 8 维评分；LLM 增强含反思循环（最多 2 次重试 + 质量门控）|
| 证据审查 | 证据或可复现性信号不足时自动降低置信度 |
| 趋势 | 7 / 30 / 90 天主题频率窗口，结合用户画像交叉比对 |
| 报告 | 论文、仓库、工具、趋势排序，附 5 条可操作建议 |
| 问答 | 混合 dense + BM25 RAG 检索；LLM 回答附 grounding score（幻觉检测）|
| Supervisor | 动态策略节点：高优先级溢出时提升 LLM 上限，跳过闲置工具 |

---

## 产品界面

单页应用包含 7 个视图：

| 视图 | 用途 |
|------|------|
| Brief | 每日行动建议、信号分布、最高价值条目 |
| Papers | 带价值分析的论文排序情报 |
| Repos | Baseline 可用性与实现可复现性评估 |
| Trends | 7 / 30 / 90 天主题信号及影响 |
| Filtered | 审计记录：已接受、已拒绝、低优先级 |
| Saved | 本地反馈与后续跟进队列 |
| Profile | 可编辑的研究方向、方法、应用领域与目标 |

![研究助手抽屉](docs/images/assistant.png)

---

## 快速开始

```bash
# 安装
pip install -e .

# 使用示例数据运行（离线）
research-intel run-daily --source sample

# 使用在线数据源运行
research-intel run-daily --source hybrid

# 使用 LangGraph 状态机流水线
research-intel run-daily --source hybrid --use-langgraph

# 启动 Web 界面
research-intel serve-web
```

---

## 配置

复制 `.env.example` 为 `.env` 并按需填写：

```env
# 流水线
USE_LANGGRAPH_PIPELINE=false   # true = 启用 LangGraph 状态机

# LLM 增强（可选）
ENABLE_LLM_ANALYSIS=false
DASHSCOPE_API_KEY=

# 数据源（可选，建议填写）
GITHUB_TOKEN=
SEMANTIC_SCHOLAR_API_KEY=

# 向量嵌入（可选，提升 RAG 质量）
EMBEDDING_PROVIDER=local_hash  # 或 sentence_transformers
```

**安装 sentence-transformers：**
```bash
pip install -e .[embeddings]
```

**启用 PostgreSQL + pgvector：**
```bash
pip install -e .[pgvector]
research-intel init-pgvector
```

---

## 项目结构

```
src/research_intel/
├── agents/          # 10 个 Agent（流水线 + 按需调用）
├── connectors/      # 5 个数据源 Connector
├── tools/           # 工具注册表 + 论文/仓库工具
├── rag/             # 混合 dense+BM25 RAG 索引
├── llm/             # Qwen/DashScope 客户端
├── evaluation/      # 回答质量评估
├── web/static/      # 静态 Web 界面（HTML/CSS/JS）
├── pipeline.py      # 原始顺序流水线
├── langgraph_pipeline.py  # LangGraph 状态机流水线
├── mcp_server.py    # MCP 工具服务
└── web_server.py    # HTTP 服务
```

---

## 数据源模式

| 模式 | 行为 |
|------|------|
| `sample` | 仅使用 `data/samples/content_items.json`，完全离线 |
| `live` | 并发查询全部 5 个在线 Connector |
| `hybrid` | 优先在线；在线结果不足时混入示例数据 |
