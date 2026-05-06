# Personalized Research Intelligence Agent

面向科研人员的个性化 Research Intelligence Agent MVP。当前版本先跑通本地多 Agent 流水线：

1. 维护用户研究画像。
2. 从候选内容池发现论文、GitHub 项目、benchmark、工具报告和技术文章。
3. 过滤弱相关、营销内容、低质量 prompt collection、空泛 README、实验不足论文和过时项目。
4. 从相关性、新颖性、技术深度、证据强度、可复现性、实用性、趋势信号和科研机会进行价值分析。
5. 生成 7/30/90 天趋势信号。
6. 输出每日科研简报和行动建议。
7. 支持对样例 GitHub 项目做 Repo QA。
8. 支持过滤结果审计、详情证据查看、用户反馈闭环。

当前实现不依赖第三方 Python 包。需要本机安装 Python 3.11+。真实数据源已经接入 arXiv、Semantic Scholar、Papers with Code 和 GitHub Search API。`GITHUB_TOKEN`、`SEMANTIC_SCHOLAR_API_KEY` 和 `DASHSCOPE_API_KEY` 都是可选配置。

## 快速运行

在 PowerShell 中执行：

```powershell
$env:PYTHONPATH="src"
python -m research_intel.cli run-daily --profile default_user --report latest --source hybrid
```

生成文件：

- `reports/latest.md`
- `reports/latest.json`

项目问答示例：

```powershell
$env:PYTHONPATH="src"
python -m research_intel.cli ask-repo --repo-id repo_videditflow --question "这个项目适合作为我的 video editing baseline 吗？"
```

也可以运行脚本：

```powershell
.\scripts\run_daily.ps1
```

## 启动网页版

```powershell
.\scripts\serve_web.ps1
```

然后打开：

```text
http://127.0.0.1:8765
```

网页当前包含：

- Daily Brief：行动建议、信号分布、最高价值内容。
- 全局 Assistant 抽屉：任意页面都可以打开，可以围绕当前日报、论文、趋势、repo 和行动建议问答。
- Papers：论文推荐和价值分析。
- Repos：GitHub 项目推荐；每个项目卡片可一键交给全局助手追问。
- Trends：7/30/90 天趋势信号。
- Filtered：查看被拒绝、低优先级、候选、高优先级内容和过滤原因。
- Saved：查看用户反馈和后续阅读队列。
- Profile：编辑用户研究画像、内容偏好、当前目标和技术水平。

## 数据源模式

`run-daily` 和网页里的 `Source` 支持三种模式：

- `sample`：只使用 `data/samples/content_items.json`，适合离线调试。
- `live`：只抓真实 arXiv 和 GitHub 数据。
- `hybrid`：先抓真实数据，如果结果太少则混入样例数据，适合开发阶段。

真实源说明：

- arXiv：使用 `https://export.arxiv.org/api/query`，返回 Atom XML。
- Semantic Scholar：使用 `https://api.semanticscholar.org/graph/v1/paper/search`。
- Papers with Code：使用 `https://paperswithcode.com/api/v1` 的 papers/repositories 读接口。
- GitHub：使用 `https://api.github.com/search/repositories`。
- `GITHUB_TOKEN` 可选；未配置时只能使用 GitHub 未认证搜索限额。
- `CONNECTOR_TIMEOUT_SECONDS` 控制单个外部请求的超时，默认 8 秒。
- `LIVE_MAX_QUERIES_PER_SOURCE` 控制每个数据源最多生成几条 query，默认 3。

## 可选 LLM 深度分析

默认仍然使用规则版分析。要开启 LLM 增强，需要配置：

```powershell
Copy-Item .env.example .env
$env:ENABLE_LLM_ANALYSIS="true"
$env:DASHSCOPE_API_KEY="..."
$env:DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL="qwen-plus"
$env:LLM_ANALYSIS_LIMIT="10"
```

也可以直接把这些值写入项目根目录的 `.env`。开启后，`ValueAnalysisAgent` 会先用规则给所有候选排序，再对 top candidates 调用千问 DashScope 兼容模式接口做 JSON 价值分析。LLM 失败不会中断日报，会自动回退到规则分析。

同一个千问配置也会用于全局 Assistant。Assistant 会先从当前日报、候选内容、趋势、行动建议和用户选中的 item 中检索相关上下文，再让千问基于这些上下文回答。未开启 LLM 时，Assistant 只会返回本地检索和规则能够支持的有限回答。

## 用户反馈

网页上的 `Relevant`、`Not relevant`、`Save`、`Deepen`、`Baseline` 等按钮会写入：

```text
data/feedback/default_user.json
```

同时会轻量更新 `default_user.json` 里的 `feedback_weights`，作为后续个性化排序和过滤优化的基础。

## 项目结构

```text
data/
  profiles/              用户画像
  samples/               本地候选内容样例
  feedback/              用户反馈事件
  runs/                  最近一次候选、过滤和分析结果
docs/
  architecture.md        架构设计说明
  roadmap.md             生产化路线图
reports/                 每日简报输出
scripts/
  run_daily.ps1          本地运行脚本
  serve_web.ps1          网页启动脚本
src/research_intel/
  agents/                多 Agent 实现
  connectors/            真实数据源接口
  web/static/            无构建工具的网页前端
  cli.py                 命令行入口
  models.py              核心数据模型
  pipeline.py            每日流水线
  scoring.py             评分规则
  storage.py             JSON 本地存储
tests/
  test_pipeline.py       最小回归测试
```

## 当前 Agent

- `ProfileAgent`：加载或创建用户研究画像。
- `DiscoveryAgent`：发现候选内容。支持 sample/live/hybrid，当前 live 接 arXiv、Semantic Scholar、Papers with Code 和 GitHub。
- `FilteringAgent`：去噪与优先级判断。
- `ValueAnalysisAgent`：判断论文、repo、工具和文章的真实价值；可选 LLM 深度分析。
- `EvidenceAgent`：检查价值判断是否有证据支撑。
- `TrendAgent`：基于 7/30/90 天窗口识别趋势信号。
- `RecommendationAgent`：生成每日简报和行动建议。
- `RepoQAAgent`：回答 GitHub 项目运行、baseline、代码质量、二次开发等问题。

## 下一步建议

优先扩展顺序：

1. 接 OpenReview、Hugging Face Models / Datasets / Spaces、重点 research blog。
2. 增加 embedding 检索和去重：`pgvector` 或 `Qdrant`。
3. 将反馈权重真正纳入排序模型，而不只是记录。
4. 让 Repo QA 读取真实 README、目录结构和核心代码。
5. 如果要多人使用，升级为 FastAPI + PostgreSQL + React/Next.js。
