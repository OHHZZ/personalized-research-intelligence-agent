# Production Roadmap

当前版本保持标准库实现，适合单用户本地运行和快速验证研究情报流程。多人使用或长期部署时，建议按以下顺序升级。

## Phase 1：稳定本地单用户版

- 保留 JSON 存储。
- 完善反馈闭环。
- 让 LLM 只分析 top candidates。
- 增加 OpenReview、Hugging Face、research blog 数据源。
- 让 Repo QA 读取 README、文件树、requirements、examples 和 tests。

## Phase 2：服务化后端

- 将 `web_server.py` 替换为 FastAPI。
- 将 JSON 文件迁移到 PostgreSQL。
- 用 `pgvector` 或 Qdrant 存储 content/profile embeddings。
- 用 Celery、Temporal 或 APScheduler 跑每日任务。
- 增加登录、多用户 profile、日报历史和反馈历史。

## Phase 3：正式前端

- 将静态 HTML/CSS/JS 迁移为 React 或 Next.js。
- 保留当前信息架构：Brief、Papers、Repos、Trends、Filtered、Saved、Profile。
- 增加详情抽屉、批量反馈、搜索、排序、收藏夹和日报历史。
- 支持移动端阅读和邮件/Slack 推送。

## Phase 4：研究工作流集成

- Paper QA：读取 PDF、方法、实验、相关工作。
- Repo QA：克隆 repo，检查代码结构和可运行性。
- Baseline Planner：自动生成复现实验计划。
- Research Gap Finder：结合趋势和过滤结果给出选题机会。

