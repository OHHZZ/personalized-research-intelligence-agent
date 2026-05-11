from __future__ import annotations

from pathlib import Path

from research_intel.agents.base import BaseAgent
from research_intel.connectors.github_repo import GitHubRepoInspector, RepoSnapshot
from research_intel.models import ContentItem, ContentType


class RepoQAAgent(BaseAgent):
    name = "repo-qa-agent"

    def __init__(self, project_root: Path | str | None = None) -> None:
        self.inspector = GitHubRepoInspector(project_root)

    def answer(self, repo: ContentItem, question: str) -> str:
        if repo.content_type != ContentType.REPO:
            return "This item is not a GitHub repository, so Repo QA cannot analyze it."

        snapshot = self.inspector.inspect(repo)
        q = question.lower()
        if any(term in q for term in ("run", "install", "运行", "安装", "demo", "怎么跑")):
            return self._running_answer(repo, snapshot)
        if any(term in q for term in ("baseline", "基线", "适合")):
            return self._baseline_answer(repo, snapshot)
        if any(term in q for term in ("quality", "bug", "代码质量", "潜在", "风险")):
            return self._quality_answer(repo, snapshot)
        if any(term in q for term in ("develop", "二次开发", "extend", "扩展", "核心模块")):
            return self._extension_answer(repo, snapshot)
        return self._overview_answer(repo, snapshot)

    def _running_answer(self, repo: ContentItem, snapshot: RepoSnapshot) -> str:
        if snapshot.fetched and snapshot.has_readme:
            commands = _extract_command_lines(snapshot.readme)
            files = _interesting_files(snapshot.files)
            lines = [f"`{repo.title}` 已读取真实 README 和文件树。"]
            if commands:
                lines.append("README 中最值得先检查的运行/安装命令：")
                lines.extend(f"- `{command}`" for command in commands[:5])
            else:
                lines.append("README 没有清晰的命令块，建议先找 installation、quick start、demo 或 inference 小节。")
            if files:
                lines.append("可优先查看的文件：")
                lines.extend(f"- `{path}`" for path in files[:6])
            return "\n".join(lines)
        return self._metadata_running_answer(repo, snapshot)

    def _baseline_answer(self, repo: ContentItem, snapshot: RepoSnapshot) -> str:
        if snapshot.fetched:
            positives: list[str] = []
            risks: list[str] = []
            if snapshot.has_examples:
                positives.append("包含 examples/demo/inference 类文件")
            else:
                risks.append("没有明显 examples/demo/inference 文件")
            if snapshot.has_tests:
                positives.append("包含测试或评测相关文件")
            else:
                risks.append("测试信号弱")
            if snapshot.has_license:
                positives.append("license 文件可见")
            else:
                risks.append("license 不明确")
            if _has_dependency_file(snapshot.files):
                positives.append("依赖配置文件可见")
            else:
                risks.append("依赖入口不清晰")

            lines = [f"`{repo.title}` 可以作为 baseline 候选，但需要先跑通最小 demo。"]
            if positives:
                lines.append("支持信号：" + "; ".join(positives))
            if risks:
                lines.append("主要风险：" + "; ".join(risks))
            lines.append("建议：先复现 README 的最小推理/评测路径，再确认输入输出是否匹配你的实验。")
            return "\n".join(lines)
        return self._metadata_baseline_answer(repo, snapshot)

    def _quality_answer(self, repo: ContentItem, snapshot: RepoSnapshot) -> str:
        if snapshot.fetched:
            risks: list[str] = []
            if not snapshot.has_tests:
                risks.append("缺少 tests/eval 的文件树信号")
            if not snapshot.has_readme or len(snapshot.readme) < 800:
                risks.append("README 信息偏少")
            if not snapshot.has_license:
                risks.append("license 不清晰")
            if not _has_dependency_file(snapshot.files):
                risks.append("依赖配置入口不明显")
            risk_text = "; ".join(risks) if risks else "未发现明显 metadata/file-tree 风险"
            return f"`{repo.title}` 的真实仓库级质量检查：{risk_text}。"
        return self._metadata_quality_answer(repo, snapshot)

    def _extension_answer(self, repo: ContentItem, snapshot: RepoSnapshot) -> str:
        if snapshot.fetched:
            module_candidates = _module_candidates(snapshot.files)
            lines = [f"基于 `{repo.title}` 做二次开发，建议先定位最小运行链路，再包一层 adapter/evaluation wrapper。"]
            if module_candidates:
                lines.append("优先查看这些可能的核心模块或入口：")
                lines.extend(f"- `{path}`" for path in module_candidates[:8])
            if snapshot.key_files:
                lines.append("依赖/示例文件已经缓存，可先看：" + ", ".join(f"`{path}`" for path in snapshot.key_files.keys()))
            return "\n".join(lines)
        return self._metadata_extension_answer(repo, snapshot)

    def _overview_answer(self, repo: ContentItem, snapshot: RepoSnapshot) -> str:
        if snapshot.fetched:
            return (
                f"`{repo.title}`: {snapshot.description or repo.summary}\n"
                f"- README: {'available' if snapshot.has_readme else 'missing'}\n"
                f"- files scanned: {len(snapshot.files)}\n"
                f"- examples: {snapshot.has_examples}; tests: {snapshot.has_tests}; license: {snapshot.has_license}\n"
                "可以继续问：怎么运行、是否适合作 baseline、代码质量如何、核心模块在哪里。"
            )
        return (
            f"`{repo.title}`: {repo.summary}\n"
            f"真实仓库读取失败：{snapshot.error or 'unknown error'}\n"
            "当前只能基于 GitHub search metadata 回答。"
        )

    def _metadata_running_answer(self, repo: ContentItem, snapshot: RepoSnapshot) -> str:
        has_examples = repo.technical_signals.get("has_examples")
        readme_quality = repo.technical_signals.get("readme_quality", "unknown")
        suffix = f" 真实仓库读取失败：{snapshot.error}" if snapshot.error else ""
        if has_examples:
            return (
                f"`{repo.title}` looks runnable enough for a first demo because examples are present in metadata. "
                f"Start from README setup, then locate the smallest inference or demo script.{suffix}"
            )
        return (
            f"`{repo.title}` does not expose strong runnable-example signals. "
            f"README quality is `{readme_quality}`, so treat setup cost as a risk.{suffix}"
        )

    def _metadata_baseline_answer(self, repo: ContentItem, snapshot: RepoSnapshot) -> str:
        suffix = f" 真实仓库读取失败：{snapshot.error}" if snapshot.error else ""
        if repo.technical_signals.get("baseline_ready"):
            return (
                f"`{repo.title}` is a plausible baseline candidate from metadata. "
                f"Validate license, data requirements, and input/output contract before using it.{suffix}"
            )
        return (
            f"`{repo.title}` is not clearly baseline-ready yet. "
            f"Use it for idea inspection unless the core inference path is easy to reproduce.{suffix}"
        )

    def _metadata_quality_answer(self, repo: ContentItem, snapshot: RepoSnapshot) -> str:
        signals = repo.technical_signals
        risks: list[str] = []
        if not signals.get("has_tests"):
            risks.append("limited test signal")
        if signals.get("last_commit_days", 999) > 180:
            risks.append("stale commit activity")
        if signals.get("readme_quality") == "thin":
            risks.append("thin README")
        if not signals.get("has_license"):
            risks.append("license unclear")
        if snapshot.error:
            risks.append(f"real repo fetch failed: {snapshot.error}")
        risk_text = "; ".join(risks) if risks else "no major metadata-level risk found"
        return f"Metadata-level quality check for `{repo.title}`: {risk_text}."

    def _metadata_extension_answer(self, repo: ContentItem, snapshot: RepoSnapshot) -> str:
        core = repo.technical_signals.get("technical_core", repo.summary)
        suffix = f" 真实仓库读取失败：{snapshot.error}" if snapshot.error else ""
        return (
            f"For second-stage development on `{repo.title}`, first isolate the core path: {core} "
            f"Then add your own adapter or evaluation wrapper instead of modifying model internals immediately.{suffix}"
        )


def _extract_command_lines(readme: str) -> list[str]:
    commands: list[str] = []
    in_fence = False
    for raw_line in readme.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if line.startswith("$ "):
            line = line[2:].strip()
        if not line:
            continue
        if in_fence or _looks_like_command(line):
            if _looks_like_command(line):
                commands.append(line)
    return _unique(commands)[:10]


def _looks_like_command(line: str) -> bool:
    prefixes = (
        "pip install",
        "conda ",
        "python ",
        "python3 ",
        "uv ",
        "poetry ",
        "pipenv ",
        "npm ",
        "yarn ",
        "pnpm ",
        "docker ",
        "git clone",
    )
    return line.lower().startswith(prefixes)


def _interesting_files(files: list[str]) -> list[str]:
    return [
        path
        for path in files
        if path.split("/")[-1] in {"pyproject.toml", "requirements.txt", "environment.yml", "package.json", "Dockerfile"}
        or any(part in path.lower() for part in ("example", "demo", "inference", "eval"))
    ]


def _module_candidates(files: list[str]) -> list[str]:
    return [
        path
        for path in files
        if path.endswith((".py", ".ipynb", ".js", ".ts"))
        and not any(part in path.lower() for part in ("test", "docs", ".github"))
        and (
            "/" not in path
            or path.lower().startswith(("src/", "research_", "examples/", "demo/", "demos/", "eval", "scripts/"))
        )
    ]


def _has_dependency_file(files: list[str]) -> bool:
    names = {path.split("/")[-1] for path in files}
    return bool(names & {"pyproject.toml", "requirements.txt", "setup.py", "environment.yml", "package.json"})


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output
