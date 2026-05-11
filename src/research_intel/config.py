from __future__ import annotations

import os
from pathlib import Path


PROXY_ENV_KEYS = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "GIT_HTTP_PROXY",
    "GIT_HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "git_http_proxy",
    "git_https_proxy",
}


def load_dotenv(project_root: Path | str, override: bool = True) -> None:
    """Load simple KEY=VALUE pairs from .env without adding a dependency."""

    path = Path(project_root) / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or (not override and key in os.environ):
            continue
        if key in PROXY_ENV_KEYS and value == "":
            os.environ.pop(key, None)
            continue
        os.environ[key] = value
