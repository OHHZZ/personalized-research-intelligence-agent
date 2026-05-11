from __future__ import annotations

import json
from json import JSONDecodeError
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


class ConnectorError(RuntimeError):
    pass


@dataclass(slots=True)
class HttpResponse:
    url: str
    status: int
    body: bytes
    headers: dict[str, str]

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        try:
            return json.loads(self.text())
        except JSONDecodeError as exc:
            preview = self.text()[:220].replace("\n", " ")
            raise ConnectorError(f"Invalid JSON response from {self.url}: {preview}") from exc


def build_url(base: str, params: dict[str, Any]) -> str:
    clean_params = {
        key: value
        for key, value in params.items()
        if value is not None and value != ""
    }
    return f"{base}?{urllib.parse.urlencode(clean_params)}"


def get_url(url: str, headers: dict[str, str] | None = None, timeout: int = 20) -> HttpResponse:
    timeout = min(timeout, int(os.getenv("CONNECTOR_TIMEOUT_SECONDS", "8")))
    request_headers = {
        "User-Agent": os.getenv(
            "RESEARCH_INTEL_USER_AGENT",
            "PersonalizedResearchIntelligenceAgent/0.1",
        )
    }
    request_headers.update(headers or {})
    request = urllib.request.Request(url, headers=request_headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                url=url,
                status=response.status,
                body=response.read(),
                headers={key.lower(): value for key, value in response.headers.items()},
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ConnectorError(f"HTTP {exc.code} for {url}: {body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise ConnectorError(f"Network error for {url}: {exc.reason}") from exc
    except (TimeoutError, socket.timeout, OSError) as exc:
        raise ConnectorError(f"Network timeout for {url}: {exc}") from exc


def stable_id(prefix: str, value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in value)
    safe = "_".join(part for part in safe.split("_") if part)
    return f"{prefix}_{safe[:96]}"
