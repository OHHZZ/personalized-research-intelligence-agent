from __future__ import annotations

import os

from research_intel.connectors.base import ContentConnector
from research_intel.connectors.http_client import ConnectorError, build_url, get_url, stable_id
from research_intel.connectors.signal_helpers import dedupe_items, enrich_with_profile_tags, unique
from research_intel.models import ContentItem, ContentType, UserProfile


class HuggingFaceConnector(ContentConnector):
    source_name = "huggingface"

    def fetch(self, profile: UserProfile) -> list[ContentItem]:
        self.last_errors = []
        items: list[ContentItem] = []
        for query in self._queries(profile):
            try:
                items.extend(self._search_models(query))
            except ConnectorError as exc:
                self.last_errors.append(f"models query={query}: {exc}")
            try:
                items.extend(self._search_datasets(query))
            except ConnectorError as exc:
                self.last_errors.append(f"datasets query={query}: {exc}")
        if self.last_errors and not items:
            raise ConnectorError("; ".join(self.last_errors))
        return enrich_with_profile_tags(dedupe_items(items), profile)

    def _queries(self, profile: UserProfile) -> list[str]:
        terms = [
            *profile.research_domains[:2],
            *profile.methods[:2],
        ]
        limit = int(os.getenv("HF_MAX_QUERIES", os.getenv("LIVE_MAX_QUERIES_PER_SOURCE", "2")))
        return [" ".join(t.strip().split()) for t in terms if t.strip()][:limit]

    def _search_models(self, query: str) -> list[ContentItem]:
        limit = int(os.getenv("HF_MODELS_PER_QUERY", "6"))
        url = build_url(
            "https://huggingface.co/api/models",
            {"search": query, "sort": "downloads", "direction": -1, "limit": limit},
        )
        results = get_url(url, timeout=12).json()
        if not isinstance(results, list):
            return []
        return [self._model_to_item(m) for m in results if isinstance(m, dict)]

    def _search_datasets(self, query: str) -> list[ContentItem]:
        limit = int(os.getenv("HF_DATASETS_PER_QUERY", "4"))
        url = build_url(
            "https://huggingface.co/api/datasets",
            {"search": query, "sort": "downloads", "direction": -1, "limit": limit},
        )
        results = get_url(url, timeout=12).json()
        if not isinstance(results, list):
            return []
        return [self._dataset_to_item(d) for d in results if isinstance(d, dict)]

    def _model_to_item(self, model: dict[str, object]) -> ContentItem:
        model_id = str(model.get("modelId") or model.get("id") or "")
        title = model_id.split("/")[-1] if "/" in model_id else model_id or "Unnamed HF model"
        card_data = model.get("cardData") if isinstance(model.get("cardData"), dict) else {}
        description = str(card_data.get("description") or card_data.get("summary") or "")
        if not description:
            tags_raw = model.get("tags") if isinstance(model.get("tags"), list) else []
            description = f"HuggingFace model: {', '.join(str(t) for t in tags_raw[:6])}"
        downloads = float(model.get("downloads") or 0)
        likes = float(model.get("likes") or 0)
        tags_raw = model.get("tags") if isinstance(model.get("tags"), list) else []
        tags = unique([str(t) for t in tags_raw[:8] if str(t).strip()])
        url = f"https://huggingface.co/{model_id}" if model_id else ""
        return ContentItem(
            item_id=stable_id("hf_model", model_id or title),
            content_type=ContentType.TOOL,
            title=title,
            url=url,
            source="huggingface",
            summary=description,
            tags=tags,
            authors=[str(model.get("author") or model_id.split("/")[0])] if model_id else [],
            published_at=str(model.get("lastModified") or model.get("createdAt") or ""),
            metrics={"downloads": downloads, "likes": likes},
            technical_signals={
                "has_code": True,
                "has_examples": any("example" in str(t).lower() or "demo" in str(t).lower() for t in tags_raw),
                "baseline_ready": downloads > 1000,
                "trend_signal": min(9.0, 4.5 + downloads / 500_000.0 + likes / 1000.0),
                "technical_core": description[:500],
                "hf_downloads": downloads,
                "hf_likes": likes,
            },
            links={
                "huggingface": url,
                "model_card": f"{url}/blob/main/README.md" if url else "",
            },
            raw={
                "modelId": model_id,
                "pipeline_tag": model.get("pipeline_tag"),
                "library_name": model.get("library_name"),
            },
        )

    def _dataset_to_item(self, dataset: dict[str, object]) -> ContentItem:
        dataset_id = str(dataset.get("id") or dataset.get("datasetId") or "")
        title = dataset_id.split("/")[-1] if "/" in dataset_id else dataset_id or "Unnamed HF dataset"
        card_data = dataset.get("cardData") if isinstance(dataset.get("cardData"), dict) else {}
        description = str(card_data.get("description") or card_data.get("summary") or "")
        if not description:
            tags_raw = dataset.get("tags") if isinstance(dataset.get("tags"), list) else []
            description = f"HuggingFace dataset: {', '.join(str(t) for t in tags_raw[:6])}"
        downloads = float(dataset.get("downloads") or 0)
        likes = float(dataset.get("likes") or 0)
        tags_raw = dataset.get("tags") if isinstance(dataset.get("tags"), list) else []
        tags = unique([str(t) for t in tags_raw[:8] if str(t).strip()])
        url = f"https://huggingface.co/datasets/{dataset_id}" if dataset_id else ""
        return ContentItem(
            item_id=stable_id("hf_dataset", dataset_id or title),
            content_type=ContentType.BENCHMARK,
            title=title,
            url=url,
            source="huggingface",
            summary=description,
            tags=tags,
            authors=[str(dataset.get("author") or dataset_id.split("/")[0])] if dataset_id else [],
            published_at=str(dataset.get("lastModified") or dataset.get("createdAt") or ""),
            metrics={"downloads": downloads, "likes": likes},
            technical_signals={
                "has_leaderboard": any("leaderboard" in str(t).lower() for t in tags_raw),
                "has_benchmark": True,
                "has_metrics": any("metric" in str(t).lower() for t in tags_raw),
                "trend_signal": min(9.0, 4.5 + downloads / 200_000.0 + likes / 500.0),
                "technical_core": description[:500],
                "hf_downloads": downloads,
                "hf_likes": likes,
            },
            links={"huggingface": url},
            raw={
                "datasetId": dataset_id,
                "task_categories": card_data.get("task_categories"),
            },
        )
