# Architecture

## Goal

The system is a local-first research intelligence agent. It is designed to help a researcher decide what to read, what to skip, which repositories may work as baselines, and which trend signals deserve follow-up.

## Pipeline

```text
ProfileAgent
  -> DiscoveryAgent
  -> FilteringAgent
  -> ValueAnalysisAgent
  -> EvidenceAgent
  -> TrendAgent
  -> RecommendationAgent
  -> DailyReport
  -> RagIndex
```

## Data Objects

Core objects are structured dataclasses rather than long free-form prompts:

- `UserProfile`
- `ContentItem`
- `FilterDecision`
- `ValueAnalysis`
- `TrendInsight`
- `DailyReport`
- `RagChunk`
- `RagSearchResult`

This keeps the pipeline testable, serializable, and easy to display in the web UI.

## Connectors

Live source connectors currently include:

- arXiv
- Semantic Scholar
- OpenAlex
- Papers with Code
- GitHub Search

Each connector records detailed errors locally. Public web responses expose only status counts so search failures do not leak into the frontend.

## Assistant And RAG

The assistant loads the latest report, candidate payloads, selected item context, and local RAG chunks. It can answer locally with rule-based synthesis, or call a DashScope/Qwen-compatible model when configured.

Default retrieval uses `local_hash` embeddings. Optional providers include `sentence_transformers` and PostgreSQL + pgvector for persisted vector chunks.

## Storage

The MVP uses JSON files:

- `data/samples/` for public sample content
- `data/profiles/` for profile data
- `data/runs/` for generated run artifacts
- `reports/` for generated Markdown and JSON reports
- `data/feedback/` for local feedback events

Generated run artifacts, reports, feedback, traces, and logs should not be committed.
