# Roadmap

## Phase 1: Stable Local MVP

- Keep the JSON-backed local workflow reliable.
- Keep sample mode fully offline and deterministic.
- Harden live connectors with timeouts, rate limiting, and backend-only diagnostics.
- Keep assistant responses grounded in report and RAG sources.
- Add focused tests for retrieval, assistant context, and pipeline behavior.

## Phase 2: Better Retrieval

- Add stronger multilingual embedding options.
- Add reranking after dense and keyword retrieval.
- Expand pgvector usage from storage sync to primary retrieval.
- Improve chunking for papers, repositories, trends, and feedback history.

## Phase 3: Richer Source Coverage

- Add OpenReview.
- Add Hugging Face models, datasets, and spaces.
- Add selected research blogs.
- Add PDF and repository deep inspection workflows.

## Phase 4: Service Backend

- Move the web API from `http.server` to FastAPI.
- Move JSON storage to PostgreSQL.
- Add scheduled daily runs and job status tracking.
- Add multi-user profile support and report history.

## Phase 5: Production Frontend

- Move the static frontend to React or Next.js.
- Add report history, global search, saved collections, and richer assistant citations.
- Add mobile-friendly reading and optional Slack or email delivery.
