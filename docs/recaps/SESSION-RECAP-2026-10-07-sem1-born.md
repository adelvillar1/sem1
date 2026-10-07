# Session recap — sem1 born, 2026-10-07

Repo born public this session: [github.com/adelvillar1/sem1](https://github.com/adelvillar1/sem1), the semantic embeddings lane of the sys1 house pattern. Plan contract: `docs/plans/2026-10-07-semantic-embeddings-lane.md` (mirrored from dev-decisions, where it was gated 12/12 and activated with recorded dispositions).

## Shipped

- **v0.1.0** (`60e3783`): stdlib-only core — `types.py` (EmbedResult contract, RESULT_KEYS frozen, ProviderCapabilityError/ProviderUnavailableError taxonomy), `telemetry.py` (structurally redacted JSONL), `store.py` (packed float32 + manifest, idempotent rebuild, per-entry meta), `similarity.py` (pure cosine/top-k), `registry.py` (register + enforce — raises before any I/O on unproven modalities), `probe.py` (reachability/dims + the opt-in mirror battery), `providers/http_llama.py` (text-only by declaration) and `providers/st_worker.py` + `worker_st.py` (sentence-transformers subprocess, text + images), `cli.py` (`embed`/`probe`/`doctor`). 19 selftests green.
- **LaunchAgent** (`5114ae0`): `deploy/com.adelvillar1.sem1-llama.plist` — the native llama-server (llama.cpp b11443, EmbeddingGemma-2 BF16, text embeddings on 127.0.0.1:8901) runs under launchd with RunAtLoad + KeepAlive; the model auto-loads at login after a reboot. Verified serving after cutover.
- **TECH-DOC + diagram** (`7f759895`, `ff1bbea`): TECHNICAL-DOCUMENTATION.md §1–§7 (the capability-probe contract is §3, the wire-tell is recorded: image payloads tokenize as text — a 213 KB PNG reports 203,341 tokens); archify `docs/architecture/system.{candidate.json,html,png}`, all four finalize gates passing, visual review of the rendered PNG.

## Live verification (real endpoints, real model)

- Mirror battery through the library: pass — img_red→text_red 0.7170 > text_blue 0.6336, mirrored blue, gradient correct, images distinguishable.
- Cross-provider parity on the sanity five: max pairwise cosine delta **0.0010** (llama-server HTTP vs sentence-transformers), ordering correct on both.
- dev-decisions consumers live: `semantic-index` (31/6394 rows recoverable, the rest redacted-only and skipped), `semantic-dedup` (22 pairs, graded context joined), `semantic-nn` (a plan's own graded gate rows returned first with disposition annotations), `docs-gate --via-semantic` (verdict equals unshortlisted baseline; shortlist + `sem1_raw` tag logged).

## Lessons

- `record()` must mkdir `path.parent` — the `YYYY/MM/DD.jsonl` layout has two levels below the log dir.
- A package-level function named like a submodule (`probe`) shadows it; `from .probe import probe` then returns the module. Implementation renamed `run_probe`.
- The devdec shim's `except Exception: pass` fallback turned a downstream `KeyError: 'op'` into a misleading monolith argparse error. Store manifests now carry per-entry metadata (op/ts) so consumers never index absent keys.
- The st venv must live durably (`.venv-st`, gitignored) — the `/private/tmp` probe venv from the morning assessment would be wiped by macOS.
- torchvision is required by the EmbeddingGemma-2 processor, not optional.
