# TECHNICAL-DOCUMENTATION — sem1

The semantic embeddings lane of the sys1 house pattern. Normative for the
arch-gate: code that contradicts a section here is drift; an area not covered
here is a conventions gap, not a drift accusation.

## 1. Architecture

sem1 is a stdlib-only Python library (`src/sem1/`) with two provider modules
and a thin CLI. It emits geometry — vectors, cosine, top-k — and never
verdicts. Consumers (dev-decisions via `_bootstrap_sem1`, anything else via
the registered API) own every decision.

```
caller ──> sem1.embed(items, provider=...)
              │ capability gate (registry.enforce — raises BEFORE any I/O)
              ├─> providers/http_llama.py ── HTTP ──> native llama-server (:8901, LaunchAgent)
              └─> providers/st_worker.py ── subprocess ──> .venv-st python worker_st.py
                                                            └─ sentence-transformers (text + images)
store.py: packed float32 + manifest      telemetry.py: redacted JSONL rows
probe.py: reachability + mirror battery  similarity.py: pure cosine/top-k
```

## 2. Providers

| id | transport | text | image | notes |
|---|---|---|---|---|
| `llama-server` | stdlib urllib, OpenAI-style `POST /v1/embeddings` | yes | **no (declared)** | default endpoint `http://127.0.0.1:8901`; optional Bearer; batch |
| `st-worker` | subprocess into `.venv-st` (sdm1 nvidia-sdm pattern) | yes | yes (model-card `<|image|>` interleaved format) | one-shot per invocation (~9 s model load); batch the corpus per call |

New providers register via `sem1.registry.register(name, factory,
capabilities, description)`. Capabilities are declarations the probe verifies,
not server marketing.

## 3. Wire + capability probe

The HTTP provider declares **text-only** regardless of server claims. Ground
truth (2026-10-07): llama.cpp b11443's `/v1/embeddings` never routes through
mtmd even with `--mmproj` loaded — non-text payloads are tokenized as text and
embedded, returning confident garbage (a 213 KB PNG reports "input (203341
tokens) is too large"; small payloads answer as text-of-base64). `/props`
advertising `vision: true` is not evidence.

`sem1 probe --provider P [--battery]` reports reachability, dims, and declared
capabilities; the opt-in battery is the mirror test (solid red/blue squares +
a red→blue gradient must each prefer their own description from BOTH
directions, and distinct images must not land at ≈1.0 cosine). A provider may
declare image capability in the registry only because this battery has passed
against it live. `registry.enforce()` raises `ProviderCapabilityError` before
any I/O for unproven modalities.

## 4. Store format

`~/.local/share/sem1/vectors/<model-slug>/{vectors.bin, manifest.json}`.
`vectors.bin` is packed float32 rows (row i at offset i·dims·4, `array('f')`).
`manifest.json` records model, dims, provider, count, and per-entry
`{key, sha256, **meta}`. `rebuild()` replaces the store wholesale and dedups
by key, so rebuilds are idempotent. Roundtrip is bit-exact through
`array('f').tobytes()`.

## 5. Telemetry

One JSONL row per operation under `~/.local/share/sem1/logs/YYYY/MM/DD.jsonl`:
ts, op, provider, model, per-item `input_sha256`, char counts, dims, count,
latency_ms. `record()` structurally strips FORBIDDEN top-level keys (text,
input, prompt, key, api_key, vectors) — input content and key material never
reach a row, and the selftest greps the raw file to prove it.

## 6. CLI

`sem1 embed [--provider P] [--model M] [--endpoint URL] [--stdin] [--image P]
[--json] TEXT...`; `sem1 probe --provider P [--battery]`; `sem1 doctor`.
Exit 0 ok, 1 battery failure, 3 error/unavailable.

## 7. Limits (standing)

- Eval-only until calibrated: similarity thresholds do not transfer between
  surfaces; consumers must grade per surface before any verdict or join
  trusts a score. Embeddings propose, sys1/sdm1 dispose.
- Batch-only: no sem1 call in synchronous hook paths.
- No verdicts, no joins, no caches of verdicts (graded-equality requirement).
- Audio/video modalities: accepted nowhere in this version (both providers
  declare them false).
