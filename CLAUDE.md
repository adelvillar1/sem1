# sem1

> The semantic embeddings lane of the sys1 house pattern: language and images as geometry. Vectors in, vectors out — never verdicts.
> Repo: https://github.com/adelvillar1/sem1

**Operating rule: embeddings propose, sys1/sdm1 dispose.** This library ranks, clusters, and retrieves candidates; every verdict stays with a decision model and the JSONL calibration loop.

Local endpoint facts (the LaunchAgent, the model, ports) live in `CLAUDE.local.md` (gitignored). Read it before any backend or credential operation.

---

## Hard rules (non-negotiable)

- **Stdlib-only at runtime.** `dependencies = []` in `pyproject.toml` is a contract. HTTP is `urllib`; vector math is `array`/`math`. Providers that need heavier stacks (sentence-transformers) live behind a `[models]`-style extra in their own venv, reached by subprocess — `src/sem1` never imports them.
- **Embeddings propose, sys1/sdm1 dispose.** Nothing in this library returns a verdict, a label, or a threshold decision — return scores and neighbors, and leave the judgment to a decision model.
- **The capability probe is first-class.** Two different local servers confidently returned text-of-base64 vectors for image requests with no error (the design driver, 2026-10-07). A provider serves only the modalities it can prove (the mirror cross-modal battery); everything else raises instead of answering. Never trust a 200 as a capability.
- **The normalized result shape is a public contract.** `embed()`'s result (vectors, dims, model, telemetry, latency) and the store manifest's shape are consumed by blend1's corpus and fleet's recall path. Adding keys is fine; renaming or removing is a breaking change.
- **Never log input text, never print key values.** Telemetry records hashes, shapes, dims, and latencies — never the text that was embedded. Provider config views expose whether a key is *set*, never its value.
- **Fail open, never raise out of logging.** A broken provider must not break the caller's path; `telemetry` swallows its own write errors. Preserve both invariants.
- Default operation target is `main`. Feature branches are normal for work; merging to `main` requires explicit user approval *in the current turn*.

---

## Branch → environment topology

| Branch | Environment | Auto-deploy | Notes |
|--------|-------------|-------------|-------|
| `main` | library release | no | `pip install sem1`; the vendored lane copies pin a sha |
| `feature/*` | local dev | no | |

(No remote environments — sem1 is a library. The MCP stdio server in `service/` runs wherever a lane consumer launches it.)

---

## Where to find things

- Architecture & module map → `TECHNICAL-DOCUMENTATION.md`
- The design contract → `docs/plans/2026-10-07-semantic-embeddings-lane.md`
- Provider capability matrix & wire facts → `README.md`
- LaunchAgent recipe for the local llama-server → `deploy/`
- The MCP stdio adapter → `service/mcp_server.py` (embed / probe / index / search)
- Selftests → `tests/test_sem1.py` (stdlib unittest; no live endpoints, no model downloads)

---

## Common commands

```bash
uv run --with pytest -- python -m pytest tests/ -q   # selftests (20; the venv is runtime-only, pytest rides in via uv)
.venv/bin/sem1 probe                                 # provider reachability, dims, capabilities
.venv/bin/sem1 embed --text "…"                      # one embed through the default provider
.venv/bin/sem1 doctor                                # provider + store snapshot
```

The live checks (the mirror cross-modal battery, parity against a second provider) are separate verification commands run on demand — not part of the selftest suite, which stays offline by design.

---

## Today's state

- v0.1.0 on `main`: embed/probe/similarity/store/telemetry/registry/cli, the llama-server provider (default endpoint `127.0.0.1:8901`, text-only by declaration), the st-worker provider pattern, and the MCP stdio adapter. 20 selftests green, stdlib-only core.
- The store is a packed-float32 `vectors.bin` + JSON manifest under `~/.local/share/sem1/` (or `SEM1_VECTORS_DIR`), keyed by model slug — blend1's corpus and fleet's recall path consume it read-only.
- Known environmental watch: the 8901 endpoint can flap — `/health` answers while `/v1/embeddings` hangs (fleet KI-296 records the shape and the paused-retry remedy on the consumer side).

---

## Session protocol

1. Read this file (auto-loaded) and `CLAUDE.local.md` for endpoint facts.
2. For task-specific context, read `TECHNICAL-DOCUMENTATION.md` or the plan doc.
3. Trivial changes go direct; non-trivial work drafts a short plan note in `docs/plans/` first and writes a recap after — the sibling repos' cycle (`warmup → plan → build → recap`), compressed to fit this repo's size.
4. Keep this file ≤ ~150 lines; push detail into the docs it points at.
