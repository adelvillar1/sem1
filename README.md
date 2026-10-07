# sem1

The semantic embeddings lane of the sys1 house pattern. [sys1](https://github.com/adelvillar1/sys1) judges what the work says (decision models over text), [sdm1](https://github.com/adelvillar1/sdm1) scores what the work measures (tabular foundation models), and sem1 indexes what the work looks like: language and images as geometry. Vectors in, vectors out — never verdicts.

Operating rule: **embeddings propose, sys1/sdm1 dispose.** This library ranks, clusters, and retrieves candidates; every verdict stays with a decision model and the JSONL calibration loop.

## Status

Pre-implementation. The contract is the plan: [`docs/plans/2026-10-07-semantic-embeddings-lane.md`](docs/plans/2026-10-07-semantic-embeddings-lane.md) (drafted and plan-gated in [adelvillar1/dev-decisions](https://github.com/adelvillar1/dev-decisions), same file mirrored here).

## Planned shape

- **Stdlib-only core**: one `embed()` API, normalized result contract, packed-float32 vector store with JSON manifest, pure cosine/top-k, JSONL telemetry with redaction (never input text, never key values), pluggable provider registry.
- **`llama-server` provider** — a native llama-server speaking OpenAI-style `/v1/embeddings`. On this machine it runs LaunchAgent-owned at `127.0.0.1:8901` (recipe in [`deploy/`](deploy/): RunAtLoad + KeepAlive, so the model auto-loads at login after a reboot). Text-only by declaration.
- **`st-worker` provider** — subprocess worker into a venv (the sdm1 `nvidia-sdm` pattern) running sentence-transformers; text and images via the `<|image|>` interleaved format.
- **Capability probe, first-class.** The design driver, live-proven on 2026-10-07: two different local servers happily returned confident text-of-base64 vectors for image requests with no error. A provider serves only the modalities it can prove (the mirror cross-modal battery, passed live), and everything else raises instead of answering.

## Wire facts (verified 2026-10-07, local EmbeddingGemma-2)

| Lane | Text | Images | Latency |
|---|---|---|---|
| HTTP (`/v1/embeddings`, 768 dims) | works, sane cosines | **no** — tokenizes input as text | 86 ms single, 46 ms batch-of-5 |
| sentence-transformers | works, 4-decimal parity with HTTP | **works** — mirror battery passes; re-encode deterministic at 1.0000 | ~2 s per 2048px PNG |

## Running the local endpoint

```bash
cp deploy/com.adelvillar1.sem1-llama.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.adelvillar1.sem1-llama.plist
# restart by hand:  launchctl kickstart -k gui/$(id -u)/com.adelvillar1.sem1-llama
# logs:             ~/.local/share/sem1/llama-server.log
```

## House pattern

One Python API, one wire shape, normalized results, JSONL telemetry, pluggable registry, stdlib-only core. Siblings: [sys1](https://github.com/adelvillar1/sys1) (text judgment), [sdm1](https://github.com/adelvillar1/sdm1) (tabular), [dev-decisions](https://github.com/adelvillar1/dev-decisions) (the verification and calibration layer that consumes all three).

## License

Apache-2.0
