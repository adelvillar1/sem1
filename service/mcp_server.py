#!/usr/bin/env python3
"""
sem1 MCP server: exposes `sem1.embed` as the `sem1_embed` tool and
`sem1.probe` as `sem1_probe`.

Transport: stdio JSON-RPC (MCP 2024-11-05).

Default provider is `llama-server` (local HTTP endpoint, text-only, default
http://127.0.0.1:8901/v1/embeddings). Images require the `st-worker`
provider (sentence-transformers in a subprocess venv). House rule: vectors
in, vectors out — embeddings propose, the decision layers dispose.

Dependencies: sem1 package on PYTHONPATH (use `pip install -e` or set
SEM1_PYTHONPATH to the `src` directory).
"""

from __future__ import annotations

import json
import os
import sys


def _add_sem1_src_to_path() -> None:
    """Ensure `import sem1` resolves from the repo src layout."""
    env_path = os.environ.get("SEM1_PYTHONPATH")
    if env_path:
        sys.path.insert(0, env_path)


def _normalize_items_argument(value):
    """Accept a bare string or a list of strings / {text}/{image} objects."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(
        isinstance(item, (str, dict)) for item in value
    ):
        return value
    return None


def main() -> None:
    _add_sem1_src_to_path()
    from sem1 import embed, probe  # noqa: E402  (import after path patch)

    def respond(req_id, result):
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
        sys.stdout.flush()

    def error(req_id, code, msg):
        respond(req_id, {"error": {"code": code, "message": msg}})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {}) or {}

        if method == "initialize":
            respond(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "sem1", "version": "0.1.0"},
            })
        elif method == "tools/list":
            respond(req_id, {
                "tools": [
                    {
                        "name": "sem1_embed",
                        "description": (
                            "Embed text (or images via the st-worker provider) "
                            "into vectors. Returns the vectors, dims, model, "
                            "provider, and telemetry. Raises on unsupported "
                            "modality instead of answering with a wrong vector."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "items": {
                                    "type": ["string", "array"],
                                    "items": {
                                        "type": ["string", "object"],
                                        "properties": {
                                            "text": {"type": "string"},
                                            "image": {
                                                "type": "string",
                                                "description": "Path to an image file (st-worker provider only).",
                                            },
                                        },
                                    },
                                    "description": (
                                        "A single string or a list of strings / "
                                        "{\"text\": …} / {\"image\": path} items."
                                    ),
                                },
                                "provider": {
                                    "type": "string",
                                    "description": "llama-server (text, default) or st-worker (text + images).",
                                    "default": "llama-server",
                                },
                                "model": {
                                    "type": "string",
                                    "description": "Optional model override.",
                                },
                                "endpoint": {
                                    "type": "string",
                                    "description": "llama-server base URL override (default http://127.0.0.1:8901).",
                                },
                            },
                            "required": ["items"],
                        },
                    },
                    {
                        "name": "sem1_probe",
                        "description": (
                            "Capability and reachability probe for a provider: "
                            "declared modalities, whether the endpoint answers, "
                            "dims, model, latency."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "provider": {
                                    "type": "string",
                                    "description": "Default: llama-server.",
                                    "default": "llama-server",
                                },
                                "endpoint": {
                                    "type": "string",
                                    "description": "llama-server base URL override.",
                                },
                            },
                        },
                    },
                ],
            })
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {}) or {}

            provider = arguments.get("provider") or "llama-server"
            model = arguments.get("model")
            endpoint = arguments.get("endpoint")

            if tool_name == "sem1_embed":
                items = _normalize_items_argument(arguments.get("items"))
                if items is None:
                    error(req_id, -32602, "`items` must be a string or a non-empty list of strings / {text|image} objects.")
                    continue
                kwargs = {}
                if model is not None:
                    kwargs["model"] = str(model)
                if endpoint is not None:
                    kwargs["endpoint"] = str(endpoint)
                try:
                    result = embed(items, provider=str(provider), **kwargs)
                except TypeError as exc:
                    error(req_id, -32602, f"Provider {provider} does not accept these arguments: {exc}")
                    continue
                except Exception as exc:  # transport boundary: capability refusals, endpoint down
                    error(req_id, -32603, str(exc))
                    continue
                respond(req_id, {
                    "content": [{"type": "text", "text": json.dumps(result.to_dict())}],
                })
                continue

            if tool_name == "sem1_probe":
                kwargs = {"provider": str(provider)}
                if endpoint is not None:
                    kwargs["endpoint"] = str(endpoint)
                try:
                    report = probe(**kwargs)
                except Exception as exc:  # pragma: no cover — transport boundary
                    error(req_id, -32603, str(exc))
                    continue
                respond(req_id, {
                    "content": [{"type": "text", "text": json.dumps(report)}],
                })
                continue

            error(req_id, -32601, f"Unknown tool: {tool_name}")
        elif method == "initialized":
            # Client notification; no response required.
            pass
        else:
            error(req_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    main()
