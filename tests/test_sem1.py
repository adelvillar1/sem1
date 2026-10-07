"""sem1 selftests: redaction, capability-raise, store roundtrip, stubbed worker,
probe battery gating. Stdlib unittest; NO live endpoints or model downloads —
the live checks (mirror battery, parity) are separate verification commands."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sem1  # noqa: E402
from sem1 import store as store_mod  # noqa: E402
from sem1 import telemetry as tel  # noqa: E402
from sem1.providers import http_llama, st_worker  # noqa: E402
from sem1.similarity import cosine, top_k  # noqa: E402
from sem1.types import (  # noqa: E402
    EmbedResult,
    ProviderCapabilityError,
    ProviderUnavailableError,
)


class TestNormalize(unittest.TestCase):
    def test_str_becomes_text_item(self):
        self.assertEqual(sem1.normalize_items(["a"]), [{"text": "a"}])

    def test_rejects_empty_dict(self):
        with self.assertRaises(sem1.Sem1Error):
            sem1.normalize_items([{}])

    def test_rejects_non_dict(self):
        with self.assertRaises(sem1.Sem1Error):
            sem1.normalize_items([42])


class TestCapabilityGate(unittest.TestCase):
    """C1: image requests against a text-only provider RAISE — before any
    network work, so the 2026-10-07 silent text-of-base64 answer is impossible."""

    def test_enforce_raises_for_image(self):
        items = sem1.normalize_items([{"image": "/tmp/x.png"}])
        with self.assertRaises(ProviderCapabilityError):
            sem1.enforce("llama-server", items)

    def test_http_provider_raises_without_network(self):
        # No server needed: the refusal must happen before any request is made.
        with self.assertRaises(ProviderCapabilityError):
            http_llama.embed([{"image": "/tmp/x.png"}])

    def test_text_items_pass_gate(self):
        items = sem1.normalize_items(["hello"])
        sem1.enforce("llama-server", items)  # must not raise

    def test_st_worker_declares_image(self):
        sem1.enforce("st-worker", sem1.normalize_items([{"image": "/tmp/x.png"}]))


class _StubHandler:
    """Minimal HTTP stub: fixed embeddings, records the last request."""

    last_body: bytes = b""
    status = 200
    payload = {"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}],
               "model": "stub-model", "usage": {"prompt_tokens": 9}}


class TestHttpProvider(unittest.TestCase):
    def setUp(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        stub = _StubHandler

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                stub.last_body = self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps(stub.payload).encode()
                self.send_response(stub.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self._tmp = tempfile.TemporaryDirectory()
        self._srv = HTTPServer(("127.0.0.1", 0), H)
        self._port = self._srv.server_address[1]
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

    def tearDown(self):
        self._srv.shutdown()
        self._tmp.cleanup()

    def test_embed_roundtrip_and_redaction(self):
        # C0: end-to-end embed through a real HTTP server, telemetry redacted.
        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td) / "logs"
            secret = "sk-secret-key-abc123"
            r = http_llama.embed(
                ["my secret document text about auth keys", "second line"],
                endpoint=f"http://127.0.0.1:{self._port}",
                api_key=secret,
            )
            self.assertIsInstance(r, EmbedResult)
            self.assertEqual(r.dims, 3)
            self.assertEqual(r.count, 2)
            tel.record({"op": "embed", "provider": "llama-server",
                        "input_sha256": [tel.sha256("x")], "latency_ms": 1.0},
                       log_dir=log_dir)
            # the leaked-key row: record() must strip FORBIDDEN keys structurally
            row = tel.record({"op": "embed", "text": "SHOULD NOT APPEAR",
                              "api_key": secret, "vectors": [[1.0]]}, log_dir=log_dir)
            self.assertNotIn("text", row)
            self.assertNotIn("api_key", row)
            self.assertNotIn("vectors", row)
            raw = b"".join(p.read_bytes() for p in log_dir.rglob("*.jsonl"))
            self.assertNotIn(b"SHOULD NOT APPEAR", raw)
            self.assertNotIn(b"sk-secret-key-abc123", raw)
            self.assertNotIn(b"secret document text", raw)

    def test_unavailable_raises_named_error(self):
        # C0: unreachable endpoint -> ProviderUnavailableError naming the fix.
        with self.assertRaises(ProviderUnavailableError) as cm:
            http_llama.embed(["x"], endpoint="http://127.0.0.1:1", timeout=0.5)
        self.assertIn("llama-server", str(cm.exception))


class TestStore(unittest.TestCase):
    def test_roundtrip_bit_exact(self):
        with tempfile.TemporaryDirectory() as td:
            s = store_mod.VectorStore(td)
            vecs = [[0.5, -0.25, 1.0 / 3.0, 1e-7], [3.5, 2.25, -1.5, 0.0]]
            entries = [(f"k{i}", f"sha{i}", v) for i, v in enumerate(vecs)]
            s.rebuild("test/model", 4, "stub", entries)
            manifest, loaded = s.load("test/model")
            self.assertEqual(manifest["count"], 2)
            self.assertEqual(manifest["dims"], 4)
            self.assertEqual(manifest["model"], "test/model")
            # bit-exact through array('f') bytes
            for orig, got in zip(vecs, loaded):
                self.assertEqual(array("f", orig).tobytes(), array("f", got).tobytes())

    def test_rebuild_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            s = store_mod.VectorStore(td)
            entries = [("a", "sha_a", [1.0, 2.0]), ("b", "sha_b", [3.0, 4.0])]
            m1 = s.rebuild("m", 2, "stub", entries)
            m2 = s.rebuild("m", 2, "stub", entries)
            self.assertEqual(m1["count"], m2["count"])
            self.assertEqual([e["key"] for e in m1["entries"]], [e["key"] for e in m2["entries"]])

    def test_rebuild_dedupes_and_validates_dims(self):
        with tempfile.TemporaryDirectory() as td:
            s = store_mod.VectorStore(td)
            m = s.rebuild("m", 2, "stub", [("a", "s", [1.0, 2.0]), ("a", "s", [9.0, 9.0])])
            self.assertEqual(m["count"], 1)
            with self.assertRaises(store_mod.Sem1Error):
                s.rebuild("m2", 3, "stub", [("a", "s", [1.0, 2.0])])

    def test_load_missing_store(self):
        with tempfile.TemporaryDirectory() as td:
            manifest, vectors = store_mod.VectorStore(td).load("nope")
            self.assertIsNone(manifest)
            self.assertEqual(vectors, [])


class TestStubbedWorker(unittest.TestCase):
    """C2/C3 unit layer: the host side parses a worker response correctly,
    using a stub venv python (no sentence-transformers, no downloads)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        venv = Path(self._tmp.name) / "venv-st"
        (venv / "bin").mkdir(parents=True)
        stub = venv / "bin" / "python"
        stub.write_text(
            "#!/bin/sh\n"
            f'exec {sys.executable} -c \''
            'import json,sys\n'
            'req=json.loads(sys.stdin.read())\n'
            'vecs=[[0.9,0.1] if "image" in it else [0.2,0.8] for it in req["items"]]\n'
            'json.dump({"ok":True,"vectors":vecs,"dims":2,"model":req["model"]},sys.stdout)\n'
            "'\n"
        )
        stub.chmod(0o755)
        self._old_env = dict(__import__("os").environ)
        __import__("os").environ["SEM1_ST_VENV"] = str(venv)

    def tearDown(self):
        __import__("os").environ.clear()
        __import__("os").environ.update(self._old_env)
        self._tmp.cleanup()

    def test_host_parses_worker_response(self):
        r = st_worker.embed(["a text", {"image": "/tmp/anything.png"}], model="stub/model")
        self.assertEqual(r.provider, "st-worker")
        self.assertEqual(r.dims, 2)
        self.assertEqual(r.vectors[0], [0.2, 0.8])
        self.assertEqual(r.vectors[1], [0.9, 0.1])

    def test_missing_venv_raises_unavailable(self):
        __import__("os").environ["SEM1_ST_VENV"] = "/nonexistent/venv"
        with self.assertRaises(ProviderUnavailableError) as cm:
            st_worker.embed(["x"])
        self.assertIn("uv venv", str(cm.exception))


class TestSimilarity(unittest.TestCase):
    def test_cosine_known_values(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine([1, 0], [-1, 0]), -1.0)
        self.assertEqual(cosine([0, 0], [1, 0]), 0.0)

    def test_top_k_order(self):
        q = [1.0, 0.0]
        vs = [[0.5, 0.5], [0.9, 0.1], [0.0, 1.0]]
        got = top_k(q, vs, 2)
        self.assertEqual([i for i, _s in got], [1, 0])
        self.assertGreater(got[0][1], got[1][1])


class TestContract(unittest.TestCase):
    def test_result_keys_frozen_set(self):
        r = EmbedResult(vectors=[[1.0]], dims=1, model="m", provider="p")
        d = r.to_dict()
        self.assertEqual(sorted(d.keys()), sorted(sem1_types_keys()))

    def test_registry_known_providers(self):
        ids = registry_ids()
        self.assertIn("llama-server", ids)
        self.assertIn("st-worker", ids)
        self.assertFalse(sem1.capabilities("llama-server")["image"])
        self.assertTrue(sem1.capabilities("st-worker")["image"])


def sem1_types_keys():
    from sem1.types import RESULT_KEYS
    return RESULT_KEYS


def registry_ids():
    from sem1 import registry
    return registry.all_ids()


if __name__ == "__main__":
    unittest.main()
