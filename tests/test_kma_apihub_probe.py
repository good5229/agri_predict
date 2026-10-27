import datetime as dt
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from agri_data import kma_apihub_probe as probe


class FakeResponse:
    def __init__(self, body=b"TMP data", status=200, headers=None):
        self.body, self.status_code = body, status
        self.headers = headers or {"Content-Type": "text/plain"}
        self.closed = False

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        self.closed = True


class ProbeTests(unittest.TestCase):
    def test_temporal_validation(self):
        probe.validate_times("2025090108", "2025090206")
        for issue, valid in [("2025090106", "2025090206"),
                             ("2025023008", "2025030106"),
                             ("2025090108", "2025090108"),
                             ("2025090108", "2025083108")]:
            with self.subTest(issue=issue, valid=valid), self.assertRaises(ValueError):
                probe.validate_times(issue, valid)

    def test_metadata_redacts_auth_and_preserves_exact_raw_and_hash(self):
        key = "sensitive-token-123456"
        response = FakeResponse(b"2025090108,2025090206,12.3\n")
        with tempfile.TemporaryDirectory() as td:
            with patch.object(probe.time, "sleep"):
                result = probe.probe("2025090108", "2025090206", key=key, raw_dir=td,
                                     session=SimpleNamespace(get=lambda *a, **kw: response),
                                     now=dt.datetime(2025, 9, 3, tzinfo=dt.timezone.utc))
            raw = (Path(td) / result["raw_file"]).read_bytes()
            meta_bytes = (Path(td) / result["metadata_file"]).read_bytes()
            self.assertEqual(raw, response.body)
            self.assertEqual(result["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertNotIn(key.encode(), meta_bytes)
            self.assertEqual(result["classification"], "plausible_payload")
            self.assertIs(result["complete"], False)
            self.assertTrue(response.closed)

    def test_retries_transport_failure_then_preserves_first_http_error(self):
        first_http = FakeResponse(b"rate limited", 429, {"Content-Length": "12"})
        send = SimpleNamespace(get=unittest.mock.Mock(side_effect=[requests.Timeout(), first_http]))
        with tempfile.TemporaryDirectory() as td, patch.object(probe.time, "sleep"):
            result = probe.probe("2025090108", "2025090206", key="abcd", raw_dir=td, session=send)
            self.assertEqual((Path(td) / result["raw_file"]).read_bytes(), b"rate limited")
        self.assertEqual(send.get.call_count, 2)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["http_status"], 429)
        self.assertEqual(result["classification"], "http_error")

    def test_stream_failure_has_sanitized_error(self):
        key = "secret-in-url-98765"

        class BrokenStream(FakeResponse):
            def iter_content(self, chunk_size):
                raise requests.ConnectionError(f"failed fetching {probe.URL}?authKey={key}")

        response = BrokenStream()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(probe.ProbeError) as caught:
                probe.probe("2025090108", "2025090206", key=key, raw_dir=td,
                            session=SimpleNamespace(get=lambda *a, **kw: response))
        self.assertEqual(str(caught.exception), "KMA APIHub response stream failed")
        self.assertNotIn(key, str(caught.exception))
        self.assertTrue(response.closed)

    def test_api_error_is_saved_as_raw(self):
        response = FakeResponse(b"ERROR: invalid authKey", 200)
        with tempfile.TemporaryDirectory() as td:
            result = probe.probe("2025090108", "2025090206", key="abcd", raw_dir=td,
                                 session=SimpleNamespace(get=lambda *a, **kw: response))
            self.assertEqual((Path(td) / result["raw_file"]).read_bytes(), response.body)
        self.assertEqual(result["classification"], "api_error")

    def test_oversize_rejected_without_persisting_partial_payload(self):
        response = FakeResponse(b"x" * (probe.MAX_RESPONSE_BYTES + 1))
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(probe.ProbeError, "maximum size"):
                probe.probe("2025090108", "2025090206", key="abcd", raw_dir=td,
                            session=SimpleNamespace(get=lambda *a, **kw: response))
            self.assertEqual(list(Path(td).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
