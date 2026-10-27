"""One-shot diagnostic probe for KMA APIHub historical short forecasts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_shrt_grd"
DEFAULT_ISSUE = "2025090108"
DEFAULT_VALID = "2025090206"
MAX_RESPONSE_BYTES = 2_000_000
TIMEOUT_SECONDS = 20
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0


class ProbeError(RuntimeError):
    pass


def validate_times(issue: str, valid: str) -> None:
    """Require KST hourly timestamps, 08 issue cycle, and a future valid time."""
    if not re.fullmatch(r"\d{10}", issue) or not re.fullmatch(r"\d{10}", valid):
        raise ValueError("tmfc and tmef must use YYYYMMDDHH")
    try:
        issued = dt.datetime.strptime(issue, "%Y%m%d%H")
        effective = dt.datetime.strptime(valid, "%Y%m%d%H")
    except ValueError as exc:
        raise ValueError("invalid tmfc or tmef calendar time") from exc
    if issued.hour != 8:
        raise ValueError("tmfc must be an 08 KST issue cycle")
    if effective <= issued:
        raise ValueError("tmef must be later than tmfc")


def _read_key(env_file: str | None) -> str:
    key = os.environ.get("KMA_APIHUB_KEY", "")
    if env_file:
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                if name.strip() == "KMA_APIHUB_KEY":
                    key = value.strip().strip("\"'")
    if not key:
        raise ProbeError("KMA_APIHUB_KEY is not configured")
    return key


def _classify(status: int, body: bytes) -> str:
    if status < 200 or status >= 300:
        return "http_error"
    if not body:
        return "empty"
    text = body[:MAX_RESPONSE_BYTES].decode("utf-8", errors="replace").lstrip()
    if re.search(r"(?:ERROR|인증키|서비스키|잘못된 요청|INVALID|FAIL)", text, re.I):
        return "api_error"
    # A non-empty response is only a plausible payload; this probe cannot establish completeness.
    return "plausible_payload"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def probe(issue: str, valid: str, *, key: str, raw_dir: str | Path = "data/raw/kma_apihub",
          session: Any = requests, now: dt.datetime | None = None) -> dict[str, Any]:
    validate_times(issue, valid)
    parsed = urlsplit(URL)
    params = {"tmfc": issue, "tmef": valid, "vars": "TMP", "authKey": key}
    response = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = session.get(URL, params=params, timeout=TIMEOUT_SECONDS, verify=True,
                                   allow_redirects=False, stream=True)
            break
        except requests.RequestException:
            if attempt + 1 == MAX_ATTEMPTS:
                raise ProbeError("KMA APIHub request failed after bounded retries") from None
            time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
    assert response is not None
    body = bytearray()
    try:
        for chunk in response.iter_content(chunk_size=65536):
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ProbeError("response exceeded configured maximum size")
    except ProbeError:
        raise
    except (requests.RequestException, OSError) as exc:
        raise ProbeError("KMA APIHub response stream failed") from None
    finally:
        response.close()
    status = int(response.status_code)
    headers = {k.lower(): v for k, v in response.headers.items()
               if k.lower() in {"content-type", "content-length", "date", "server"}}
    # Do not follow a redirect or include its Location value in saved metadata.
    retrieved = now or dt.datetime.now(dt.timezone.utc)
    stamp = retrieved.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest = hashlib.sha256(body).hexdigest()
    base = Path(raw_dir) / f"kma_shrt_grd_{issue}_{valid}_{stamp}_{digest[:12]}"
    raw_path = base.with_suffix(".bin")
    metadata_path = base.with_suffix(".meta.json")
    _atomic_write(raw_path, bytes(body))
    # Only fixed, allowlisted request fields are recorded. The auth key is never serialized.
    metadata = {
        "source": URL,
        "request": {"tmfc": issue, "tmef": valid, "vars": "TMP"},
        "retrieved_at_utc": retrieved.astimezone(dt.timezone.utc).isoformat(),
        "http_status": status,
        "response_headers": headers,
        "response_bytes": len(body),
        "sha256": digest,
        "classification": _classify(status, bytes(body)),
        "complete": False,
        "attempts": attempt + 1,
        "redirect_refused": status in {301, 302, 303, 307, 308},
        "raw_file": raw_path.name,
    }
    _atomic_write(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"))
    metadata["metadata_file"] = metadata_path.name
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", default=DEFAULT_ISSUE)
    parser.add_argument("--valid", default=DEFAULT_VALID)
    parser.add_argument("--env-file")
    parser.add_argument("--raw-dir", default="data/raw/kma_apihub")
    parser.add_argument("--execute", action="store_true", help="send the single diagnostic request")
    args = parser.parse_args(argv)
    try:
        validate_times(args.issue, args.valid)
        if not args.execute:
            print(json.dumps({"mode": "dry_run", "source": URL,
                              "request": {"tmfc": args.issue, "tmef": args.valid, "vars": "TMP"},
                              "request_sent": False}, ensure_ascii=False))
            return 0
        result = probe(args.issue, args.valid, key=_read_key(args.env_file), raw_dir=args.raw_dir)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ProbeError, ValueError, OSError) as exc:
        print(f"probe failed: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
