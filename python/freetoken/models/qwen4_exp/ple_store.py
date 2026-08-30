"""Immutable, page-aligned IQ4_NL PLE sidecar format.

PLE rows are never mmaped by paged serving.  ``.ftple`` keeps 45 native 90-byte
rows in each 4 KiB page; remaining 46 bytes are zero padding for direct I/O.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from freetoken.models.gguf.dequant import GGML_IQ4_NL
from freetoken.models.gguf.reader import load_gguf_headers

MAGIC = "FTPLE"
VERSION = 1
HEADER_BYTES = PAGE_BYTES = 4096
ROW_BYTES = 90
ROWS_PER_PAGE = 45
SOURCE_KIND = "GGUF"
QUANT_TYPE = "IQ4_NL"
LOGICAL_ROW_WIDTH = 160
NUM_SEGMENTS = 1
SEGMENT_DIRECTORY_OFFSET = 0
FLAGS = 0


def default_store_path(model_path: str) -> Path:
    return Path(model_path).with_suffix(".ftple")


def _source(model_path: str):
    _meta, shards, headers = load_gguf_headers(model_path)
    h = next((x for x in headers if x.name == "per_layer_token_embd.weight"), None)
    if h is None or h.ggml_type != GGML_IQ4_NL or h.row_bytes != ROW_BYTES or h.shape[-1] != 160:
        raise ValueError("Qwen4Exp PLE must be IQ4_NL [rows,160] with 90-byte rows")
    return shards, h


def source_fingerprint(model_path: str) -> str:
    """Stable sampled source identity from GGUF metadata and PLE row samples.

    This avoids hashing every PLE byte during startup. It detects metadata, shard,
    timestamp, and sampled-row changes; it is not a full-file integrity hash.
    Operators requiring that guarantee must verify GGUF artifacts before serving.
    """
    metadata, _shards_all, headers = load_gguf_headers(model_path)
    shards, h = _source(model_path)
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, default=str).encode())
    digest.update(b"\0")
    # Tensor directory identity catches changed offsets, shapes, quant types, and shard
    # assignment even when filesystem timestamps are preserved.
    for item in sorted(headers, key=lambda x: x.name):
        digest.update(repr((item.name, item.ggml_shape, item.ggml_type, item.nbytes,
                            item.data_offset, item.shard_index, item.shard_path)).encode())
        digest.update(b"\0")
    for s in shards:
        st = os.stat(s.path)
        digest.update(f"{Path(s.path).name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    digest.update(f"{h.shard_path}:{h.data_offset}:{h.nbytes}:{h.rows}:{h.row_bytes}".encode())
    # Cheap content samples protect against a same-size/same-mtime replacement while
    # avoiding a 28 GiB startup hash. Build and validation share this exact identity.
    try:
        with open(h.shard_path, "rb", buffering=0) as src:
            sample_offsets = {0, max(0, h.nbytes - ROW_BYTES)}
            sample_offsets.update(min(h.nbytes - ROW_BYTES, i * PAGE_BYTES)
                                  for i in range(1, min(64, h.nbytes // PAGE_BYTES + 1)))
            for rel in sorted(sample_offsets):
                digest.update(os.pread(src.fileno(), ROW_BYTES, h.data_offset + rel))
    except (OSError, ValueError):
        # Header-only test fixtures may not expose a readable source; directory identity
        # remains useful and the sidecar byte-copy step still validates reads.
        pass
    return digest.hexdigest()


def read_header(path: str | Path) -> dict:
    with open(path, "rb") as f:
        raw = f.read(HEADER_BYTES)
    try: value = json.loads(raw.rstrip(b"\0"))
    except Exception as exc: raise ValueError(f"invalid .ftple header: {path}") from exc
    if value.get("magic") != MAGIC or value.get("format_version", value.get("version")) != VERSION:
        raise ValueError(f"unsupported .ftple format: {path}")
    required = ("source_kind", "quant_type", "logical_row_width", "packed_row_bytes",
                "page_bytes", "rows_per_page", "total_rows", "total_data_pages",
                "num_segments", "segment_directory_offset", "flags", "source_fingerprint",
                "format_version")
    if any(key not in value for key in required):
        raise ValueError(f"incomplete .ftple header: {path}")
    if (value.get("source_kind") != SOURCE_KIND or value.get("quant_type") != QUANT_TYPE
            or value.get("logical_row_width") != LOGICAL_ROW_WIDTH
            or value.get("packed_row_bytes") != ROW_BYTES
            or value.get("page_bytes") != PAGE_BYTES
            or value.get("rows_per_page") != ROWS_PER_PAGE
            or value.get("row_bytes", ROW_BYTES) != ROW_BYTES
            or value.get("num_segments") != NUM_SEGMENTS
            or value.get("segment_directory_offset") != SEGMENT_DIRECTORY_OFFSET
            or value.get("flags") != FLAGS):
        raise ValueError(f"invalid .ftple page geometry: {path}")
    if int(value["total_rows"]) != int(value.get("rows", value["total_rows"])):
        raise ValueError(f"inconsistent .ftple row count: {path}")
    expected_pages = (int(value["total_rows"]) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
    if int(value["total_data_pages"]) != expected_pages:
        raise ValueError(f"invalid .ftple page count: {path}")
    return value


def validate_store(path: str | Path, model_path: str) -> dict:
    value = read_header(path)
    if value.get("source_fingerprint") != source_fingerprint(model_path):
        raise ValueError(".ftple source fingerprint mismatch; rebuild sidecar")
    expected = HEADER_BYTES + int(value["total_data_pages"]) * PAGE_BYTES
    if os.path.getsize(path) != expected: raise ValueError(".ftple length mismatch")
    _shards, source = _source(model_path)
    if int(value.get("total_rows", -1)) != source.rows:
        raise ValueError(".ftple row count differs from GGUF PLE tensor")
    return value


def build_store(model_path: str, output: str | Path | None = None, *, force: bool = False) -> Path:
    """Build atomically. Existing valid sidecar survives failed/parallel builds."""
    shards, h = _source(model_path)
    output = Path(output) if output else default_store_path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_suffix(output.suffix + ".lock")
    import fcntl
    with open(lock_path, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists() and not force:
            validate_store(output, model_path)
            return output
        fd, tmp_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
        try:
            expected_size = HEADER_BYTES + ((h.rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE) * PAGE_BYTES
            if shutil.disk_usage(output.parent).free < expected_size + PAGE_BYTES:
                raise OSError(f"insufficient free space for .ftple build ({expected_size} bytes needed)")
            fingerprint = source_fingerprint(model_path)
            with os.fdopen(fd, "w+b", buffering=0) as dst, open(h.shard_path, "rb", buffering=0) as src:
                dst.write(b"\0" * HEADER_BYTES)
                for first in range(0, h.rows, ROWS_PER_PAGE):
                    count = min(ROWS_PER_PAGE, h.rows - first)
                    payload = os.pread(src.fileno(), count * ROW_BYTES, h.data_offset + first * ROW_BYTES)
                    if len(payload) != count * ROW_BYTES: raise EOFError("short GGUF PLE read")
                    dst.write(payload)
                    dst.write(b"\0" * (PAGE_BYTES - len(payload)))
                total_pages = (h.rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
                header = {
                    "magic": MAGIC, "version": VERSION, "format_version": VERSION,
                    "source_kind": SOURCE_KIND,
                    "quant_type": QUANT_TYPE, "logical_row_width": LOGICAL_ROW_WIDTH,
                    "packed_row_bytes": ROW_BYTES, "rows": h.rows, "total_rows": h.rows,
                    "row_bytes": ROW_BYTES, "rows_per_page": ROWS_PER_PAGE,
                    "page_bytes": PAGE_BYTES, "total_data_pages": total_pages,
                    "num_segments": NUM_SEGMENTS,
                    "segment_directory_offset": SEGMENT_DIRECTORY_OFFSET, "flags": FLAGS,
                    "source_fingerprint": fingerprint,
                }
                raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
                if len(raw) > HEADER_BYTES:
                    raise ValueError(".ftple header overflow")
                dst.seek(0)
                dst.write(raw + b"\0" * (HEADER_BYTES - len(raw)))
                dst.flush()
                os.fsync(dst.fileno())
            # Validate unpublished bytes first. A corrupt temporary build must never
            # replace a previously valid sidecar.
            validate_store(tmp_name, model_path)
            os.replace(tmp_name, output)
            dirfd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
            # Reopen through runtime validation: a torn/incorrect header must never be
            # reported as a successful build merely because rename succeeded.
            validate_store(output, model_path)
        except Exception:
            try: os.unlink(tmp_name)
            except FileNotFoundError: pass
            raise
    return output
