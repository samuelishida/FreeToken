"""Shared GGUF access helpers: detection, metadata, and tensor enumeration.

Thin layer over ``gguf.GGUFReader`` (gguf-py). Metadata is read into a plain dict
keyed by the GGUF field name (``general.architecture``, ``gemma4.block_count`` ...);
tensors are exposed as ``GgufTensor`` records carrying the *torch* shape (ggml dims
reversed), the ggml quant type, and a zero-copy ``uint8`` view of the packed block
bytes laid out as ``[rows, row_bytes]`` (rows = product of all but the fastest ggml
dim; row_bytes spans whole quant blocks of the fastest dim).
"""

from __future__ import annotations

import functools
import os
import re
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def is_gguf_path(model_path: str) -> bool:
    """A local GGUF entrypoint, including the first file of a split family."""
    return isinstance(model_path, str) and os.path.isfile(model_path) and model_path.endswith(
        ".gguf"
    )


# Canonical name of the metadata-only GGUF that ``convert_checkpoint`` drops into an FTW
# dir built from a bare ``.gguf`` source. A GGUF carries its config AND tokenizer in the
# file's KV section, not sibling files, so a converted checkpoint has nowhere else to read
# them from -- this file is the header + KV bytes verbatim (tensor_count patched to 0, no
# tensor infos, no weight data), letting the FTW dir resolve config/tokenizer the exact
# same way the original ``.gguf`` file does.
FTW_METADATA_GGUF = "source_metadata.gguf"
# Records whether the source carried an untied "output.weight" head (the tensor table
# is stripped from metadata-only gguf files, so the fact travels as a KV).
OUTPUT_WEIGHT_PRESENT_KV = "freetoken.output_weight_present"


def gguf_config_source(model_path: str) -> str | None:
    """The ``.gguf`` file to source config/tokenizer/metadata from, or ``None``.

    A bare ``.gguf`` file resolves to itself; an FTW dir carrying a
    :data:`FTW_METADATA_GGUF` resolves to that embedded metadata file. This is the single
    seam config/tokenizer dispatch uses to decide "this checkpoint is GGUF-config-sourced"
    -- a real file and a converted-FTW dir both land on a genuine ``.gguf`` path the reader
    can parse, so no downstream code learns about the FTW wrapper.
    """
    if is_gguf_path(model_path):
        return model_path
    if isinstance(model_path, str) and os.path.isdir(model_path):
        cand = os.path.join(model_path, FTW_METADATA_GGUF)
        if os.path.isfile(cand):
            return cand
    return None


def write_metadata_gguf(source_gguf: str, dest_path: str) -> None:
    """Write a metadata-only GGUF: the source's header + KV section byte-for-byte, with
    ``tensor_count`` patched to 0 (no tensor infos, no weight data). Reading only the
    header+KV is cheap; the multi-GB tensor data is never touched.

    Validates by re-parsing: the copy must list zero tensors and expose the identical KV
    key set (the KV *bytes* are copied verbatim, so identical keys imply identical values).
    """
    import gguf

    reader = gguf.GGUFReader(source_gguf)
    assert reader.tensors, f"{source_gguf}: no tensors to bound the KV section"
    # The first tensor-info record starts exactly where the KV section ends (GGUF places no
    # padding between KV and tensor infos; padding is only before the tensor *data*).
    kv_end = int(reader.tensors[0].field.offset)
    buf = bytearray(reader.data[:kv_end].tobytes())  # header + all KV, verbatim
    buf[8:16] = b"\x00" * 8  # tensor_count is a u64 at byte 8; 0 is byte-order agnostic
    # The tensor table is dropped, but config derivation needs one fact from it (an
    # untied output head shows up only as an "output.weight" tensor). Append it as an
    # extra KV and bump kv_count (u64 at byte 16). Little-endian only -- the re-parse
    # below fails loudly on a big-endian source.
    key = OUTPUT_WEIGHT_PRESENT_KV.encode()
    present = any(t.name == "output.weight" for t in reader.tensors)
    buf += struct.pack("<Q", len(key)) + key
    buf += struct.pack("<I", int(gguf.GGUFValueType.BOOL)) + bytes([1 if present else 0])
    struct.pack_into("<Q", buf, 16, struct.unpack_from("<Q", buf, 16)[0] + 1)
    tmp = dest_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf)
    os.replace(tmp, dest_path)

    check = gguf.GGUFReader(dest_path)
    assert not check.tensors, "metadata gguf still lists tensors after patch"
    src_keys = {k for k in reader.fields if not k.startswith("GGUF.")}
    dst_keys = {k for k in check.fields if not k.startswith("GGUF.")}
    assert dst_keys == src_keys | {OUTPUT_WEIGHT_PRESENT_KV}, (
        f"metadata gguf KV keys differ from source: "
        f"missing {sorted(src_keys - dst_keys)}, extra {sorted(dst_keys - src_keys - {OUTPUT_WEIGHT_PRESENT_KV})}"
    )


@dataclass(frozen=True)
class GgufTensor:
    name: str
    shape: tuple[int, ...]  # torch order (ggml dims reversed)
    ggml_type: int
    rows: int  # product of shape[:-1] over the *ggml* layout = blocks-major rows
    row_bytes: int  # packed bytes per row (whole quant blocks of the fastest dim)
    _raw: np.ndarray  # uint8 view, shape [rows, row_bytes]

    def packed(self) -> torch.Tensor:
        """Zero-copy ``[rows, row_bytes]`` uint8 tensor of the native block bytes."""
        # GGUFReader exposes read-only mmap-backed arrays. Preserve zero-copy storage;
        # PyTorch's warning describes a write hazard that this packed path never uses.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given NumPy array is not writable",
                category=UserWarning,
            )
            return torch.from_numpy(self._raw)


@dataclass(frozen=True)
class GgufTensorHeader:
    """Header-only tensor location.

    ``data_offset`` is absolute within ``shard_path``. GGUF split offsets are not
    offsets into the entrypoint and must never be resolved against another shard.
    """

    name: str
    ggml_shape: tuple[int, ...]
    shape: tuple[int, ...]  # torch order (ggml dims reversed)
    ggml_type: int
    n_elements: int
    nbytes: int
    data_offset: int
    shard_index: int
    shard_path: str
    block_size: int | None = None
    type_size: int | None = None

    @property
    def rows(self) -> int:
        return int(np.prod(self.ggml_shape[1:])) if len(self.ggml_shape) > 1 else 1

    @property
    def row_bytes(self) -> int | None:
        if self.block_size is None or self.type_size is None:
            return None
        fastest = self.ggml_shape[0]
        if fastest % self.block_size:
            return None
        return fastest // self.block_size * self.type_size


@dataclass(frozen=True)
class GgufShard:
    path: str
    index: int
    count: int
    tensor_count: int
    data_offset: int
    file_size: int


_SPLIT_NAME = re.compile(r"^(?P<prefix>.+)-(?P<part>\d{5})-of-(?P<count>\d{5})\.gguf$")


def gguf_split_paths(model_path: str) -> tuple[str, ...]:
    """Resolve complete split family from shard 1 without accepting broad matches."""
    path = Path(model_path)
    if path.name == FTW_METADATA_GGUF:
        return (str(path),)
    if not path.is_file() or path.suffix != ".gguf":
        raise ValueError(f"GGUF path is not a file: {model_path}")

    import gguf

    first = _reader(str(path))
    split_no = int(_field_value(first, "split.no") or 0)
    split_count = int(_field_value(first, "split.count") or 1)
    if split_count <= 1:
        if split_no != 0:
            raise ValueError(f"single-file GGUF has invalid split.no={split_no}: {path}")
        return (str(path),)
    if split_no != 0:
        raise ValueError(
            f"multi-part GGUF must be loaded from first shard (-00001-...), got split.no={split_no}: {path}"
        )
    match = _SPLIT_NAME.fullmatch(path.name)
    if match is None or int(match.group("part")) != 1:
        raise ValueError(
            f"multi-part GGUF entrypoint must end with -00001-of-{split_count:05d}.gguf: {path}"
        )
    if int(match.group("count")) != split_count:
        raise ValueError(
            f"GGUF filename split count disagrees with metadata: {path} vs {split_count}"
        )
    return tuple(
        str(path.with_name(f"{match.group('prefix')}-{i:05d}-of-{split_count:05d}.gguf"))
        for i in range(1, split_count + 1)
    )


@functools.cache
def load_gguf_headers(model_path: str) -> tuple[dict[str, Any], tuple[GgufShard, ...], tuple[GgufTensorHeader, ...]]:
    """Parse all GGUF split headers; never materialize tensor payloads."""
    import gguf

    paths = gguf_split_paths(model_path)
    metadata: dict[str, Any] | None = None
    shards: list[GgufShard] = []
    headers: list[GgufTensorHeader] = []
    names: set[str] = set()
    expected_count: int | None = None
    for shard_index, shard_path in enumerate(paths):
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(
                f"GGUF split missing shard {shard_index + 1}/{len(paths)}: {shard_path}"
            )
        reader = _reader(shard_path)
        split_no = int(_field_value(reader, "split.no") or 0)
        split_count = int(_field_value(reader, "split.count") or 1)
        if split_no != shard_index or split_count != len(paths):
            raise ValueError(
                f"GGUF split metadata mismatch in {shard_path}: "
                f"split.no={split_no}, split.count={split_count}, expected {shard_index}/{len(paths)}"
            )
        if metadata is None:
            metadata = {name: field.contents() for name, field in reader.fields.items()}
            expected_count = split_count
        elif split_count != expected_count:
            raise ValueError(f"GGUF split count changed between shards: {shard_path}")

        file_size = os.path.getsize(shard_path)
        shards.append(GgufShard(shard_path, shard_index, split_count, len(reader.tensors),
                                int(reader.data_offset), file_size))
        for tensor in reader.tensors:
            if tensor.name in names:
                raise ValueError(f"duplicate GGUF tensor across split family: {tensor.name}")
            names.add(tensor.name)
            ggml_shape = tuple(int(value) for value in tensor.shape)
            ggml_type = int(tensor.tensor_type)
            try:
                block_size, type_size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
            except KeyError:
                block_size, type_size = None, None
            data_offset = int(tensor.data_offset)
            nbytes = int(tensor.n_bytes)
            if data_offset < int(reader.data_offset) or data_offset + nbytes > file_size:
                raise ValueError(
                    f"GGUF tensor range outside owning shard: {tensor.name} in {shard_path} "
                    f"offset={data_offset} bytes={nbytes} file_size={file_size}"
                )
            if block_size is not None and ggml_shape[0] % block_size:
                raise ValueError(
                    f"{tensor.name}: fastest dim {ggml_shape[0]} is not a multiple of "
                    f"quant block {block_size} for GGML type {ggml_type}"
                )
            headers.append(GgufTensorHeader(
                name=tensor.name,
                ggml_shape=ggml_shape,
                shape=tuple(reversed(ggml_shape)),
                ggml_type=ggml_type,
                n_elements=int(tensor.n_elements),
                nbytes=nbytes,
                data_offset=data_offset,
                shard_index=shard_index,
                shard_path=shard_path,
                block_size=block_size,
                type_size=type_size,
            ))
    assert metadata is not None
    return metadata, tuple(shards), tuple(headers)


def drop_gguf_page_cache(model_path: str) -> None:
    """Advise Linux to drop source pages after dense H2D materialization.

    GGUF readers keep file mappings cached for metadata reuse, but dense tensors
    have already moved to VRAM at this point. ``POSIX_FADV_DONTNEED`` releases
    those source pages without closing shared readers; unsupported platforms are
    a no-op. PLE/expert pagers use their own sources and are unaffected.
    """
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        paths = gguf_split_paths(model_path)
    except (OSError, ValueError):
        return
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            continue


def _field_value(reader, name: str) -> Any:
    field = reader.fields.get(name)
    if field is None:
        return None
    return field.contents()


@functools.cache
def _reader(model_path: str):
    import gguf

    return gguf.GGUFReader(model_path)


@functools.cache
def load_gguf_metadata(model_path: str) -> dict[str, Any]:
    """All GGUF KV metadata as ``{field_name: python_value}`` (arrays -> lists)."""
    reader = _reader(model_path)
    return {name: field.contents() for name, field in reader.fields.items()}


def gguf_architecture(model_path: str) -> str:
    arch = _field_value(_reader(model_path), "general.architecture")
    if arch is None:
        raise ValueError(f"GGUF file {model_path} has no general.architecture")
    return str(arch)


def iter_gguf_tensors(model_path: str) -> Iterator[GgufTensor]:
    """Yield every tensor across complete split family with packed block bytes.

    GGUF split tensor payloads are owned by individual files. Reading only the
    entrypoint silently drops most model weights, so resolve and iterate every
    validated shard in order.
    """
    import gguf

    for shard_path in gguf_split_paths(model_path):
        reader = _reader(shard_path)
        for t in reader.tensors:
            ne = [int(s) for s in t.shape]  # ggml order, fastest dim first
            torch_shape = tuple(reversed(ne))
            block, type_size = gguf.GGML_QUANT_SIZES[t.tensor_type]
            n_fast = ne[0]
            if n_fast % block != 0:
                raise ValueError(
                    f"{t.name}: fastest dim {n_fast} not a multiple of block {block} "
                    f"for {t.tensor_type.name}"
                )
            row_bytes = n_fast // block * type_size
            rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
            # gguf-py returns quantized tensors as raw uint8 but F32/F16 as typed arrays;
            # normalize everything to a flat byte view before shaping into [rows, row_bytes].
            data = t.data
            # GGUFReader exposes file-backed contiguous arrays. Do not call
            # ascontiguousarray unconditionally: on the Qwen3.8 PLE tensor that
            # would copy tens of GiB before adapter code can skip it.
            if not data.flags.c_contiguous:
                data = np.ascontiguousarray(data)
            flat = data.reshape(-1).view(np.uint8)
            raw = flat.reshape(rows, row_bytes)
            yield GgufTensor(
                name=t.name,
                shape=torch_shape,
                ggml_type=int(t.tensor_type),
                rows=rows,
                row_bytes=row_bytes,
                _raw=raw,
            )


def gguf_tensor_names(model_path: str) -> set[str]:
    return {header.name for header in load_gguf_headers(model_path)[2]}


__all__ = [
    "is_gguf_path",
    "FTW_METADATA_GGUF",
    "OUTPUT_WEIGHT_PRESENT_KV",
    "gguf_config_source",
    "write_metadata_gguf",
    "GgufTensor",
    "GgufTensorHeader",
    "GgufShard",
    "load_gguf_metadata",
    "load_gguf_headers",
    "drop_gguf_page_cache",
    "gguf_architecture",
    "gguf_split_paths",
    "iter_gguf_tensors",
    "gguf_tensor_names",
]
