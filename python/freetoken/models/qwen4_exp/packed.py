"""Bounded, split-aware packed GGUF source for native Qwen4-Exp."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import mmap
import os
from pathlib import Path
import threading
from typing import Iterable
import warnings

import torch

from freetoken.models.gguf.reader import GgufTensorHeader, load_gguf_headers


@dataclass(frozen=True)
class PackedLocator:
    name: str
    shard: int
    offset: int
    nbytes: int
    rows: int
    row_bytes: int
    shape: tuple[int, ...]
    ggml_type: int


class PackedCache:
    """Byte-bounded LRU for packed tensor ranges; values stay packed."""

    def __init__(self, capacity_bytes: int = 512 << 20):
        if capacity_bytes <= 0: raise ValueError("packed cache capacity must be positive")
        self.capacity_bytes = int(capacity_bytes)
        self._items: OrderedDict[tuple[str, tuple[int, ...]], torch.Tensor] = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0

    def get(self, key):
        value = self._items.get(key)
        if value is None:
            self.misses += 1; return None
        self._items.move_to_end(key); self.hits += 1
        return value

    def put(self, key, value: torch.Tensor) -> torch.Tensor:
        size = value.numel() * value.element_size()
        if size > self.capacity_bytes: raise MemoryError("packed range exceeds cache capacity")
        old = self._items.pop(key, None)
        if old is not None: self.bytes -= old.numel() * old.element_size()
        while self._items and self.bytes + size > self.capacity_bytes:
            _, victim = self._items.popitem(last=False)
            self.bytes -= victim.numel() * victim.element_size(); self.evictions += 1
        self._items[key] = value; self.bytes += size
        return value

    def report(self) -> dict:
        return {"capacity_bytes": self.capacity_bytes, "resident_bytes": self.bytes,
                "entries": len(self._items), "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions}


class PackedExpertHotCache:
    """Bounded packed ``(layer, expert)`` cache.

    Entries stay in GGUF byte layout.  A small probationary queue prevents one
    large scan from evicting replay-hot entries; repeated route hits promote to
    protected LRU.  Caller owns returned tensors until the synchronous copy
    completes, so eviction never invalidates an in-use tensor.
    """

    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = max(0, int(capacity_bytes))
        self.bytes = 0
        self.hits = self.misses = self.evictions = 0
        self._items: OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = OrderedDict()
        self._probationary: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._protected: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._freq: dict[tuple[int, int], int] = {}
        self._layer_counts: dict[int, int] = {}

    @staticmethod
    def _size(value: dict[str, torch.Tensor]) -> int:
        return sum(int(t.numel() * t.element_size()) for t in value.values())

    def get(self, key: tuple[int, int]):
        value = self._items.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self._freq[key] = self._freq.get(key, 0) + 1
        self._items.move_to_end(key)
        if key in self._probationary:
            self._probationary.pop(key, None)
            self._protected[key] = None
        elif key in self._protected:
            self._protected.move_to_end(key)
        return value

    def put(self, key: tuple[int, int], value: dict[str, torch.Tensor]):
        size = self._size(value)
        if self.capacity_bytes <= 0 or size > self.capacity_bytes:
            return value
        old = self._items.pop(key, None)
        if old is not None:
            self.bytes -= self._size(old)
            self._layer_counts[key[0]] = max(0, self._layer_counts.get(key[0], 1) - 1)
        while self._items and self.bytes + size > self.capacity_bytes:
            # Prefer one-shot probationary entries. Protected entries are only
            # evicted once probationary entries are exhausted.
            victim = next(iter(self._probationary), None)
            if victim is not None and self._layer_counts.get(victim[0], 0) <= 1:
                victim = next(
                    (key for key in self._probationary if self._layer_counts.get(key[0], 0) > 1),
                    None,
                )
            if victim is None:
                victim = next(iter(self._protected), None)
                if victim is not None and self._layer_counts.get(victim[0], 0) <= 1:
                    victim = next(
                        (key for key in self._protected if self._layer_counts.get(key[0], 0) > 1),
                        None,
                    )
            if victim is None:
                victim = next(iter(self._items), None)
            if victim is None:
                break
            self._probationary.pop(victim, None)
            self._protected.pop(victim, None)
            removed = self._items.pop(victim, None)
            if removed is not None:
                self.bytes -= self._size(removed)
                self._layer_counts[victim[0]] = max(0, self._layer_counts.get(victim[0], 1) - 1)
                self.evictions += 1
        self._items[key] = value
        self._probationary[key] = None
        self._freq.setdefault(key, 0)
        self._layer_counts[key[0]] = self._layer_counts.get(key[0], 0) + 1
        self.bytes += size
        return value

    def report(self) -> dict:
        return {
            "capacity_bytes": self.capacity_bytes,
            "resident_bytes": self.bytes,
            "entries": len(self._items),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "probationary": len(self._probationary),
            "protected": len(self._protected),
            "layers": len(self._layer_counts),
        }


class Qwen4ExpPackedSource:
    def __init__(self, model_path: str, *, cache_bytes: int = 512 << 20):
        self.model_path = str(model_path)
        self.metadata, shards, headers = load_gguf_headers(self.model_path)
        self.shards = shards
        self._files = [open(shard.path, "rb") for shard in shards]
        self._maps = [mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) for stream in self._files]
        self._locators = {h.name: self._locator(h) for h in headers}
        self.cache = PackedCache(cache_bytes)
        self.host_cache_bytes = 0
        self.read_bytes = 0
        try:
            io_workers = int(os.getenv("FREETOKEN_QWEN38_EXPERT_IO_WORKERS", "16"))
        except ValueError:
            io_workers = 16
        self._expert_io_workers = max(1, min(32, io_workers))
        self._expert_io_pool: ThreadPoolExecutor | None = None
        self._read_stats_lock = threading.Lock()
        # Expert banks are the hot host tier.  Dropping every read from the
        # kernel page cache defeats that tier once an entry is evicted from the
        # bounded packed cache, causing repeated NVMe reads on decode.  Keep
        # pages cacheable by default; retain opt-in DONTNEED for diagnostics.
        self._drop_source_cache = os.getenv("FREETOKEN_QWEN38_EXPERT_CACHE_DROP", "0").lower() in {
            "1", "true", "yes", "on"
        }
        self._closed = False

    def expert_set_bytes(self, layer: int, expert: int) -> int:
        """Packed bytes for one routed expert, without touching payload."""
        prefix = f"blk.{int(layer)}."
        return sum(
            self.locate(prefix + suffix).row_bytes * self.locate(prefix + suffix).shape[1]
            for suffix in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
        )

    def _read_rows_file(self, loc: PackedLocator, ids: tuple[int, ...]) -> torch.Tensor:
        """Read packed rows through file descriptors, bypassing mapped tensors."""
        if not ids:
            return torch.empty((0, loc.row_bytes), dtype=torch.uint8)
        fd = self._files[loc.shard].fileno()
        row_bytes = int(loc.row_bytes)
        # Expert rows are contiguous. One aligned range read avoids thousands
        # of tiny pread calls on every cold expert admission.
        contiguous = all(int(row) == int(ids[0]) + i for i, row in enumerate(ids))
        if contiguous:
            offset = loc.offset + int(ids[0]) * row_bytes
            size = len(ids) * row_bytes
            raw = os.pread(fd, size, offset)
            if len(raw) != size:
                raise EOFError(f"short GGUF read for {loc.name}")
            payload = bytearray(raw)
            drop_offset, drop_size = offset, size
        else:
            payload = bytearray(len(ids) * row_bytes)
            drop_offset, drop_size = loc.offset, 0
            for index, row in enumerate(ids):
                offset = loc.offset + int(row) * row_bytes
                raw = os.pread(fd, row_bytes, offset)
                if len(raw) != row_bytes:
                    raise EOFError(f"short GGUF read for {loc.name}")
                start = index * row_bytes
                payload[start:start + row_bytes] = raw
                drop_offset = min(drop_offset, offset) if drop_size else offset
                drop_size += row_bytes
        with self._read_stats_lock:
            self.read_bytes += len(payload)
        if self._drop_source_cache and hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(fd, max(0, drop_offset), drop_size, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        return torch.frombuffer(payload, dtype=torch.uint8).reshape(len(ids), loc.row_bytes)

    def _read_expert_entry(self, layer: int, expert: int, names: dict[str, PackedLocator]) -> dict[str, torch.Tensor]:
        """Read one routed expert's three packed banks.

        Each bank is one contiguous range, so a worker performs only three
        pread calls. Workers are bounded and return to the caller before cache
        insertion; ``PackedExpertHotCache`` remains single-owner and requires
        no locking or duplicate-entry reconciliation.
        """
        return {
            name: self._read_rows_file(
                loc,
                tuple(range(expert * int(loc.shape[1]), (expert + 1) * int(loc.shape[1]))),
            )
            for name, loc in names.items()
        }

    def _expert_pool(self) -> ThreadPoolExecutor:
        pool = self._expert_io_pool
        if pool is None:
            pool = ThreadPoolExecutor(
                max_workers=self._expert_io_workers,
                thread_name_prefix="qwen38-expert-io",
            )
            self._expert_io_pool = pool
        return pool

    def read_experts(
        self,
        layer: int,
        ids: Iterable[int],
        *,
        cache_bytes: int = 0,
        pin_memory: bool = False,
    ):
        """Return packed bank rows for routed expert IDs with bounded host cache."""
        ids = tuple(dict.fromkeys(int(i) for i in ids))
        if any(i < 0 for i in ids):
            raise IndexError("expert id outside bank")
        cache = getattr(self, "expert_cache", None)
        if cache is None or cache.capacity_bytes != max(0, int(cache_bytes)):
            self.expert_cache = cache = PackedExpertHotCache(cache_bytes)
        names = {
            "gate": self.locate(f"blk.{layer}.ffn_gate_exps.weight"),
            "up": self.locate(f"blk.{layer}.ffn_up_exps.weight"),
            "down": self.locate(f"blk.{layer}.ffn_down_exps.weight"),
        }
        result = {name: [] for name in names}
        missing: list[int] = []
        entries: dict[int, dict[str, torch.Tensor]] = {}
        for expert in ids:
            key = (int(layer), expert)
            entry = cache.get(key)
            if entry is None:
                missing.append(expert)
            else:
                entries[expert] = entry
        if missing:
            # Cold prefill can admit thousands of experts. Serial pread made
            # each 1K-token chunk wait behind tens of thousands of syscalls.
            # Bounded parallel reads keep NVMe busy without unbounded staging.
            if len(missing) == 1 or self._expert_io_workers == 1:
                loaded = ((expert, self._read_expert_entry(layer, expert, names)) for expert in missing)
                for expert, entry in loaded:
                    entries[expert] = entry
                    cache.put((int(layer), expert), entry)
            else:
                futures = {
                    self._expert_pool().submit(self._read_expert_entry, layer, expert, names): expert
                    for expert in missing
                }
                for future, expert in futures.items():
                    entry = future.result()
                    entries[expert] = entry
                    cache.put((int(layer), expert), entry)
        for expert in ids:
            entry = entries[expert]
            for name in names:
                result[name].append(entry[name])
        output = {}
        for name, loc in names.items():
            values = result[name]
            if not values:
                output[name] = torch.empty(
                    (0, int(loc.shape[1]), loc.row_bytes), dtype=torch.uint8
                )
                continue
            # Build transfer staging in final pinned form. The old path stacked
            # into pageable storage, then pin_memory() copied every selected bank
            # a second time before H2D.
            if pin_memory:
                staged = torch.empty(
                    (len(values), int(loc.shape[1]), loc.row_bytes),
                    dtype=torch.uint8,
                    pin_memory=True,
                )
                for index, value in enumerate(values):
                    staged[index].copy_(value)
                output[name] = staged
            else:
                output[name] = torch.stack(values, dim=0)
        return output

    def _locator(self, header: GgufTensorHeader) -> PackedLocator:
        if header.row_bytes is None:
            raise ValueError(f"{header.name}: unsupported GGUF row geometry for packed source")
        if header.nbytes != header.rows * header.row_bytes:
            raise ValueError(f"{header.name}: tensor byte size does not match row geometry")
        return PackedLocator(header.name, header.shard_index, header.data_offset,
                             header.nbytes, header.rows, header.row_bytes, header.shape,
                             header.ggml_type)

    def locate(self, name: str) -> PackedLocator:
        try: return self._locators[name]
        except KeyError as exc: raise KeyError(f"Qwen4-Exp tensor not found: {name}") from exc

    def _read(self, loc: PackedLocator, first: int, count: int) -> torch.Tensor:
        if first < 0 or count < 0 or first + count > loc.rows:
            raise IndexError(f"{loc.name}: row range {first}:{first + count} outside {loc.rows}")
        start = loc.offset + first * loc.row_bytes; size = count * loc.row_bytes
        payload = memoryview(self._maps[loc.shard])[start:start + size]
        if len(payload) != size: raise EOFError(f"short GGUF read for {loc.name}")
        with self._read_stats_lock:
            self.read_bytes += size
        # Copy from read-only mmap before exposing tensor storage; frombuffer warns on
        # read-only buffers even when the tensor is immediately cloned.
        return torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(count, loc.row_bytes)

    def _read_rows(self, loc: PackedLocator, ids: tuple[int, ...]) -> torch.Tensor:
        """Copy arbitrary rows into one packed buffer.

        PLE hashes produce mostly non-contiguous row IDs. Building one tensor per
        row and calling ``torch.cat`` made prefill spend minutes in Python and
        allocator overhead before layer 1 could consume PLE. Keep requested order,
        but perform one allocation and one tensor construction.
        """
        row_bytes = loc.row_bytes
        packed = bytearray(len(ids) * row_bytes)
        source = memoryview(self._maps[loc.shard])
        for index, row in enumerate(ids):
            start = loc.offset + row * row_bytes
            stop = start + row_bytes
            payload = source[start:stop]
            if len(payload) != row_bytes:
                raise EOFError(f"short GGUF read for {loc.name}")
            dst = index * row_bytes
            packed[dst:dst + row_bytes] = payload
        with self._read_stats_lock:
            self.read_bytes += len(packed)
        return torch.frombuffer(packed, dtype=torch.uint8).reshape(len(ids), row_bytes)

    def read_rows(self, name: str, rows: Iterable[int], *, device: torch.device | str = "cpu") -> torch.Tensor:
        loc = self.locate(name); ids = tuple(int(x) for x in rows)
        if not ids:
            return torch.empty((0, loc.row_bytes), dtype=torch.uint8, device=device)
        if any(x < 0 or x >= loc.rows for x in ids): raise IndexError(f"{name}: row outside tensor")
        key = (name, ids)
        packed = self.cache.get(key)
        if packed is None:
            runs = _runs(ids)
            if len(runs) == 1 and runs[0] == (ids[0], ids[-1] + 1):
                packed = self._read(loc, ids[0], len(ids))
            else:
                packed = self._read_rows(loc, ids)
            self.cache.put(key, packed)
        return packed.to(device=device, non_blocking=device != "cpu")

    def read_tensor(self, name: str, *, device: torch.device | str = "cpu") -> torch.Tensor:
        loc = self.locate(name)
        if "_exps.weight" in name or name == "per_layer_token_embd.weight":
            raise ValueError(f"{name}: full bank materialization forbidden; use read_rows")
        if loc.ggml_type in (0, 1, 30):
            raw = self._read(loc, 0, loc.rows)
            dtype = {0: torch.float32, 1: torch.float16, 30: torch.bfloat16}[loc.ggml_type]
            value = raw.reshape(-1).view(dtype).reshape(loc.shape)
            return value.to(device=device, non_blocking=device != "cpu")
        return self.read_rows(name, range(loc.rows), device=device)

    def mapped_tensor(self, name: str, *, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        """Expose packed tensor bytes directly from its file mmap.

        Used only for bounded GGUF expert banks. No bytes are copied or pinned; the
        mmap remains owned by this source for the process lifetime and pageable H2D
        staging happens when routed rows enter the GPU cache.
        """
        loc = self.locate(name)
        if shape is None:
            shape = (loc.rows, loc.row_bytes)
        if int(torch.tensor(shape).prod().item()) * 1 != loc.nbytes:
            raise ValueError(f"{name}: mapped shape {shape} does not match {loc.nbytes} bytes")
        view = memoryview(self._maps[loc.shard])[loc.offset:loc.offset + loc.nbytes]
        # This tensor is intentionally read-only and file-backed. PyTorch warns because
        # its generic frombuffer API cannot express read-only storage; suppress only this
        # known warning instead of copying the expert bank into RAM.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            return torch.frombuffer(view, dtype=torch.uint8, count=loc.nbytes).reshape(*shape)

    def materialize_tensor(self, name: str, *, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        """Copy one packed GGUF tensor into contiguous anonymous CPU ``uint8`` storage.

        This is intentionally a byte copy: routed expert kernels consume GGUF
        blocks directly, so dequantizing to BF16 would multiply host residency.
        The source mmap remains private to this call and can be closed after all
        requested banks have been materialized.
        """
        loc = self.locate(name)
        if not name.endswith("_exps.weight"):
            raise ValueError(f"{name}: resident materialization is only for expert banks")
        if shape is None:
            shape = (*loc.shape[:-1], loc.row_bytes)
        expected = loc.nbytes
        actual = 1
        for dim in shape:
            actual *= int(dim)
        if actual != expected:
            raise ValueError(f"{name}: materialized shape {shape} does not match {expected} bytes")
        packed = torch.empty((expected,), dtype=torch.uint8, device="cpu")
        source = memoryview(self._maps[loc.shard])[loc.offset:loc.offset + expected]
        if len(source) != expected:
            raise EOFError(f"short GGUF read for {name}")
        # frombuffer is a read-only view over the mmap; copy_ transfers bytes into
        # owned writable storage and does not retain that view after return.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            packed.copy_(torch.frombuffer(source, dtype=torch.uint8, count=expected))
        self.read_bytes += expected
        return packed.reshape(*shape)

    def report(self) -> dict:
        expert_cache = getattr(self, "expert_cache", None)
        return {
            "model": self.model_path,
            "tensor_count": len(self._locators),
            "expert_io_workers": self._expert_io_workers,
            "read_bytes": self.read_bytes,
            "cache": self.cache.report(),
            "expert_cache": expert_cache.report() if expert_cache is not None else None,
            "expert_cache_drop": self._drop_source_cache,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._expert_io_pool is not None:
            self._expert_io_pool.shutdown(wait=True, cancel_futures=True)
            self._expert_io_pool = None
        for mapped in self._maps:
            try:
                mapped.close()
            except BufferError:
                # A caller may still hold a mapped_tensor view.  Keep that
                # mapping alive until its exported buffer is released instead
                # of turning shutdown into a process-level failure.
                continue
        for stream in self._files: stream.close()


def _runs(ids: tuple[int, ...]):
    if not ids: return ()
    out = []; start = previous = ids[0]
    for value in ids[1:]:
        if value != previous + 1: out.append((start, previous + 1)); start = value
        previous = value
    out.append((start, previous + 1)); return tuple(out)


__all__ = ["PackedLocator", "PackedCache", "PackedExpertHotCache", "Qwen4ExpPackedSource"]
