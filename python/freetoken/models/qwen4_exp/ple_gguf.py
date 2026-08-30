"""Bounded GGUF IQ4_NL PLE backends.

``paged`` owns every PLE byte it keeps: fixed 4 KiB RAM pages, a bounded read
queue, a pinned transfer ring, and an optional fixed 96-byte packed-row GPU
cache. It never maps the PLE source GGUF tensor.
"""
from __future__ import annotations

import errno
import mmap
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

from freetoken.models.gguf.reader import load_gguf_headers
from freetoken.utils import init_logger

from .ple_store import (
    HEADER_BYTES,
    PAGE_BYTES,
    ROW_BYTES,
    ROWS_PER_PAGE,
    _source,
    build_store,
    default_store_path,
    validate_store,
)

_MAX_IO_WORKERS = 16
_DIRECT_FALLBACK_ERRNOS = frozenset({
    errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENODEV, errno.ENXIO,
})


@dataclass(frozen=True)
class PagePlan:
    """Stable request row dedup plus page grouping; inverse restores caller order."""

    unique_rows: tuple[int, ...]
    inverse: torch.Tensor
    pages: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    # ``pages`` deliberately retains first-seen order.  Consumers which copy rows
    # use it to preserve caller ordering; I/O consumers use these locality views.
    sorted_pages: tuple[int, ...]
    page_spans: tuple[tuple[int, int], ...]

    @classmethod
    def build(cls, ids: list[int]) -> "PagePlan":
        positions: dict[int, int] = {}
        unique: list[int] = []
        inverse: list[int] = []
        grouped: "OrderedDict[int, list[tuple[int, int]]]" = OrderedDict()
        for row in ids:
            index = positions.get(row)
            if index is None:
                index = len(unique)
                positions[row] = index
                unique.append(row)
                grouped.setdefault(row // ROWS_PER_PAGE, []).append(
                    (index, row % ROWS_PER_PAGE)
                )
            inverse.append(index)
        pages = tuple((page, tuple(rows)) for page, rows in grouped.items())
        sorted_pages = tuple(sorted(grouped))
        spans: list[tuple[int, int]] = []
        for page in sorted_pages:
            if spans and page == spans[-1][1] + 1:
                spans[-1] = (spans[-1][0], page)
            else:
                spans.append((page, page))
        return cls(
            tuple(unique), torch.tensor(inverse, dtype=torch.long), pages,
            sorted_pages, tuple(spans),
        )


class PackedRowCache:
    """Byte-bounded cache for packed 90-byte IQ4_NL rows."""

    def __init__(self, capacity_bytes: int, policy: str = "2q") -> None:
        self.capacity_bytes = max(0, int(capacity_bytes))
        self.policy = policy
        self._items: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._probationary: OrderedDict[int, None] = OrderedDict()
        self._protected: OrderedDict[int, None] = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0

    def get(self, row: int):
        value = self._items.get(int(row))
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self._items.move_to_end(int(row))
        if self.policy == "2q" and int(row) in self._probationary:
            self._probationary.pop(int(row), None)
            self._protected[int(row)] = None
        return value

    def put(self, row: int, value: torch.Tensor) -> None:
        if self.capacity_bytes < ROW_BYTES:
            return
        row = int(row)
        old = self._items.pop(row, None)
        if old is not None:
            self.bytes -= ROW_BYTES
            self._probationary.pop(row, None)
            self._protected.pop(row, None)
        while self._items and self.bytes + ROW_BYTES > self.capacity_bytes:
            queues = (self._probationary, self._protected) if self.policy == "2q" else (self._items,)
            victim = next((key for queue in queues for key in queue), None)
            if victim is None:
                break
            self._items.pop(victim, None)
            self._probationary.pop(victim, None)
            self._protected.pop(victim, None)
            self.bytes -= ROW_BYTES
            self.evictions += 1
        self._items[row] = value.detach().contiguous()
        if self.policy == "2q":
            self._probationary[row] = None
        self.bytes += ROW_BYTES

    def report(self) -> dict:
        return {
            "capacity_bytes": self.capacity_bytes,
            "resident_bytes": self.bytes,
            "entries": len(self._items),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


class _BasePLETable:
    dtype = torch.bfloat16
    head_dim = 160
    host_bytes = 0
    pageable_host_bytes = 0
    pinned_host_bytes = 0
    device_bytes = 0

    def set_status_observer(self, observer) -> None:
        """Install best-effort stage callback; callback must never own table locks."""
        self._status_observer = observer

    def _emit_status(self, stage: str) -> None:
        callback = getattr(self, "_status_observer", None)
        if callback is None:
            return
        try:
            callback(stage)
        except Exception:
            return

    @staticmethod
    def _ids(row_ids: torch.Tensor) -> list[int]:
        return [int(x) for x in row_ids.detach().reshape(-1).to("cpu").tolist()]

    def mark_forward(self) -> None:
        """Hook called at model-forward boundary, separate from prefetch calls."""

    def _inc(self, **counts: int) -> None:
        """Best-effort counter update shared by paged and direct backends."""
        lock = getattr(self, "_stats_lock", None)
        report = getattr(self, "_report", None)
        if lock is None or report is None:
            return
        with lock:
            for name, value in counts.items():
                if name in report:
                    report[name] += value

    def _dequant(self, packed: torch.Tensor, row_ids: torch.Tensor, out=None):
        from freetoken.kernel.triton.ple_iq4_nl import (
            dequant_iq4_nl_rows,
            gather_dequant_iq4_nl_rows,
        )

        inc = getattr(self, "_inc", None)
        if inc is not None:
            inc(dequant_calls=1)
        self._emit_status("dequant_started")
        try:
            # Native paged reads are [N,90]; GPU cache/staging slots are aligned
            # [N,96].  The helper validates both and consumes only first 90 bytes.
            aligned = packed[:, :ROW_BYTES] if packed.shape[1] == ROW_BYTES else packed
            if getattr(self, "_fused_dequant", False):
                try:
                    value = gather_dequant_iq4_nl_rows(aligned, out_dtype=self.dtype)
                    if inc is not None:
                        inc(fused_dequant_calls=1)
                except Exception:
                    # Experimental path must never make an unsupported ROCm compiler
                    # or device a serving failure.
                    if inc is not None:
                        inc(fused_dequant_fallbacks=1)
                    value = dequant_iq4_nl_rows(aligned, out_dtype=self.dtype)
            else:
                value = dequant_iq4_nl_rows(aligned, out_dtype=self.dtype)
            value = value.view(*row_ids.shape[:-1], -1)
            if value.device != row_ids.device:
                value = value.to(device=row_ids.device, non_blocking=row_ids.device.type == "cuda")
        except Exception:
            if inc is not None:
                inc(dequant_errors=1)
            raise
        if out is not None:
            out.copy_(value)
            return out
        return value

    def report(self) -> dict:
        with self._stats_lock:
            report = dict(self._report)
        # Stable semantic aliases consumed by live status clients.
        report.setdefault("prefetch_calls", report.get("prefetch_requests", 0))
        report.setdefault("pages_read", report.get("ssd_read_ops", 0))
        report.setdefault("cache_hits", report.get("ram_page_hits", 0) + report.get("gpu_hits", 0))
        report.setdefault("cache_misses", report.get("ram_page_misses", 0) + report.get("gpu_misses", 0))
        report.setdefault("waits", report.get("io_wait_us", 0) + report.get("lookup_wait_us", 0))
        return report


class PackedPagedPLETable(_BasePLETable):
    """4 KiB sidecar page LRU with bounded in-flight dedup and fixed budgets."""

    def __init__(self, model_path: str, args) -> None:
        store = getattr(args, "ple_store", None) or str(default_store_path(model_path))
        policy = getattr(args, "ple_store_build", "auto")
        if policy == "force":
            build_store(model_path, store, force=True)
        elif not os.path.exists(store):
            if policy == "never":
                raise ValueError(f"PLE sidecar missing: {store}")
            build_store(model_path, store)
        else:
            try:
                validate_store(store, model_path)
            except ValueError:
                if policy == "never":
                    raise
                init_logger(__name__).warning("rebuilding stale/legacy PLE sidecar: %s", store)
                build_store(model_path, store, force=True)

        self.header = validate_store(store, model_path)
        self.path = store
        self.num_rows = int(self.header["total_rows"])
        page_budget = getattr(args, "ple_ram_cache_mib", 512)
        if isinstance(page_budget, str):
            page_budget = 512 if page_budget in ("auto", "adaptive") else int(page_budget)
        row_budget = getattr(args, "ple_row_cache_mib", 0)
        if isinstance(row_budget, str):
            row_budget = 0 if row_budget in ("auto", "adaptive") else int(row_budget)
        self.capacity = max(0, int(page_budget) * (1 << 20) // PAGE_BYTES)
        self._row_cache_bytes = max(0, int(row_budget)) * (1 << 20)
        self._gpu_capacity = max(0, int(getattr(args, "ple_gpu_cache_mib", 128)) * (1 << 20) // 96)
        self._staging_bytes = max(0, int(getattr(args, "ple_staging_mib", 32)) * (1 << 20))
        self._prefetch_enabled = bool(getattr(args, "ple_prefetch", True))
        self._cache_policy = str(getattr(args, "ple_cache_policy", "lru"))
        # GPU packed rows are part of normal paged operation whenever budget is
        # non-zero.  Keep an explicit false switch for rollback/debugging, while
        # making omitted legacy args use the production cache path.
        self._batched_cache = bool(getattr(args, "ple_batched_cache", True))
        self._fused_dequant = bool(getattr(args, "ple_fused_dequant", False))
        if self._cache_policy not in ("lru", "2q"):
            raise ValueError(f"unsupported PLE cache policy: {self._cache_policy}")
        requested_depth = getattr(args, "ple_io_depth", 64)
        if isinstance(requested_depth, str):
            requested_depth = 64 if requested_depth in ("auto", "adaptive") else int(requested_depth)
        requested_depth = max(1, int(requested_depth))
        self._io_depth = min(requested_depth, 256)
        prefetch_depth = getattr(args, "ple_prefetch_depth", self._io_depth)
        if isinstance(prefetch_depth, str):
            prefetch_depth = (
                self._io_depth if prefetch_depth in ("auto", "adaptive") else int(prefetch_depth)
            )
        self._prefetch_depth = min(max(0, int(prefetch_depth)), 256)
        self._io_workers = min(self._io_depth, _MAX_IO_WORKERS)
        self._queue_capacity = max(1, self._io_depth)
        self._stats_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._lookup_lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._pages: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._rows = PackedRowCache(self._row_cache_bytes, self._cache_policy)
        self._probationary: OrderedDict[int, None] = OrderedDict()
        self._protected: OrderedDict[int, None] = OrderedDict()
        self._prefetch_pages: set[int] = set()
        self._prefetched_resident: set[int] = set()
        self._inflight: dict[int, Future[torch.Tensor]] = {}
        self._page_refs: dict[int, int] = {}
        self._read_slots = threading.BoundedSemaphore(self._queue_capacity)
        # Prefetch itself is a bounded work queue.  Read workers live in a separate
        # executor so a coordinator waiting for a read slot can never deadlock the
        # workers which release that slot.
        self._prefetch_slots = threading.BoundedSemaphore(self._queue_capacity)
        self._executor = ThreadPoolExecutor(
            max_workers=self._io_workers, thread_name_prefix="ftple"
        )
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=self._io_workers, thread_name_prefix="ftple-prefetch"
        )
        self._pending: dict[
            int, tuple[torch.Tensor, torch.Tensor | None, torch.cuda.Event | None, Future[None]]
        ] = {}
        self._closed = False
        self._staging = self._allocate_staging()
        self.pinned_host_bytes = (
            self._staging.numel() * self._staging.element_size()
            if self._staging is not None and self._staging.is_pinned() else 0
        )
        self._staging_cursor = 0
        self._staging_event: torch.cuda.Event | None = None
        self._device = (torch.device("cuda", torch.cuda.current_device())
                        if torch.cuda.is_available() else None)
        self._gpu_slots, self._gpu_rows, self._free_slots = self._allocate_gpu_cache()
        self.device_bytes = self._gpu_capacity * 96

        requested = getattr(args, "ple_io", "auto")
        self._direct_fd: int | None = None
        self._buffered_fd: int | None = os.open(store, os.O_RDONLY)
        actual = "buffered"
        if requested in ("auto", "direct") and hasattr(os, "O_DIRECT"):
            try:
                self._direct_fd = os.open(store, os.O_RDONLY | os.O_DIRECT)
                actual = "direct"
            except OSError as exc:
                if requested == "direct":
                    self.close()
                    raise RuntimeError("PLE direct I/O unavailable") from exc
                actual = "buffered-fallback"
                init_logger(__name__).warning(
                    "PLE direct I/O unavailable; using buffered pread. Linux file-page "
                    "cache may exceed --ple-ram-cache-mib."
                )
        elif requested == "direct":
            self.close()
            raise RuntimeError("PLE direct I/O is unavailable on this platform")
        if actual != "direct" and self._buffered_fd is not None and hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(self._buffered_fd, 0, 0, os.POSIX_FADV_RANDOM)
            except OSError:
                pass

        self._report = {
            "backend": "ftple-paged", "mode": "paged", "sidecar": store,
            "source_policy": "sidecar_ssd_bounded_ram_cache",
            "sidecar_fingerprint": self.header.get("source_fingerprint"),
            "num_rows": self.num_rows, "ram_budget_bytes": self.capacity * PAGE_BYTES,
            "gpu_budget_bytes": self.device_bytes, "staging_budget_bytes": self._staging_bytes,
            "pageable_ram_bytes": self.capacity * PAGE_BYTES,
            "row_cache_budget_bytes": self._row_cache_bytes,
            "pinned_staging_bytes": self.pinned_host_bytes,
            "io": actual, "io_depth": self._io_depth, "io_workers": self._io_workers,
            "queue_capacity": self._queue_capacity,
            "prefetch_workers": self._io_workers,
            "prefetch_queue_capacity": self._queue_capacity,
            "prefetch_read_ahead_pages": self._prefetch_depth,
            "cache_policy": self._cache_policy,
            "batched_cache": self._batched_cache,
            "fused_dequant": self._fused_dequant,
            "prefetch_enabled": self._prefetch_enabled, "forward_calls": 0,
            "lookup_calls": 0, "lookup_rows": 0, "ram_page_hits": 0,
            "ram_page_misses": 0, "ram_page_evictions": 0, "gpu_hits": 0,
            "gpu_misses": 0, "gpu_evictions": 0, "ssd_read_ops": 0,
            "ssd_read_bytes": 0, "prefetch_requests": 0, "prefetch_hits": 0,
            "prefetch_late": 0, "requested_packed_bytes": 0,
            "d2h_id_bytes": 0, "h2d_row_bytes": 0, "io_wait_us": 0,
            "lookup_wait_us": 0, "dequant_calls": 0, "dequant_errors": 0,
            # Host-clock submission/planning telemetry. These never synchronize HIP/CUDA.
            "prefetch_plan_us": 0, "prefetch_d2h_us": 0, "io_queue_wait_us": 0,
            "ram_gather_us": 0, "h2d_submit_us": 0, "io_failed_ops": 0,
            "io_queue_starvation": 0,
            "prefetch_queue_saturation": 0,
            "prefetch_joins": 0,
            "prefetch_pages_submitted": 0,
            "prefetch_pages_skipped": 0,
            "prefetch_pages_planned": 0,
            "prefetch_failures": 0,
            "prefetch_demand_hits": 0, "prefetch_pages_evicted": 0,
            "planned_rows": 0, "planned_unique_rows": 0, "planned_pages": 0,
            "gpu_batched_rows": 0, "gpu_batch_ops": 0, "gpu_reconstruct_ops": 0,
            "fused_dequant_calls": 0, "fused_dequant_fallbacks": 0,
            "row_cache_hits": 0, "row_cache_misses": 0, "row_cache_evictions": 0,
        }
        self.pageable_host_bytes = self.capacity * PAGE_BYTES
        self.host_bytes = self.pageable_host_bytes + self._row_cache_bytes + self._staging_bytes
        init_logger(__name__).info(
            "PLE storage: mode=%s sidecar=%s IQ4_NL rows=%d page=4096x45 "
            "ram=%dMiB gpu=%dMiB staging=%dMiB io=%s depth=%d workers=%d prefetch=%s",
            "paged", store, self.num_rows, self.capacity * PAGE_BYTES >> 20,
            self.device_bytes >> 20, self._staging_bytes >> 20, actual,
            self._io_depth, self._io_workers, self._prefetch_enabled,
        )

    def _allocate_staging(self) -> torch.Tensor | None:
        if self._staging_bytes < 96:
            return None
        try:
            return torch.empty((self._staging_bytes // 96, 96), dtype=torch.uint8, pin_memory=True)
        except (RuntimeError, OSError):
            return torch.empty((self._staging_bytes // 96, 96), dtype=torch.uint8)

    def _allocate_gpu_cache(self):
        if not self._gpu_capacity or not torch.cuda.is_available():
            return None, OrderedDict(), []
        assert self._device is not None
        slots = torch.empty((self._gpu_capacity, 96), dtype=torch.uint8, device=self._device)
        return slots, OrderedDict(), list(range(self._gpu_capacity - 1, -1, -1))

    def _inc(self, **counts: int) -> None:
        with self._stats_lock:
            for name, value in counts.items():
                if name in self._report:
                    self._report[name] += value

    def mark_forward(self) -> None:
        self._inc(forward_calls=1)

    def _read_page(self, page: int) -> torch.Tensor:
        offset = HEADER_BYTES + page * PAGE_BYTES
        started = time.perf_counter_ns()
        fd = self._direct_fd
        if fd is not None:
            aligned = mmap.mmap(-1, PAGE_BYTES)
            try:
                read = os.preadv(fd, [aligned], offset)
                raw = aligned[:read]
            except OSError as exc:
                if exc.errno not in _DIRECT_FALLBACK_ERRNOS:
                    raise
                with self._io_lock:
                    if self._direct_fd is not None:
                        os.close(self._direct_fd)
                        self._direct_fd = None
                        with self._stats_lock:
                            self._report["io"] = "buffered-fallback"
                        init_logger(__name__).warning(
                            "PLE direct I/O unavailable; buffered reads may use Linux file-page cache"
                        )
                raw = os.pread(self._buffered_fd, PAGE_BYTES, offset)
                self._advise_drop(offset)
            finally:
                aligned.close()
        else:
            if self._buffered_fd is None:
                raise RuntimeError("PLE table is closed")
            raw = os.pread(self._buffered_fd, PAGE_BYTES, offset)
            self._advise_drop(offset)
        if len(raw) != PAGE_BYTES:
            raise EOFError("short .ftple page read")
        self._inc(ssd_read_ops=1, ssd_read_bytes=PAGE_BYTES,
                  io_wait_us=(time.perf_counter_ns() - started) // 1000)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8)

    def _read_span(self, first: int, last: int) -> dict[int, torch.Tensor]:
        """Read adjacent 4 KiB pages with one aligned physical operation."""
        if first > last:
            return {}
        size = (last - first + 1) * PAGE_BYTES
        offset = HEADER_BYTES + first * PAGE_BYTES
        started = time.perf_counter_ns()
        fd = self._direct_fd
        if fd is not None:
            aligned = mmap.mmap(-1, size)
            try:
                read = os.preadv(fd, [aligned], offset)
                raw = bytes(aligned[:read])
            except OSError as exc:
                if exc.errno not in _DIRECT_FALLBACK_ERRNOS:
                    raise
                with self._io_lock:
                    if self._direct_fd is not None:
                        os.close(self._direct_fd)
                        self._direct_fd = None
                        with self._stats_lock:
                            self._report["io"] = "buffered-fallback"
                if self._buffered_fd is None:
                    raise RuntimeError("PLE table is closed") from exc
                raw = os.pread(self._buffered_fd, size, offset)
                self._advise_drop(offset, size)
            finally:
                aligned.close()
        else:
            if self._buffered_fd is None:
                raise RuntimeError("PLE table is closed")
            raw = os.pread(self._buffered_fd, size, offset)
            self._advise_drop(offset, size)
        if len(raw) != size:
            raise EOFError("short .ftple span read")
        self._inc(ssd_read_ops=1, ssd_read_bytes=size,
                  io_wait_us=(time.perf_counter_ns() - started) // 1000)
        return {
            page: torch.frombuffer(
                bytearray(raw[(page - first) * PAGE_BYTES:(page - first + 1) * PAGE_BYTES]),
                dtype=torch.uint8,
            )
            for page in range(first, last + 1)
        }

    def _advise_drop(self, offset: int, length: int = PAGE_BYTES) -> None:
        fd = self._buffered_fd
        if fd is None or not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
            return
        try:
            os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass

    def _touch_page_locked(self, page: int, *, demand: bool) -> None:
        self._pages.move_to_end(page)
        if self._cache_policy != "2q":
            return
        if page in self._protected:
            self._protected.move_to_end(page)
            return
        if page in self._probationary:
            self._probationary.pop(page)
            if demand:
                self._protected[page] = None
            else:
                self._probationary[page] = None
        elif demand:
            self._protected[page] = None
        else:
            self._probationary[page] = None
        protected_limit = max(1, self.capacity * 3 // 4)
        while len(self._protected) > protected_limit:
            demoted, _ = self._protected.popitem(last=False)
            self._probationary[demoted] = None

    def _evict_one_locked(self) -> bool:
        queues = (self._probationary, self._protected) if self._cache_policy == "2q" else (self._pages,)
        victim = next(
            (key for queue in queues for key in queue if self._page_refs.get(key, 0) == 0), None
        )
        if victim is None:
            return False
        self._pages.pop(victim, None)
        if victim in self._prefetched_resident:
            self._prefetched_resident.discard(victim)
            self._inc(prefetch_pages_evicted=1)
        self._probationary.pop(victim, None)
        self._protected.pop(victim, None)
        self._inc(ram_page_evictions=1)
        return True

    def _admit_page_locked(self, page: int, value: torch.Tensor) -> None:
        if not self.capacity:
            # A zero-budget table still completes demand reads, but must not
            # retain prefetch bookkeeping indefinitely.
            self._prefetch_pages.discard(page)
            return
        if page in self._pages:
            self._prefetch_pages.discard(page)
            return
        while len(self._pages) >= self.capacity:
            if not self._evict_one_locked():
                return
        self._pages[page] = value
        if page in self._prefetch_pages:
            self._prefetched_resident.add(page)
        self._touch_page_locked(page, demand=page not in self._prefetch_pages)
        self._prefetch_pages.discard(page)

    def _read_done(
        self, page: int, future: Future[torch.Tensor], waiter: Future[torch.Tensor]
    ) -> None:
        try:
            value = future.result()
        except BaseException as exc:
            self._inc(io_failed_ops=1)
            with self._cache_lock:
                if self._inflight.get(page) is waiter:
                    self._inflight.pop(page, None)
                self._prefetch_pages.discard(page)
            self._read_slots.release()
            if not waiter.done():
                waiter.set_exception(exc)
            return
        with self._cache_lock:
            if self._inflight.get(page) is waiter:
                self._inflight.pop(page, None)
            self._admit_page_locked(page, value)
        self._read_slots.release()
        if not waiter.done():
            waiter.set_result(value)

    def _submit_read(self, page: int, *, block: bool, prefetch: bool = False) -> Future[torch.Tensor] | None:
        """Deduplicate one page read without waiting under ``_cache_lock``."""
        with self._cache_lock:
            if page in self._pages:
                return None
            existing = self._inflight.get(page)
            if existing is not None:
                return existing
            waiter: Future[torch.Tensor] = Future()
            self._inflight[page] = waiter
            if prefetch:
                self._prefetch_pages.add(page)
        wait_started = time.perf_counter_ns()
        acquired = self._read_slots.acquire(blocking=block)
        self._inc(io_queue_wait_us=(time.perf_counter_ns() - wait_started) // 1000)
        if not acquired:
            self._inc(io_queue_starvation=1)
            with self._cache_lock:
                if self._inflight.get(page) is waiter:
                    self._inflight.pop(page, None)
                self._prefetch_pages.discard(page)
            waiter.cancel()
            return None
        try:
            future = self._executor.submit(self._read_page, page)
        except BaseException as exc:
            self._read_slots.release()
            with self._cache_lock:
                if self._inflight.get(page) is waiter:
                    self._inflight.pop(page, None)
                self._prefetch_pages.discard(page)
            waiter.set_exception(exc)
            raise
        future.add_done_callback(lambda done, p=page, w=waiter: self._read_done(p, done, w))
        return waiter

    def _start_read(self, page: int, waiter: Future[torch.Tensor], *, block: bool) -> None:
        """Start read for placeholder installed by a synchronous lookup.

        Caller must not hold ``_cache_lock``: blocking on ``_read_slots`` while holding
        it prevents completed-read callbacks from releasing that same semaphore.
        """
        if not self._read_slots.acquire(blocking=block):
            with self._cache_lock:
                if self._inflight.get(page) is waiter:
                    self._inflight.pop(page, None)
            waiter.cancel()
            return
        try:
            future = self._executor.submit(self._read_page, page)
        except BaseException as exc:
            self._read_slots.release()
            with self._cache_lock:
                if self._inflight.get(page) is waiter:
                    self._inflight.pop(page, None)
            waiter.set_exception(exc)
            raise
        future.add_done_callback(lambda done, p=page, w=waiter: self._read_done(p, done, w))

    def _submit_span(self, first: int, last: int) -> int:
        """Submit one bounded span and create per-page join futures."""
        pages = list(range(first, last + 1))
        waiters: dict[int, Future[torch.Tensor]] = {}
        with self._cache_lock:
            for page in pages:
                if page in self._pages or page in self._inflight:
                    continue
                waiter: Future[torch.Tensor] = Future()
                self._inflight[page] = waiter
                self._prefetch_pages.add(page)
                waiters[page] = waiter
        if not waiters:
            return 0
        if not self._read_slots.acquire(blocking=True):
            with self._cache_lock:
                for page, waiter in waiters.items():
                    if self._inflight.get(page) is waiter:
                        self._inflight.pop(page, None)
                    self._prefetch_pages.discard(page)
            return 0
        try:
            future = self._executor.submit(self._read_span, min(waiters), max(waiters))
        except BaseException as exc:
            self._read_slots.release()
            with self._cache_lock:
                for page, waiter in waiters.items():
                    if self._inflight.get(page) is waiter:
                        self._inflight.pop(page, None)
                    self._prefetch_pages.discard(page)
                    waiter.set_exception(exc)
            raise

        def done(span_future):
            try:
                values = span_future.result()
            except BaseException as exc:
                self._inc(io_failed_ops=1)
                with self._cache_lock:
                    for page, waiter in waiters.items():
                        if self._inflight.get(page) is waiter:
                            self._inflight.pop(page, None)
                        self._prefetch_pages.discard(page)
                        if not waiter.done():
                            waiter.set_exception(exc)
            else:
                with self._cache_lock:
                    for page, waiter in waiters.items():
                        if self._inflight.get(page) is waiter:
                            self._inflight.pop(page, None)
                        self._admit_page_locked(page, values[page])
                        if not waiter.done():
                            waiter.set_result(values[page])
            finally:
                self._read_slots.release()

        future.add_done_callback(done)
        return len(waiters)

    def _release_page(self, page: int) -> None:
        with self._cache_lock:
            refs = self._page_refs.get(page, 0)
            if refs <= 1:
                self._page_refs.pop(page, None)
            else:
                self._page_refs[page] = refs - 1

    def _page(self, page: int) -> torch.Tensor:
        launch = False
        with self._cache_lock:
            value = self._pages.get(page)
            if value is not None:
                self._touch_page_locked(page, demand=True)
                self._page_refs[page] = self._page_refs.get(page, 0) + 1
                if page in self._prefetched_resident:
                    self._prefetched_resident.discard(page)
                    self._inc(prefetch_demand_hits=1, prefetch_hits=1)
                self._inc(ram_page_hits=1)
                return value
            future = self._inflight.get(page)
            if future is None:
                # Reserve placeholder and consumer reference before launching. Read
                # submission can block on bounded I/O capacity, so launch outside lock;
                # callbacks need this lock to admit pages and release semaphore.
                future = Future()
                self._inflight[page] = future
                launch = True
                self._inc(ram_page_misses=1)
            # Reserve a consumer reference before waiting. Completion callbacks must not
            # evict this page while the caller is about to copy row bytes from it.
            self._page_refs[page] = self._page_refs.get(page, 0) + 1
        if launch:
            self._start_read(page, future, block=True)
        started = time.perf_counter_ns()
        try:
            value = future.result()
        except BaseException:
            # A demand reference is reserved before launching/waiting.  Do not
            # strand it when a short read, cancellation, or close fails; stranded
            # refs disable every later LRU eviction.
            self._release_page(page)
            raise
        self._inc(lookup_wait_us=(time.perf_counter_ns() - started) // 1000)
        with self._cache_lock:
            cached = self._pages.get(page)
            if cached is not None:
                value = cached
                self._touch_page_locked(page, demand=True)
            else:
                self._admit_page_locked(page, value)
        return value

    def _prefetch_ids(
        self,
        host_ids: torch.Tensor,
        ready: torch.cuda.Event | None,
        *,
        is_decode: bool = False,
    ) -> None:
        started = time.perf_counter_ns()
        if ready is not None:
            ready.synchronize()
        ids = [int(x) for x in host_ids.reshape(-1).tolist()]
        plan = PagePlan.build([row for row in ids if 0 <= row < self.num_rows])
        # A prefetch call is one complete known window.  Decode has no future
        # token IDs here, so never invent/speculate rows; caller-supplied rows
        # are capped to the bounded read-ahead budget only.
        pages = plan.sorted_pages
        # Prefetch only bounded read-ahead.  A full 1K/24K prefill can hash
        # tens of thousands of random pages before layer 1 is reached; that
        # floods NVMe and evicts replay-hot rows. Demand lookup still reads
        # remaining pages and joins any in-flight futures.
        limit = self._prefetch_depth if is_decode else max(128, self._prefetch_depth * 4)
        pages = pages[:max(0, limit)]
        page_set = set(pages)
        self._inc(
            planned_rows=len(ids), planned_unique_rows=len(plan.unique_rows),
            planned_pages=len(pages), prefetch_pages_planned=len(pages),
        )
        # Iterate sorted pages in bounded 64-page (256 KiB) spans. Per-page
        # futures remain separate so demand lookup joins exactly one page.
        for first, last in plan.page_spans:
            cursor = first
            while cursor <= last:
                span_last = min(last, cursor + 63)
                span_pages = [page for page in range(cursor, span_last + 1) if page in page_set]
                if span_pages:
                    submitted = self._submit_span(min(span_pages), max(span_pages))
                    if submitted:
                        self._inc(prefetch_pages_submitted=submitted)
                    else:
                        self._inc(prefetch_pages_skipped=len(span_pages))
                cursor = span_last + 1
        self._inc(prefetch_plan_us=(time.perf_counter_ns() - started) // 1000)

    def _run_prefetch(
        self,
        host_ids: torch.Tensor,
        ready: torch.cuda.Event | None,
        is_decode: bool,
    ) -> None:
        self._prefetch_ids(host_ids, ready, is_decode=is_decode)

    def _prefetch_done(self, key: int, future: Future[None]) -> None:
        # Do not retain row tensors/host staging after coordinator completion.
        with self._pending_lock:
            pending = self._pending.get(key)
            # Future may finish before submit() caller records it in _pending.  In
            # that race callback leaves no record; post-submit cleanup handles it.
            if pending is None or pending[3] is not future:
                return
            self._pending.pop(key, None)
        self._prefetch_slots.release()
        if future.cancelled() or future.exception() is not None:
            self._inc(prefetch_failures=1)

    def prefetch(self, row_ids: torch.Tensor, *, is_decode: bool = False) -> None:
        if not self._prefetch_enabled or row_ids.numel() == 0:
            return
        key = id(row_ids)
        with self._pending_lock:
            if key in self._pending:
                self._inc(prefetch_joins=1)
                return
        if not self._prefetch_slots.acquire(blocking=False):
            self._inc(prefetch_queue_saturation=1)
            return
        self._inc(prefetch_requests=1)
        ready: torch.cuda.Event | None = None
        try:
            if row_ids.device.type != "cuda":
                host_ids = row_ids.detach().reshape(-1).contiguous()
                pending = self._prefetch_executor.submit(self._run_prefetch, host_ids, None, is_decode)
            elif torch.version.hip is not None:
                # ROCm 7.2/gfx1100 can leave a cross-thread Event wait unsignaled.
                started = time.perf_counter_ns()
                host_ids = row_ids.detach().reshape(-1).to("cpu")
                self._inc(prefetch_d2h_us=(time.perf_counter_ns() - started) // 1000)
                self._inc(d2h_id_bytes=row_ids.numel() * row_ids.element_size())
                pending = self._prefetch_executor.submit(self._run_prefetch, host_ids, None, is_decode)
            else:
                host_ids = torch.empty(row_ids.numel(), dtype=row_ids.dtype, device="cpu", pin_memory=True)
                host_ids.copy_(row_ids.detach().reshape(-1), non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(row_ids.device))
                self._inc(d2h_id_bytes=row_ids.numel() * row_ids.element_size())
                pending = self._prefetch_executor.submit(self._run_prefetch, host_ids, ready, is_decode)
            with self._pending_lock:
                # Keep row_ids alive until its asynchronous host copy/stream event
                # is consumed.  Completion callback removes this bounded record.
                self._pending[key] = (row_ids, host_ids, ready, pending)
            pending.add_done_callback(lambda done, k=key: self._prefetch_done(k, done))
            # A very small host request can complete before _pending is installed;
            # callback then intentionally did nothing.  Re-check after registration
            # without double-releasing semaphore when callback won the race.
            if pending.done():
                self._prefetch_done(key, pending)
        except BaseException:
            self._prefetch_slots.release()
            raise

    def _packed_cpu(self, ids: list[int]) -> torch.Tensor:
        started = time.perf_counter_ns()
        plan = PagePlan.build(ids)
        unique = torch.empty((len(plan.unique_rows), ROW_BYTES), dtype=torch.uint8)
        missing_positions: dict[int, list[int]] = {}
        for index, row in enumerate(plan.unique_rows):
            cached = self._rows.get(row)
            if cached is None:
                missing_positions.setdefault(row // ROWS_PER_PAGE, []).append(index)
            else:
                unique[index].copy_(cached)
        for page_no, rows in plan.pages:
            positions = missing_positions.get(page_no, [])
            if not positions:
                continue
            page = self._page(page_no)
            try:
                # Page payload is 45 tightly packed rows followed by 46 bytes of
                # alignment padding.  View only payload and copy all requested
                # rows for this page in one vectorized operation; caller order is
                # restored below through ``plan.inverse``.
                positions_t = torch.tensor(positions, dtype=torch.long)
                row_offsets = torch.tensor(
                    [plan.unique_rows[index] % ROWS_PER_PAGE for index in positions],
                    dtype=torch.long,
                )
                payload = page[:ROWS_PER_PAGE * ROW_BYTES].view(ROWS_PER_PAGE, ROW_BYTES)
                values = payload.index_select(0, row_offsets)
                unique.index_copy_(0, positions_t, values)
                for index, value in zip(positions, values):
                    self._rows.put(plan.unique_rows[index], value)
            finally:
                self._release_page(page_no)
        packed = unique.index_select(0, plan.inverse) if ids else unique
        self._inc(requested_packed_bytes=len(ids) * ROW_BYTES)
        self._inc(planned_rows=len(ids), planned_unique_rows=len(plan.unique_rows), planned_pages=len(plan.pages))
        self._inc(ram_gather_us=(time.perf_counter_ns() - started) // 1000)
        return packed

    def _stage(self, packed: torch.Tensor) -> torch.Tensor:
        if self._staging is None or packed.shape[0] > self._staging.shape[0]:
            raise MemoryError("PLE lookup exceeds fixed --ple-staging-mib ring")
        with self._lookup_lock:
            if self._staging_cursor + packed.shape[0] > self._staging.shape[0]:
                if self._staging_event is not None:
                    self._staging_event.synchronize()
                self._staging_cursor = 0
            end = self._staging_cursor + packed.shape[0]
            slot = self._staging[self._staging_cursor:end]
            self._staging_cursor = end
            slot.zero_()
            slot[:, :ROW_BYTES].copy_(packed)
            return slot

    def _record_stage_event(self, device: torch.device) -> None:
        if device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            self._staging_event = event

    def _lookup_impl(self, row_ids: torch.Tensor, out=None):
        self._emit_status("ple_lookup")
        self._inc(lookup_calls=1, lookup_rows=row_ids.numel())
        with self._pending_lock:
            # Keep coordinator record until its completion callback releases
            # ``_prefetch_slots``.  Popping here leaked one slot per lookup and
            # silently disabled prefetch after queue capacity was exhausted.
            pending = self._pending.get(id(row_ids))
        wait_started = time.perf_counter_ns()
        same_pending = pending is not None and pending[0] is row_ids
        if same_pending:
            if not pending[3].done():
                self._inc(prefetch_late=1)
            # Host ID/event readiness is separate from I/O completion.  Joining the
            # coordinator future here serialized lookup behind unrelated pages and
            # recreated the old one-prefetch bottleneck.  Demand ``_page`` calls
            # below join only pages needed by this lookup.
            if pending[2] is not None:
                pending[2].synchronize()
            ids = [int(x) for x in pending[1].reshape(-1).tolist()]
        else:
            ids = self._ids(row_ids)
        if same_pending:
            self._inc(lookup_wait_us=(time.perf_counter_ns() - wait_started) // 1000)
        if any(row < 0 or row >= self.num_rows for row in ids):
            raise IndexError("PLE row outside table")
        if row_ids.device.type != "cuda" or self._gpu_slots is None or not self._batched_cache:
            if row_ids.device.type != "cuda" and self._gpu_slots is None:
                return self._dequant(self._packed_cpu(ids), row_ids, out)
            # GPU zero-cache path: dequantize bounded chunks, never allocate all rows
            # in the transfer ring at once.
            chunks = []
            step = max(1, self._staging.shape[0] if self._staging is not None else 1024)
            flat = row_ids.reshape(-1)
            for start in range(0, len(ids), step):
                end = min(len(ids), start + step)
                chunk_ids = flat[start:end].reshape(1, -1)
                chunks.append(self._dequant(self._packed_cpu(ids[start:end]), chunk_ids))
            # Chunks split rows, not embedding channels; concatenate row axis before restoring
            # the caller's ``row_ids.shape[:-1]`` prefix.
            value = torch.cat(chunks, dim=1).view(*row_ids.shape[:-1], -1)
            if out is not None:
                out.copy_(value)
                return out
            return value
        if row_ids.device != self._device:
            raise ValueError(f"PLE table bound to {self._device}, got {row_ids.device}")

        # Process request in fixed-size batches so packed-row staging stays bounded even
        # when prefill contains millions of hashed rows.
        step = max(1, self._staging.shape[0] if self._staging is not None else 1024)
        flat = row_ids.reshape(-1)
        values = []
        for start in range(0, len(ids), step):
            end = min(len(ids), start + step)
            chunk_ids = ids[start:end]
            missing: dict[int, list[int]] = {}
            row_slots: list[int | None] = [None] * len(chunk_ids)
            for index, row in enumerate(chunk_ids):
                slot = self._gpu_rows.get(row)
                if slot is None:
                    missing.setdefault(row, []).append(index)
                else:
                    self._gpu_rows.move_to_end(row)
                    row_slots[index] = slot
                if slot is not None:
                    self._inc(gpu_hits=1)
            self._inc(gpu_misses=len(missing))
            if missing:
                missing_rows = list(missing)
                packed_cpu = self._packed_cpu(missing_rows)
                if self._staging is None:
                    started = time.perf_counter_ns()
                    staged = torch.zeros((len(missing_rows), 96), dtype=torch.uint8)
                    staged[:, :ROW_BYTES].copy_(packed_cpu)
                    copied = staged.to(row_ids.device, non_blocking=True)
                    self._inc(h2d_submit_us=(time.perf_counter_ns() - started) // 1000)
                else:
                    started = time.perf_counter_ns()
                    copied = self._stage(packed_cpu).to(row_ids.device, non_blocking=True)
                    self._inc(h2d_submit_us=(time.perf_counter_ns() - started) // 1000)
                    self._inc(h2d_row_bytes=len(missing_rows) * ROW_BYTES)
                    self._record_stage_event(row_ids.device)
            else:
                missing_rows, copied = [], None
            inserted_slots: list[int] = []
            for local, row in enumerate(missing_rows):
                if self._free_slots:
                    slot = self._free_slots.pop()
                else:
                    _, slot = self._gpu_rows.popitem(last=False)
                    self._inc(gpu_evictions=1)
                self._gpu_rows[row] = slot
                inserted_slots.append(slot)
                for index in missing[row]:
                    row_slots[index] = slot
            if inserted_slots:
                assert copied is not None
                slot_tensor = torch.tensor(inserted_slots, dtype=torch.long, device=row_ids.device)
                self._gpu_slots.index_copy_(0, slot_tensor, copied)
                self._inc(gpu_batched_rows=len(inserted_slots), gpu_batch_ops=1)
            assert all(slot is not None for slot in row_slots)
            request_slots = torch.tensor(row_slots, dtype=torch.long, device=row_ids.device)
            packed = self._gpu_slots.index_select(0, request_slots)
            self._inc(gpu_reconstruct_ops=1)
            values.append(self._dequant(packed, flat[start:end].reshape(1, -1)))
        value = torch.cat(values, dim=1).view(*row_ids.shape[:-1], -1)
        if out is not None:
            out.copy_(value)
            return out
        return value

    def lookup(self, row_ids: torch.Tensor, out=None):
        with self._lookup_lock:
            return self._lookup_impl(row_ids, out)

    def report(self) -> dict:
        value = super().report()
        with self._cache_lock:
            value["cached_pages"] = len(self._pages)
            value["inflight_pages"] = len(self._inflight)
            value["page_refs"] = sum(self._page_refs.values())
        with self._pending_lock:
            value["pending_prefetches"] = len(self._pending)
        row_report = self._rows.report()
        value.update({
            "row_cache": row_report,
            "row_cache_hits": row_report["hits"],
            "row_cache_misses": row_report["misses"],
            "row_cache_evictions": row_report["evictions"],
            "row_cache_resident_bytes": row_report["resident_bytes"],
        })
        requested = value.get("requested_packed_bytes", 0)
        value["read_amplification"] = (value.get("ssd_read_bytes", 0) / requested
                                        if requested else 0.0)
        value["io_backend"] = value.get("io")
        value["gpu_cache_hits"] = value.get("gpu_hits", 0)
        value["gpu_cache_misses"] = value.get("gpu_misses", 0)
        value["gpu_cache_evictions"] = value.get("gpu_evictions", 0)
        prefetched = value.get("prefetch_pages_submitted", 0)
        value["useful_prefetch_ratio"] = (
            value.get("prefetch_demand_hits", 0) / prefetched if prefetched else 0.0
        )
        value["ssd_bytes_per_lookup_row"] = (
            value.get("ssd_read_bytes", 0) / max(1, value.get("lookup_rows", 0))
        )
        return value

    def close(self) -> None:
        with self._pending_lock:
            self._pending.clear()
        if getattr(self, "_closed", False):
            return
        self._closed = True
        executor = getattr(self, "_executor", None)
        prefetch_executor = getattr(self, "_prefetch_executor", None)
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True)
            self._prefetch_executor = None
        if executor is not None:
            executor.shutdown(wait=True)
            self._executor = None
        direct_fd = getattr(self, "_direct_fd", None)
        if direct_fd is not None:
            os.close(direct_fd)
            self._direct_fd = None
        buffered_fd = getattr(self, "_buffered_fd", None)
        if buffered_fd is not None:
            os.close(buffered_fd)
            self._buffered_fd = None


def estimate_gguf_ple_host_bytes(model_path: str, args) -> int:
    """Return maximum application-owned PLE host bytes for startup budgeting.

    Paged/direct modes reserve only configured RAM-cache plus pinned staging;
    resident mode reserves the complete packed IQ4_NL payload. No payload is
    read or allocated by this estimator.
    """
    _shards, header = _source(model_path)
    mode = getattr(args, "ple_mode", "auto")
    if mode == "auto":
        mode = "paged"
    if mode == "resident":
        store_bytes = int(header.nbytes)
    elif mode in ("paged", "direct-gguf"):
        configured = getattr(args, "ple_ram_cache_mib", 512)
        if isinstance(configured, str):
            configured = 512 if configured in ("auto", "adaptive") else int(configured)
        store_bytes = max(0, int(configured)) * (1 << 20)
    else:
        raise ValueError(f"unsupported GGUF PLE mode {mode!r}")
    staging = max(0, int(getattr(args, "ple_staging_mib", 32))) * (1 << 20)
    row = getattr(args, "ple_row_cache_mib", 0)
    if isinstance(row, str):
        row = 0 if row in ("auto", "adaptive") else int(row)
    return store_bytes + max(0, int(row)) * (1 << 20) + staging


def _available_ram_bytes() -> int | None:
    from freetoken.engine.host_tier import cgroup_memory_limit_bytes

    try:
        available = int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        limit = cgroup_memory_limit_bytes()
        return min(available, limit) if limit is not None else available
    except (OSError, ValueError, AttributeError):
        return None


class PackedResidentPLETable(PackedPagedPLETable):
    def __init__(self, model_path: str, args) -> None:
        super().__init__(model_path, args)
        needed = ((self.num_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE) * PAGE_BYTES
        configured = getattr(args, "ple_ram_cache_mib", 512)
        if isinstance(configured, str):
            if configured in ("auto", "adaptive"):
                # Resident mode is explicit: derive minimum cap from sidecar
                # geometry, then enforce available-RAM headroom below.
                configured = (needed + (1 << 20) - 1) // (1 << 20)
            else:
                configured = int(configured)
        configured = int(configured) * (1 << 20)
        available = _available_ram_bytes()
        # Keep production headroom proportional to a large resident table, but do not
        # reject a valid tiny/test sidecar solely because this process has <1GiB free.
        reserve = max(64 << 20, needed // 20)
        if configured < needed:
            self.close()
            raise MemoryError(f"resident PLE needs {needed} bytes of application RAM")
        if available is not None and available < needed + self._staging_bytes + reserve:
            self.close()
            raise MemoryError(
                f"resident PLE needs {needed} bytes plus headroom; only {available} bytes available"
            )
        try:
            self._resident = torch.empty((needed,), dtype=torch.uint8)
            self.capacity = needed // PAGE_BYTES
            for page in range(self.capacity):
                value = self._read_page(page)
                self._resident[page * PAGE_BYTES:(page + 1) * PAGE_BYTES].copy_(value)
        except BaseException:
            self.close()
            raise
        super().close()
        self._closed = False
        self._report.update(
            backend="ftple-resident", mode="resident",
            source_policy="resident_application_ram",
            sidecar_fingerprint=self.header.get("source_fingerprint"),
            ram_budget_bytes=needed,
        )
        self._report.update(pageable_ram_bytes=needed, pinned_staging_bytes=self.pinned_host_bytes)
        self.pageable_host_bytes = needed
        self.host_bytes = needed + self._staging_bytes

    def _page(self, page: int) -> torch.Tensor:
        with self._cache_lock:
            self._inc(ram_page_hits=1)
            self._page_refs[page] = self._page_refs.get(page, 0) + 1
        return self._resident[page * PAGE_BYTES:(page + 1) * PAGE_BYTES]

    def close(self) -> None:
        super().close()
        if hasattr(self, "_resident"):
            self._resident = torch.empty((0,), dtype=torch.uint8)


class DirectGGUFPLETable(_BasePLETable):
    """Experimental compatibility path. Production serving uses ``paged``."""

    def __init__(self, model_path: str, args) -> None:
        from .packed import Qwen4ExpPackedSource

        _source(model_path)  # shared IQ4_NL/90-byte/160-wide validation
        _metadata, _shards, headers = load_gguf_headers(model_path)
        header = next(x for x in headers if x.name == "per_layer_token_embd.weight")
        self.num_rows = int(header.rows)
        self.source = Qwen4ExpPackedSource(
            model_path, cache_bytes=int(getattr(args, "ple_ram_cache_mib", 512)) * (1 << 20)
        )
        self._stats_lock = threading.Lock()
        self._report = {
            "backend": "direct-gguf", "mode": "direct-gguf", "experimental": True,
            "source_policy": "direct_gguf_file_backed",
            "num_rows": self.num_rows, "forward_calls": 0, "lookup_calls": 0,
            "lookup_rows": 0, "requested_packed_bytes": 0,
            "dequant_calls": 0, "dequant_errors": 0,
        }

    def mark_forward(self) -> None:
        with self._stats_lock:
            self._report["forward_calls"] += 1

    def prefetch(self, row_ids, *, is_decode: bool = False):
        return None

    def lookup(self, row_ids, out=None):
        ids = self._ids(row_ids)
        if any(row < 0 or row >= self.num_rows for row in ids):
            raise IndexError("PLE row outside table")
        with self._stats_lock:
            self._report["lookup_calls"] += 1
            self._report["lookup_rows"] += len(ids)
            self._report["requested_packed_bytes"] += len(ids) * ROW_BYTES
        return self._dequant(self.source.read_rows("per_layer_token_embd.weight", ids), row_ids, out)

    def close(self):
        source = getattr(self, "source", None)
        if source is not None:
            source.close()
            self.source = None


GGUFPLETable = PackedPagedPLETable
__all__ = [
    "PackedPagedPLETable", "PackedResidentPLETable", "DirectGGUFPLETable", "GGUFPLETable",
    "estimate_gguf_ple_host_bytes",
]
