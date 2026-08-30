"""FreeToken engine adapter for native Qwen4-Exp ROCm execution."""
from __future__ import annotations

import torch

from .state import Qwen4ExpState


class Qwen4ExpRocmEngine:
    """Single-request native GGUF engine."""

    def __init__(self, config):
        if torch.version.hip is None:
            raise RuntimeError("native Qwen4-Exp route requires ROCm/HIP")
        self.config = config
        self.state = Qwen4ExpState()
        self.state.reserve(min(int(config.max_seq_len), 32768))
        self.source = None
        self.model = None
        if hasattr(config, "model_path"):
            from .native_model import Qwen4ExpNativeModel
            from .packed import Qwen4ExpPackedSource
            self.source = Qwen4ExpPackedSource(config.model_path)
            self.model = Qwen4ExpNativeModel(self.source, config, torch.device("cuda"))
        self._closed = False

    def forward_batch(self, batch, sampling_args):
        if self.model is None: raise RuntimeError("native Qwen4-Exp model is not loaded")
        from .state import assert_batch_one
        assert_batch_one(batch.input_ids[:batch.reqs[0].extend_len].reshape(1, -1))
        req = batch.reqs[0]
        if self.state.live and self.state.request_id != req.uid:
            self.reset_request()
        if req.cached_len and not self.state.live:
            raise RuntimeError("native Qwen4-Exp cannot resume a cached prefix without state")
        if not self.state.live: self.state.begin(req.uid)
        count = int(req.extend_len)
        ids = batch.input_ids[:count].reshape(1, count)
        logits = self.model.forward(ids, req.cached_len)
        self.state.advance(count)
        return logits

    def reset_request(self) -> None:
        self.state.reset()
        if self.model is not None: self.model.reset()

    def memory_report(self) -> dict:
        report = {"native_qwen4exp": True, "max_seq_len": self.state.max_seq_len,
                  "position": self.state.position, "closed": self._closed}
        if self.source is not None: report["packed_source"] = self.source.report()
        if self.model is not None:
            cache = self.model.expert_cache
            report["expert_cache"] = {"resident_bytes": cache.bytes,
                                       "capacity_bytes": cache.limit_bytes,
                                       "entries": len(cache.items)}
        return report

    def shutdown(self) -> None:
        self.reset_request()
        if self.source is not None: self.source.close()
        self._closed = True


__all__ = ["Qwen4ExpRocmEngine"]
