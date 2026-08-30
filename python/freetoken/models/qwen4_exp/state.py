"""Explicit single-request state for native Qwen4-Exp execution."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Qwen4ExpState:
    max_seq_len: int = 0
    request_id: int | None = None
    position: int = 0
    live: bool = False

    def reserve(self, max_seq_len: int) -> None:
        if max_seq_len <= 0:
            raise ValueError("Qwen4-Exp max_seq_len must be positive")
        self.max_seq_len = int(max_seq_len)
        self.reset()

    def begin(self, request_id: int = 0) -> None:
        if self.max_seq_len <= 0:
            raise RuntimeError("Qwen4-Exp state is not reserved")
        if self.live:
            raise RuntimeError("Qwen4-Exp state already owns a live request")
        self.request_id = int(request_id)
        self.position = 0
        self.live = True

    def advance(self, count: int) -> None:
        self.assert_live()
        if count < 0 or self.position + count > self.max_seq_len:
            raise ValueError(f"Qwen4-Exp context overflow: {self.position}+{count}>{self.max_seq_len}")
        self.position += int(count)

    def assert_live(self) -> None:
        if not self.live:
            raise RuntimeError("Qwen4-Exp request state is not live")

    def reset(self) -> None:
        self.request_id = None
        self.position = 0
        self.live = False


def assert_batch_one(input_ids: torch.Tensor) -> None:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Qwen4-Exp native route requires input_ids [1,T], got {tuple(input_ids.shape)}")


__all__ = ["Qwen4ExpState", "assert_batch_one"]
