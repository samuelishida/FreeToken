"""Sampler content-guard tests.

The tinygrad server crashed with ``multinomial: probability tensor contains
either inf, nan ...`` when a long-context forward produced NaN logits. The
guard in ``sample_cpu`` clamps instead of raising so one bad forward never
kills the scheduler; these tests pin that contract (root cause in the model
forward is a separate fix — see .plans/decode-nan-captured-jit/).
"""

import logging
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from freetoken.engine.sample import BatchSamplingArgs, Sampler


VOCAB = 1024


def _args(temperature: float = 1.0, top_k: int = 20, top_p: float | None = None):
    return BatchSamplingArgs(
        temperatures=torch.tensor([temperature], dtype=torch.float32),
        top_k=torch.tensor([top_k], dtype=torch.int32),
        top_p=None if top_p is None else torch.tensor([top_p], dtype=torch.float32),
    )


def _batch():
    from freetoken.core import Batch, Req, SamplingParams

    req = Req(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        table_idx=0, cached_len=2, output_len=1, uid=7,
        sampling_params=SamplingParams(), cache_handle=None,
    )
    return Batch(reqs=[req], phase="decode")


class TestSampleCpuNanGuard(unittest.TestCase):
    def test_nan_logits_sampled_topk(self):
        s = Sampler(torch.device("cpu"), VOCAB)
        logits = torch.full((1, VOCAB), float("nan"))
        with self.assertLogs("freetoken.engine.sample", level="ERROR") as cap:
            tok = s.sample_cpu(logits, _args(), _batch())
        self.assertEqual(tok.shape[0], 1)
        self.assertTrue(0 <= tok.item() < VOCAB)
        self.assertIn("non-finite logits", cap.output[0])

    def test_partially_nan_logits_favor_finite_tail(self):
        s = Sampler(torch.device("cpu"), VOCAB)
        logits = torch.full((1, VOCAB), float("nan"))
        logits[0, 500] = 5.0
        tok = s.sample_cpu(logits, _args(top_k=1), _batch())
        self.assertEqual(int(tok.item()), 500)

    def test_nan_logits_greedy(self):
        s = Sampler(torch.device("cpu"), VOCAB)
        logits = torch.full((1, VOCAB), float("nan"))
        logits[0, 77] = 2.5
        tok = s.sample_cpu(logits, BatchSamplingArgs(temperatures=None), _batch())
        self.assertEqual(int(tok.item()), 77)

    def test_finite_logits_do_not_log(self):
        s = Sampler(torch.device("cpu"), VOCAB)
        with self.assertNoLogs("freetoken.engine.sample", level="ERROR"):
            tok = s.sample_cpu(torch.randn(1, VOCAB), _args(), _batch())
        self.assertEqual(tok.shape[0], 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    torch.manual_seed(0)