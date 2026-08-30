from types import SimpleNamespace

import torch

from freetoken.models.qwen4_exp.native_gdn import Qwen4ExpGDN
from freetoken.models.qwen4_exp.native_ops import HyperConnection
from freetoken.models.qwen4_exp.qsa import Qwen4ExpQSA
from freetoken.models.qwen4_exp.native_ple import Qwen4ExpPLE
def reference_ple_row_ids(tokens):
    primes = (20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
              20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171)
    offsets = (0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
               160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275)
    mult = (23703573157769, 20109073645365, 8052911324071); eos = 248044
    history = []; out = []
    for token in tokens:
        prev = history[-1] if history else eos
        prev2 = history[-2] if len(history) > 1 and history[-1] != eos else eos
        out.append([((token * mult[0]) ^ (prev * mult[1]) ^ (prev2 * mult[2] if h >= 8 else 0)) % primes[h] + offsets[h] for h in range(16)])
        history.append(int(token))
    return torch.tensor(out, dtype=torch.int64).numpy()


class FakeSource:
    metadata = {}

    def __init__(self): self.values = {}

    def add(self, name, shape):
        self.values[name] = torch.randn(shape)

    def locate(self, name):
        value = self.values[name]
        return SimpleNamespace(name=name, ggml_type=0, shape=tuple(value.shape), rows=value.numel(), row_bytes=value.numel() * 4)

    def read_tensor(self, name, device="cpu"):
        return self.values[name].to(device)

    def read_rows(self, name, rows, device="cpu"):
        return self.values[name][list(rows)].to(device)


def gdn_source():
    s = FakeSource(); p = "blk.0."
    for n, shape in (("attn_qkv.weight", (10240, 2560)), ("attn_gate.weight", (6144, 2560)),
                     ("ssm_alpha.weight", (48, 2560)), ("ssm_beta.weight", (48, 2560)),
                     ("ssm_out.weight", (2560, 6144)), ("ssm_conv1d.weight", (10240, 4)),
                     ("ssm_dt.bias", (48,)), ("ssm_a", (48,)), ("ssm_norm.weight", (128,))):
        s.add(p + n, shape)
    return s


def test_gdn_shape_and_reset_cpu():
    torch.manual_seed(1); op = Qwen4ExpGDN(gdn_source(), 0, torch.device("cpu"))
    x = torch.randn(1, 2, 2560)
    y = op(x); assert y.shape == x.shape and torch.isfinite(y).all()
    op.reset(); z = op(x)
    torch.testing.assert_close(y, z)


def test_gdn_extend_decode_matches_single_extend():
    torch.manual_seed(2); x = torch.randn(1, 3, 2560)
    whole = Qwen4ExpGDN(gdn_source(), 0, torch.device("cpu")); expected = whole(x)
    split = Qwen4ExpGDN(gdn_source(), 0, torch.device("cpu"))
    # Share weights to make this a state-boundary check, not two random models.
    split.qkv, split.gate, split.alpha, split.beta, split.out = whole.qkv, whole.gate, whole.alpha, whole.beta, whole.out
    split.conv, split.dt, split.a, split.norm = whole.conv, whole.dt, whole.a, whole.norm
    actual = torch.cat((split(x[:, :1]), split(x[:, 1:])), dim=1)
    torch.testing.assert_close(expected, actual, rtol=2e-3, atol=1e-2)


def test_hyperconnection_four_stream_geometry():
    s = FakeSource()
    for n, shape in (("norm.weight", (10240,)), ("down.weight", (320, 10240)),
                     ("up.weight", (10240, 320)), ("inject.weight", (4, 10240))): s.add("hc_" + n, shape)
    op = HyperConnection(s, "hc_", torch.device("cpu")); streams = torch.randn(1, 3, 4, 2560)
    branch = op.read(streams); out = op.write(streams, branch)
    assert branch.shape == (1, 3, 2560) and out.shape == streams.shape
    assert torch.isfinite(out).all()


def test_qsa_keeps_causal_state_cpu():
    s = FakeSource(); p = "blk.3."
    for n, shape in (("attn_q.weight", (12288, 2560)), ("attn_k.weight", (512, 2560)),
                     ("attn_v.weight", (512, 2560)), ("attn_output.weight", (2560, 6144)),
                     ("indexer.q_proj.weight", (512, 2560)), ("indexer.k_proj.weight", (128, 2560)),
                     ("attn_q_norm.weight", (256,)), ("attn_k_norm.weight", (256,)),
                     ("indexer.q_norm.weight", (128,)), ("indexer.k_norm.weight", (128,))): s.add(p + n, shape)
    op = Qwen4ExpQSA(s, 3, torch.device("cpu")); x = torch.randn(1, 2, 2560)
    y = op(x); assert y.shape == x.shape and torch.isfinite(y).all()
    op.reset(); z = op(x); torch.testing.assert_close(y, z)


def test_ple_device_row_ids_match_eos_bounded_reference():
    op = Qwen4ExpPLE.__new__(Qwen4ExpPLE)
    op.table = SimpleNamespace(rows=320001536); op.device = torch.device("cpu"); op.history = []
    tokens = torch.tensor([[11, 22, 248044, 33, 44]], dtype=torch.long)
    got = op.row_ids(tokens).cpu().numpy()
    expected = reference_ple_row_ids(tokens.reshape(-1).numpy())
    assert (got == expected).all()
