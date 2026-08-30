import struct

import pytest

from freetoken.models.gguf.reader import (
    gguf_split_paths,
    gguf_tensor_names,
    load_gguf_headers,
)


def _str(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _write_shard(path, index: int, count: int, tensors: list[tuple[str, tuple[int, ...], int, bytes]]) -> None:
    kv = [
        (_str("general.architecture"), struct.pack("<i", 8) + _str("qwen4exp")),
        (_str("split.no"), struct.pack("<iI", 4, index)),
        (_str("split.count"), struct.pack("<iI", 4, count)),
    ]
    tensor_infos = []
    payload = bytearray()
    for name, dims, ggml_type, data in tensors:
        offset = len(payload)
        tensor_infos.append(_str(name) + struct.pack("<I", len(dims)))
        tensor_infos[-1] += b"".join(struct.pack("<Q", dim) for dim in dims)
        tensor_infos[-1] += struct.pack("<iQ", ggml_type, offset)
        payload += data
    header = bytearray(b"GGUF")
    header += struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, value in kv:
        header += key + value
    header += b"".join(tensor_infos)
    header += b"\x00" * ((32 - len(header) % 32) % 32)
    path.write_bytes(header + payload)


def test_metadata_only_first_shard_indexes_complete_family(tmp_path):
    prefix = "qwen3.8-UD-Q2_K_XL"
    paths = [tmp_path / f"{prefix}-{i:05d}-of-00003.gguf" for i in range(1, 4)]
    _write_shard(paths[0], 0, 3, [])
    _write_shard(paths[1], 1, 3, [("token_embd.weight", (32, 8), 8, bytes(272))])
    _write_shard(paths[2], 2, 3, [("output.weight", (32, 8), 8, bytes(272))])

    metadata, shards, headers = load_gguf_headers(str(paths[0]))
    assert metadata["general.architecture"] == "qwen4exp"
    assert [(shard.index, shard.tensor_count) for shard in shards] == [(0, 0), (1, 1), (2, 1)]
    assert len(headers) == 2
    assert gguf_tensor_names(str(paths[0])) == {"token_embd.weight", "output.weight"}
    assert headers[0].shard_index == 1
    assert headers[1].shard_index == 2


def test_split_entrypoint_and_sibling_failures(tmp_path):
    prefix = "qwen3.8-UD-Q2_K_XL"
    paths = [tmp_path / f"{prefix}-{i:05d}-of-00003.gguf" for i in range(1, 4)]
    for index, path in enumerate(paths):
        _write_shard(path, index, 3, [])

    with pytest.raises(ValueError, match="first shard"):
        gguf_split_paths(str(paths[1]))
    paths[2].unlink()
    with pytest.raises(FileNotFoundError, match="GGUF split missing shard"):
        load_gguf_headers(str(paths[0]))
