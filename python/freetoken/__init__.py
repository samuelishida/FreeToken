"""FreeToken inference runtime."""

import importlib

from freetoken.version import __version__

__all__ = ["__version__"]

# Keep package-qualified monkeypatch/import paths reliable when optional backend probes
# temporarily replace ``sys.modules['freetoken']``.  Python does not restore parent-package
# attributes when a submodule remains cached across that replacement; lazy lookup repairs the
# attribute without eagerly importing heavy CUDA/ROCm modules during normal startup.
_LAZY_SUBMODULES = frozenset({
    "attention", "cache_report", "checkpoint", "core", "distributed", "engine",
    "kernel", "kvcache", "layers", "llm", "message", "models", "moe", "scheduler",
    "server", "tokenizer", "utils",
})


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
