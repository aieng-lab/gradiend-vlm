"""Paired image and text continuation training for GRADIEND."""

from importlib import import_module

_EXPORTS = {
    "JanusGenerationTask": ".janus",
    "PreparedJanusGeneration": ".janus",
    "VisionLanguageGenerationPair": ".data",
    "load_generation_pairs": ".data",
    "load_matched_examples": ".data",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
