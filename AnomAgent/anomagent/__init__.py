from typing import TYPE_CHECKING

__all__ = ["AIImageAnalyzer"]

if TYPE_CHECKING:
    from .analyzer import AIImageAnalyzer


def __getattr__(name):
    if name == "AIImageAnalyzer":
        from .analyzer import AIImageAnalyzer as _AIImageAnalyzer

        return _AIImageAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
