"""
Hybrid VIO model.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0
"""

from __future__ import annotations

from typing import Any, Type


from fno_evio.legacy import train_fno_vio as _LEGACY

HybridVIONet: Type[Any] = getattr(_LEGACY, "HybridVIONet")

__all__ = ["HybridVIONet"]
