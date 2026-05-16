"""Optional plug-in baselines for cliffordstf.

Exposes ``AVAILABLE_MODELS`` so ``cliffordstf.cli`` can merge third-party
model factories into ``cliffordstf.models.build_model`` via its
``extras`` argument. Heavy dependencies are gated behind
``pip install cliffordstf[baselines]`` and populated incrementally
(one baseline per commit; see the project backlog).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from cliffordstf.domain import ModelFactory

AVAILABLE_MODELS: Mapping[str, ModelFactory] = MappingProxyType({})

__all__ = ["AVAILABLE_MODELS"]
