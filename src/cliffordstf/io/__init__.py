"""I/O boundaries: config loading, manifest parsing, on-disk artifact writers.

Per ``CODING_RULES.md`` §A "Recommended package structure", boundary code
that converts untyped formats (YAML, JSON, CLI argv) into typed objects
lives here.
"""

from __future__ import annotations

__all__: list[str] = []
