"""Training loop, optimizer / scheduler helpers, and loss utilities.

``ExponentialMovingAverage`` is ported in Phase 1 Step 4 because the public
``CliffordSTFWrapper`` consumes it. The remaining helpers (optimizer,
scheduler, loss, two-phase weighting) land in Step 5.
"""

from __future__ import annotations

from cliffordstf.training.ema import ExponentialMovingAverage

__all__ = ["ExponentialMovingAverage"]
