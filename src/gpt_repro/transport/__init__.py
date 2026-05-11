"""Policy Transportation — Sec. IV of Franzese et al. (2024).

Phase 3 exposes only the linear (rigid) component γ via
:class:`LinearTransport`. The non-linear residual ψ (Sec. IV-B) is
added in Phase 4.
"""

from gpt_repro.transport.linear import LinearTransport, kabsch_svd_rotation

__all__ = ["LinearTransport", "kabsch_svd_rotation"]
