"""Policy Transportation — Sec. IV of Franzese et al. (2024).

Phase 3 added the linear (rigid) component γ via
:class:`LinearTransport`. Phase 4 adds the non-linear residual ψ
(:class:`GPNonlinearResidual`) and the full transportation map ϕ = γ + ψ ∘ γ
(:class:`PolicyTransport`), along with velocity / orientation /
stiffness / damping transport.
"""

from gpt_repro.transport.linear import LinearTransport, kabsch_svd_rotation
from gpt_repro.transport.nonlinear_gp import GPNonlinearResidual
from gpt_repro.transport.policy_transport import (
    PolicyTransport,
    _nearest_proper_rotation,
)
from gpt_repro.transport.uncertainty import (
    total_velocity_variance,
    transportation_velocity_variance,
)

__all__ = [
    "LinearTransport",
    "kabsch_svd_rotation",
    "GPNonlinearResidual",
    "PolicyTransport",
    "_nearest_proper_rotation",
    "transportation_velocity_variance",
    "total_velocity_variance",
]
