"""Gaussian Process regression module — Sec. III-B of Franzese et al. (2024).

Exposes the exact GP regressor (Eqs. 2, 3, 16) and the SVGP approximation.
"""

from gpt_repro.gp.exact_gp import ExactGPRegressor
from gpt_repro.gp.svgp import SVGPRegressor

__all__ = ["ExactGPRegressor", "SVGPRegressor"]
