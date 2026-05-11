"""Phase 6 baseline registry for the Sec. V-A 2D cleaning comparison."""

from gpt_repro.baselines.base import BaseTransportBaseline
from gpt_repro.baselines.ensemble_nf import EnsembleNFBaseline
from gpt_repro.baselines.ensemble_nn import EnsembleNNBaseline
from gpt_repro.baselines.ensemble_rf import EnsembleRFBaseline
from gpt_repro.baselines.gp_baseline import GPTransportBaseline
from gpt_repro.baselines.kmp import KMPBaseline
from gpt_repro.baselines.laplacian_editing import LaplacianEditingBaseline

BASELINES = {
    "kmp": KMPBaseline,
    "le":  LaplacianEditingBaseline,
    "erf": EnsembleRFBaseline,
    "enn": EnsembleNNBaseline,
    "enf": EnsembleNFBaseline,
    "gp":  GPTransportBaseline,
}

BASELINE_NAMES = {
    "kmp": "KMP",
    "le":  "LE",
    "erf": "E-RF",
    "enn": "E-NN",
    "enf": "E-NF",
    "gp":  "GP",
}

__all__ = [
    "BaseTransportBaseline",
    "BASELINES",
    "BASELINE_NAMES",
    "KMPBaseline",
    "LaplacianEditingBaseline",
    "EnsembleRFBaseline",
    "EnsembleNNBaseline",
    "EnsembleNFBaseline",
    "GPTransportBaseline",
]
