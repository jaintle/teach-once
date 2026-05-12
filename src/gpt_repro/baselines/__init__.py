"""Phase 6 baseline registry for the Sec. V-A 2D cleaning comparison.

Phase 7 adds multi-reference-frame baselines (DMP, TP-GMM, HMM, GPT)
that share a ``fit / rollout`` interface but are not part of the
Sec. V-A residual-transport ``BASELINES`` dict.
"""

from gpt_repro.baselines.base import BaseTransportBaseline
from gpt_repro.baselines.dmp import DMPBaseline
from gpt_repro.baselines.multisource_dmp import MultiSourceDMP
from gpt_repro.baselines.multisource_gpt import MultiSourceGPT
from gpt_repro.baselines.ensemble_nf import EnsembleNFBaseline
from gpt_repro.baselines.ensemble_nn import EnsembleNNBaseline
from gpt_repro.baselines.ensemble_rf import EnsembleRFBaseline
from gpt_repro.baselines.gp_baseline import GPTransportBaseline
from gpt_repro.baselines.gpt_adapter import GPTBaseline
from gpt_repro.baselines.hmm import HMMBaseline
from gpt_repro.baselines.kmp import KMPBaseline
from gpt_repro.baselines.laplacian_editing import LaplacianEditingBaseline
from gpt_repro.baselines.tpgmm import TPGMMBaseline

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
    # Phase 7 multi-frame baselines.
    "DMPBaseline",
    "GPTBaseline",
    # Phase 8 multi-source baselines.
    "MultiSourceGPT",
    "MultiSourceDMP",
    "TPGMMBaseline",
    "HMMBaseline",
]
