"""Task-Parameterised GMM (TP-GMM) baseline — paper ref. [2].

Fits a Gaussian mixture in each reference frame's local coordinates,
then fuses the per-frame component means and covariances via the
Product of Gaussians (PoG) at rollout time. PoG fusion:

    .. math::
        P_{\\mathrm{fused}} = \\sum_f P_f, \\quad
        \\mu_{\\mathrm{fused}} = P_{\\mathrm{fused}}^{-1} \\sum_f P_f \\mu_f

with :math:`P_f = \\Sigma_f^{-1}` and means / covariances transformed to
world coordinates of the queried frame configuration.

Rollout: piecewise-linear interpolation through the temporally-ordered
fused component means (greedy proxy for an LQR controller — the paper
references LQR, which is explicitly out of scope here).
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
from sklearn.mixture import GaussianMixture

from gpt_repro.policies.multiframe_demos import FrameConfig


def _rot(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _pog_fuse(
    mus_world: List[np.ndarray],
    covs_world: List[np.ndarray],
    ridge: float = 1e-6,
    cond_threshold: float = 1e8,
) -> tuple:
    precisions = [np.linalg.inv(c) for c in covs_world]
    P_fused = np.sum(precisions, axis=0)
    if np.linalg.cond(P_fused) > cond_threshold:
        P_fused = P_fused + ridge * np.eye(P_fused.shape[0])
        warnings.warn(
            "PoG fusion precision near-singular (cond > 1e8); added ridge."
        )
    Sigma_fused = np.linalg.inv(P_fused)
    rhs = np.sum([P @ mu for P, mu in zip(precisions, mus_world)], axis=0)
    mu_fused = Sigma_fused @ rhs
    return mu_fused, Sigma_fused


class TPGMMBaseline:
    """Per-frame GMMs fused via PoG (paper ref. [2])."""

    def __init__(self, n_components: int = 5, random_state: int = 0) -> None:
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.gmm_start: Optional[GaussianMixture] = None
        self.gmm_goal: Optional[GaussianMixture] = None
        self.order: Optional[np.ndarray] = None

    def fit(
        self,
        demos: List[dict],
        frame_configs: List[FrameConfig],
        n_components: Optional[int] = None,
    ) -> "TPGMMBaseline":
        if n_components is not None:
            self.n_components = int(n_components)
        if len(demos) != len(frame_configs):
            raise ValueError("demos and frame_configs must have the same length")

        all_local_start: List[np.ndarray] = []
        all_local_goal: List[np.ndarray] = []
        time_axes: List[np.ndarray] = []
        for demo, cfg in zip(demos, frame_configs):
            R_s = _rot(cfg.start_angle)
            R_g = _rot(cfg.goal_angle)
            all_local_start.append((demo["x"] - cfg.start_pos) @ R_s)
            all_local_goal.append((demo["x"] - cfg.goal_pos) @ R_g)
            time_axes.append(np.asarray(demo["t"], dtype=float))

        Xs = np.concatenate(all_local_start, axis=0)
        Xg = np.concatenate(all_local_goal, axis=0)
        kw = dict(
            n_components=self.n_components,
            random_state=self.random_state,
            reg_covar=1e-3,
            max_iter=100,
        )
        self.gmm_start = GaussianMixture(**kw).fit(Xs)
        self.gmm_goal = GaussianMixture(**kw).fit(Xg)

        # Temporal ordering of components based on first demo.
        resp0 = self.gmm_start.predict_proba(all_local_start[0])
        t0 = time_axes[0]
        time_score = (resp0 * t0[:, None]).sum(0) / (resp0.sum(0) + 1e-12)
        self.order = np.argsort(time_score)
        return self

    def _fused_means(
        self, start_frame: FrameConfig, goal_frame: FrameConfig,
    ) -> np.ndarray:
        K = self.n_components
        R_s = _rot(start_frame.start_angle)
        R_g = _rot(goal_frame.goal_angle)
        fused = np.zeros((K, 2))
        for k in range(K):
            mu_s_w = R_s @ self.gmm_start.means_[k] + start_frame.start_pos
            mu_g_w = R_g @ self.gmm_goal.means_[k]  + goal_frame.goal_pos
            cov_s_w = R_s @ self.gmm_start.covariances_[k] @ R_s.T
            cov_g_w = R_g @ self.gmm_goal.covariances_[k]  @ R_g.T
            mu, _ = _pog_fuse([mu_s_w, mu_g_w], [cov_s_w, cov_g_w])
            fused[k] = mu
        return fused

    def rollout(
        self,
        start_frame: FrameConfig,
        goal_frame: FrameConfig,
        x0: np.ndarray,
        dt: float = 0.05,
        n_steps: int = 200,
    ) -> np.ndarray:
        if self.gmm_start is None:
            raise RuntimeError("TPGMMBaseline.fit must be called before rollout.")
        fused = self._fused_means(start_frame, goal_frame)
        ordered = fused[self.order]
        # No artificial goal-append — let the last component mean
        # be the endpoint, so final-position metrics reflect the
        # GMM's actual generalization quality.
        path = np.vstack([np.asarray(x0)[None, :], ordered])
        diffs = np.diff(path, axis=0)
        chord = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(chord)])
        total = float(cum[-1]) if cum[-1] > 1e-9 else 1.0
        s = np.linspace(0.0, total, n_steps + 1)
        traj = np.zeros((n_steps + 1, 2))
        for d in range(2):
            traj[:, d] = np.interp(s, cum, path[:, d])
        return traj
