"""Gaussian HMM baseline (Sec. V-B) — paper ref. [36].

Like :class:`TPGMMBaseline` but the per-frame model is a Gaussian HMM
fit with `hmmlearn.hmm.GaussianHMM`. Emission means + covariances are
fused across frames via the same PoG construction as TP-GMM.

Rollout: piecewise-linear path through the fused emission means in
their temporal order. The brief mentions a Viterbi initial guess; we
extract the temporal ordering of states by averaging the demo time
indices of each Viterbi-decoded state on the first demo. LQR rollout is
explicitly out of scope.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from hmmlearn.hmm import GaussianHMM

from gpt_repro.baselines.tpgmm import _pog_fuse, _rot
from gpt_repro.policies.multiframe_demos import FrameConfig


class HMMBaseline:
    """Per-frame Gaussian HMMs + PoG fusion."""

    def __init__(self, n_states: int = 5, random_state: int = 0,
                 n_iter: int = 50) -> None:
        self.n_states = int(n_states)
        self.random_state = int(random_state)
        self.n_iter = int(n_iter)
        self.hmm_start: Optional[GaussianHMM] = None
        self.hmm_goal:  Optional[GaussianHMM] = None
        self.order: Optional[np.ndarray] = None

    def fit(
        self,
        demos: List[dict],
        frame_configs: List[FrameConfig],
        n_states: Optional[int] = None,
    ) -> "HMMBaseline":
        if n_states is not None:
            self.n_states = int(n_states)
        if len(demos) != len(frame_configs):
            raise ValueError("demos and frame_configs must have the same length")

        all_local_start: List[np.ndarray] = []
        all_local_goal: List[np.ndarray] = []
        lengths: List[int] = []
        time_axes: List[np.ndarray] = []
        for demo, cfg in zip(demos, frame_configs):
            R_s = _rot(cfg.start_angle)
            R_g = _rot(cfg.goal_angle)
            all_local_start.append((demo["x"] - cfg.start_pos) @ R_s)
            all_local_goal.append((demo["x"] - cfg.goal_pos) @ R_g)
            lengths.append(int(demo["x"].shape[0]))
            time_axes.append(np.asarray(demo["t"], dtype=float))

        Xs = np.concatenate(all_local_start, axis=0)
        Xg = np.concatenate(all_local_goal, axis=0)
        common = dict(
            n_components=self.n_states,
            covariance_type="full",
            random_state=self.random_state,
            n_iter=self.n_iter,
            tol=1e-3,
            init_params="stmc",
        )
        self.hmm_start = GaussianHMM(**common).fit(Xs, lengths)
        self.hmm_goal  = GaussianHMM(**common).fit(Xg, lengths)

        # Temporal ordering of states via the first demo's Viterbi path.
        seq = self.hmm_start.predict(all_local_start[0])
        t0 = time_axes[0]
        time_score = np.full(self.n_states, np.inf)
        for k in range(self.n_states):
            mask = seq == k
            if mask.any():
                time_score[k] = float(t0[mask].mean())
        self.order = np.argsort(time_score)
        return self

    def _fused_means(
        self, start_frame: FrameConfig, goal_frame: FrameConfig,
    ) -> np.ndarray:
        K = self.n_states
        R_s = _rot(start_frame.start_angle)
        R_g = _rot(goal_frame.goal_angle)
        fused = np.zeros((K, 2))
        for k in range(K):
            mu_s_w = R_s @ self.hmm_start.means_[k] + start_frame.start_pos
            mu_g_w = R_g @ self.hmm_goal.means_[k]  + goal_frame.goal_pos
            cov_s_w = R_s @ self.hmm_start.covars_[k] @ R_s.T
            cov_g_w = R_g @ self.hmm_goal.covars_[k]  @ R_g.T
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
        if self.hmm_start is None:
            raise RuntimeError("HMMBaseline.fit must be called before rollout.")
        fused = self._fused_means(start_frame, goal_frame)
        ordered = fused[self.order]
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
