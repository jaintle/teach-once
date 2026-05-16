// gp_infer.js — Pure JavaScript TP-GPT inference
// Implements Eqs. (2), (3), (7), (11)-(13) from Franzese et al. 2024
// Coordinate system: Three.js Y-up (x = left-right, y = height, z = depth)
'use strict';

// =============================================================================
// MATRIX UTILITIES — row-major, nested Float64Array rows
// =============================================================================
const Mat = {
  /** Create m×n zero matrix */
  zeros(m, n) {
    return Array.from({ length: m }, () => new Float64Array(n));
  },

  /** Identity n×n */
  eye(n) {
    const I = Mat.zeros(n, n);
    for (let i = 0; i < n; i++) I[i][i] = 1.0;
    return I;
  },

  /** Matrix multiply A(m×k) @ B(k×n) → (m×n) */
  mul(A, B) {
    const m = A.length, k = A[0].length, n = B[0].length;
    const C = Mat.zeros(m, n);
    for (let i = 0; i < m; i++)
      for (let l = 0; l < k; l++) {
        if (Math.abs(A[i][l]) < 1e-300) continue;
        for (let j = 0; j < n; j++)
          C[i][j] += A[i][l] * B[l][j];
      }
    return C;
  },

  /** Transpose (m×n) → (n×m) */
  T(A) {
    const m = A.length, n = A[0].length;
    const B = Mat.zeros(n, m);
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        B[j][i] = A[i][j];
    return B;
  },

  /** Matrix-vector multiply (m×n) @ (n,) → (m,) */
  mulVec(A, v) {
    const m = A.length, n = A[0].length;
    const w = new Float64Array(m);
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        w[i] += A[i][j] * v[j];
    return w;
  },

  dot(u, v) {
    let s = 0;
    for (let i = 0; i < u.length; i++) s += u[i] * v[i];
    return s;
  },

  norm(v) { return Math.sqrt(v.reduce((s, x) => s + x * x, 0)); },

  add(a, b) { return Array.from(a, (v, i) => v + b[i]); },
  sub(a, b) { return Array.from(a, (v, i) => v - b[i]); },
  scale(a, s) { return Array.from(a, (v) => v * s); },

  /** A + lambda * I (copy) */
  addRidge(A, lambda) {
    const n = A.length;
    const B = A.map((row) => Float64Array.from(row));
    for (let i = 0; i < n; i++) B[i][i] += lambda;
    return B;
  },

  /** Determinant of 3×3 */
  det3(A) {
    return (
      A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
      A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
      A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    );
  },

  /** Invert n×n matrix via Gauss-Jordan with partial pivoting */
  inv(A) {
    const n = A.length;
    // Build augmented [A | I]
    const M = A.map((row, i) => {
      const r = new Float64Array(2 * n);
      for (let j = 0; j < n; j++) r[j] = row[j];
      r[n + i] = 1.0;
      return r;
    });
    for (let col = 0; col < n; col++) {
      // Partial pivot
      let maxRow = col, maxVal = Math.abs(M[col][col]);
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(M[row][col]) > maxVal) {
          maxVal = Math.abs(M[row][col]);
          maxRow = row;
        }
      }
      [M[col], M[maxRow]] = [M[maxRow], M[col]];
      const pivot = M[col][col];
      if (Math.abs(pivot) < 1e-14) continue; // singular column
      const inv_p = 1.0 / pivot;
      for (let j = col; j < 2 * n; j++) M[col][j] *= inv_p;
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const f = M[row][col];
        if (Math.abs(f) < 1e-15) continue;
        for (let j = col; j < 2 * n; j++) M[row][j] -= f * M[col][j];
      }
    }
    return M.map((row) => row.slice(n));
  },

  /**
   * Jacobi eigendecomposition of symmetric n×n matrix.
   * Returns {values: (n,), vectors: (n×n)} where vectors[:,i] = i-th eigenvector.
   * Suitable for n ≤ 10.
   */
  jacobiEigen(A) {
    const n = A.length;
    let M = A.map((row) => Array.from(row));
    let V = Mat.eye(n).map((row) => Array.from(row)); // columns = eigenvectors

    const maxIter = 200 * n * n;
    for (let iter = 0; iter < maxIter; iter++) {
      // Find largest off-diagonal
      let p = 0, q = 1, maxVal = 0;
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++)
          if (Math.abs(M[i][j]) > maxVal) { maxVal = Math.abs(M[i][j]); p = i; q = j; }
      if (maxVal < 1e-13) break;

      const tau = (M[q][q] - M[p][p]) / (2.0 * M[p][q]);
      const t = tau >= 0
        ? 1.0 / (tau + Math.sqrt(1.0 + tau * tau))
        : 1.0 / (tau - Math.sqrt(1.0 + tau * tau));
      const c = 1.0 / Math.sqrt(1.0 + t * t);
      const s = t * c;

      // Update diagonal and zero out (p,q)
      const Mpp = M[p][p], Mqq = M[q][q], Mpq = M[p][q];
      M[p][p] = Mpp - t * Mpq;
      M[q][q] = Mqq + t * Mpq;
      M[p][q] = M[q][p] = 0.0;

      // Update off-diagonal rows/cols
      for (let r = 0; r < n; r++) {
        if (r === p || r === q) continue;
        const Mrp = M[r][p], Mrq = M[r][q];
        M[r][p] = M[p][r] = c * Mrp - s * Mrq;
        M[r][q] = M[q][r] = s * Mrp + c * Mrq;
      }

      // Accumulate rotation into V
      for (let r = 0; r < n; r++) {
        const Vrp = V[r][p], Vrq = V[r][q];
        V[r][p] = c * Vrp - s * Vrq;
        V[r][q] = s * Vrp + c * Vrq;
      }
    }
    return { values: M.map((row, i) => row[i]), vectors: V };
  },

  /**
   * SVD via eigendecomposition of A^T A.
   * Returns {U: (m×k), S: (k,), V: (n×k)} where A ≈ U diag(S) V^T.
   * k = min(m, n). Suitable for small matrices.
   */
  svd(A) {
    const m = A.length, n = A[0].length;
    const AT = Mat.T(A);
    const ATA = Mat.mul(AT, A); // n×n symmetric

    const { values, vectors } = Mat.jacobiEigen(ATA);

    // Sort by descending eigenvalue
    const order = values
      .map((_, i) => i)
      .sort((a, b) => values[b] - values[a]);

    // Singular values
    const S = order.map((i) => Math.sqrt(Math.max(0.0, values[i])));

    // V: columns = right singular vectors, reordered
    const Vmat = Array.from({ length: n }, (_, r) =>
      Float64Array.from(order.map((i) => vectors[r][i]))
    );

    // U = A @ V / S (left singular vectors)
    const AV = Mat.mul(A, Vmat);
    const U = AV.map((row) =>
      Float64Array.from(row.map((v, j) => (S[j] > 1e-12 ? v / S[j] : 0.0)))
    );

    return { U, S, V: Vmat };
  },
};

// =============================================================================
// RBF KERNEL
// Squared exponential: k(xi, xj) = sigmaP² · exp(−||xi−xj||² / (2·l²))
// =============================================================================
function rbfKernel(xi, xj, l, sigmaP) {
  const d2 = xi.reduce((s, v, k) => s + (v - xj[k]) ** 2, 0);
  return sigmaP * sigmaP * Math.exp(-d2 / (2.0 * l * l));
}

/** Build N×N kernel matrix K(X, X) */
function buildKernelMatrix(X, l, sigmaP) {
  const N = X.length;
  const K = Mat.zeros(N, N);
  for (let i = 0; i < N; i++)
    for (let j = i; j < N; j++) {
      const k = rbfKernel(X[i], X[j], l, sigmaP);
      K[i][j] = K[j][i] = k;
    }
  return K;
}

/** Build M×N cross-kernel K(X_star, X) */
function buildCrossKernel(Xstar, X, l, sigmaP) {
  const M = Xstar.length, N = X.length;
  const K = Mat.zeros(M, N);
  for (let i = 0; i < M; i++)
    for (let j = 0; j < N; j++)
      K[i][j] = rbfKernel(Xstar[i], X[j], l, sigmaP);
  return K;
}

// =============================================================================
// GP REGRESSOR — Sec. III-B, Eqs. (2) and (3)
// =============================================================================
class GPRegressor {
  /**
   * @param {number} l      — RBF lengthscale
   * @param {number} sigmaP — output scale
   * @param {number} sigmaN — noise std
   */
  constructor(l = 0.3, sigmaP = 1.0, sigmaN = 0.05) {
    this.l = l;
    this.sigmaP = sigmaP;
    this.sigmaN = sigmaN;
    this.alpha = null;   // (K + sigmaN² I)^-1 y
    this.K_inv = null;   // (K + sigmaN² I)^-1
    this.X_train = null;
  }

  /**
   * Fit GP to (X, y).
   * Eq. (2) pre-computation: alpha = (K + sigmaN²I)^-1 y
   * @param {Array} X — (N, d) training inputs
   * @param {Float64Array|Array} y — (N,) training targets
   */
  fit(X, y) {
    this.X_train = X;
    const K = buildKernelMatrix(X, this.l, this.sigmaP);
    const Kreg = Mat.addRidge(K, this.sigmaN * this.sigmaN);
    this.K_inv = Mat.inv(Kreg);
    this.alpha = Mat.mulVec(this.K_inv, y);
    return this;
  }

  /**
   * Predict mean and std at test points.
   * Eq. (2): mu = K(X*,X) @ alpha
   * Eq. (3): var = k(X*,X*) - K(X*,X) @ K_inv @ K(X,X*)
   * @param {Array} Xstar — (M, d)
   * @returns {{mean: Float64Array, std: Float64Array}}
   */
  predict(Xstar) {
    const M = Xstar.length;
    const Ks = buildCrossKernel(Xstar, this.X_train, this.l, this.sigmaP);
    // Eq. (2): mean = Ks @ alpha
    const mean = Mat.mulVec(Ks, this.alpha);
    // Eq. (3): var = k_self - diag(Ks @ K_inv @ Ks.T)
    const std = new Float64Array(M);
    const KsKinv = Mat.mul(Ks, this.K_inv);
    const sp2 = this.sigmaP * this.sigmaP;
    for (let i = 0; i < M; i++) {
      const v = sp2 - Mat.dot(KsKinv[i], Ks[i]);
      std[i] = Math.sqrt(Math.max(0, v));
    }
    return { mean, std };
  }
}

// =============================================================================
// MULTI-OUTPUT GP — one GPRegressor per output dimension
// =============================================================================
class MultiOutputGP {
  /**
   * @param {number} dOut — number of output dimensions
   */
  constructor(dOut, l = 0.3, sigmaP = 1.0, sigmaN = 0.05) {
    this.dOut = dOut;
    this.gps = Array.from({ length: dOut }, () => new GPRegressor(l, sigmaP, sigmaN));
  }

  /** Y: (N, dOut) */
  fit(X, Y) {
    for (let d = 0; d < this.dOut; d++) {
      const y_d = Y.map((row) => row[d]);
      this.gps[d].fit(X, y_d);
    }
    return this;
  }

  /** Returns {mean: (M, dOut), std: (M, dOut)} */
  predict(Xstar) {
    const M = Xstar.length;
    const mean = Array.from({ length: M }, () => new Float64Array(this.dOut));
    const std = Array.from({ length: M }, () => new Float64Array(this.dOut));
    for (let d = 0; d < this.dOut; d++) {
      const { mean: mu, std: sigma } = this.gps[d].predict(Xstar);
      for (let i = 0; i < M; i++) {
        mean[i][d] = mu[i];
        std[i][d] = sigma[i];
      }
    }
    return { mean, std };
  }
}

// =============================================================================
// LINEAR TRANSPORT — Sec. IV-A, Eqs. (8)-(11) — Kabsch / Arun 1987
// =============================================================================
class LinearTransport {
  constructor() {
    this.A = null;     // (d×d) rotation matrix
    this.S_bar = null; // source centroid
    this.T_bar = null; // target centroid
    this.d = 0;
  }

  /**
   * Fit Kabsch rotation aligning S → T.
   * Eq. (9): SVD of (S - S_bar)^T (T - T_bar)
   * Eq. (10): A = V U^T with reflection fix
   * @param {Array} S — (N, d) source keypoints
   * @param {Array} T — (N, d) target keypoints
   */
  fit(S, T) {
    const N = S.length;
    this.d = S[0].length;

    // Centroids
    const S_bar = new Float64Array(this.d);
    const T_bar = new Float64Array(this.d);
    for (let i = 0; i < N; i++)
      for (let k = 0; k < this.d; k++) {
        S_bar[k] += S[i][k] / N;
        T_bar[k] += T[i][k] / N;
      }
    this.S_bar = S_bar;
    this.T_bar = T_bar;

    // Center point clouds
    const S_c = S.map((row) => row.map((v, k) => v - S_bar[k]));
    const T_c = T.map((row) => row.map((v, k) => v - T_bar[k]));

    // Eq. (9): H = S_c^T @ T_c  (d×d cross-covariance)
    const H = Mat.mul(Mat.T(S_c), T_c);

    // SVD: H = U diag(S) V^T  →  A = V U^T
    const { U, V } = Mat.svd(H);

    // Eq. (10): A = V @ U^T
    let A = Mat.mul(V, Mat.T(U));

    // Reflection fix: ensure det(A) = +1
    if (Mat.det3(A) < 0) {
      // Flip last column of V
      const Vmod = V.map((row) => Float64Array.from(row));
      for (let r = 0; r < this.d; r++) Vmod[r][this.d - 1] *= -1;
      A = Mat.mul(Vmod, Mat.T(U));
    }

    this.A = A;
    return this;
  }

  /**
   * Eq. (11): γ(x) = A(x − S_bar) + T_bar
   * @param {Array} X — (M, d)
   * @returns {Array} (M, d)
   */
  transform(X) {
    return X.map((row) => {
      const centered = row.map((v, k) => v - this.S_bar[k]);
      const rotated = Mat.mulVec(this.A, centered);
      return Array.from(rotated, (v, k) => v + this.T_bar[k]);
    });
  }

  /** Returns constant Jacobian A (d×d) */
  jacobian() { return this.A; }
}

// =============================================================================
// NONLINEAR RESIDUAL — Sec. IV-B, Eq. (12)
// GP fitted on residual: T - gamma(S)
// =============================================================================
class NonlinearResidual {
  constructor(d) {
    this.d = d;
    // Tighter lengthscale for residual GP, small noise (near-interpolation)
    this.gp = new MultiOutputGP(d, 0.25, 0.5, 0.02);
  }

  /**
   * Fit GP on (S_lin, T) residual.
   * Eq. (12): ψ fitted on T - γ(S)
   * @param {Array} S_lin — (N, d) linearly-transformed source points γ(S)
   * @param {Array} T     — (N, d) target points
   */
  fit(S_lin, T) {
    const residual = S_lin.map((row, i) => row.map((v, k) => T[i][k] - v));
    this.gp.fit(S_lin, residual);
    return this;
  }

  /** @returns {{mean: (M,d), std: (M,d)}} */
  predict(Xstar) { return this.gp.predict(Xstar); }
}

// =============================================================================
// TP-GPT TRANSPORT — Sec. IV, Eq. (7)
// phi(x) = gamma(x) + psi(gamma(x))
// =============================================================================
class TPGPTTransport {
  constructor() {
    this.linear = new LinearTransport();
    this.nonlinear = null;
    this.d = 0;
    this._fitted = false;
  }

  /**
   * Fit the two-stage transport map.
   * Step 1: linear Kabsch alignment (Eqs. 8-11)
   * Step 2: GP residual on gamma(S) → T (Eq. 12)
   * @param {Array} S — (N, d) source keypoints
   * @param {Array} T — (N, d) target keypoints
   */
  fit(S, T) {
    this.d = S[0].length;
    // Step 1: linear transport
    this.linear.fit(S, T);
    // Step 2: nonlinear residual
    const S_lin = this.linear.transform(S);
    this.nonlinear = new NonlinearResidual(this.d);
    this.nonlinear.fit(S_lin, T);
    this._fitted = true;
    return this;
  }

  /**
   * Eq. (7): phi(x) = gamma(x) + psi(gamma(x))
   * @param {Array} X — (M, d) source-frame points
   * @returns {Array} (M, d) transported points
   */
  transform(X) {
    const X_lin = this.linear.transform(X);
    const { mean: psi } = this.nonlinear.predict(X_lin);
    // phi = X_lin + psi
    return X_lin.map((row, i) => row.map((v, k) => v + psi[i][k]));
  }

  /**
   * Eq. (13): velocity transport via finite-difference Jacobian
   * xdot_hat[i] = J(x[i]) @ xdot[i]
   * @param {Array} X    — (M, d)
   * @param {Array} Xdot — (M, d) velocities
   * @param {number} eps — finite-difference step
   * @returns {Array} (M, d) transported velocities
   */
  transformVelocity(X, Xdot, eps = 1e-4) {
    return X.map((x, i) => {
      const vhat = new Float64Array(this.d);
      for (let j = 0; j < this.d; j++) {
        const xp = x.map((v, k) => k === j ? v + eps : v);
        const xm = x.map((v, k) => k === j ? v - eps : v);
        const phip = this.transform([xp])[0];
        const phim = this.transform([xm])[0];
        // J[:,j] = (phi(x+eps*ej) - phi(x-eps*ej)) / (2*eps)
        for (let dim = 0; dim < this.d; dim++)
          vhat[dim] += ((phip[dim] - phim[dim]) / (2 * eps)) * Xdot[i][j];
      }
      return Array.from(vhat);
    });
  }

  /**
   * Simplified transport uncertainty (from Eq. 17).
   * Returns mean std across output dims per input point.
   * @param {Array} X — (M, d)
   * @returns {Float64Array} (M,) scalar uncertainty per point
   */
  getUncertainty(X) {
    const X_lin = this.linear.transform(X);
    const { std } = this.nonlinear.predict(X_lin);
    return std.map((row) =>
      Math.sqrt(row.reduce((s, v) => s + v * v, 0) / row.length)
    );
  }
}

// =============================================================================
// DEMO DATA — pre-computed source scene (Three.js Y-up coordinates)
// Matches scene.js default layout exactly.
// Box center: (0.5, 0.785, 0.0)  Shelf center: (0.5, 0.9, -0.7)
// =============================================================================
const DEMO_DATA = {
  // Source keypoints S: 8 points — 4 box corners + 4 shelf slot corners
  // Shape (8, 3) in Three.js (x, height-y, depth-z)
  S: [
    // Box corners (±0.035 in x and z, fixed height 0.785)
    [0.465, 0.785, -0.035],
    [0.535, 0.785, -0.035],
    [0.535, 0.785,  0.035],
    [0.465, 0.785,  0.035],
    // Shelf slot corners (±0.05 in x, ±0.035 in z, height 0.9)
    [0.45, 0.9, -0.735],
    [0.55, 0.9, -0.735],
    [0.55, 0.9, -0.665],
    [0.45, 0.9, -0.665],
  ],

  // 6-phase EE trajectory waypoints (Three.js x, height-y, depth-z)
  waypoints: [
    [0.50, 0.960, 0.000],  // 0: pre-grasp — above box
    [0.50, 0.785, 0.000],  // 1: grasp     — at box
    [0.50, 1.060, 0.000],  // 2: lift      — lifted clear of table
    [0.50, 1.060, -0.700], // 3: carry     — above shelf
    [0.50, 0.823, -0.700], // 4: place     — onto lower plank (shelfY - 0.077 = 0.823)
    [0.50, 1.060, -0.700], // 5: retreat   — lift back up
  ],

  // Phase metadata for animation
  phases: [
    { label: 'APPROACH', steps: 40 },
    { label: 'GRASP',    steps: 20 },
    { label: 'LIFT',     steps: 35 },
    { label: 'CARRY',    steps: 55 },
    { label: 'PLACE',    steps: 30 },
    { label: 'RETREAT',  steps: 25 },
  ],
};

// =============================================================================
// HELPER — build target keypoints from new box/shelf positions
// All positions in Three.js Y-up: (x, height-y, depth-z)
// =============================================================================
function buildTargetKeypoints(bx, bz, sheight, sdepth) {
  const BOX_Y = 0.785; // box stays on table surface
  const SHELF_X = 0.5; // shelf x is fixed in scene
  return [
    // Box corners
    [bx - 0.035, BOX_Y, bz - 0.035],
    [bx + 0.035, BOX_Y, bz - 0.035],
    [bx + 0.035, BOX_Y, bz + 0.035],
    [bx - 0.035, BOX_Y, bz + 0.035],
    // Shelf slot corners
    [SHELF_X - 0.05, sheight, sdepth - 0.035],
    [SHELF_X + 0.05, sheight, sdepth - 0.035],
    [SHELF_X + 0.05, sheight, sdepth + 0.035],
    [SHELF_X - 0.05, sheight, sdepth + 0.035],
  ];
}

// =============================================================================
// SELF-TEST (runs on script load, results in console)
// =============================================================================
(function selfTest() {
  try {
    // Test 1: identity transport — phi(x) ≈ x
    const S_id = [
      [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
      [1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
      [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
    ];
    const t1 = new TPGPTTransport();
    t1.fit(S_id, S_id);
    const r1 = t1.transform([[0.5, 0.5, 0.5]]);
    const err1 = Math.abs(r1[0][0] - 0.5) + Math.abs(r1[0][1] - 0.5) + Math.abs(r1[0][2] - 0.5);
    console.assert(err1 < 0.05, 'Test 1 FAILED: identity error=' + err1.toFixed(4));
    console.log('[gp_infer] Test 1 passed: identity transport, err=' + err1.toFixed(4));

    // Test 2: pure translation (+0.2 in x, +0.1 in y)
    const T_tr = S_id.map((p) => [p[0] + 0.2, p[1] + 0.1, p[2]]);
    const t2 = new TPGPTTransport();
    t2.fit(S_id, T_tr);
    const r2 = t2.transform([[0.5, 0.5, 0.5]]);
    const err2x = Math.abs(r2[0][0] - 0.7);
    const err2y = Math.abs(r2[0][1] - 0.6);
    console.assert(err2x < 0.05, 'Test 2 FAILED: translation X error=' + err2x.toFixed(4));
    console.assert(err2y < 0.05, 'Test 2 FAILED: translation Y error=' + err2y.toFixed(4));
    console.log('[gp_infer] Test 2 passed: translation transport, X_err=' + err2x.toFixed(4) + ' Y_err=' + err2y.toFixed(4));

    // Test 3: full demo pipeline
    const T3 = buildTargetKeypoints(0.60, 0.10, 0.95, -0.65);
    const t3 = new TPGPTTransport();
    t3.fit(DEMO_DATA.S, T3);
    const wp3 = t3.transform(DEMO_DATA.waypoints);
    console.assert(wp3.length === 6, 'Test 3 FAILED: wrong waypoint count');
    console.log('[gp_infer] Test 3 passed: full pipeline, place_wp=' +
      wp3[4].map((v) => v.toFixed(3)).join(','));
    console.log('[gp_infer] teach-once GP inference loaded ✓');
  } catch (e) {
    console.error('[gp_infer] self-test ERROR:', e);
  }
})();
