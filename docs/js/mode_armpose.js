// mode_armpose.js — Mode 3: Arm-pose following demo
// User adjusts 4 keypoint positions via sliders or presets.
// TP-GPT transports the tracing path to the new arm configuration.
// Coordinate system: Three.js Y-up (x, height-y, depth-z)
'use strict';

/* global Scene, TPGPTTransport */

const ModeArmpose = (() => {
  // ---------------------------------------------------------------------------
  // Keypoint colors (match scene.js sphere colors)
  // ---------------------------------------------------------------------------
  const KP_COLORS = {
    shoulder: '#00eeff',
    elbow:    '#ff44aa',
    wrist:    '#ffdd00',
    hand:     '#4488ff',
  };

  // ---------------------------------------------------------------------------
  // Default positions — Three.js Y-up
  // ---------------------------------------------------------------------------
  const DEFAULTS = {
    shoulder: [0.38, 1.05, 0.00],
    elbow:    [0.48, 0.95, 0.00],
    wrist:    [0.58, 0.85, 0.00],
    hand:     [0.63, 0.78, 0.00],
  };

  // ---------------------------------------------------------------------------
  // Presets
  // ---------------------------------------------------------------------------
  const PRESETS = {
    default: {
      shoulder: [0.38, 1.05, 0.00],
      elbow:    [0.48, 0.95, 0.00],
      wrist:    [0.58, 0.85, 0.00],
      hand:     [0.63, 0.78, 0.00],
    },
    raised: {
      shoulder: [0.35, 1.10, 0.00],
      elbow:    [0.45, 1.05, 0.00],
      wrist:    [0.55, 0.95, 0.00],
      hand:     [0.60, 0.88, 0.00],
    },
    extended: {
      shoulder: [0.40, 1.00, 0.00],
      elbow:    [0.52, 0.92, 0.00],
      wrist:    [0.63, 0.85, 0.00],
      hand:     [0.70, 0.80, 0.00],
    },
    bent: {
      shoulder: [0.35, 1.08, 0.00],
      elbow:    [0.43, 0.88, 0.00],
      wrist:    [0.52, 0.82, 0.00],
      hand:     [0.58, 0.79, 0.00],
    },
  };

  // ---------------------------------------------------------------------------
  // Source keypoints: 3 points per keypoint (cross pattern)
  // Captures position + local orientation per paper Sec. V-B
  // ---------------------------------------------------------------------------
  const S = [
    // Shoulder
    [0.38, 1.05, -0.03], [0.38, 1.05,  0.03], [0.38, 1.08, 0.00],
    // Elbow
    [0.48, 0.95, -0.03], [0.48, 0.95,  0.03], [0.48, 0.98, 0.00],
    // Wrist
    [0.58, 0.85, -0.03], [0.58, 0.85,  0.03], [0.58, 0.88, 0.00],
    // Hand
    [0.63, 0.78, -0.03], [0.63, 0.78,  0.03], [0.63, 0.81, 0.00],
  ];

  // ---------------------------------------------------------------------------
  // Demo waypoints: approach → shoulder → elbow → wrist → hand → retreat
  // ---------------------------------------------------------------------------
  const WAYPOINTS = [
    [0.38, 1.20, 0.00],  // approach above shoulder
    [0.38, 1.05, 0.00],  // shoulder
    [0.48, 0.95, 0.00],  // elbow
    [0.58, 0.85, 0.00],  // wrist
    [0.63, 0.78, 0.00],  // hand
    [0.63, 0.93, 0.00],  // retreat
  ];

  const PHASES = [
    { label: 'APPROACH',   steps: 30 },
    { label: 'shoulder ✓', steps: 25 },
    { label: 'elbow ✓',    steps: 25 },
    { label: 'wrist ✓',    steps: 25 },
    { label: 'hand ✓',     steps: 25 },
    { label: 'RETREAT',    steps: 25 },
  ];

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  let kp = JSON.parse(JSON.stringify(DEFAULTS));
  let isRunning = false;
  let _genId    = 0;

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  function init() {
    kp = JSON.parse(JSON.stringify(DEFAULTS));
    isRunning = false;
    renderControlPanel();
    Scene.updateKeypointSpheres(kp);
    updateMetrics(null);
    updatePhaseBadge('READY');
    // Armpose is always ready to generalize
    const btn = document.getElementById('btn-generalize');
    if (btn) btn.disabled = false;
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  function isMobile() { return window.innerWidth <= 768; }

  function renderMobilePanel() {
    const presetHTML = Object.keys(PRESETS).map(p => `
      <button class="mobile-preset-btn${p === 'default' ? ' running' : ''}"
              data-preset="${p}">
        ${p.charAt(0).toUpperCase() + p.slice(1)}
      </button>
    `).join('');

    document.getElementById('panel-body').innerHTML = `
      <p class="mobile-note">
        🖥️ Fine-tune each joint on desktop — tap a pose below to see TP-GPT adapt the path:
      </p>
      <div class="mobile-preset-grid" id="mobile-presets-armpose">
        ${presetHTML}
      </div>
    `;

    document.querySelectorAll('#mobile-presets-armpose .mobile-preset-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        if (isRunning) return;
        document.querySelectorAll('#mobile-presets-armpose .mobile-preset-btn')
          .forEach(b => b.classList.remove('running'));
        e.currentTarget.classList.add('running');

        // Apply pose preset — updates kp state and scene spheres
        applyPreset(e.currentTarget.dataset.preset);

        // Enable Generalize button — user taps it when ready
        const genBtn = document.getElementById('btn-generalize');
        if (genBtn) genBtn.disabled = false;
      });
    });
  }

  // ---------------------------------------------------------------------------
  // Control panel HTML
  // ---------------------------------------------------------------------------
  function renderControlPanel() {
    if (isMobile()) { renderMobilePanel(); return; }
    const names = ['shoulder', 'elbow', 'wrist', 'hand'];

    const kpHTML = names.map(name => `
      <div class="keypoint-group">
        <div class="keypoint-header">
          <span class="keypoint-dot" style="background:${KP_COLORS[name]}"></span>
          <span class="keypoint-name">${name.charAt(0).toUpperCase() + name.slice(1)}</span>
        </div>
        <label class="slider-label">
          X <span class="slider-value" id="arm-${name}-x-val">${DEFAULTS[name][0].toFixed(2)}</span>
        </label>
        <input type="range" id="arm-${name}-x"
               min="0.25" max="0.75" step="0.01"
               value="${DEFAULTS[name][0]}"
               class="tp-slider" data-kp="${name}" data-axis="0">
        <label class="slider-label">
          Height <span class="slider-value" id="arm-${name}-y-val">${DEFAULTS[name][1].toFixed(2)}</span>
        </label>
        <input type="range" id="arm-${name}-y"
               min="0.70" max="1.20" step="0.01"
               value="${DEFAULTS[name][1]}"
               class="tp-slider" data-kp="${name}" data-axis="1">
      </div>
    `).join('');

    const presetHTML = Object.keys(PRESETS).map(p => `
      <button class="surface-btn${p === 'default' ? ' active' : ''}" data-preset="${p}">
        ${p.charAt(0).toUpperCase() + p.slice(1)}
      </button>
    `).join('');

    document.getElementById('panel-body').innerHTML = `
      <div class="drag-instructions">
        <p>🎯 Adjust keypoint positions or pick a preset.</p>
        <p>TP-GPT transports the path to the new arm configuration.</p>
      </div>

      <div class="slider-group">
        <label class="slider-label">Presets</label>
        <div class="surface-selector" id="arm-presets">
          ${presetHTML}
        </div>
      </div>

      <div class="keypoint-controls">
        ${kpHTML}
      </div>
    `;
    setupListeners();
  }

  // ---------------------------------------------------------------------------
  // Listeners
  // ---------------------------------------------------------------------------
  function setupListeners() {
    // Keypoint sliders
    document.querySelectorAll('.tp-slider[data-kp]').forEach(slider => {
      slider.addEventListener('input', () => {
        const name = slider.dataset.kp;
        const axis = parseInt(slider.dataset.axis);
        kp[name][axis] = parseFloat(slider.value);
        const axLabel = axis === 0 ? 'x' : 'y';
        const valEl = document.getElementById(`arm-${name}-${axLabel}-val`);
        if (valEl) valEl.textContent = kp[name][axis].toFixed(2);
        Scene.updateKeypointSpheres(kp);
        onConfigChanged();
      });
    });

    // Preset buttons
    document.querySelectorAll('#arm-presets .surface-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        document.querySelectorAll('#arm-presets .surface-btn')
          .forEach(b => b.classList.remove('active'));
        e.currentTarget.classList.add('active');
        applyPreset(e.currentTarget.dataset.preset);
      });
    });
  }

  function applyPreset(name) {
    const preset = PRESETS[name];
    if (!preset) return;
    kp = JSON.parse(JSON.stringify(preset));

    ['shoulder', 'elbow', 'wrist', 'hand'].forEach(kpName => {
      ['x', 'y'].forEach((axis, axIdx) => {
        const slider = document.getElementById(`arm-${kpName}-${axis}`);
        const valEl  = document.getElementById(`arm-${kpName}-${axis}-val`);
        if (slider) slider.value = kp[kpName][axIdx];
        if (valEl)  valEl.textContent = kp[kpName][axIdx].toFixed(2);
      });
    });

    Scene.updateKeypointSpheres(kp);
    onConfigChanged();
  }

  // ---------------------------------------------------------------------------
  // Configuration change
  // ---------------------------------------------------------------------------
  function onConfigChanged() {
    const btn = document.getElementById('btn-generalize');
    if (btn && !isRunning) btn.disabled = false;
    updateTransportPreview();
  }

  function updateTransportPreview() {
    try {
      const T = buildArmposeKeypoints(kp);
      const transport = new TPGPTTransport();
      transport.fit(S, T);
      const u = transport.getUncertainty(WAYPOINTS);
      const mean = u.reduce((s, v) => s + v, 0) / u.length;
      updateMetrics({ sigma: mean, error: null });
    } catch (e) {
      console.warn('[armpose] preview error:', e.message);
    }
  }

  // ---------------------------------------------------------------------------
  // Build target keypoints from current kp state
  // Expand each keypoint to 3 points (cross pattern) matching S structure
  // ---------------------------------------------------------------------------
  function buildArmposeKeypoints(keypoints) {
    const names = ['shoulder', 'elbow', 'wrist', 'hand'];
    const T = [];
    names.forEach(name => {
      const [x, y, z] = keypoints[name];
      T.push([x, y, z - 0.03]);
      T.push([x, y, z + 0.03]);
      T.push([x, y + 0.03, z]);
    });
    return T;
  }

  // ---------------------------------------------------------------------------
  // Interpolate waypoints into dense trajectory
  // ---------------------------------------------------------------------------
  function interpolateWaypoints(waypoints, phases) {
    const positions = [];
    const labels    = [];
    const n = Math.min(waypoints.length - 1, phases.length);
    for (let i = 0; i < n; i++) {
      const start = waypoints[i];
      const end   = waypoints[i + 1];
      const steps = phases[i].steps;
      const label = phases[i].label;
      for (let t = 0; t < steps; t++) {
        const a = t / steps;
        positions.push(start.map((v, k) => v + a * (end[k] - v)));
        labels.push(label);
      }
    }
    positions.push(waypoints[waypoints.length - 1]);
    labels.push(phases[phases.length - 1].label);
    return { positions, labels };
  }

  // ---------------------------------------------------------------------------
  // Generalize — full TP-GPT pipeline (Eq. 7)
  // ---------------------------------------------------------------------------
  async function generalize() {
    if (isRunning) return;
    isRunning = true;
    const myId = ++_genId;

    const btn = document.getElementById('btn-generalize');
    if (btn) { btn.disabled = true; btn.textContent = 'Computing…'; }

    try {
      const T = buildArmposeKeypoints(kp);
      const transport = new TPGPTTransport();
      transport.fit(S, T);

      const transported = transport.transform(WAYPOINTS);
      const { positions, labels } = interpolateWaypoints(transported, PHASES);

      const u = transport.getUncertainty(WAYPOINTS);
      const meanSigma = u.reduce((s, v) => s + v, 0) / u.length;

      // EE error: mean distance from transported keypoint waypoints to target keypoints
      const kpNames = ['shoulder', 'elbow', 'wrist', 'hand'];
      const errors = kpNames.map((name, i) => {
        const kpArr = kp[name];
        const wp    = transported[i + 1];  // indices 1-4 are the 4 keypoints
        return Math.sqrt(
          (wp[0] - kpArr[0]) ** 2 + (wp[1] - kpArr[1]) ** 2 + (wp[2] - kpArr[2]) ** 2
        );
      });
      const meanErr = errors.reduce((s, v) => s + v, 0) / errors.length;

      updateMetrics({ sigma: meanSigma, error: meanErr });

      if (btn) btn.textContent = 'Executing…';
      await Scene.playTrajectoryArmpose(positions, labels, 50);

      if (_genId !== myId) return; // cancelled by mode switch

      updatePhaseBadge('DONE ✓');
    } catch (e) {
      if (_genId === myId) {
        console.error('[armpose] generalize error:', e);
        updatePhaseBadge('ERROR');
      }
    } finally {
      if (_genId === myId) {
        isRunning = false;
        const btn2 = document.getElementById('btn-generalize');
        if (btn2) { btn2.disabled = false; btn2.textContent = 'Generalize TP-GPT →'; }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Cancel — called by ui.js on mode switch
  // ---------------------------------------------------------------------------
  function cancel() {
    _genId++;
    isRunning = false;
  }

  // ---------------------------------------------------------------------------
  // Metrics display (null-safe)
  // ---------------------------------------------------------------------------
  function updateMetrics(opts) {
    const sigma = opts ? opts.sigma : null;
    const error = opts ? opts.error : null;

    const sigmaEl = document.getElementById('metric-sigma');
    const errorEl = document.getElementById('metric-error');
    const confEl  = document.getElementById('metric-conf');

    if (sigmaEl) sigmaEl.textContent = sigma != null ? sigma.toFixed(4) : '—';
    if (errorEl) {
      errorEl.textContent = error != null ? error.toFixed(4) + ' m' : '—';
      if (error != null) {
        errorEl.style.color = error < 0.08
          ? 'var(--success)' : error < 0.15
          ? 'var(--warning)' : 'var(--error)';
      } else {
        errorEl.style.color = '';
      }
    }
    // Restore label for armpose mode
    const errorLabelEl = document.querySelector(
      '#metrics-panel .metric:nth-child(2) .metric-label');
    if (errorLabelEl) errorLabelEl.textContent = 'EE error';
    if (confEl) confEl.textContent = error != null
      ? Math.max(0, (1 - error * 5) * 100).toFixed(0) + '%' : '—';
  }

  function updatePhaseBadge(label) {
    const badge = document.getElementById('mode-badge');
    if (badge) badge.textContent = 'Arm-pose · ' + label;
  }

  // ---------------------------------------------------------------------------
  // Reset
  // ---------------------------------------------------------------------------
  function reset() {
    isRunning = false;
    kp = JSON.parse(JSON.stringify(DEFAULTS));

    try { Scene.resetScene(); } catch (e) { /* Scene not ready */ }

    const btn = document.getElementById('btn-generalize');
    if (btn) { btn.disabled = false; btn.textContent = 'Generalize TP-GPT →'; }

    updateMetrics(null);
    updatePhaseBadge('READY');
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------
  return { init, generalize, reset, cancel };
})();
