// mode_cleaning.js — Mode 2: Surface cleaning demo
// Flat surface only. Tilt angle is the single task parameter.
// TP-GPT transports the raster-scan demo path to the tilted surface.
// Coverage metric reads actual canvas pixel erasure from scene.js.
// Coordinate system: Three.js Y-up (x, height-y, depth-z)
'use strict';

/* global Scene, TPGPTTransport */

const ModeCleaning = (() => {
  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  let currentTilt = 0;
  let currentMess = 'scatter';
  let isRunning   = false;

  // Source keypoints: 8 points on flat table surface — Three.js Y-up
  const S_FLAT = [
    [0.25, 0.755, -0.17], [0.50, 0.755, -0.17],
    [0.75, 0.755, -0.17], [0.75, 0.755,  0.00],
    [0.75, 0.755,  0.17], [0.50, 0.755,  0.17],
    [0.25, 0.755,  0.17], [0.25, 0.755,  0.00],
  ];

  // ---------------------------------------------------------------------------
  // Mess presets — each defines the spill shape + matching raster scan path.
  //
  // Coordinate mapping  (PlaneGeometry 0.5×0.35, canvas 256×256):
  //   world x ∈ [0.25, 0.75] → canvas u ∈ [0, 256]
  //   world z ∈ [-0.175, 0.175] → canvas v ∈ [0, 256]
  //   erase arc radius = 26px ≈ 0.036 world-z units
  //
  // Six-stroke raster spacing = 0.056 world-z units ≈ 40px — arcs overlap by ~12px.
  // ---------------------------------------------------------------------------
  const MESS_PRESETS = {
    scatter: {
      label: 'Scatter',
      drawType: 'scatter',
      // 12 waypoints → 6 horizontal strokes covering full table (z: -0.140 … +0.140)
      waypoints: [
        [0.30, 0.775, -0.140], [0.70, 0.775, -0.140],
        [0.70, 0.775, -0.084], [0.30, 0.775, -0.084],
        [0.30, 0.775, -0.028], [0.70, 0.775, -0.028],
        [0.70, 0.775,  0.028], [0.30, 0.775,  0.028],
        [0.30, 0.775,  0.084], [0.70, 0.775,  0.084],
        [0.70, 0.775,  0.140], [0.30, 0.775,  0.140],
      ],
      // 13 phases: APPROACH + 6×(STROKE+TRANSIT) collapsed to 11 mid-segments + RETREAT
      phases: [
        { label: 'APPROACH', steps: 22 },
        { label: 'STROKE 1', steps: 38 }, { label: 'TRANSIT',  steps: 10 },
        { label: 'STROKE 2', steps: 38 }, { label: 'TRANSIT',  steps: 10 },
        { label: 'STROKE 3', steps: 38 }, { label: 'TRANSIT',  steps: 10 },
        { label: 'STROKE 4', steps: 38 }, { label: 'TRANSIT',  steps: 10 },
        { label: 'STROKE 5', steps: 38 }, { label: 'TRANSIT',  steps: 10 },
        { label: 'STROKE 6', steps: 38 },
        { label: 'RETREAT',  steps: 22 },
      ],
    },

    puddle: {
      label: 'Puddle',
      drawType: 'puddle',
      // 6 waypoints → 3 strokes covering central blob (z: -0.065 … +0.065, x: 0.35–0.65)
      waypoints: [
        [0.35, 0.775, -0.065], [0.65, 0.775, -0.065],
        [0.65, 0.775,  0.000], [0.35, 0.775,  0.000],
        [0.35, 0.775,  0.065], [0.65, 0.775,  0.065],
      ],
      phases: [
        { label: 'APPROACH', steps: 22 },
        { label: 'STROKE 1', steps: 38 }, { label: 'TRANSIT',  steps: 10 },
        { label: 'STROKE 2', steps: 38 }, { label: 'TRANSIT',  steps: 10 },
        { label: 'STROKE 3', steps: 38 },
        { label: 'RETREAT',  steps: 22 },
      ],
    },

    line: {
      label: 'Line',
      drawType: 'line',
      // 4 waypoints → 2 passes along the horizontal streak (z ≈ 0, x: 0.28–0.72)
      waypoints: [
        [0.28, 0.775, -0.008], [0.72, 0.775, -0.008],
        [0.72, 0.775,  0.008], [0.28, 0.775,  0.008],
      ],
      phases: [
        { label: 'APPROACH', steps: 22 },
        { label: 'STROKE 1', steps: 52 }, { label: 'TRANSIT',  steps: 8 },
        { label: 'STROKE 2', steps: 52 },
        { label: 'RETREAT',  steps: 22 },
      ],
    },
  };

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  function init() {
    currentTilt = 0;
    currentMess = 'scatter';
    isRunning   = false;
    renderControlPanel();
    updatePhaseBadge('READY');
    Scene.updateSurfaceMesh({ type: 'flat', tilt: 0 });
    // Show live path shift immediately (0.0 mm at flat — grows as user tilts)
    updateTransportPreview();
    // Cleaning is always ready — button enabled immediately
    const btn = document.getElementById('btn-generalize');
    if (btn) btn.disabled = false;
  }

  // ---------------------------------------------------------------------------
  // Control panel HTML — mess selector + tilt slider
  // ---------------------------------------------------------------------------
  function renderControlPanel() {
    const messHTML = Object.keys(MESS_PRESETS).map(key => `
      <button class="surface-btn${key === currentMess ? ' active' : ''}"
              data-mess="${key}">
        ${MESS_PRESETS[key].label}
      </button>
    `).join('');

    document.getElementById('panel-body').innerHTML = `
      <div class="drag-instructions">
        <p>🧹 Pick a mess shape, then tilt the surface.</p>
        <p>TP-GPT adapts the raster-scan path — watch the spill disappear.</p>
      </div>

      <div class="slider-group">
        <label class="slider-label">Mess shape</label>
        <div class="surface-selector" id="clean-mess-selector">
          ${messHTML}
        </div>
      </div>

      <div class="slider-group">
        <label class="slider-label">
          Surface tilt
          <span class="slider-value" id="clean-tilt-val">${currentTilt}°</span>
        </label>
        <input type="range" id="clean-tilt"
               min="0" max="12" step="1" value="${currentTilt}"
               class="tp-slider">
      </div>

      <div class="config-summary">
        <span class="config-label">Configuration:</span>
        <span class="config-value" id="clean-config-text">${MESS_PRESETS[currentMess].label} · ${currentTilt === 0 ? 'Flat (0° tilt)' : `Tilted ${currentTilt}°`}</span>
      </div>
    `;
    setupListeners();
  }

  // ---------------------------------------------------------------------------
  // Listeners
  // ---------------------------------------------------------------------------
  function setupListeners() {
    // Mess shape selector
    document.querySelectorAll('#clean-mess-selector .surface-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        document.querySelectorAll('#clean-mess-selector .surface-btn')
          .forEach(b => b.classList.remove('active'));
        e.currentTarget.classList.add('active');
        currentMess = e.currentTarget.dataset.mess;
        updateConfigText();
        // Reset spill canvas to the new mess shape (keeps tilt, refreshes pixels)
        Scene.resetSpillCanvas(MESS_PRESETS[currentMess].drawType);
        updateTransportPreview();
      });
    });

    // Tilt slider
    const tiltSlider = document.getElementById('clean-tilt');
    if (tiltSlider) {
      tiltSlider.addEventListener('input', () => {
        currentTilt = parseInt(tiltSlider.value);
        const tv = document.getElementById('clean-tilt-val');
        if (tv) tv.textContent = currentTilt + '°';
        Scene.updateSurfaceMesh({ type: currentTilt > 0 ? 'tilted' : 'flat', tilt: currentTilt });
        // Reset spill so user sees a fresh mess after changing tilt
        Scene.resetSpillCanvas(MESS_PRESETS[currentMess].drawType);
        updateConfigText();
        updateTransportPreview();
      });
    }
  }

  function updateConfigText() {
    const ct = document.getElementById('clean-config-text');
    if (!ct) return;
    const messLabel = MESS_PRESETS[currentMess] ? MESS_PRESETS[currentMess].label : currentMess;
    ct.textContent = currentTilt === 0
      ? `${messLabel} · Flat (0° tilt)`
      : `${messLabel} · Tilted ${currentTilt}° — TP-GPT will adapt`;
  }

  // ---------------------------------------------------------------------------
  // Quick uncertainty + path-shift preview (runs live as sliders move)
  // ---------------------------------------------------------------------------
  function updateTransportPreview() {
    try {
      const preset = MESS_PRESETS[currentMess];
      const T = getSurfaceKeypoints(currentTilt);
      const transport = new TPGPTTransport();
      transport.fit(S_FLAT, T);
      const transported = transport.transform(preset.waypoints);
      const u = transport.getUncertainty(preset.waypoints);
      const sigma = u.reduce((s, v) => s + v, 0) / u.length;
      // Mean Euclidean displacement of waypoints due to transport (metres)
      const pathShift = preset.waypoints.reduce((sum, wp, i) => {
        const tp = transported[i];
        return sum + Math.sqrt(
          (tp[0]-wp[0])**2 + (tp[1]-wp[1])**2 + (tp[2]-wp[2])**2
        );
      }, 0) / preset.waypoints.length;
      updateMetrics({ sigma, pathShift, coverage: null });
    } catch (e) {
      console.warn('[cleaning] preview error:', e.message);
    }
  }

  // ---------------------------------------------------------------------------
  // Surface keypoints for given tilt angle (flat at tilt=0)
  // ---------------------------------------------------------------------------
  function getSurfaceKeypoints(tiltDeg) {
    if (tiltDeg === 0) return S_FLAT.map(p => [...p]);
    const tilt = tiltDeg * Math.PI / 180;
    return S_FLAT.map(p => [
      p[0],
      p[1] - p[2] * Math.sin(tilt),
      p[2] * Math.cos(tilt),
    ]);
  }

  // ---------------------------------------------------------------------------
  // Interpolate waypoints into dense trajectory
  // ---------------------------------------------------------------------------
  function interpolateWaypoints(waypoints, phases) {
    const positions = [], labels = [];
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

    const btn = document.getElementById('btn-generalize');
    if (btn) { btn.disabled = true; btn.textContent = 'Computing…'; }

    try {
      const preset = MESS_PRESETS[currentMess];
      const T = getSurfaceKeypoints(currentTilt);
      const transport = new TPGPTTransport();
      transport.fit(S_FLAT, T);
      const transported = transport.transform(preset.waypoints);

      // Clamp all waypoint y-values so the arm never goes through the table (y < 0.780).
      const MIN_Y = 0.780;
      const safe = transported.map(p => [p[0], Math.max(MIN_Y, p[1]), p[2]]);

      // Approach above first waypoint, retreat above last
      const first = safe[0];
      const last  = safe[safe.length - 1];
      const fullPath = [
        [first[0], first[1] + 0.12, first[2]],
        ...safe,
        [last[0],  last[1]  + 0.12, last[2]],
      ];

      const { positions, labels } = interpolateWaypoints(fullPath, preset.phases);

      const u = transport.getUncertainty(preset.waypoints);
      const meanSigma = u.reduce((s, v) => s + v, 0) / u.length;

      // Mean Euclidean displacement of waypoints due to transport (metres → mm display)
      const pathShift = preset.waypoints.reduce((sum, wp, i) => {
        const tp = transported[i];
        return sum + Math.sqrt(
          (tp[0]-wp[0])**2 + (tp[1]-wp[1])**2 + (tp[2]-wp[2])**2
        );
      }, 0) / preset.waypoints.length;

      if (btn) btn.textContent = 'Executing…';
      await Scene.playTrajectoryClean(positions, labels, 30);

      // Coverage = fraction of spill actually erased from the canvas
      const coverage = Scene.getSpillCoverage();
      updateMetrics({ sigma: meanSigma, pathShift, coverage });
      updatePhaseBadge('DONE ✓');
    } catch (e) {
      console.error('[cleaning] generalize error:', e);
      updatePhaseBadge('ERROR');
    } finally {
      isRunning = false;
      const btn2 = document.getElementById('btn-generalize');
      if (btn2) { btn2.disabled = false; btn2.textContent = 'Generalize TP-GPT →'; }
    }
  }

  // ---------------------------------------------------------------------------
  // Metrics display (null-safe)
  // ---------------------------------------------------------------------------
  function updateMetrics(opts) {
    const sigma     = opts ? opts.sigma     : null;
    const coverage  = opts ? opts.coverage  : null;
    const pathShift = opts ? opts.pathShift : null;  // metres; display in mm

    const sigmaEl = document.getElementById('metric-sigma');
    const errorEl = document.getElementById('metric-error');
    const confEl  = document.getElementById('metric-conf');

    if (sigmaEl) sigmaEl.textContent = sigma != null ? sigma.toFixed(4) : '—';

    // Rename "EE error" → "Cleaned" for cleaning mode
    const errorLabelEl = document.querySelector(
      '#metrics-panel .metric:nth-child(2) .metric-label');
    if (errorLabelEl) errorLabelEl.textContent = 'Cleaned';

    if (errorEl) {
      errorEl.textContent = coverage != null ? (coverage * 100).toFixed(0) + '%' : '—';
      if (coverage != null) {
        errorEl.style.color = coverage > 0.75
          ? 'var(--success)' : coverage > 0.40
          ? 'var(--warning)' : 'var(--error)';
      } else {
        errorEl.style.color = '';
      }
    }

    // Path shift: mean displacement of demo waypoints due to TP-GPT transport.
    // 0 mm = flat surface (no adaptation needed). Grows with tilt.
    // Replaces "GP confidence" which was constant across tilt values.
    const confLabelEl = document.querySelector(
      '#metrics-panel .metric:nth-child(3) .metric-label');
    if (confLabelEl) confLabelEl.textContent = 'Path shift';
    if (confEl) confEl.textContent = pathShift != null
      ? (pathShift * 1000).toFixed(1) + ' mm' : '—';
  }

  function updatePhaseBadge(label) {
    const badge = document.getElementById('mode-badge');
    if (badge) badge.textContent = 'Cleaning · ' + label;
  }

  // ---------------------------------------------------------------------------
  // Reset
  // ---------------------------------------------------------------------------
  function reset() {
    isRunning   = false;
    currentTilt = 0;
    currentMess = 'scatter';

    // loadScene re-creates the spill mesh with a fresh canvas
    try { Scene.loadScene('cleaning'); } catch (e) { /* not ready */ }

    // Re-render control panel so mess selector + tilt reflect reset state
    renderControlPanel();

    const btn = document.getElementById('btn-generalize');
    if (btn) { btn.disabled = false; btn.textContent = 'Generalize TP-GPT →'; }

    updateMetrics(null);
    updatePhaseBadge('READY');
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------
  return { init, generalize, reset };
})();
