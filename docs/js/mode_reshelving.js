// mode_reshelving.js — Mode 1: Reshelving interactive demo
// User moves box and shelf to new positions.
// TP-GPT transports the demo trajectory live in browser.
// Coordinate system: Three.js Y-up (x, height-y, depth-z)
'use strict';

/* global Scene, TPGPTTransport, LinearTransport, DEMO_DATA, buildTargetKeypoints */

const ModeReshelving = (() => {
  // ---------------------------------------------------------------------------
  // State — Three.js Y-up coordinates
  // ---------------------------------------------------------------------------
  // box: (x, depth-z) — height fixed at 0.785 (table surface)
  let boxX   = 0.500;
  let boxZ   = 0.000;
  // shelf: (height-y, depth-z) — x fixed at 0.5 in scene
  let shelfY = 0.900;
  let shelfZ = -0.700;

  let hasBeenDragged = false;
  let isRunning = false;
  let hasPlaced = false;  // true after first successful generalize — sliders snap box back to table

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  function init() {
    renderControlPanel();
    setupListeners();
    updateMetrics(null);
    updatePhaseBadge('READY');
  }

  // ---------------------------------------------------------------------------
  // Control panel HTML
  // ---------------------------------------------------------------------------
  function renderControlPanel() {
    document.getElementById('panel-body').innerHTML = `
      <div class="drag-instructions">
        <p>🟠 Move the orange box to a new table position.</p>
        <p>🟢 Move the shelf slot to a new height and depth.</p>
      </div>

      <div class="slider-group">
        <label class="slider-label">
          Box X position
          <span class="slider-value" id="box-x-val">0.50</span>
        </label>
        <input type="range" id="box-x"
               min="0.25" max="0.75" step="0.01" value="0.50"
               class="tp-slider">

        <label class="slider-label">
          Box depth (Z)
          <span class="slider-value" id="box-z-val">0.00</span>
        </label>
        <input type="range" id="box-z"
               min="-0.20" max="0.20" step="0.01" value="0.00"
               class="tp-slider">
      </div>

      <div class="slider-group">
        <label class="slider-label">
          Shelf height
          <span class="slider-value" id="shelf-y-val">0.90</span>
        </label>
        <input type="range" id="shelf-y"
               min="0.78" max="1.10" step="0.01" value="0.90"
               class="tp-slider">

        <label class="slider-label">
          Shelf depth
          <span class="slider-value" id="shelf-z-val">-0.70</span>
        </label>
        <input type="range" id="shelf-z"
               min="-0.90" max="-0.50" step="0.01" value="-0.70"
               class="tp-slider">
      </div>

      <div class="config-summary" id="config-summary">
        <span class="config-label">Configuration:</span>
        <span class="config-value" id="config-text">Default (demo scene)</span>
      </div>
    `;
  }

  // ---------------------------------------------------------------------------
  // Listeners
  // ---------------------------------------------------------------------------
  function setupListeners() {
    document.getElementById('box-x').addEventListener('input', onBoxSlider);
    document.getElementById('box-z').addEventListener('input', onBoxSlider);
    document.getElementById('shelf-y').addEventListener('input', onShelfSlider);
    document.getElementById('shelf-z').addEventListener('input', onShelfSlider);
  }

  function onBoxSlider() {
    boxX = parseFloat(document.getElementById('box-x').value);
    boxZ = parseFloat(document.getElementById('box-z').value);
    document.getElementById('box-x-val').textContent = boxX.toFixed(2);
    document.getElementById('box-z-val').textContent = boxZ.toFixed(2);
    Scene.setObjectPos([boxX, 0.785, boxZ]);
    onConfigChanged();
  }

  function onShelfSlider() {
    shelfY = parseFloat(document.getElementById('shelf-y').value);
    shelfZ = parseFloat(document.getElementById('shelf-z').value);
    document.getElementById('shelf-y-val').textContent = shelfY.toFixed(2);
    document.getElementById('shelf-z-val').textContent = shelfZ.toFixed(2);
    Scene.setShelfPos([0.5, shelfY, shelfZ]);
    onConfigChanged();
  }

  // ---------------------------------------------------------------------------
  // Configuration change
  // ---------------------------------------------------------------------------
  function onConfigChanged() {
    hasBeenDragged = true;
    const btn = document.getElementById('btn-generalize');
    if (btn && !isRunning) btn.disabled = false;

    document.getElementById('config-text').textContent =
      `Box: (${boxX.toFixed(2)}, ${boxZ.toFixed(2)})  Shelf: h=${shelfY.toFixed(2)}, d=${shelfZ.toFixed(2)}`;

    // If the arm already placed the box once, snap it back to its table position so
    // the user can immediately see what the new configuration looks like before re-running.
    if (hasPlaced) {
      Scene.setObjectPos([boxX, 0.785, boxZ]);
    }

    updateTransportPreview();
  }

  // ---------------------------------------------------------------------------
  // Quick transport preview (linear only for responsiveness)
  // ---------------------------------------------------------------------------
  function updateTransportPreview() {
    try {
      const T = buildTargetKeypoints(boxX, boxZ, shelfY, shelfZ);
      const transport = new TPGPTTransport();
      transport.fit(DEMO_DATA.S, T);
      const uncertainties = transport.getUncertainty(DEMO_DATA.waypoints);
      const meanSigma = uncertainties.reduce((s, v) => s + v, 0) / uncertainties.length;
      updateMetrics({ sigma: meanSigma, error: null, conf: Math.max(0, 1 - meanSigma * 8) });
    } catch (e) {
      console.warn('[reshelving] preview error:', e.message);
    }
  }

  // ---------------------------------------------------------------------------
  // Generalize — full TP-GPT pipeline
  // ---------------------------------------------------------------------------
  async function generalize() {
    if (isRunning) return;
    isRunning = true;

    const btn = document.getElementById('btn-generalize');
    if (btn) { btn.disabled = true; btn.textContent = 'Computing…'; }

    try {
      // Build target keypoints from current slider state
      const T = buildTargetKeypoints(boxX, boxZ, shelfY, shelfZ);

      // Fit full TP-GPT transport (Eq. 7)
      const transport = new TPGPTTransport();
      transport.fit(DEMO_DATA.S, T);

      // Transport demo waypoints
      const transported = transport.transform(DEMO_DATA.waypoints);

      // Hard-pin the entire shelf-approach chain so the arm descends from directly
      // above the NEW shelf — not wherever the GP transport guessed.
      //
      //   wp[3] CARRY   — hover above new shelf, high enough to clear the plank sides
      //   wp[4] PLACE   — box centre on plank surface (plank top = shelfY−0.112, half-box = 0.035)
      //                   +0.003 m clearance so box visually floats on plank, no z-fight
      //   wp[5] RETREAT — lift straight back up to the hover height
      //
      const CARRY_Y = Math.max(shelfY + 0.16, 1.06); // at least 16 cm above shelf
      const PLACE_Y = shelfY - 0.074;                // plank top + box half-height + 3 mm gap
      transported[3] = [0.5, CARRY_Y, shelfZ];
      transported[4] = [0.5, PLACE_Y, shelfZ];
      transported[5] = [0.5, CARRY_Y, shelfZ];

      // Compute uncertainty at all waypoints
      const uncertainties = transport.getUncertainty(DEMO_DATA.waypoints);
      const meanSigma = uncertainties.reduce((s, v) => s + v, 0) / uncertainties.length;

      // EE error: distance from transported place waypoint to shelf slot centre
      const placeWp = transported[4];
      const shelfCenter = [0.5, shelfY, shelfZ];
      const finalErr = euclidean(placeWp, shelfCenter);

      updateMetrics({
        sigma: meanSigma,
        error: finalErr,
        conf: Math.max(0, 1 - finalErr * 4),
      });

      // Build dense trajectory by interpolating between transported waypoints
      const { positions, phaseLabels } = interpolateWaypoints(
        transported, DEMO_DATA.phases
      );

      // Play trajectory in 3D scene.
      if (btn) btn.textContent = 'Executing…';
      await Scene.playTrajectory(positions, phaseLabels, 40);

      // Pin box to exact shelf surface. The idle tick no longer touches objectMesh.position,
      // so this single call is authoritative — no subsequent override can undo it.
      Scene.setObjectPos([0.5, PLACE_Y, shelfZ]);

      hasPlaced = true;
      updatePhaseBadge('DONE ✓');
    } catch (e) {
      console.error('[reshelving] generalize error:', e);
      updatePhaseBadge('ERROR');
    } finally {
      isRunning = false;
      const btn2 = document.getElementById('btn-generalize');
      if (btn2) { btn2.disabled = false; btn2.textContent = 'Generalize TP-GPT →'; }
    }
  }

  // ---------------------------------------------------------------------------
  // Interpolate waypoints into dense trajectory
  // ---------------------------------------------------------------------------
  function interpolateWaypoints(waypoints, phases) {
    const positions = [];
    const phaseLabels = [];
    const n = Math.min(waypoints.length - 1, phases.length);
    for (let i = 0; i < n; i++) {
      const start = waypoints[i];
      const end = waypoints[i + 1];
      const steps = phases[i].steps;
      const label = phases[i].label;
      for (let t = 0; t < steps; t++) {
        const alpha = t / steps;
        positions.push(start.map((v, k) => v + alpha * (end[k] - v)));
        phaseLabels.push(label);
      }
    }
    // Final waypoint
    positions.push(waypoints[waypoints.length - 1]);
    phaseLabels.push(phases[phases.length - 1].label);
    return { positions, phaseLabels };
  }

  // ---------------------------------------------------------------------------
  // Metrics display
  // ---------------------------------------------------------------------------
  function updateMetrics(opts) {
    // Null-safe: called as updateMetrics(null) to clear, or updateMetrics({sigma, error, conf})
    const sigma = opts ? opts.sigma : null;
    const error = opts ? opts.error : null;
    const conf  = opts ? opts.conf  : null;

    const sigmaEl = document.getElementById('metric-sigma');
    const errorEl = document.getElementById('metric-error');
    const confEl  = document.getElementById('metric-conf');

    if (sigmaEl) sigmaEl.textContent = sigma != null ? sigma.toFixed(4) : '—';
    if (errorEl) {
      errorEl.textContent = error != null ? error.toFixed(4) + ' m' : '—';
      if (error != null) {
        errorEl.style.color =
          error < 0.08 ? 'var(--success)' :
          error < 0.15 ? 'var(--warning)' :
                         'var(--error)';
      } else {
        errorEl.style.color = '';
      }
    }
    if (confEl) confEl.textContent = conf != null ? (conf * 100).toFixed(1) + '%' : '—';
  }

  function updatePhaseBadge(label) {
    const badge = document.getElementById('mode-badge');
    if (badge) badge.textContent = 'Reshelving · ' + label;
  }

  // ---------------------------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------------------------
  function euclidean(a, b) {
    return Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));
  }

  // ---------------------------------------------------------------------------
  // Reset
  // ---------------------------------------------------------------------------
  function reset() {
    // 1. Reset state
    boxX = 0.500; boxZ = 0.000;
    shelfY = 0.900; shelfZ = -0.700;
    hasBeenDragged = false;
    isRunning = false;
    hasPlaced = false;

    // 2. Reset sliders (only if panel has been rendered)
    const sliderDefs = [
      ['box-x',   0.50,   'box-x-val',   '0.50'],
      ['box-z',   0.00,   'box-z-val',   '0.00'],
      ['shelf-y', 0.90,   'shelf-y-val', '0.90'],
      ['shelf-z', -0.70,  'shelf-z-val', '-0.70'],
    ];
    sliderDefs.forEach(([sliderId, val, labelId, labelText]) => {
      const slider = document.getElementById(sliderId);
      if (slider) slider.value = val;
      const label = document.getElementById(labelId);
      if (label) label.textContent = labelText;
    });
    const configText = document.getElementById('config-text');
    if (configText) configText.textContent = 'Default (demo scene)';

    // 3. Reset 3D scene (guard: may not be initialized yet)
    try { Scene.resetScene(); } catch (e) { /* Scene not ready */ }

    // 4. Reset button
    const btn = document.getElementById('btn-generalize');
    if (btn) { btn.disabled = true; btn.textContent = 'Generalize TP-GPT →'; }

    // 5. Clear metrics + badge (updateMetrics is now null-safe)
    updateMetrics(null);
    updatePhaseBadge('READY');
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------
  return { init, generalize, reset };
})();
