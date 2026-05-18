// ui.js — Mode switching, button wiring, fallback GIF toggle
// teach-once · TP-GPT · W4
'use strict';

/* global Scene, ModeReshelving, ModeCleaning, ModeArmpose */

const FALLBACK_GIFS = {
  reshelving: 'assets/gifs/final_reshelving.gif',
  cleaning:   'assets/gifs/final_cleaning.gif',
  armpose:    'assets/gifs/final_armpose.gif',
};

let currentMode    = 'reshelving';
let showingFallback = false;

// ---------------------------------------------------------------------------
// Mode switching
// ---------------------------------------------------------------------------
function switchMode(mode) {
  if (mode === currentMode && !showingFallback) return;

  // Cancel any in-flight generalize() on the departing mode so its async
  // continuation doesn't overwrite the new mode's scene or badge.
  try {
    if (currentMode === 'reshelving' && typeof ModeReshelving !== 'undefined') ModeReshelving.cancel();
    else if (currentMode === 'cleaning'   && typeof ModeCleaning   !== 'undefined') ModeCleaning.cancel();
    else if (currentMode === 'armpose'    && typeof ModeArmpose    !== 'undefined') ModeArmpose.cancel();
  } catch (e) { /* ignore */ }

  currentMode = mode;

  // Update tab active state
  document.querySelectorAll('.mode-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.mode === mode);
  });

  // Update mode badge
  const modeNames = {
    reshelving: 'Mode 1: Reshelving',
    cleaning:   'Mode 2: Cleaning',
    armpose:    'Mode 3: Arm-pose',
  };
  const badge = document.getElementById('mode-badge');
  if (badge) badge.textContent = modeNames[mode] || mode;

  // Stop any active freehand drawing before resetting scene
  try { Scene.disableDrawing(); } catch (e) { /* ignore */ }

  // Reset scene geometry + clear overlays, then load new scene
  try { Scene.resetScene(); } catch (e) { console.warn('[ui] resetScene error:', e); }
  try { Scene.loadScene(mode); } catch (e) { console.warn('[ui] loadScene error:', e); }

  // Reset generalize button BEFORE mode init() runs, so init() can override the state.
  // Reshelving: starts disabled (user must move objects before generalizing).
  // Cleaning + Armpose: init() will enable the button immediately.
  const btn = document.getElementById('btn-generalize');
  if (btn) {
    btn.textContent = 'Generalize TP-GPT →';
    btn.disabled = (mode === 'reshelving');
  }

  // Init mode-specific control panel — may enable/disable button
  try {
    if      (mode === 'reshelving' && typeof ModeReshelving !== 'undefined') ModeReshelving.init();
    else if (mode === 'cleaning'   && typeof ModeCleaning   !== 'undefined') ModeCleaning.init();
    else if (mode === 'armpose'    && typeof ModeArmpose    !== 'undefined') ModeArmpose.init();
  } catch (e) { console.warn('[ui] mode init error:', e); }

  // Update fallback GIF src (preload)
  const gifImg = document.querySelector('#gif-fallback img');
  if (gifImg) gifImg.src = FALLBACK_GIFS[mode] || '';
}

// ---------------------------------------------------------------------------
// GIF fallback toggle
// ---------------------------------------------------------------------------
function setupFallbackToggle() {
  const btn = document.getElementById('btn-fallback');
  if (!btn) return;

  btn.addEventListener('click', () => {
    showingFallback = !showingFallback;
    const wrapper  = document.querySelector('.canvas-wrapper');
    const fallback = document.getElementById('gif-fallback');

    if (showingFallback) {
      if (wrapper)  wrapper.style.display  = 'none';
      if (fallback) fallback.style.display = 'block';
      btn.textContent = 'Show live demo ↑';
    } else {
      if (wrapper)  wrapper.style.display  = 'block';
      if (fallback) fallback.style.display = 'none';
      btn.textContent = 'Show pre-recorded ↓';
    }
  });
}

// ---------------------------------------------------------------------------
// DOMContentLoaded — wire everything
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  console.log('teach-once UI loaded (W5)');

  // Default mode init
  if (typeof ModeReshelving !== 'undefined') {
    ModeReshelving.init();
  }

  // Generalize button — delegates to current mode
  const btnGen = document.getElementById('btn-generalize');
  if (btnGen) {
    btnGen.addEventListener('click', () => {
      if      (currentMode === 'reshelving' && typeof ModeReshelving !== 'undefined') ModeReshelving.generalize();
      else if (currentMode === 'cleaning'   && typeof ModeCleaning   !== 'undefined') ModeCleaning.generalize();
      else if (currentMode === 'armpose'    && typeof ModeArmpose    !== 'undefined') ModeArmpose.generalize();
    });
  }

  // Reset button — delegates to current mode
  const btnReset = document.getElementById('btn-reset');
  if (btnReset) {
    btnReset.addEventListener('click', () => {
      if      (currentMode === 'reshelving' && typeof ModeReshelving !== 'undefined') ModeReshelving.reset();
      else if (currentMode === 'cleaning'   && typeof ModeCleaning   !== 'undefined') ModeCleaning.reset();
      else if (currentMode === 'armpose'    && typeof ModeArmpose    !== 'undefined') ModeArmpose.reset();
    });
  }

  // Mode tabs
  document.querySelectorAll('.mode-tab').forEach(tab => {
    tab.addEventListener('click', e => {
      switchMode(e.currentTarget.dataset.mode);
    });
  });

  // Fallback GIF toggle
  setupFallbackToggle();

  // ---------------------------------------------------------------------------
  // W5 — Scroll-triggered step card animations
  // ---------------------------------------------------------------------------
  (function initScrollAnimations() {
    const cards = document.querySelectorAll('#how-it-works .step-card');
    if (!cards.length) return;
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    }, { threshold: 0.12 });
    cards.forEach(card => observer.observe(card));
  })();

  // ---------------------------------------------------------------------------
  // W5 — BibTeX copy button
  // ---------------------------------------------------------------------------
  const bibBtn = document.getElementById('btn-copy-bib');
  if (bibBtn) {
    bibBtn.addEventListener('click', () => {
      const code = document.querySelector('.bibtex-code');
      if (!code) return;
      navigator.clipboard.writeText(code.textContent)
        .then(() => {
          bibBtn.textContent = 'Copied ✓';
          setTimeout(() => { bibBtn.textContent = 'Copy'; }, 2000);
        })
        .catch(() => {
          bibBtn.textContent = 'Failed';
          setTimeout(() => { bibBtn.textContent = 'Copy'; }, 2000);
        });
    });
  }
});
