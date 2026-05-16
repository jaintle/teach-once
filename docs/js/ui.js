// ui.js — Interactive controls, metrics panel, mode switching
// teach-once · TP-GPT demo · Franzese et al. 2024
// Populated in W3 (reshelving), W4 (cleaning, armpose)
'use strict';

/* global Scene, ModeReshelving */

document.addEventListener('DOMContentLoaded', function () {
  console.log('teach-once UI loaded');

  // Initialize reshelving mode (default)
  if (typeof ModeReshelving !== 'undefined') {
    ModeReshelving.init();
  }

  // Generalize button
  const btnGen = document.getElementById('btn-generalize');
  if (btnGen) {
    btnGen.addEventListener('click', function () {
      if (typeof ModeReshelving !== 'undefined') {
        ModeReshelving.generalize();
      }
    });
  }

  // Reset button
  const btnReset = document.getElementById('btn-reset');
  if (btnReset) {
    btnReset.addEventListener('click', function () {
      if (typeof ModeReshelving !== 'undefined') {
        ModeReshelving.reset();
      } else if (typeof Scene !== 'undefined') {
        Scene.resetScene();
      }
    });
  }

  // Mode tabs — visual switch; W4 will wire cleaning + armpose
  document.querySelectorAll('.mode-tab').forEach(function (tab) {
    tab.addEventListener('click', function (e) {
      document.querySelectorAll('.mode-tab').forEach(function (t) {
        t.classList.remove('active');
      });
      e.currentTarget.classList.add('active');

      const mode = e.currentTarget.dataset.mode;
      const badge = document.getElementById('mode-badge');
      const modeLabels = {
        reshelving: 'Mode 1: Reshelving',
        cleaning:   'Mode 2: Cleaning',
        armpose:    'Mode 3: Arm-pose',
      };
      if (badge) badge.textContent = modeLabels[mode] || mode;

      if (typeof Scene !== 'undefined') Scene.loadScene(mode);

      // W4 will replace this
      if (mode !== 'reshelving') {
        const pb = document.getElementById('panel-body');
        if (pb) pb.innerHTML = '<p class="panel-placeholder">Coming in W4…</p>';
        const btn = document.getElementById('btn-generalize');
        if (btn) btn.disabled = true;
      } else {
        if (typeof ModeReshelving !== 'undefined') ModeReshelving.init();
      }
    });
  });
});
