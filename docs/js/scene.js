// scene.js — Three.js 3D scene for teach-once
// TP-GPT interactive demo · Franzese et al. 2024
// W2: Franka arm FK, table, shelf, orange box, lighting, idle animation

/* global THREE */

const Scene = (() => {
  'use strict';

  // ---------------------------------------------------------------------------
  // Private state
  // ---------------------------------------------------------------------------
  let renderer, camera, threeScene, controls;
  let armGroup, tableGroup, shelfGroup, objectMesh, eeSphere;
  let eeWorldMarker = null;  // world-space sphere used during playTrajectory
  let joints = [];           // 7 THREE.Group pivot points
  let animationId = null;
  let idleRunning = false;
  let canvas;

  // ---------------------------------------------------------------------------
  // Arm parameters (simplified Franka Panda, visual only)
  // ---------------------------------------------------------------------------
  const LINK_LENGTHS = [0.20, 0.28, 0.05, 0.28, 0.22, 0.10, 0.09];
  const LINK_RADII   = [0.055, 0.050, 0.048, 0.048, 0.042, 0.038, 0.030];
  const LINK_COLORS  = [
    0xeff0f0, 0xeff0f0, 0xe4e4e8, 0xe4e4e8,
    0xeff0f0, 0xe0e0e4, 0xd8d8de,
  ];
  // Joint rotation axes in Three.js convention (Y = world-up spin, Z = pitch)
  const JOINT_AXES  = ['y', 'z', 'y', 'z', 'y', 'z', 'y'];
  const JOINT_COLOR = 0x3a3a50;

  // Standard Franka Panda home pose (radians)
  const HOME_ANGLES = [0, -0.785, 0, -2.356, 0, 1.571, 0.785];

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  /** Canvas texture for the checker floor. */
  function createCheckerTexture(c1, c2, size, squares) {
    const cv = document.createElement('canvas');
    cv.width = cv.height = size;
    const ctx = cv.getContext('2d');
    const step = size / squares;
    for (let r = 0; r < squares; r++) {
      for (let c = 0; c < squares; c++) {
        ctx.fillStyle = (r + c) % 2 === 0 ? c1 : c2;
        ctx.fillRect(c * step, r * step, step, step);
      }
    }
    const tex = new THREE.CanvasTexture(cv);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    return tex;
  }

  /** Standard material for Franka white plastic links. */
  function linkMat(color) {
    return new THREE.MeshStandardMaterial({
      color,
      roughness: 0.25,
      metalness: 0.55,
    });
  }

  // ---------------------------------------------------------------------------
  // Scene geometry builders
  // ---------------------------------------------------------------------------

  function buildFloor() {
    const geo = new THREE.PlaneGeometry(5, 5);
    const mat = new THREE.MeshLambertMaterial({
      map: createCheckerTexture('#111118', '#1a1a28', 512, 10),
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.receiveShadow = true;
    threeScene.add(mesh);
  }

  function buildWalls() {
    const wallMat = new THREE.MeshLambertMaterial({ color: 0x0d0d1a });
    // Back wall
    const backWall = new THREE.Mesh(new THREE.PlaneGeometry(5, 3), wallMat);
    backWall.position.set(0, 1.5, -1.5);
    threeScene.add(backWall);
    // Left wall
    const leftWall = new THREE.Mesh(new THREE.PlaneGeometry(3, 3), wallMat);
    leftWall.position.set(-1.5, 1.5, 0);
    leftWall.rotation.y = Math.PI / 2;
    threeScene.add(leftWall);
  }

  function buildTable() {
    tableGroup = new THREE.Group();
    const topMat = new THREE.MeshStandardMaterial({
      color: 0x4a3526, roughness: 0.6, metalness: 0.05,
    });
    const legMat = new THREE.MeshStandardMaterial({
      color: 0x3d2b1f, roughness: 0.8, metalness: 0.0,
    });

    // Table top: 0.6 w × 0.02 h × 0.4 d, top surface at y = 0.75
    const topGeo = new THREE.BoxGeometry(0.6, 0.02, 0.4);
    const top = new THREE.Mesh(topGeo, topMat);
    top.position.set(0, 0.74, 0);   // top face sits at y = 0.75
    top.castShadow = true;
    top.receiveShadow = true;
    tableGroup.add(top);

    // Four legs (0.04 × 0.75 × 0.04)
    const legH = 0.74;
    const legGeo = new THREE.BoxGeometry(0.04, legH, 0.04);
    [[-1, -1], [-1, 1], [1, -1], [1, 1]].forEach(([sx, sz]) => {
      const leg = new THREE.Mesh(legGeo, legMat);
      leg.position.set(sx * 0.27, legH / 2, sz * 0.17);
      leg.castShadow = true;
      tableGroup.add(leg);
    });

    // Table is centered at x=0.5 in world, centered at z=0
    tableGroup.position.set(0.5, 0, 0);
    threeScene.add(tableGroup);
  }

  function buildShelf() {
    shelfGroup = new THREE.Group();
    const shelfMat = new THREE.MeshStandardMaterial({
      color: 0x6b4c2a, roughness: 0.9, metalness: 0.0,
    });
    const slotMat = new THREE.MeshStandardMaterial({
      color: 0x00ff88,
      roughness: 0.5,
      transparent: true,
      opacity: 0.7,
      emissive: 0x00ff88,
      emissiveIntensity: 0.35,
    });

    // Two horizontal planks (0.5 w × 0.016 h × 0.16 d)
    [-0.12, 0.12].forEach((dy) => {
      const plank = new THREE.Mesh(
        new THREE.BoxGeometry(0.50, 0.016, 0.16), shelfMat,
      );
      plank.position.y = dy;
      plank.castShadow = true;
      shelfGroup.add(plank);
    });

    // Two vertical supports (0.016 w × 0.28 h × 0.16 d)
    [-0.24, 0.24].forEach((dx) => {
      const sup = new THREE.Mesh(
        new THREE.BoxGeometry(0.016, 0.28, 0.16), shelfMat,
      );
      sup.position.x = dx;
      sup.castShadow = true;
      shelfGroup.add(sup);
    });

    // Green goal slot marker (front face)
    const slot = new THREE.Mesh(
      new THREE.BoxGeometry(0.12, 0.10, 0.004), slotMat,
    );
    slot.position.set(0, -0.01, -0.082);
    shelfGroup.add(slot);

    // Shelf world position — on back wall, elevated
    // MuJoCo: (0.0, 0.7, 0.9) [Z-up] → Three.js: (0.5, 0.9, -0.7)
    shelfGroup.position.set(0.5, 0.9, -0.7);
    threeScene.add(shelfGroup);
  }

  function buildObject() {
    const geo = new THREE.BoxGeometry(0.07, 0.07, 0.07);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xff6b1a,
      roughness: 0.65,
      metalness: 0.1,
      emissive: 0xff6b1a,
      emissiveIntensity: 0.08,
    });
    objectMesh = new THREE.Mesh(geo, mat);
    objectMesh.castShadow = true;
    // On table surface: table top at y=0.75 in world → object center at y=0.785
    objectMesh.position.set(0.5, 0.785, 0.0);
    threeScene.add(objectMesh);
  }

  function buildFrankaArm() {
    armGroup = new THREE.Group();
    joints = [];

    // Base disc — mounted on table top (world y=0.75)
    const baseGeo = new THREE.CylinderGeometry(0.065, 0.075, 0.10, 24);
    const baseMat = new THREE.MeshStandardMaterial({
      color: 0xcccccc, roughness: 0.4, metalness: 0.55,
    });
    const baseMesh = new THREE.Mesh(baseGeo, baseMat);
    baseMesh.position.y = 0.05;
    baseMesh.castShadow = true;
    armGroup.add(baseMesh);

    // Build 7-joint chain
    let parentGroup = armGroup;
    let prevLen = 0.10;  // top of base disc

    for (let i = 0; i < 7; i++) {
      const pivot = new THREE.Group();
      pivot.position.y = prevLen;  // at tip of previous link
      parentGroup.add(pivot);
      joints.push(pivot);

      // Joint sphere
      const jSphere = new THREE.Mesh(
        new THREE.SphereGeometry(LINK_RADII[i] * 1.18, 16, 16),
        new THREE.MeshStandardMaterial({
          color: JOINT_COLOR, roughness: 0.4, metalness: 0.7,
        }),
      );
      jSphere.castShadow = true;
      pivot.add(jSphere);

      // Link cylinder extending along local +Y
      const len = LINK_LENGTHS[i];
      const cyl = new THREE.Mesh(
        new THREE.CylinderGeometry(
          LINK_RADII[i] * 0.90,
          LINK_RADII[i],
          len,
          16,
        ),
        linkMat(LINK_COLORS[i]),
      );
      cyl.position.y = len / 2;
      cyl.castShadow = true;
      pivot.add(cyl);

      parentGroup = pivot;
      prevLen = len;
    }

    // EE glow sphere at tip of last link
    eeSphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.022, 16, 16),
      new THREE.MeshStandardMaterial({
        color: 0x00d4ff,
        emissive: 0x00d4ff,
        emissiveIntensity: 1.2,
        transparent: true,
        opacity: 0.92,
      }),
    );
    eeSphere.position.y = LINK_LENGTHS[6] + 0.04;
    joints[6].add(eeSphere);

    // Small EE point light for glow effect
    const eeLight = new THREE.PointLight(0x00d4ff, 0.5, 0.4);
    eeSphere.add(eeLight);

    // World-space EE trajectory marker — shown during playTrajectory
    eeWorldMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.025, 16, 16),
      new THREE.MeshStandardMaterial({
        color: 0x00d4ff,
        emissive: 0x00d4ff,
        emissiveIntensity: 1.5,
        transparent: true,
        opacity: 0.88,
      }),
    );
    eeWorldMarker.visible = false;
    const eeWorldLight = new THREE.PointLight(0x00d4ff, 0.8, 0.6);
    eeWorldMarker.add(eeWorldLight);
    threeScene.add(eeWorldMarker);

    // Arm base sits on the table surface (world y=0.75)
    armGroup.position.set(0.0, 0.75, 0.0);
    threeScene.add(armGroup);
  }

  function buildLighting() {
    // Ambient
    threeScene.add(new THREE.AmbientLight(0x334466, 0.65));

    // Key light — warm, upper left, casts shadows
    const key = new THREE.DirectionalLight(0xfff5e0, 1.3);
    key.position.set(-2, 3, 4);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 20;
    key.shadow.camera.left = -3;
    key.shadow.camera.right = 3;
    key.shadow.camera.top = 3;
    key.shadow.camera.bottom = -3;
    key.shadow.bias = -0.001;
    threeScene.add(key);

    // Fill light — cool blue, right side
    const fill = new THREE.DirectionalLight(0x4466ff, 0.30);
    fill.position.set(3, 1, -1);
    threeScene.add(fill);

    // Rim light — accent color on arm
    const rim = new THREE.DirectionalLight(0x00d4ff, 0.18);
    rim.position.set(0, -2, 2);
    threeScene.add(rim);
  }

  // ---------------------------------------------------------------------------
  // Arm pose
  // ---------------------------------------------------------------------------

  function setArmPose(angles) {
    if (!joints || joints.length === 0) return;
    for (let i = 0; i < Math.min(7, angles.length, joints.length); i++) {
      if (!joints[i]) continue;
      const ax = JOINT_AXES[i];
      if (ax === 'y') {
        joints[i].rotation.y = angles[i];
        joints[i].rotation.z = 0;
      } else {
        joints[i].rotation.z = angles[i];
        joints[i].rotation.y = 0;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Idle animation
  // ---------------------------------------------------------------------------

  let _idleT0 = 0;

  function idleTick(ts) {
    if (!idleRunning || !renderer || !threeScene || !camera) return;
    const t = ts * 0.001;
    const angles = [...HOME_ANGLES];
    angles[1] += Math.sin(t * 0.45) * 0.05;
    angles[3] += Math.sin(t * 0.30) * 0.04;
    angles[5] += Math.sin(t * 0.55) * 0.025;
    setArmPose(angles);

    // Subtle box hover (y oscillation, cosmetic only)
    if (objectMesh) {
      objectMesh.position.y = 0.785 + Math.sin(t * 0.8) * 0.0025;
    }

    controls.update();
    renderer.render(threeScene, camera);
    animationId = requestAnimationFrame(idleTick);
  }

  function startIdle() {
    if (idleRunning) return;
    idleRunning = true;
    animationId = requestAnimationFrame(idleTick);
  }

  function stopIdle() {
    idleRunning = false;
    if (animationId !== null) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Resize
  // ---------------------------------------------------------------------------

  function onResize() {
    if (!renderer || !canvas) return;
    const wrapper = canvas.parentElement;
    const w = wrapper.clientWidth;
    const h = Math.round(w * 9 / 16);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  function init(canvasId) {
    canvas = document.getElementById(canvasId);
    if (!canvas) { console.error('scene.js: canvas not found:', canvasId); return; }

    // Renderer
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.outputEncoding = THREE.sRGBEncoding;

    // Size from wrapper (CSS aspect-ratio: 16/9)
    const wrapper = canvas.parentElement;
    const w = wrapper.clientWidth || 800;
    const h = Math.round(w * 9 / 16);
    renderer.setSize(w, h, false);

    // Scene
    threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color(0x0a0a0f);
    threeScene.fog = new THREE.FogExp2(0x0a0a0f, 0.18);

    // Camera
    camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 50);
    // Spec positions converted to Three.js Y-up:
    // MuJoCo (1.4, -0.8, 1.3) → Three.js (1.4, 1.3, 0.8)
    camera.position.set(1.4, 1.3, 0.8);
    // Always set lookAt so the scene renders correctly even if OrbitControls fails
    camera.lookAt(0.35, 0.85, 0.0);

    // OrbitControls — graceful fallback if CDN misses
    try {
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.target.set(0.35, 0.85, 0.0);
      controls.minDistance = 0.5;
      controls.maxDistance = 4.0;
      controls.maxPolarAngle = Math.PI * 0.85;
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.enablePan = false;
      controls.update();
    } catch (e) {
      console.warn('[scene] OrbitControls unavailable:', e.message);
      controls = { update: () => {}, target: new THREE.Vector3(0.35, 0.85, 0) };
    }

    // Build scene
    buildLighting();
    buildFloor();
    buildWalls();
    buildTable();
    buildShelf();
    buildObject();
    buildFrankaArm();

    // Apply home pose
    setArmPose(HOME_ANGLES);

    // Resize handler
    window.addEventListener('resize', onResize);

    // Draw one frame immediately so canvas is never black during first load
    renderer.render(threeScene, camera);

    // Start idle
    startIdle();

    console.log('teach-once scene loaded');
  }

  function loadScene(task) {
    // In W3/W4 this will reconfigure objects per task.
    // For W2 the default reshelving layout is always shown.
    console.log('Scene task:', task);
  }

  function setObjectPos(pos) {
    if (objectMesh) objectMesh.position.set(pos[0], pos[1], pos[2]);
  }

  function setShelfPos(pos) {
    if (shelfGroup) shelfGroup.position.set(pos[0], pos[1], pos[2]);
  }

  function highlightObject(name, color) {
    if (name === 'object' && objectMesh) {
      objectMesh.material.emissive.set(color);
      objectMesh.material.emissiveIntensity = 0.4;
    }
  }

  // ---------------------------------------------------------------------------
  // Visual IK approximation — makes the arm reach toward a world-space target
  // Not mathematically exact; purely cosmetic for browser demo
  // ---------------------------------------------------------------------------
  function _visualIKApprox(target) {
    if (!joints || joints.length < 7 || !armGroup) return;
    const [tx, ty, tz] = target;

    // Horizontal offset from arm base (armGroup at world y=0.75, x=0, z=0)
    const dx = tx; // arm base at x=0
    const dz = tz; // arm base at z=0
    const dy = ty - 0.75; // vertical above mount

    // Joint 0 (Y rotation): yaw arm to face target in XZ plane
    // When rot.y=0, arm's reach is along +X after shoulder pitch
    joints[0].rotation.y = Math.atan2(-dz, Math.max(0.01, dx));

    // Joint 1 (Z rotation, shoulder pitch): blend toward target height+distance
    const horizDist = Math.sqrt(dx * dx + dz * dz);
    const targetPitch = -Math.atan2(dy, Math.max(0.15, horizDist) * 0.85);
    joints[1].rotation.z = HOME_ANGLES[1] * 0.35 + targetPitch * 0.65;
    joints[1].rotation.y = 0;

    // Joint 3 (Z rotation, elbow): extend/contract based on reach distance
    const reach = Math.sqrt(dx * dx + dz * dz + dy * dy);
    joints[3].rotation.z = HOME_ANGLES[3] + Math.min(1.4, reach * 0.8) * 0.45;
    joints[3].rotation.y = 0;

    // Keep remaining joints near home
    joints[2].rotation.y = HOME_ANGLES[2];  joints[2].rotation.z = 0;
    joints[4].rotation.y = HOME_ANGLES[4];  joints[4].rotation.z = 0;
    joints[5].rotation.z = HOME_ANGLES[5];  joints[5].rotation.y = 0;
    joints[6].rotation.y = HOME_ANGLES[6];  joints[6].rotation.z = 0;
  }

  // ---------------------------------------------------------------------------
  // Trajectory playback — W3
  // positions: (N, 3) world-space EE positions (Three.js Y-up)
  // phaseLabels: (N,) string label per frame
  // ---------------------------------------------------------------------------
  async function playTrajectory(positions, phaseLabels, dt = 50) {
    stopIdle();
    const badge = document.getElementById('mode-badge');

    // Show world-space EE marker, dim arm's own EE sphere
    if (eeWorldMarker) eeWorldMarker.visible = true;
    if (eeSphere) eeSphere.visible = false;

    // Remember original box y (table surface)
    const origBoxY = objectMesh ? objectMesh.position.y : 0.785;

    for (let i = 0; i < positions.length; i++) {
      const pos = positions[i];
      const label = (phaseLabels && phaseLabels[i]) || '';

      // Update world-space EE marker
      if (eeWorldMarker) eeWorldMarker.position.set(pos[0], pos[1], pos[2]);

      // Approximate visual IK
      _visualIKApprox(pos);

      // Update mode badge
      if (badge && label) badge.textContent = 'Reshelving · ' + label;

      // Box follows EE during CARRY and PLACE
      if (objectMesh && (label === 'CARRY' || label === 'PLACE')) {
        objectMesh.position.set(pos[0], pos[1] - 0.045, pos[2]);
      }

      controls.update();
      renderer.render(threeScene, camera);
      await new Promise((r) => setTimeout(r, dt));
    }

    // Restore
    if (eeWorldMarker) eeWorldMarker.visible = false;
    if (eeSphere) eeSphere.visible = true;

    setArmPose(HOME_ANGLES);
    startIdle();
  }

  function resetScene() {
    // Stop any running trajectory / idle loop first
    stopIdle();

    // Reset arm, object, shelf
    setArmPose(HOME_ANGLES);
    if (objectMesh) objectMesh.position.set(0.5, 0.785, 0.0);
    if (shelfGroup) shelfGroup.position.set(0.5, 0.9, -0.7);

    // Restore EE marker visibility (may have been left visible by interrupted trajectory)
    if (eeWorldMarker) eeWorldMarker.visible = false;
    if (eeSphere) eeSphere.visible = true;

    // Draw once immediately so the reset is visible right away
    if (renderer && threeScene && camera) renderer.render(threeScene, camera);

    // Restart idle animation
    startIdle();
  }

  function dispose() {
    stopIdle();
    window.removeEventListener('resize', onResize);
    if (renderer) renderer.dispose();
  }

  return {
    init,
    loadScene,
    setArmPose,
    setObjectPos,
    setShelfPos,
    highlightObject,
    playTrajectory,
    resetScene,
    dispose,
    // expose for W3 inspection
    get joints() { return joints; },
    get objectMesh() { return objectMesh; },
    get shelfGroup() { return shelfGroup; },
    get eeSphere() { return eeSphere; },
  };
})();
