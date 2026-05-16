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
  let joints = [];  // 7 THREE.Group pivot points
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

  // Franka Panda hardware joint limits (radians)
  const JOINT_LIMITS = [
    [-2.8973,  2.8973],  // q0 — base rotation
    [-1.7628,  1.7628],  // q1 — shoulder pitch
    [-2.8973,  2.8973],  // q2 — upper-arm rotation
    [-3.0718, -0.0698],  // q3 — elbow (always bent, negative)
    [-2.8973,  2.8973],  // q4 — forearm rotation
    [-0.0175,  3.7525],  // q5 — wrist pitch
    [-2.8973,  2.8973],  // q6 — hand rotation
  ];

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

    // EE indicator sphere at tip of last link — subtle during idle, brightens during trajectory
    eeSphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.020, 16, 16),
      new THREE.MeshStandardMaterial({
        color: 0x00d4ff,
        emissive: 0x00d4ff,
        emissiveIntensity: 0.25,  // subtle during idle — not a blinding orb
        transparent: true,
        opacity: 0.80,
      }),
    );
    eeSphere.position.y = LINK_LENGTHS[6] + 0.04;
    joints[6].add(eeSphere);

    // Soft EE point light — just enough to hint at the end-effector
    const eeLight = new THREE.PointLight(0x00d4ff, 0.15, 0.3);
    eeSphere.add(eeLight);

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
  // Jacobian IK — accurate end-effector positioning via Damped Least Squares
  // Mirrors buildFrankaArm() chain exactly: no Three.js objects, pure math.
  // ---------------------------------------------------------------------------

  // Link lengths along local +Y for each joint step.
  // Index 6 includes the eeSphere offset (LINK_LENGTHS[6] + 0.04 = 0.13).
  const _FK_LINKS = [0.20, 0.28, 0.05, 0.28, 0.22, 0.10, 0.13];

  // 3×3 rotation about local Y
  function _Ry(t) {
    const c = Math.cos(t), s = Math.sin(t);
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
  }

  // 3×3 rotation about local Z
  function _Rz(t) {
    const c = Math.cos(t), s = Math.sin(t);
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
  }

  // 3×3 matrix multiply
  function _m3mul(A, B) {
    const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        for (let k = 0; k < 3; k++)
          C[i][j] += A[i][k] * B[k][j];
    return C;
  }

  // Apply 3×3 rotation to a 3-vector
  function _m3v(R, v) {
    return [
      R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
      R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
      R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
    ];
  }

  // 3×3 matrix inverse via Cramer's rule — safe for well-conditioned 3×3
  function _inv3(M) {
    const [[a, b, c], [d, e, f], [g, h, i]] = M;
    const det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
    if (Math.abs(det) < 1e-12) return [[1,0,0],[0,1,0],[0,0,1]];
    const s = 1.0 / det;
    return [
      [(e*i-f*h)*s, (c*h-b*i)*s, (b*f-c*e)*s],
      [(f*g-d*i)*s, (a*i-c*g)*s, (c*d-a*f)*s],
      [(d*h-e*g)*s, (b*g-a*h)*s, (a*e-b*d)*s],
    ];
  }

  // Forward kinematics — returns world-space EE position [x, y, z].
  // Chain: armGroup(0,0.75,0) → +0.10(base) → joint[0..6] rotations + links
  function _fk(angles) {
    let p = [0, 0.85, 0];  // armGroup.y=0.75 + joint[0] offset=0.10
    let R = [[1,0,0],[0,1,0],[0,0,1]];
    for (let i = 0; i < 7; i++) {
      const Rj = JOINT_AXES[i] === 'y' ? _Ry(angles[i]) : _Rz(angles[i]);
      R = _m3mul(R, Rj);
      const v = _m3v(R, [0, _FK_LINKS[i], 0]);
      p = [p[0]+v[0], p[1]+v[1], p[2]+v[2]];
    }
    return p;
  }

  // Numerical Jacobian — 3×7 matrix (rows=xyz, cols=joints)
  function _jac(angles) {
    const eps = 0.001;
    const ee0 = _fk(angles);
    const J = [[], [], []];
    for (let j = 0; j < 7; j++) {
      const qp = [...angles];
      qp[j] += eps;
      const ee1 = _fk(qp);
      J[0].push((ee1[0]-ee0[0]) / eps);
      J[1].push((ee1[1]-ee0[1]) / eps);
      J[2].push((ee1[2]-ee0[2]) / eps);
    }
    return J;
  }

  // Damped Least Squares IK (Levenberg–Marquardt).
  // Warm-start from q0; iterates until EE error < tol or maxIter reached.
  // dq = J^T (J J^T + λ²I)^{-1} · err
  function _solveIK(target, q0, maxIter = 14, tol = 0.004) {
    const lambda = 0.06;   // damping — raises stability in near-singular configs
    const l2 = lambda * lambda;
    const maxStep = 0.25;  // max radians per joint per iteration
    let q = [...q0];

    for (let iter = 0; iter < maxIter; iter++) {
      const ee  = _fk(q);
      const err = [target[0]-ee[0], target[1]-ee[1], target[2]-ee[2]];
      if (err[0]**2 + err[1]**2 + err[2]**2 < tol*tol) break;

      const J = _jac(q);

      // JJT = J · J^T  (3×3)
      const JJT = [[0,0,0],[0,0,0],[0,0,0]];
      for (let i = 0; i < 3; i++)
        for (let k = 0; k < 3; k++)
          for (let j = 0; j < 7; j++)
            JJT[i][k] += J[i][j] * J[k][j];

      // M = JJT + λ²I,  then M^{-1} · err
      const M = JJT.map((row, i) => row.map((v, k) => v + (i===k ? l2 : 0)));
      const Minv = _inv3(M);
      const Me = [0, 0, 0];
      for (let i = 0; i < 3; i++)
        for (let k = 0; k < 3; k++)
          Me[i] += Minv[i][k] * err[k];

      // dq = J^T · Me,  clamped + joint-limited
      for (let j = 0; j < 7; j++) {
        let dqj = 0;
        for (let i = 0; i < 3; i++) dqj += J[i][j] * Me[i];
        q[j] = Math.max(JOINT_LIMITS[j][0],
               Math.min(JOINT_LIMITS[j][1],
               q[j] + Math.max(-maxStep, Math.min(maxStep, dqj))));
      }
    }
    return q;
  }

  // ---------------------------------------------------------------------------
  // Trajectory playback — W3
  // positions: (N, 3) world-space EE positions (Three.js Y-up)
  // phaseLabels: (N,) string label per frame
  // Uses Jacobian IK so the arm genuinely reaches each EE target.
  // Warm-starts each frame from the previous frame's joint angles for smooth motion.
  // ---------------------------------------------------------------------------
  async function playTrajectory(positions, phaseLabels, dt = 50) {
    stopIdle();
    const badge = document.getElementById('mode-badge');

    // Brighten EE sphere during active execution
    if (eeSphere) {
      eeSphere.material.emissiveIntensity = 1.0;
      eeSphere.material.opacity = 0.95;
    }

    // Warm-start IK from current arm pose
    let currentAngles = [...HOME_ANGLES];

    for (let i = 0; i < positions.length; i++) {
      const pos = positions[i];
      const label = (phaseLabels && phaseLabels[i]) || '';

      // Solve IK — warm-started from previous frame → smooth joint motion
      currentAngles = _solveIK(pos, currentAngles);
      setArmPose(currentAngles);

      // Update mode badge
      if (badge && label) badge.textContent = 'Reshelving · ' + label;

      // Box follows EE from GRASP onwards (EE is at box center — offset 0).
      // GRASP: arm is at box, box stays put visually (they coincide).
      // LIFT:  arm rises, box rises with it → visible pickup.
      // CARRY/PLACE: box moves with arm to shelf.
      // RETREAT: not included → box stays on shelf as arm pulls away.
      if (objectMesh && (label === 'GRASP' || label === 'LIFT' || label === 'CARRY' || label === 'PLACE')) {
        objectMesh.position.set(pos[0], pos[1], pos[2]);
      }

      controls.update();
      renderer.render(threeScene, camera);
      await new Promise((r) => setTimeout(r, dt));
    }

    // Restore EE sphere to idle appearance
    if (eeSphere) {
      eeSphere.material.emissiveIntensity = 0.25;
      eeSphere.material.opacity = 0.80;
    }

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

    // Restore EE sphere to idle appearance (may have been brightened during trajectory)
    if (eeSphere) {
      eeSphere.material.emissiveIntensity = 0.25;
      eeSphere.material.opacity = 0.80;
    }

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
