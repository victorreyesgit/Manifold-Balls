/* ==========================================================================
   Manifold Explorer
   A smooth surface z = f(u,v) = A * sin(k1*u) * cos(k2*v), embedded in R^3
   as X(u,v) = (u, f(u,v), v)  [y is "up" in three.js].

   All differential-geometric objects (tangent vectors, normal, first
   fundamental form, Christoffel symbols, geodesics) are computed from the
   analytic partial derivatives of f, not approximated visually.
   ========================================================================== */

const DOMAIN = 4.5;       // u, v range covered by the rendered surface mesh
const GRID_RES = 90;      // surface mesh resolution (per side)
const GEO_DIRECTIONS = 48; // number of geodesics traced to build the geodesic ball
const GEO_RINGS = 14;      // radial steps per geodesic (resolution of the ball fill)
const AMB_SURF_DIRECTIONS = 64; // angular samples for the ambient-ball/surface curve
const AMB_SURF_RINGS = 14;      // radial steps per direction (resolution of the extrinsic ball fill)
const REF_POINT_RADIUS = 0.045 * 1.5; // reference point c's sphere radius (1.5x the original)
const POS_RANGE = 4.5; // u0,v0 clamp range: the whole rendered manifold (= DOMAIN); must match the slider min/max in index.html

const state = {
  amplitude: 1,
  kx: 0.5,
  ky: 0.5,
  u0: 0.7,
  v0: -0.5,
  ballRadius: 1.0, // shared by the ambient ball, Tangent Ball, Extrinsic ball and Intrinsic ball
  tangentSize: 1.5,   // fixed (no longer a slider)
  normalLength: 1.0,  // fixed (no longer a slider)
  basisLength: 1.0,   // fixed (no longer a slider)
  numSamplePoints: 1000,
  vis: {
    surface: true,
    points: true,
    point: true,
    tangentPlane: false,
    normal: false,
    basis: false,
    geodesicBall: false,
    geodesicBoundary: false,
    ambientBall: true,
    ambientPlaneCircle: false,
    ambientSurfaceCurve: true,
    axes: true,
    grid: false,
  }
};

let needsSurfaceRebuild = true;
let needsIndicatorRebuild = true;

/* ==========================================================================
   Differential geometry: the surface is the graph z = f(u,v) mapped into
   R^3 via X(u,v) = (u, f(u,v), v). All derivatives below are analytic.
   ========================================================================== */

function f(u, v) {
  const { amplitude: A, kx, ky } = state;
  return A * Math.sin(kx * u) * Math.cos(ky * v);
}
function fu(u, v) {
  const { amplitude: A, kx, ky } = state;
  return A * kx * Math.cos(kx * u) * Math.cos(ky * v);
}
function fv(u, v) {
  const { amplitude: A, kx, ky } = state;
  return -A * ky * Math.sin(kx * u) * Math.sin(ky * v);
}
function fuu(u, v) {
  const { amplitude: A, kx, ky } = state;
  return -A * kx * kx * Math.sin(kx * u) * Math.cos(ky * v);
}
function fvv(u, v) {
  const { amplitude: A, kx, ky } = state;
  return -A * ky * ky * Math.sin(kx * u) * Math.cos(ky * v);
}
function fuv(u, v) {
  const { amplitude: A, kx, ky } = state;
  return -A * kx * ky * Math.cos(kx * u) * Math.sin(ky * v);
}

function point3D(u, v) {
  return new THREE.Vector3(u, f(u, v), v);
}

// Tangent vectors Xu = dX/du, Xv = dX/dv
function tangentVectors(u, v) {
  const Fu = fu(u, v), Fv = fv(u, v);
  return {
    Xu: new THREE.Vector3(1, Fu, 0),
    Xv: new THREE.Vector3(0, Fv, 1),
  };
}

// Unit normal, oriented "upward" for a nearly-flat surface
function normalAt(u, v) {
  const { Xu, Xv } = tangentVectors(u, v);
  return new THREE.Vector3().crossVectors(Xv, Xu).normalize();
}

// Orthonormal basis {e1, e2} for the tangent plane (Gram-Schmidt on Xu, Xv)
function orthonormalBasis(u, v) {
  const { Xu, Xv } = tangentVectors(u, v);
  const e1 = Xu.clone().normalize();
  const e2 = Xv.clone().sub(e1.clone().multiplyScalar(Xv.dot(e1))).normalize();
  return { e1, e2 };
}

// First fundamental form (Riemannian metric induced from R^3)
function metricAt(u, v) {
  const Fu = fu(u, v), Fv = fv(u, v);
  return { E: 1 + Fu * Fu, F: Fu * Fv, G: 1 + Fv * Fv };
}

// Gaussian curvature of a Monge patch z = f(u,v)
function gaussianCurvatureAt(u, v) {
  const Fu = fu(u, v), Fv = fv(u, v);
  const Fuu = fuu(u, v), Fuv = fuv(u, v), Fvv = fvv(u, v);
  return (Fuu * Fvv - Fuv * Fuv) / Math.pow(1 + Fu * Fu + Fv * Fv, 2);
}

// Christoffel symbols of the second kind, derived from the metric's
// derivatives (which come from f's second derivatives).
function christoffelAt(u, v) {
  const Fu = fu(u, v), Fv = fv(u, v);
  const Fuu = fuu(u, v), Fuv = fuv(u, v), Fvv = fvv(u, v);
  const E = 1 + Fu * Fu, F = Fu * Fv, G = 1 + Fv * Fv;
  const Eu = 2 * Fu * Fuu, Ev = 2 * Fu * Fuv;
  const Fu_ = Fuu * Fv + Fu * Fuv;   // dF/du
  const Fv_ = Fuv * Fv + Fu * Fvv;   // dF/dv
  const Gu = 2 * Fv * Fuv, Gv = 2 * Fv * Fvv;
  const det = 2 * (E * G - F * F);

  return {
    G1_11: (G * Eu - 2 * F * Fu_ + F * Ev) / det,
    G1_12: (G * Ev - F * Gu) / det,
    G1_22: (2 * G * Fv_ - G * Gu - F * Gv) / det,
    G2_11: (2 * E * Fu_ - E * Ev - F * Eu) / det,
    G2_12: (E * Gu - F * Ev) / det,
    G2_22: (E * Gv - 2 * F * Fv_ + F * Gu) / det,
  };
}

/* ==========================================================================
   Geodesics: solve the geodesic ODE
     u'' + G1_11 u'^2 + 2 G1_12 u'v' + G1_22 v'^2 = 0
     v'' + G2_11 u'^2 + 2 G2_12 u'v' + G2_22 v'^2 = 0
   via RK4, starting from a unit-speed direction in the tangent plane.
   ========================================================================== */

function geodesicDeriv(y) {
  const [u, v, du, dv] = y;
  const c = christoffelAt(u, v);
  const d2u = -(c.G1_11 * du * du + 2 * c.G1_12 * du * dv + c.G1_22 * dv * dv);
  const d2v = -(c.G2_11 * du * du + 2 * c.G2_12 * du * dv + c.G2_22 * dv * dv);
  return [du, dv, d2u, d2v];
}

function rk4Step(y, h) {
  const add = (a, b, s) => a.map((val, i) => val + b[i] * s);
  const k1 = geodesicDeriv(y);
  const k2 = geodesicDeriv(add(y, k1, h / 2));
  const k3 = geodesicDeriv(add(y, k2, h / 2));
  const k4 = geodesicDeriv(add(y, k3, h));
  return y.map((val, i) => val + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

// Given a direction angle theta in the orthonormal tangent basis {e1,e2},
// find the (du,dv) coordinate velocity that is unit-speed under the metric.
function initialVelocity(u0, v0, theta) {
  const { Xu, Xv } = tangentVectors(u0, v0);
  const { e1, e2 } = orthonormalBasis(u0, v0);
  const w = e1.clone().multiplyScalar(Math.cos(theta)).add(e2.clone().multiplyScalar(Math.sin(theta)));
  const { E, F, G } = metricAt(u0, v0);
  const a = Xu.dot(w), b = Xv.dot(w);
  const det = E * G - F * F;
  return [(G * a - F * b) / det, (E * b - F * a) / det];
}

// Trace a geodesic of given arc length from (u0,v0) in direction theta.
// Returns `steps+1` points including the start, spaced by equal arc length.
function traceGeodesic(u0, v0, theta, length, steps) {
  const [du0, dv0] = initialVelocity(u0, v0, theta);
  let y = [u0, v0, du0, dv0];
  const h = length / steps;
  const pts = [{ u: u0, v: v0 }];
  for (let i = 0; i < steps; i++) {
    y = rk4Step(y, h);
    // Renormalize speed under the metric to curb numerical drift.
    const { E, F, G } = metricAt(y[0], y[1]);
    const speed2 = E * y[2] * y[2] + 2 * F * y[2] * y[3] + G * y[3] * y[3];
    if (speed2 > 1e-10) {
      const s = 1 / Math.sqrt(speed2);
      y[2] *= s; y[3] *= s;
    }
    // Geodesics are NOT stopped at the edge of the rendered manifold: f is
    // defined everywhere, so they simply continue, and the per-pixel overlay
    // only ever looks at the part that falls inside the visible surface.
    pts.push({ u: y[0], v: y[1] });
  }
  return pts;
}

/* ==========================================================================
   Ambient ball intersections.

   The ambient ball is a literal Euclidean sphere centered at p. Its
   intersection with the tangent plane is exact and needs no numerics: since
   the plane passes through the sphere's center, the intersection is always
   a circle of radius = ambient radius, lying in the plane (see
   ambientPlaneCircle below).

   Its intersection with the curved surface has no closed form in general,
   so it's found numerically: walk outward along straight rays in (u,v)
   parameter space (NOT geodesics) and bisect for the point where the 3D
   distance to p equals the ambient radius.
   ========================================================================== */

// Distance along the ray (u0,v0)+t*(du,dv) at which it leaves the rendered
// domain [-DOMAIN,DOMAIN]^2.
function rayDomainExitT(u0, v0, du, dv) {
  let t = Infinity;
  const EPS = 1e-9;
  if (du > EPS) t = Math.min(t, (DOMAIN - u0) / du);
  else if (du < -EPS) t = Math.min(t, (-DOMAIN - u0) / du);
  if (dv > EPS) t = Math.min(t, (DOMAIN - v0) / dv);
  else if (dv < -EPS) t = Math.min(t, (-DOMAIN - v0) / dv);
  return Math.max(0, t);
}

function findAmbientSurfaceCrossingT(u0, v0, P, phi, R) {
  const du = Math.cos(phi), dv = Math.sin(phi);
  const maxT = rayDomainExitT(u0, v0, du, dv);
  if (maxT <= 0) return 0;
  // If even the domain edge is within R of c, the ball spills past the
  // visible manifold: clip it exactly at the edge.
  if (point3D(u0 + du * maxT, v0 + dv * maxT).distanceTo(P) <= R) return maxT;
  let tHi = Math.min(0.1, maxT);
  while (tHi < maxT) {
    const d = point3D(u0 + du * tHi, v0 + dv * tHi).distanceTo(P);
    if (d > R) break;
    tHi = Math.min(tHi * 1.5, maxT);
  }
  let lo = 0, hi = tHi;
  for (let i = 0; i < 30; i++) {
    const mid = (lo + hi) / 2;
    const d = point3D(u0 + du * mid, v0 + dv * mid).distanceTo(P);
    if (d < R) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

/* ==========================================================================
   Geodesic-distance map. Shoots GEO_FIELD_DIRS geodesics out of c, each
   sampled at fine arc-length steps, and rasterizes the resulting polar grid
   (a triangle mesh in (u,v), carrying "distance travelled" at each vertex)
   into a GEO_FIELD_SIZE^2 texture covering the whole rendered domain. A
   texel's value is then the geodesic distance from c to that point of the
   manifold (1000 where no geodesic of length <= R reached it). Triangles
   that stick out past the domain edge are simply clipped by the texture
   bounds, so the map is complete right up to the border.
   ========================================================================== */

function computeGeodesicDistanceField(u0, v0, R) {
  const S = GEO_FIELD_SIZE, cell = (2 * DOMAIN) / S;
  const data = geoFieldData;
  for (let i = 0; i < S * S; i++) data[i * 4] = 1000;

  const STEPS = Math.max(8, Math.ceil(R / 0.03));
  const h = R / STEPS;
  const paths = [];
  for (let d = 0; d < GEO_FIELD_DIRS; d++) {
    paths.push(traceGeodesic(u0, v0, (2 * Math.PI * d) / GEO_FIELD_DIRS, R, STEPS));
  }

  const toGrid = (w) => (w + DOMAIN) / cell - 0.5; // world coord -> texel-center coord
  const BARY_EPS = 1e-3;

  function rasterTri(A, dA, B, dB, C, dC) {
    const ax = toGrid(A.u), ay = toGrid(A.v);
    const bx = toGrid(B.u), by = toGrid(B.v);
    const cx = toGrid(C.u), cy = toGrid(C.v);
    const minX = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
    const maxX = Math.min(S - 1, Math.ceil(Math.max(ax, bx, cx)));
    const minY = Math.max(0, Math.floor(Math.min(ay, by, cy)));
    const maxY = Math.min(S - 1, Math.ceil(Math.max(ay, by, cy)));
    if (minX > maxX || minY > maxY) return;
    if ((maxX - minX) * (maxY - minY) > 6000) return; // runaway triangle, ignore
    const den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    if (Math.abs(den) < 1e-12) return;
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const w0 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / den;
        const w1 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / den;
        const w2 = 1 - w0 - w1;
        if (w0 >= -BARY_EPS && w1 >= -BARY_EPS && w2 >= -BARY_EPS) {
          const dist = w0 * dA + w1 * dB + w2 * dC;
          const idx = (y * S + x) * 4;
          if (dist < data[idx]) data[idx] = dist;
        }
      }
    }
  }

  for (let d = 0; d < GEO_FIELD_DIRS; d++) {
    const dn = (d + 1) % GEO_FIELD_DIRS;
    for (let r = 0; r < STEPS; r++) {
      const A = paths[d][r], B = paths[dn][r], C = paths[d][r + 1], D = paths[dn][r + 1];
      const d0 = r * h, d1 = (r + 1) * h;
      rasterTri(A, d0, B, d0, C, d1);
      rasterTri(B, d0, D, d1, C, d1);
    }
  }
  geoFieldTex.needsUpdate = true;
}

const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161c);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 100);
const DEFAULT_CAM_POS = new THREE.Vector3(5.5, 4.5, 6.5);
camera.position.copy(DEFAULT_CAM_POS);

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
dirLight.position.set(6, 10, 4);
scene.add(dirLight);
const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.35);
dirLight2.position.set(-6, -4, -6);
scene.add(dirLight2);

const gridHelper = new THREE.GridHelper(10, 20, 0x2b2f3a, 0x22252e);
scene.add(gridHelper);

/* ---- Surface mesh ---- */
let surfaceGeom = new THREE.BufferGeometry();
const surfaceMat = new THREE.MeshStandardMaterial({
  color: 0x8892a0, side: THREE.DoubleSide, metalness: 0.05, roughness: 0.85,
  transparent: true, opacity: 0.92,
});
const surfaceMesh = new THREE.Mesh(surfaceGeom, surfaceMat);
scene.add(surfaceMesh);

function rebuildSurface() {
  const n = GRID_RES;
  const positions = new Float32Array((n + 1) * (n + 1) * 3);
  let idx = 0;
  for (let i = 0; i <= n; i++) {
    const v = -DOMAIN + (2 * DOMAIN * i) / n;
    for (let j = 0; j <= n; j++) {
      const u = -DOMAIN + (2 * DOMAIN * j) / n;
      positions[idx++] = u;
      positions[idx++] = f(u, v);
      positions[idx++] = v;
    }
  }
  const indices = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const a = i * (n + 1) + j;
      const b = a + 1;
      const c = a + (n + 1);
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }
  surfaceGeom.dispose();
  surfaceGeom = new THREE.BufferGeometry();
  surfaceGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  surfaceGeom.setIndex(indices);
  surfaceGeom.computeVertexNormals();
  surfaceMesh.geometry = surfaceGeom;
  regionMesh.geometry = surfaceGeom; // the ball overlay shares the manifold's mesh
}

/* ---- Sample points: a layer of points scattered on the manifold,
   independent of the movable reference point c. Their (u,v) locations are
   generated once and cached; the count slider just adjusts how many are
   drawn (via setDrawRange), and their height is recomputed only when the
   surface shape changes. ---- */
const MAX_SAMPLE_POINTS = 5000;
const SAMPLE_POINT_LIFT = 0.015; // nudge above the surface to avoid z-fighting with the mesh
const sampleUV = (() => {
  // Cover the FULL rendered domain [-DOMAIN, DOMAIN] so there's no empty
  // border ring between the sample points and the edge of the manifold.
  const arr = [];
  for (let i = 0; i < MAX_SAMPLE_POINTS; i++) {
    arr.push({ u: (Math.random() * 2 - 1) * DOMAIN, v: (Math.random() * 2 - 1) * DOMAIN });
  }
  return arr;
})();

// A small hard-edged circular sprite, so points render as round dots
// instead of the default square/cube-looking point sprites.
function makeCircleTexture() {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 1, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

let samplePointsGeom = new THREE.BufferGeometry();
// Sized to match the reference point's sphere diameter. alphaTest (rather
// than blending) keeps the dots crisp and fully solid-colored.
const samplePointsMat = new THREE.PointsMaterial({
  color: 0x4da6ff, size: REF_POINT_RADIUS * 2, sizeAttenuation: true,
  map: makeCircleTexture(), alphaTest: 0.5,
});
const samplePointsObj = new THREE.Points(samplePointsGeom, samplePointsMat);
scene.add(samplePointsObj);

function rebuildSamplePoints() {
  const positions = new Float32Array(MAX_SAMPLE_POINTS * 3);
  for (let i = 0; i < MAX_SAMPLE_POINTS; i++) {
    const { u, v } = sampleUV[i];
    positions[i * 3] = u;
    positions[i * 3 + 1] = f(u, v) + SAMPLE_POINT_LIFT;
    positions[i * 3 + 2] = v;
  }
  samplePointsGeom.dispose();
  samplePointsGeom = new THREE.BufferGeometry();
  samplePointsGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  samplePointsGeom.setDrawRange(0, state.numSamplePoints);
  samplePointsObj.geometry = samplePointsGeom;
}

/* ==========================================================================
   Color palette: Intrinsic = orange, Extrinsic = purple, Ambient/Tangent
   Ball = blue -- these three are pushed to more saturated, more widely
   separated hues (orange ~28°, blue ~218° deep royal, purple ~275° vivid
   violet) so they stay distinguishable from each other even as translucent
   overlapping fills, which was the problem with the lighter/closer-hued
   version. Tangent plane keeps its own cyan, separate from the Ambient
   ball/Tangent Ball blue (they're a different concept: a plane, not a
   cross-section of the ball).
   ========================================================================== */
const COLOR_REF_POINT = 0xff4d4d;
const COLOR_TANGENT_PLANE = 0x42c7ff;   // the plane itself
const COLOR_AMBIENT_GROUP = 0x2f5dff;   // ambient ball, Tangent Ball -- deep saturated blue
const COLOR_INTRINSIC = 0xff8c2b;       // geodesic ball -- orange
const COLOR_EXTRINSIC = 0xa855f7;       // ambient ball ∩ manifold -- vivid violet-purple
const COLOR_NORMAL = 0xffd23f;
const COLOR_E1 = 0xff5fd1;
const COLOR_E2 = 0x63ff8a;

/* ---- Reference point c ---- */
const pointMesh = new THREE.Mesh(
  new THREE.SphereGeometry(REF_POINT_RADIUS, 20, 20),
  new THREE.MeshStandardMaterial({ color: COLOR_REF_POINT, emissive: 0x551515 })
);
scene.add(pointMesh);

/* ---- Tangent plane ---- */
const tangentGeom = new THREE.PlaneGeometry(1, 1);
const tangentMat = new THREE.MeshBasicMaterial({
  color: COLOR_TANGENT_PLANE, side: THREE.DoubleSide, transparent: true, opacity: 0.28,
  depthWrite: false,
});
const tangentMesh = new THREE.Mesh(tangentGeom, tangentMat);
const tangentEdges = new THREE.LineSegments(
  new THREE.EdgesGeometry(tangentGeom),
  new THREE.LineBasicMaterial({ color: COLOR_TANGENT_PLANE, transparent: true, opacity: 0.7 })
);
tangentMesh.add(tangentEdges);
scene.add(tangentMesh);

/* ---- Arrows: normal + basis ---- */
function makeArrow(color) {
  return new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 1, color, 0.09, 0.05);
}
const normalArrow = makeArrow(COLOR_NORMAL);
const e1Arrow = makeArrow(COLOR_E1);
const e2Arrow = makeArrow(COLOR_E2);
scene.add(normalArrow, e1Arrow, e2Arrow);


/* ---- Ambient (Euclidean) ball ----
   A faint solid translucent sphere, not a wireframe cage: wireframe lines
   compete visually with the manifold grid and the intersection curves, so
   a very low-opacity solid fill reads as "a ball is here" without clutter.
   depthWrite is off so the transparent sphere doesn't occlude things behind
   it in the depth buffer (avoids harsh self-intersection artifacts). */
const ambientGeom = new THREE.SphereGeometry(1, 32, 24);
const ambientMat = new THREE.MeshBasicMaterial({
  color: COLOR_AMBIENT_GROUP, wireframe: false, transparent: true, opacity: 0.12,
  side: THREE.DoubleSide, depthWrite: false,
});
const ambientMesh = new THREE.Mesh(ambientGeom, ambientMat);
scene.add(ambientMesh);

/* ---- Ambient ball intersections: Tangent Ball and Extrinsic ball ----
   Filled discs (not just boundary curves), plus a thin boundary outline
   for crispness. Tangent Ball = ambient ball ∩ tangent plane: a
   cross-section of the ambient ball, so it shares the ambient group's
   blue. Extrinsic ball = ambient ball ∩ manifold: its own vivid purple,
   distinguishable from both the blue (ambient/Tangent Ball) and the
   orange (Intrinsic ball) it's most often compared against. */
let ambPlaneGeom = new THREE.BufferGeometry();
const ambPlaneFillMat = new THREE.MeshBasicMaterial({
  color: COLOR_AMBIENT_GROUP, side: THREE.DoubleSide, transparent: true, opacity: 0.55, depthWrite: false,
});
const ambPlaneMesh = new THREE.Mesh(ambPlaneGeom, ambPlaneFillMat);
scene.add(ambPlaneMesh);

let ambPlaneBoundaryGeom = new THREE.BufferGeometry();
const ambPlaneMat = new THREE.LineBasicMaterial({ color: COLOR_AMBIENT_GROUP, linewidth: 2 });
const ambPlaneLine = new THREE.LineLoop(ambPlaneBoundaryGeom, ambPlaneMat);
scene.add(ambPlaneLine);

/* ---- Intrinsic & Extrinsic balls, painted per pixel ----
   Instead of building each ball as a fan of rays (which tears and leaves
   gaps wherever a ball runs into the edge of the manifold), an overlay
   shares the manifold's own mesh and decides for EVERY PIXEL whether it
   lies inside each ball:
     - Extrinsic: is the straight-line 3D distance to c within R?  (exact)
     - Intrinsic: is the geodesic distance to c within R?  (looked up in a
       distance map computed on the CPU, see computeGeodesicDistanceField)
   Because the overlay IS the manifold surface, regions are clipped by the
   border exactly, and every pixel that should be painted gets painted.
   Where a pixel is in both balls, a diamond (harlequin) pattern of the two
   colors is drawn. */
const GEO_FIELD_SIZE = 256;
const GEO_FIELD_DIRS = 240;
const DIAMOND_TILE = 0.16; // size of one diamond tile, in (u,v) units
const geoFieldData = new Float32Array(GEO_FIELD_SIZE * GEO_FIELD_SIZE * 4);
const geoFieldTex = new THREE.DataTexture(
  geoFieldData, GEO_FIELD_SIZE, GEO_FIELD_SIZE, THREE.RGBAFormat, THREE.FloatType
);
geoFieldTex.minFilter = THREE.NearestFilter;  // bilinear is done by hand in the shader,
geoFieldTex.magFilter = THREE.NearestFilter;  // so float textures need no linear-filter extension
geoFieldTex.generateMipmaps = false;
geoFieldTex.needsUpdate = true;

const regionMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, side: THREE.DoubleSide,
  polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
  extensions: { derivatives: true },
  uniforms: {
    uC: { value: new THREE.Vector3() },
    uR: { value: 1 },
    uDomain: { value: DOMAIN },
    uSize: { value: GEO_FIELD_SIZE },
    uTile: { value: DIAMOND_TILE },
    uShowI: { value: 0 },
    uShowE: { value: 0 },
    uGeo: { value: geoFieldTex },
    uColI: { value: new THREE.Color(COLOR_INTRINSIC) },
    uColE: { value: new THREE.Color(COLOR_EXTRINSIC) },
  },
  vertexShader: `
    varying vec3 vPos;
    void main() {
      vPos = position;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    precision highp float;
    uniform vec3 uC;
    uniform float uR, uDomain, uSize, uTile, uShowI, uShowE;
    uniform sampler2D uGeo;
    uniform vec3 uColI, uColE;
    varying vec3 vPos;

    // Bilinear lookup of the geodesic-distance map at surface coords (u,v).
    float geoDist(vec2 uv) {
      vec2 t = (uv + uDomain) / (2.0 * uDomain) * uSize - 0.5;
      vec2 i0 = floor(t);
      vec2 fr = t - i0;
      vec2 lo = clamp(i0, 0.0, uSize - 1.0);
      vec2 hi = clamp(i0 + 1.0, 0.0, uSize - 1.0);
      float a = texture2D(uGeo, (vec2(lo.x, lo.y) + 0.5) / uSize).r;
      float b = texture2D(uGeo, (vec2(hi.x, lo.y) + 0.5) / uSize).r;
      float c = texture2D(uGeo, (vec2(lo.x, hi.y) + 0.5) / uSize).r;
      float d = texture2D(uGeo, (vec2(hi.x, hi.y) + 0.5) / uSize).r;
      return mix(mix(a, b, fr.x), mix(c, d, fr.x), fr.y);
    }

    void main() {
      vec2 uv = vPos.xz;
      float dE = length(vPos - uC);
      float dI = geoDist(uv);
      bool inE = uShowE > 0.5 && dE <= uR;
      bool inI = uShowI > 0.5 && dI <= uR;
      if (!inE && !inI) discard;

      vec3 col;
      if (inE && inI) {
        vec2 p = fract(uv / uTile);
        col = (abs(p.x - 0.5) + abs(p.y - 0.5) < 0.5) ? uColE : uColI;
      } else if (inI) {
        col = uColI;
      } else {
        col = uColE;
      }

      // Thin outline along each ball's own boundary (constant pixel width).
      float wE = clamp(fwidth(dE) * 1.3, 0.0005, 0.03);
      float wI = clamp(fwidth(dI) * 1.3, 0.0005, 0.03);
      if (uShowE > 0.5) col = mix(col, uColE * 0.65, 1.0 - smoothstep(0.0, wE, abs(dE - uR)));
      if (uShowI > 0.5 && dI < 100.0) col = mix(col, uColI * 0.65, 1.0 - smoothstep(0.0, wI, abs(dI - uR)));

      gl_FragColor = vec4(col, 0.93);
    }
  `,
});
const regionMesh = new THREE.Mesh(surfaceGeom, regionMat);
regionMesh.renderOrder = 1;
scene.add(regionMesh);

/* ---- Orientation axes: a small, unlabeled RGB triad that repositions
   itself to the far corner of the surface (relative to the camera) each
   frame, purely to help read the 3D shape without cluttering the view. ---- */
const axesHelper = new THREE.AxesHelper(3.0);
scene.add(axesHelper);

function updateAxesPlacement() {
  const dir = new THREE.Vector3().subVectors(camera.position, controls.target);
  const signX = dir.x >= 0 ? -1 : 1;
  const signZ = dir.z >= 0 ? -1 : 1;
  const x = signX * DOMAIN * 0.55;
  const z = signZ * DOMAIN * 0.55;
  const y = -Math.max(1.0, Math.abs(state.amplitude) + 0.6);
  axesHelper.position.set(x, y, z);
}

/* ==========================================================================
   Rebuild indicators (point, tangent plane, normal, basis, both balls)
   ========================================================================== */

function rebuildIndicators() {
  const { u0, v0 } = state;
  const P = point3D(u0, v0);
  const N = normalAt(u0, v0);
  const { e1, e2 } = orthonormalBasis(u0, v0);

  pointMesh.position.copy(P);

  // Tangent plane: quad spanned by e1, e2, centered at P.
  // Use e1 x e2 (not the separately-computed N) as the basis's third axis:
  // this guarantees a proper right-handed rotation matrix. Using N directly
  // can produce a left-handed (reflective, det = -1) matrix whenever the
  // Gram-Schmidt {e1,e2} pair happens to be oriented oppositely to N, which
  // makes three.js's quaternion decomposition misbehave (the plane snaps to
  // a wrong orientation). The plane material is double-sided, so the sign
  // of its face normal doesn't matter visually.
  const planeZ = new THREE.Vector3().crossVectors(e1, e2).normalize();
  tangentMesh.position.copy(P);
  const m = new THREE.Matrix4().makeBasis(e1, e2, planeZ);
  tangentMesh.setRotationFromMatrix(m);
  // tangentSize is a half-extent: size S means the square spans -S..+S
  // along e1 and e2 (a 2S x 2S square). The base PlaneGeometry is 1x1
  // (half-width 0.5), so it needs a scale factor of 2*S.
  tangentMesh.scale.set(state.tangentSize * 2, state.tangentSize * 2, 1);

  // Normal arrow
  normalArrow.position.copy(P);
  normalArrow.setDirection(N);
  normalArrow.setLength(state.normalLength, state.normalLength * 0.16, state.normalLength * 0.09);

  // Basis arrows
  e1Arrow.position.copy(P);
  e1Arrow.setDirection(e1);
  e1Arrow.setLength(state.basisLength, state.basisLength * 0.16, state.basisLength * 0.09);
  e2Arrow.position.copy(P);
  e2Arrow.setDirection(e2);
  e2Arrow.setLength(state.basisLength, state.basisLength * 0.16, state.basisLength * 0.09);

  // Ambient ball: literal Euclidean sphere in R^3. All three balls below
  // share this same radius (state.ballRadius) -- the point of the tool is
  // to compare what one radius produces intrinsically vs. extrinsically.
  ambientMesh.position.copy(P);
  ambientMesh.scale.setScalar(state.ballRadius);

  // Tangent Ball = ambient ball ∩ tangent plane. Exact and needs no
  // numerics: since the plane passes through the sphere's center P, this
  // is always a circle of radius = ballRadius, lying in the plane. It's
  // flat, so a single fan from the center to the boundary ring fills it
  // completely -- no need for multiple rings the way a curved patch does.
  {
    const N_AMB = 64;
    const verts = [P.x, P.y, P.z];
    for (let i = 0; i < N_AMB; i++) {
      const theta = (2 * Math.PI * i) / N_AMB;
      const pt = P.clone()
        .add(e1.clone().multiplyScalar(state.ballRadius * Math.cos(theta)))
        .add(e2.clone().multiplyScalar(state.ballRadius * Math.sin(theta)));
      verts.push(pt.x, pt.y, pt.z);
    }
    const faces = [];
    for (let i = 0; i < N_AMB; i++) {
      const a = 1 + i, b = 1 + ((i + 1) % N_AMB);
      faces.push(0, a, b);
    }
    ambPlaneGeom.dispose();
    ambPlaneGeom = new THREE.BufferGeometry();
    ambPlaneGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verts), 3));
    ambPlaneGeom.setIndex(faces);
    ambPlaneGeom.computeVertexNormals();
    ambPlaneMesh.geometry = ambPlaneGeom;

    const boundary = [];
    for (let i = 1; i <= N_AMB; i++) boundary.push(verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]);
    ambPlaneBoundaryGeom.dispose();
    ambPlaneBoundaryGeom = new THREE.BufferGeometry();
    ambPlaneBoundaryGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(boundary), 3));
    ambPlaneLine.geometry = ambPlaneBoundaryGeom;
  }

  // Intrinsic & Extrinsic balls are painted per pixel by regionMat (see the
  // scene setup). Extrinsic needs nothing precomputed (it tests the 3D
  // distance to c directly); Intrinsic needs a geodesic-distance map of the
  // manifold, which is only computed while that ball is switched on.
  if (state.vis.geodesicBall) computeGeodesicDistanceField(u0, v0, state.ballRadius);
  regionMat.uniforms.uC.value.copy(P);
  regionMat.uniforms.uR.value = state.ballRadius;

  // Readout
  const K = gaussianCurvatureAt(u0, v0);
  const { E, F, G } = metricAt(u0, v0);
  document.getElementById('readout').textContent =
    `c = (${u0.toFixed(2)}, ${f(u0, v0).toFixed(2)}, ${v0.toFixed(2)})\n` +
    `Gaussian curvature K = ${K.toFixed(3)}\n` +
    `metric  E=${E.toFixed(2)}  F=${F.toFixed(2)}  G=${G.toFixed(2)}`;
}

function applyVisibility() {
  const v = state.vis;
  surfaceMesh.visible = v.surface;
  pointMesh.visible = v.point;
  tangentMesh.visible = v.tangentPlane;
  normalArrow.visible = v.normal;
  e1Arrow.visible = v.basis;
  e2Arrow.visible = v.basis;
  ambientMesh.visible = v.ambientBall;
  ambPlaneMesh.visible = v.ambientPlaneCircle;
  ambPlaneLine.visible = v.ambientPlaneCircle;
  regionMat.uniforms.uShowI.value = v.geodesicBall ? 1 : 0;
  regionMat.uniforms.uShowE.value = v.ambientSurfaceCurve ? 1 : 0;
  regionMesh.visible = v.surface && (v.geodesicBall || v.ambientSurfaceCurve);
  axesHelper.visible = v.axes;
  gridHelper.visible = v.grid;
  samplePointsObj.visible = v.points;
}

/* ==========================================================================
   UI wiring
   ========================================================================== */

function bindSlider(id, stateKey, decimals, onChange) {
  const input = document.getElementById(id);
  const out = document.getElementById(id + '-val');
  const update = () => {
    const val = parseFloat(input.value);
    state[stateKey] = val;
    out.textContent = val.toFixed(decimals);
    onChange();
  };
  input.addEventListener('input', update);
  update();
}

bindSlider('u0', 'u0', 2, () => { needsIndicatorRebuild = true; });
bindSlider('v0', 'v0', 2, () => { needsIndicatorRebuild = true; });
bindSlider('amp', 'amplitude', 2, () => { needsSurfaceRebuild = true; needsIndicatorRebuild = true; });
bindSlider('kx', 'kx', 2, () => { needsSurfaceRebuild = true; needsIndicatorRebuild = true; });
bindSlider('ky', 'ky', 2, () => { needsSurfaceRebuild = true; needsIndicatorRebuild = true; });
bindSlider('ballR', 'ballRadius', 2, () => { needsIndicatorRebuild = true; });
bindSlider('numPoints', 'numSamplePoints', 0, () => {
  samplePointsObj.geometry.setDrawRange(0, state.numSamplePoints);
});

const visMap = {
  'vis-surface': 'surface', 'vis-points': 'points', 'vis-point': 'point',
  'vis-tangent': 'tangentPlane', 'vis-normal': 'normal', 'vis-basis': 'basis',
  'vis-ambient': 'ambientBall',
  'vis-amb-plane': 'ambientPlaneCircle', 'vis-amb-surf': 'ambientSurfaceCurve',
  'vis-axes': 'axes', 'vis-grid': 'grid',
};
Object.entries(visMap).forEach(([id, key]) => {
  document.getElementById(id).addEventListener('change', (e) => {
    state.vis[key] = e.target.checked;
    applyVisibility();
  });
});

// Intrinsic ball = the geodesic ball's fill + its boundary, shown/hidden
// together as one concept (both already share the orange color).
document.getElementById('vis-intrinsic').addEventListener('change', (e) => {
  state.vis.geodesicBall = e.target.checked;
  state.vis.geodesicBoundary = e.target.checked;
  applyVisibility();
  if (e.target.checked) needsIndicatorRebuild = true; // compute the geodesic-distance map now
});

document.getElementById('reset-point').addEventListener('click', () => {
  state.u0 = 0; state.v0 = 0;
  document.getElementById('u0').value = 0;
  document.getElementById('v0').value = 0;
  document.getElementById('u0-val').textContent = '0.00';
  document.getElementById('v0-val').textContent = '0.00';
  needsIndicatorRebuild = true;
});

document.getElementById('reset-view').addEventListener('click', () => {
  camera.position.copy(DEFAULT_CAM_POS);
  controls.target.set(0, 0, 0);
});

// Mobile panel toggle
const panel = document.getElementById('panel');
document.getElementById('panel-toggle').addEventListener('click', () => {
  panel.classList.toggle('open');
});

// Click-to-place point: raycast against the surface mesh. Because the
// surface is built as X(u,v) = (u, f(u,v), v), the intersection's x and z
// coordinates ARE u and v directly.
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
renderer.domElement.addEventListener('pointerdown', (ev) => {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(surfaceMesh, false)[0];
  if (!hit) return;
  const u = THREE.MathUtils.clamp(hit.point.x, -POS_RANGE, POS_RANGE);
  const v = THREE.MathUtils.clamp(hit.point.z, -POS_RANGE, POS_RANGE);
  state.u0 = u; state.v0 = v;
  document.getElementById('u0').value = u;
  document.getElementById('v0').value = v;
  document.getElementById('u0-val').textContent = u.toFixed(2);
  document.getElementById('v0-val').textContent = v.toFixed(2);
  needsIndicatorRebuild = true;
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

/* ==========================================================================
   Photo / video capture
   ========================================================================== */

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

document.getElementById('capture-photo').addEventListener('click', () => {
  renderer.render(scene, camera); // ensure the frame about to be read is fresh
  renderer.domElement.toBlob((blob) => {
    if (blob) downloadBlob(blob, `manifold-${Date.now()}.png`);
  }, 'image/png');
});

const videoBtn = document.getElementById('capture-video');
const canCaptureVideo = !!(renderer.domElement.captureStream && window.MediaRecorder);

if (!canCaptureVideo) {
  videoBtn.disabled = true;
  videoBtn.title = 'Video recording is not supported in this browser (works in Chrome/Firefox/Edge; limited on iOS Safari).';
  videoBtn.textContent = '⏺ Record Video (unsupported)';
} else {
  let mediaRecorder = null;
  let recordedChunks = [];

  videoBtn.addEventListener('click', () => {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
      const stream = renderer.domElement.captureStream(30);
      let mimeType = 'video/webm;codecs=vp9';
      if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = 'video/webm';
      try {
        mediaRecorder = new MediaRecorder(stream, { mimeType });
      } catch (err) {
        alert('Could not start video recording: ' + err.message);
        return;
      }
      recordedChunks = [];
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        downloadBlob(blob, `manifold-${Date.now()}.webm`);
        videoBtn.textContent = '⏺ Record Video';
        videoBtn.classList.remove('recording');
      };
      mediaRecorder.start();
      videoBtn.textContent = '⏹ Stop Recording';
      videoBtn.classList.add('recording');
    } else {
      mediaRecorder.stop();
    }
  });
}

/* ==========================================================================
   Main loop
   ========================================================================== */

applyVisibility();

function animate() {
  requestAnimationFrame(animate);
  if (needsSurfaceRebuild) { rebuildSurface(); rebuildSamplePoints(); needsSurfaceRebuild = false; }
  if (needsIndicatorRebuild) { rebuildIndicators(); needsIndicatorRebuild = false; }
  updateAxesPlacement();
  controls.update();
  renderer.render(scene, camera);
}
animate();
