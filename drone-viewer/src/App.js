/** @format */

import {
  useState,
  useEffect,
  useRef,
  useMemo,
  Suspense,
  useCallback,
} from "react";
import { Canvas, useLoader, useThree, useFrame } from "@react-three/fiber";
import { OrbitControls, Html, useProgress } from "@react-three/drei";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader";
import SunCalc from "suncalc";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const STORAGE_KEY = "airlink_viewer_";
const API_BASE = "http://localhost:8000";

function loadStored(key, def) {
  try {
    const v = localStorage.getItem(STORAGE_KEY + key);
    if (v == null) return def;
    const parsed = JSON.parse(v);
    return parsed;
  } catch {
    return def;
  }
}
function saveStored(key, value) {
  try {
    localStorage.setItem(STORAGE_KEY + key, JSON.stringify(value));
  } catch (_) {}
}

// Compress heat value arrays to base64-encoded Uint16 for compact localStorage storage.
// A plain-float JSON array uses ~15 bytes/value; this uses ~2.67 bytes/value (~6× smaller).
function encodeHeatArrays(heatArrays, maxHeat) {
  if (!maxHeat) return heatArrays.map(() => "");
  return heatArrays.map((arr) => {
    const u16 = new Uint16Array(arr.length);
    for (let i = 0; i < arr.length; i++)
      u16[i] = Math.min(65535, Math.round((arr[i] / maxHeat) * 65535));
    const bytes = new Uint8Array(u16.buffer);
    let bin = "";
    const CHUNK = 8192;
    for (let i = 0; i < bytes.length; i += CHUNK)
      bin += String.fromCharCode.apply(
        null,
        bytes.subarray(i, Math.min(i + CHUNK, bytes.length)),
      );
    return btoa(bin);
  });
}

function decodeHeatArrays(b64Arr, maxHeat) {
  const scale = maxHeat / 65535;
  return b64Arr.map((b64) => {
    if (!b64) return [];
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const u16 = new Uint16Array(bytes.buffer);
    const arr = new Array(u16.length);
    for (let i = 0; i < u16.length; i++) arr[i] = u16[i] * scale;
    return arr;
  });
}

function listStoredHeatmaps() {
  const prefix = STORAGE_KEY + "heatmapCache_";
  const results = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (!key || !key.startsWith(prefix)) continue;
      try {
        const val = JSON.parse(localStorage.getItem(key));
        if (
          val &&
          typeof val.dateStr === "string" &&
          typeof val.timeStep === "number"
        ) {
          results.push({
            storageKey: key,
            cacheKey: key.slice(prefix.length),
            settingsKey: val.settingsKey ?? key.slice(prefix.length),
            computedAt: val.computedAt ?? 0,
            dateStr: val.dateStr,
            timeStep: val.timeStep,
            lat: Number(val.lat),
            lng: Number(val.lng),
            sampleType: val.sampleType ?? "day",
          });
        }
      } catch (_) {}
    }
  } catch (_) {}
  // Most recently computed first
  return results.sort((a, b) => b.computedAt - a.computedAt);
}

// ── Loader overlay ───────────────────────────────────────────────────────────
function Loader() {
  const { active, progress } = useProgress();
  if (!active) return null;
  return (
    <Html center>
      <div className="loading-box">
        <div className="loading-bar" style={{ width: `${progress}%` }} />
        <p>Loading {Math.round(progress)}%</p>
      </div>
    </Html>
  );
}

// Use PCFShadowMap to avoid THREE.PCFSoftShadowMap deprecation warning
function ShadowMapType() {
  const { gl } = useThree();
  useEffect(() => {
    if (gl.shadowMap) gl.shadowMap.type = THREE.PCFShadowMap;
  }, [gl]);
  return null;
}

// ── Camera fit and presets ───────────────────────────────────────────────────
function CameraFit({ target, preset, onConsumed, orbitRef, overrideCenter }) {
  const { camera } = useThree();
  useEffect(() => {
    if (!target || !preset) return;
    const { extent } = target;
    const dist = extent * 0.8;
    // overrideCenter lets callers (e.g. "Zoom to Pivot") redirect the fit point
    const [cx, cy, cz] = overrideCenter ?? target.center;
    if (preset === "solar") {
      const domeR = Math.max(15, extent * 0.75);
      const sd = domeR * 2.8;
      camera.position.set(cx, cy + sd * 0.4, cz + sd);
    } else if (preset === "auto") {
      camera.position.set(cx, cy + dist * 0.4, cz + dist);
    } else if (preset === "top") {
      camera.position.set(cx, cy + dist, cz);
    } else if (preset === "bottom") {
      camera.position.set(cx, cy - dist, cz);
    } else if (preset === "front" || preset === "south") {
      // +Z = South in SunCalc/Three.js mapping used throughout
      camera.position.set(cx, cy, cz + dist);
    } else if (preset === "north") {
      camera.position.set(cx, cy, cz - dist);
    } else if (preset === "left" || preset === "east") {
      // -X = East per SunCalc azimuth convention
      camera.position.set(cx - dist, cy, cz);
    } else if (preset === "right" || preset === "west") {
      camera.position.set(cx + dist, cy, cz);
    }
    // Sync OrbitControls target, then flush all accumulated damping velocity
    if (orbitRef?.current) {
      const ctrl = orbitRef.current;
      ctrl.target.set(cx, cy, cz);
      if (ctrl._sphericalDelta) ctrl._sphericalDelta.set(0, 0, 0);
      if (ctrl._panOffset) ctrl._panOffset.set(0, 0, 0);
      const prevDamping = ctrl.enableDamping;
      ctrl.enableDamping = false;
      ctrl.update();
      ctrl.enableDamping = prevDamping;

      // Top/bottom: OrbitControls can derive a tilted spherical from its internal
      // quat, giving NE/SW instead of N/S. Force exact position + N/S orientation
      // and sync _spherical so the next update() doesn't snap back.
      if (preset === "top") {
        camera.position.set(cx, cy + dist, cz);
        camera.lookAt(cx, cy, cz);
        camera.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), Math.PI);
        if (ctrl._spherical) {
          ctrl._spherical.radius = dist;
          ctrl._spherical.phi = 0;
          ctrl._spherical.theta = 0;
        }
      } else if (preset === "bottom") {
        camera.position.set(cx, cy - dist, cz);
        camera.lookAt(cx, cy, cz);
        camera.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), Math.PI);
        if (ctrl._spherical) {
          ctrl._spherical.radius = dist;
          ctrl._spherical.phi = Math.PI;
          ctrl._spherical.theta = 0;
        }
      }
    } else {
      camera.lookAt(cx, cy, cz);
      if (preset === "top" || preset === "bottom") {
        camera.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), Math.PI);
      }
    }
    camera.updateProjectionMatrix();
    onConsumed?.();
  }, [camera, target, preset, onConsumed, orbitRef, overrideCenter]);
  return null;
}

// ── Point cloud (raw vertex colors) ───────────────────────────────────────────
function PointCloud({ visible, pointSize, opacity, clipY, meta }) {
  const geom = useLoader(PLYLoader, "/models/point_cloud.ply");
  const clipPlane = useMemo(() => {
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    return plane;
  }, []);

  useEffect(() => {
    if (clipY == null || meta?.bbox_min == null) return;
    clipPlane.setFromNormalAndCoplanarPoint(
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, clipY, 0),
    );
  }, [clipY, meta, clipPlane]);

  if (!visible) return null;
  return (
    <points geometry={geom} frustumCulled={false}>
      <pointsMaterial
        size={pointSize}
        vertexColors
        sizeAttenuation
        transparent
        opacity={opacity}
        depthWrite={false}
        clippingPlanes={clipY != null && meta ? [clipPlane] : []}
        clipIntersection={false}
      />
    </points>
  );
}

function BuildingMeshSimple({
  visible,
  shadowsEnabled,
  wireframe,
  opacity,
  materialMode,
  doubleSided,
  clipY,
  meta,
}) {
  const [scene, setScene] = useState(null);
  const clipPlane = useMemo(
    () => new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
    [],
  );

  useEffect(() => {
    if (clipY != null && meta?.bbox_min != null) {
      clipPlane.setFromNormalAndCoplanarPoint(
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(0, clipY, 0),
      );
    }
  }, [clipY, meta, clipPlane]);

  useEffect(() => {
    new GLTFLoader().load(
      "/models/drone_mesh.glb",
      (gltf) => {
        const root = gltf.scene;
        root.traverse((c) => {
          // Disable frustum culling throughout — the parent group may be rotated
          // and Three.js would otherwise incorrectly cull at extreme camera angles.
          c.frustumCulled = false;
          if (c.isMesh) {
            c.castShadow = shadowsEnabled;
            c.receiveShadow = shadowsEnabled;
            const g = c.geometry;
            const hasColor = g.hasAttribute("color");
            const mat = new THREE.MeshLambertMaterial({
              vertexColors: hasColor,
              color: hasColor ? 0xffffff : 0xcccccc,
              wireframe,
              transparent: true,
              opacity,
              side: doubleSided ? THREE.DoubleSide : THREE.FrontSide,
              clippingPlanes: [clipPlane],
              clipIntersection: false,
            });
            c.material = mat;
          }
        });
        setScene(root);
      },
      undefined,
      (err) => console.warn("Mesh load failed:", err),
    );
  }, [shadowsEnabled, doubleSided]);

  useEffect(() => {
    if (!scene) return;
    scene.traverse((c) => {
      if (c.isMesh && c.material) {
        c.material.wireframe = wireframe;
        c.material.opacity = opacity;
        c.material.side = doubleSided ? THREE.DoubleSide : THREE.FrontSide;
        c.material.vertexColors =
          c.geometry.hasAttribute("color") && materialMode === "vertex";
        c.material.color.setStyle(
          materialMode === "solid" ? "#cccccc" : "#ffffff",
        );
        c.material.clippingPlanes = clipY != null && meta ? [clipPlane] : [];
        c.castShadow = shadowsEnabled;
        c.receiveShadow = shadowsEnabled;
      }
    });
  }, [
    scene,
    wireframe,
    opacity,
    materialMode,
    doubleSided,
    clipY,
    meta,
    clipPlane,
    shadowsEnabled,
  ]);

  if (!scene) return null;
  return <primitive object={scene} visible={visible} frustumCulled={false} />;
}

// ── OBJ Model (client-side upload) ───────────────────────────────────────────
function ObjModel({ url, modelRotation, wireframe, opacity, doubleSided, materialMode }) {
  const [obj, setObj] = useState(null);

  // Load the OBJ once per URL; build initial materials with current settings
  useEffect(() => {
    if (!url) { setObj(null); return; }
    new OBJLoader().load(
      url,
      (object) => {
        object.traverse((c) => {
          c.frustumCulled = false;
          if (c.isMesh) {
            const hasColor = c.geometry.hasAttribute("color");
            const useVertex = hasColor && materialMode === "vertex";
            c.material = new THREE.MeshLambertMaterial({
              vertexColors: useVertex,
              color: useVertex ? 0xffffff : 0xcccccc,
              wireframe: wireframe ?? false,
              transparent: (opacity ?? 1) < 1,
              opacity: opacity ?? 1,
              side: doubleSided ? THREE.DoubleSide : THREE.FrontSide,
            });
          }
        });
        setObj(object);
      },
      undefined,
      (err) => console.warn("OBJ load failed:", err),
    );
    return () => setObj(null);
  }, [url]); // eslint-disable-line react-hooks/exhaustive-deps

  // Live-update materials whenever settings change without reloading
  useEffect(() => {
    if (!obj) return;
    obj.traverse((c) => {
      if (c.isMesh && c.material) {
        const hasColor = c.geometry.hasAttribute("color");
        const useVertex = hasColor && materialMode === "vertex";
        c.material.wireframe = wireframe ?? false;
        c.material.opacity = opacity ?? 1;
        c.material.transparent = (opacity ?? 1) < 1;
        c.material.side = doubleSided ? THREE.DoubleSide : THREE.FrontSide;
        c.material.vertexColors = useVertex;
        c.material.color.set(useVertex ? 0xffffff : 0xcccccc);
        c.material.needsUpdate = true;
      }
    });
  }, [obj, wireframe, opacity, doubleSided, materialMode]);

  if (!obj) return null;
  return <primitive object={obj} rotation={modelRotation} frustumCulled={false} />;
}

// ── Solar ────────────────────────────────────────────────────────────────────
function SolarLight({ sunAz, sunAlt, shadowsEnabled }) {
  const lightRef = useRef();
  useEffect(() => {
    if (!lightRef.current) return;
    const r = 150;
    lightRef.current.position.set(
      r * Math.cos(sunAlt) * Math.sin(sunAz),
      r * Math.sin(sunAlt),
      r * Math.cos(sunAlt) * Math.cos(sunAz),
    );
    lightRef.current.target.position.set(0, 0, 0);
    lightRef.current.target.updateMatrixWorld();
  }, [sunAz, sunAlt]);
  return (
    <>
      <directionalLight
        ref={lightRef}
        intensity={sunAlt > 0 ? 3 : 0}
        castShadow={shadowsEnabled}
        shadow-mapSize-width={4096}
        shadow-mapSize-height={4096}
        shadow-camera-far={400}
        shadow-camera-left={-100}
        shadow-camera-right={100}
        shadow-camera-top={100}
        shadow-camera-bottom={-100}
      />
      <ambientLight intensity={sunAlt > 0 ? 0.3 : 0.7} />
    </>
  );
}

function ShadowGround({ center }) {
  if (!center) return null;
  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={[center[0], center[1] - 2, center[2]]}
      receiveShadow
    >
      <planeGeometry args={[500, 500]} />
      <shadowMaterial transparent opacity={0.4} />
    </mesh>
  );
}

// ── Solar path dome helpers ────────────────────────────────────────────────────
// SunCalc azimuth convention: 0 = South, π/2 = West, π = North, -π/2 = East
// Three.js mapping: x = r·sin(az), z = r·cos(az)  →  +Z = South, -Z = North

function SunArc({ pts, color, opacity, tube }) {
  const geom = useMemo(() => {
    if (!pts || pts.length < 2) return null;
    const curve = new THREE.CatmullRomCurve3(pts);
    return new THREE.TubeGeometry(
      curve,
      Math.min(pts.length * 2, 360),
      tube,
      4,
      false,
    );
  }, [pts, tube]);
  if (!geom) return null;
  return (
    <mesh geometry={geom} frustumCulled={false}>
      <meshBasicMaterial
        color={color}
        transparent
        opacity={opacity}
        depthWrite={false}
      />
    </mesh>
  );
}

// Hour/analemma curve — dense daily sampling, raw line strip
function HourCurve({ pts, color, opacity }) {
  const geom = useMemo(() => {
    if (!pts || pts.length < 2) return null;
    return new THREE.BufferGeometry().setFromPoints(pts);
  }, [pts]);
  if (!geom) return null;
  return (
    <line geometry={geom} frustumCulled={false}>
      <lineBasicMaterial
        color={color}
        transparent
        opacity={opacity}
        depthWrite={false}
      />
    </line>
  );
}

// ── Heatmap color ramps ────────────────────────────────────────────────────────
const COLORMAPS = {
  thermal: {
    label: "Thermal",
    stops: [
      [0.0, [0.05, 0.05, 0.55]],
      [0.25, [0.0, 0.55, 0.9]],
      [0.5, [0.05, 0.8, 0.15]],
      [0.75, [0.95, 0.8, 0.0]],
      [1.0, [0.95, 0.1, 0.05]],
    ],
  },
  solar: {
    label: "Solar",
    stops: [
      [0.0, [0.04, 0.04, 0.18]],
      [0.3, [0.22, 0.08, 0.0]],
      [0.65, [0.8, 0.56, 0.0]],
      [1.0, [1.0, 0.97, 0.55]],
    ],
  },
  grayscale: {
    label: "Grayscale",
    stops: [
      [0.0, [0.08, 0.08, 0.08]],
      [1.0, [1.0, 1.0, 1.0]],
    ],
  },
  inferno: {
    label: "Inferno",
    stops: [
      [0.0, [0.0, 0.0, 0.02]],
      [0.25, [0.27, 0.0, 0.37]],
      [0.5, [0.62, 0.04, 0.18]],
      [0.75, [0.96, 0.52, 0.1]],
      [1.0, [0.99, 0.99, 0.75]],
    ],
  },
  viridis: {
    label: "Viridis",
    stops: [
      [0.0, [0.27, 0.0, 0.33]],
      [0.33, [0.19, 0.41, 0.56]],
      [0.67, [0.13, 0.57, 0.55]],
      [1.0, [0.98, 0.9, 0.14]],
    ],
  },
};

function sampleColormap(t, stops) {
  if (t <= 0) return stops[0][1];
  if (t >= 1) return stops[stops.length - 1][1];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return [
        c0[0] + f * (c1[0] - c0[0]),
        c0[1] + f * (c1[1] - c0[1]),
        c0[2] + f * (c1[2] - c0[2]),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

function heatmapColor(t, colormap = "thermal") {
  const stops = (COLORMAPS[colormap] ?? COLORMAPS.thermal).stops;
  return sampleColormap(t, stops);
}

// Build a CSS linear-gradient string from a colormap key
function colormapCssGradient(colormap) {
  const stops = (COLORMAPS[colormap] ?? COLORMAPS.thermal).stops;
  const parts = stops.map(
    ([pos, [r, g, b]]) =>
      `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}) ${(pos * 100).toFixed(0)}%`,
  );
  return `linear-gradient(to right, ${parts.join(", ")})`;
}

// ── Solar heatmap overlay on mesh (GPU shadow-map depth readback) ─────────────
//
// How it works:
//   Each frame, this component:
//     1. Reads the depth texture rendered in the PREVIOUS frame (from the
//        custom orthographic shadow camera aimed at the last sun position)
//     2. For every mesh vertex, projects it into that shadow camera space and
//        compares its window-Z against the stored depth — if it's farther away
//        the vertex is occluded (shadowed) for that sun angle.
//     3. Sets up the shadow camera for the NEXT sun position and renders a
//        fresh depth pass using Three.js's overrideMaterial.
//   After all daylight sun positions have been processed the accumulated
//   cosine-weighted exposure is mapped to a heatmap palette and applied as
//   vertex colors.  One sun sample per animation frame → typically < 2 s.
//
// normalise a raw heat value to [0,1] given a range in hours and timeStep (minutes)
function heatToT(raw, timeStep, rangeMinHrs, rangeMaxHrs) {
  const hrs = (raw * timeStep) / 60;
  const span = rangeMaxHrs - rangeMinHrs;
  if (span <= 0) return 0;
  return Math.max(0, Math.min(1, (hrs - rangeMinHrs) / span));
}

function applyHeatColors(
  meshes,
  heat,
  timeStep,
  rMin,
  rMax,
  clipY,
  meta,
  clipPlane,
  colormap = "thermal",
) {
  meshes.forEach((mesh, mi) => {
    const h = heat[mi];
    const colors = new Float32Array(h.length * 3);
    for (let i = 0; i < h.length; i++) {
      const t = heatToT(h[i], timeStep, rMin, rMax);
      const [rv, gv, bv] = heatmapColor(t, colormap);
      colors[i * 3] = rv;
      colors[i * 3 + 1] = gv;
      colors[i * 3 + 2] = bv;
    }
    mesh.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    if (mesh.material) mesh.material.dispose();
    mesh.material = new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      clippingPlanes: clipY != null && meta ? [clipPlane] : [],
      clipIntersection: false,
    });
  });
}

function SolarHeatmapMesh({
  visible,
  dateStr,
  lat,
  lng,
  modelRotation, // eslint-disable-line no-unused-vars
  timeStep,
  sampleType = "day",
  trigger,
  onStatus,
  clipY,
  meta,
  savedHeatmap,
  rangeMinHrs,
  rangeMaxHrs,
  colormap = "thermal",
  meshUrl,   // optional blob URL (OBJ); undefined → use default GLB
  meshType,  // "obj" | "glb"
}) {
  const { gl } = useThree();
  const [scene, setScene] = useState(null);
  const mountedRef = useRef(true);
  const clipPlane = useMemo(
    () => new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
    [],
  );
  // Keep range + colormap in a ref so the useFrame closure always sees the latest value
  const rangeRef = useRef({ min: rangeMinHrs, max: rangeMaxHrs });
  useEffect(() => {
    rangeRef.current = { min: rangeMinHrs, max: rangeMaxHrs };
  }, [rangeMinHrs, rangeMaxHrs]);
  const colormapRef = useRef(colormap);
  useEffect(() => {
    colormapRef.current = colormap;
  }, [colormap]);

  // Mutable computation state — lives in a ref so useFrame never causes re-renders
  const compRef = useRef({ active: false });
  // Keep the latest callback in a ref so the useFrame closure doesn't go stale
  const cbRef = useRef(onStatus);
  useEffect(() => {
    cbRef.current = onStatus;
  }, [onStatus]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (clipY != null && meta?.bbox_min != null) {
      clipPlane.setFromNormalAndCoplanarPoint(
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(0, clipY, 0),
      );
    }
  }, [clipY, meta, clipPlane]);

  // Load the display mesh whenever the source changes.
  // Meshes are placed on layer 1 so the shadow camera (also layer 1)
  // only captures them, not the solar dome or other scene objects.
  useEffect(() => {
    cbRef.current?.("loading");
    setScene(null);

    const prepareRoot = (root) => {
      if (!mountedRef.current) return;
      root.traverse((c) => {
        c.frustumCulled = false;
        if (c.isMesh) {
          c.geometry = c.geometry.clone();
          c.layers.enable(1);
          c.material = new THREE.MeshBasicMaterial({
            color: 0x888888,
            side: THREE.DoubleSide,
          });
        }
      });
      setScene(root);
      cbRef.current?.("idle");
    };

    const onError = (err) => {
      console.warn("Heatmap mesh load failed:", err);
      if (mountedRef.current) cbRef.current?.("idle");
    };

    if (meshType === "obj" && meshUrl) {
      new OBJLoader().load(meshUrl, prepareRoot, undefined, onError);
    } else {
      new GLTFLoader().load(
        "/models/drone_mesh.glb",
        (gltf) => prepareRoot(gltf.scene.clone(true)),
        undefined,
        onError,
      );
    }
  }, [meshUrl, meshType]); // eslint-disable-line

  // Initialise a new computation run whenever the trigger increments
  useEffect(() => {
    if (!trigger || !scene) return;
    cbRef.current?.("computing");

    // Build sun direction samples based on sampleType
    const sunDirs = [];
    const addDirsSingleDay = (dayBase) => {
      const dayTimes = SunCalc.getTimes(
        new Date(dayBase + 12 * 3600000),
        lat,
        lng,
      );
      // Clamp loop bounds to sunrise→sunset so nighttime minutes are never iterated
      const sunriseMin =
        dayTimes.sunrise instanceof Date && !isNaN(dayTimes.sunrise)
          ? Math.max(
              0,
              Math.floor((dayTimes.sunrise.getTime() - dayBase) / 60000),
            )
          : 0;
      const sunsetMin =
        dayTimes.sunset instanceof Date && !isNaN(dayTimes.sunset)
          ? Math.min(
              1440,
              Math.ceil((dayTimes.sunset.getTime() - dayBase) / 60000),
            )
          : 1440;
      // Align start to the next timeStep boundary at or after sunrise
      const startMin = Math.ceil(sunriseMin / timeStep) * timeStep;
      for (let m = startMin; m <= sunsetMin; m += timeStep) {
        const pos = SunCalc.getPosition(
          new Date(dayBase + m * 60000),
          lat,
          lng,
        );
        if (pos.altitude > 0) {
          sunDirs.push(
            new THREE.Vector3(
              Math.cos(pos.altitude) * Math.sin(pos.azimuth),
              Math.sin(pos.altitude),
              Math.cos(pos.altitude) * Math.cos(pos.azimuth),
            ).normalize(),
          );
        }
      }
    };

    const [y, mo] = dateStr.split("-").map(Number);
    if (sampleType === "month") {
      const daysInMonth = new Date(y, mo, 0).getDate();
      for (let d = 1; d <= daysInMonth; d++) {
        addDirsSingleDay(new Date(y, mo - 1, d).getTime());
      }
    } else if (sampleType === "year") {
      for (let doy = 0; doy < 365; doy++) {
        addDirsSingleDay(new Date(y, 0, 1 + doy).getTime());
      }
    } else {
      // "day" — current behaviour
      addDirsSingleDay(new Date(`${dateStr}T00:00:00`).getTime());
    }

    if (sunDirs.length === 0) {
      cbRef.current?.("done");
      return;
    }

    // Shadow render target — UnsignedByte RGBA, BasicDepthPacking stores (1−z) in R
    const MAP = 1024;
    const shadowRT = new THREE.WebGLRenderTarget(MAP, MAP, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      generateMipmaps: false,
    });

    // Depth material: outputs (1 − windowZ) into R,G,B via BasicDepthPacking
    const depthMat = new THREE.MeshDepthMaterial({
      depthPacking: THREE.BasicDepthPacking,
      side: THREE.DoubleSide,
    });

    // Orthographic shadow camera — large enough to cover mesh from any sun angle
    const e = meta?.extent ?? 50;
    const cx = meta?.center?.[0] ?? 0;
    const cy = meta?.center?.[1] ?? 0;
    const cz = meta?.center?.[2] ?? 0;
    const dist = e * 3;
    const shadowCam = new THREE.OrthographicCamera(
      -e * 1.5,
      e * 1.5,
      e * 1.5,
      -e * 1.5,
      dist - e * 1.5,
      dist + e * 2,
    );
    shadowCam.layers.set(1); // only renders the heatmap mesh (layer 1)

    // Collect mesh refs and allocate per-vertex heat accumulators
    const meshes = [];
    scene.traverse((c) => {
      if (!c.isMesh) return;
      if (!c.geometry.attributes.normal) c.geometry.computeVertexNormals();
      meshes.push(c);
    });
    const heat = meshes.map(
      (m) => new Float32Array(m.geometry.attributes.normal.count),
    );
    const pixels = new Uint8Array(MAP * MAP * 4); // reused every frame

    compRef.current = {
      active: true,
      step: 0,
      sunDirs,
      meshes,
      heat,
      maxHeat: 0,
      shadowRT,
      depthMat,
      shadowCam,
      savedViewProj: null, // view-proj matrix from the previous frame's render
      savedSunDir: null, // world-space sun direction from the previous frame
      pixels,
      MAP,
      cx,
      cy,
      cz,
      dist,
    };
  }, [trigger]); // eslint-disable-line react-hooks/exhaustive-deps

  // Apply saved heatmap — also re-applies when the display range changes
  const lastAppliedKeyRef = useRef(null);
  useEffect(() => {
    if (
      !scene ||
      !savedHeatmap ||
      compRef.current.active || // don't override an in-progress computation
      savedHeatmap.dateStr !== dateStr ||
      savedHeatmap.timeStep !== timeStep ||
      Number(savedHeatmap.lat) !== Number(lat) ||
      Number(savedHeatmap.lng) !== Number(lng)
    ) {
      if (!savedHeatmap) lastAppliedKeyRef.current = null;
      return;
    }
    const applyKey = `${savedHeatmap.maxHeat}_${rangeMinHrs}_${rangeMaxHrs}_${colormap}`;
    if (lastAppliedKeyRef.current === applyKey) return;
    const { heatArrays, maxHeat } = savedHeatmap;
    if (
      !Array.isArray(heatArrays) ||
      typeof maxHeat !== "number" ||
      !Number.isFinite(maxHeat)
    )
      return;
    const meshes = [];
    scene.traverse((c) => {
      if (!c.isMesh) return;
      if (!c.geometry.attributes.normal) c.geometry.computeVertexNormals();
      meshes.push(c);
    });
    if (
      heatArrays.length !== meshes.length ||
      meshes.some(
        (m, i) =>
          !Array.isArray(heatArrays[i]) ||
          heatArrays[i].length !== m.geometry.attributes.normal.count,
      )
    )
      return;
    try {
      const autoMaxHrs = (maxHeat * timeStep) / 60;
      const rMin = rangeMinHrs ?? 0;
      const rMax = rangeMaxHrs ?? autoMaxHrs;
      applyHeatColors(
        meshes,
        heatArrays,
        timeStep,
        rMin,
        rMax,
        clipY,
        meta,
        clipPlane,
        colormap,
      );
      lastAppliedKeyRef.current = applyKey;
      cbRef.current?.("done");
    } catch (err) {
      lastAppliedKeyRef.current = null;
      console.warn("Failed to apply saved heatmap:", err);
    }
  }, [
    scene,
    savedHeatmap,
    dateStr,
    timeStep,
    lat,
    lng,
    clipY,
    meta,
    clipPlane,
    rangeMinHrs,
    rangeMaxHrs,
    colormap,
  ]);

  // ── Per-frame driver ───────────────────────────────────────────────────────
  // Every frame: (A) accumulate exposure from last frame's depth map,
  //              (B) render depth map from the current sun position.
  useFrame(({ scene: r3fScene }) => {
    const c = compRef.current;
    if (!c.active || !scene) return;

    const {
      sunDirs,
      meshes,
      heat,
      shadowRT,
      depthMat,
      shadowCam,
      pixels,
      MAP,
      cx,
      cy,
      cz,
      dist,
    } = c;

    // ── A: accumulate from the depth map rendered last frame ────────────────
    if (c.savedViewProj) {
      try {
        gl.readRenderTargetPixels(shadowRT, 0, 0, MAP, MAP, pixels);
      } catch (err) {
        console.warn("Shadow depth readback failed:", err);
      }

      const vpm = c.savedViewProj;
      const sunDir = c.savedSunDir;
      const BIAS = 0.005; // avoids self-shadowing from depth quantisation

      const vPos = new THREE.Vector3();
      const vNorm = new THREE.Vector3();
      const proj = new THREE.Vector4();

      meshes.forEach((mesh, mi) => {
        // matrixWorld includes the parent group's modelRotation
        mesh.updateWorldMatrix(true, false);
        const wm = mesh.matrixWorld;
        const posAttr = mesh.geometry.attributes.position;
        const nrmAttr = mesh.geometry.attributes.normal;
        const count = posAttr.count;
        const heatArr = heat[mi];

        for (let i = 0; i < count; i++) {
          vPos.fromBufferAttribute(posAttr, i).applyMatrix4(wm);
          vNorm
            .fromBufferAttribute(nrmAttr, i)
            .transformDirection(wm)
            .normalize();

          const dot = vNorm.dot(sunDir);
          if (dot <= 0) continue; // back-facing — sun cannot reach this side

          // Project vertex into the shadow camera's clip space
          proj.set(vPos.x, vPos.y, vPos.z, 1.0).applyMatrix4(vpm);
          const w = proj.w;
          const nx = proj.x / w;
          const ny = proj.y / w;
          const nz = proj.z / w;

          // Skip if outside the shadow frustum
          if (nx < -1 || nx > 1 || ny < -1 || ny > 1 || nz < -1 || nz > 1)
            continue;

          const vertexZ = nz * 0.5 + 0.5; // window-Z in [0, 1]
          const px = Math.min(Math.floor((nx * 0.5 + 0.5) * MAP), MAP - 1);
          const py = Math.min(Math.floor((ny * 0.5 + 0.5) * MAP), MAP - 1);
          const idx = (py * MAP + px) * 4;

          // BasicDepthPacking: R channel stores (1 − windowZ)
          const shadowZ = 1.0 - pixels[idx] / 255.0;
          if (vertexZ > shadowZ + BIAS) continue; // occluded — in shadow

          heatArr[i] += dot;
          if (heatArr[i] > c.maxHeat) c.maxHeat = heatArr[i];
        }
      });

      c.savedViewProj = null;
      c.savedSunDir = null;
      cbRef.current?.(
        "computing:" + Math.round((c.step / sunDirs.length) * 100),
      );
    }

    // ── B: render depth map for the current sun direction ──────────────────
    if (c.step < sunDirs.length) {
      const sunDir = sunDirs[c.step];
      shadowCam.position.set(
        cx + sunDir.x * dist,
        cy + sunDir.y * dist,
        cz + sunDir.z * dist,
      );
      shadowCam.lookAt(cx, cy, cz);
      shadowCam.updateMatrixWorld();
      shadowCam.updateProjectionMatrix();

      // Save the view-projection matrix so Step A can use it next frame
      c.savedViewProj = new THREE.Matrix4().multiplyMatrices(
        shadowCam.projectionMatrix,
        shadowCam.matrixWorldInverse,
      );
      c.savedSunDir = sunDir;

      // Depth-only render — shadow camera only sees layer-1 objects
      const prevAutoUpdate = gl.shadowMap.autoUpdate;
      gl.shadowMap.autoUpdate = false; // skip re-rendering other shadow maps
      r3fScene.overrideMaterial = depthMat;
      gl.setRenderTarget(shadowRT);
      gl.clear();
      gl.render(r3fScene, shadowCam);
      gl.setRenderTarget(null);
      r3fScene.overrideMaterial = null;
      gl.shadowMap.autoUpdate = prevAutoUpdate;

      c.step++;
    } else {
      // ── All sun positions processed — apply heatmap vertex colors ─────────
      const autoMaxHrs = (c.maxHeat * timeStep) / 60;
      const rMin = rangeRef.current.min ?? 0;
      const rMax = rangeRef.current.max ?? autoMaxHrs;
      applyHeatColors(
        meshes,
        heat,
        timeStep,
        rMin,
        rMax,
        clipY,
        meta,
        clipPlane,
        colormapRef.current,
      );

      shadowRT.dispose();
      depthMat.dispose();
      c.active = false;

      if (mountedRef.current) {
        const heatArrays = heat.map((h) => Array.from(h));
        cbRef.current?.("done", { heatArrays, maxHeat: c.maxHeat });
      }
    }
  });

  if (!visible || !scene) return null;
  return <primitive object={scene} frustumCulled={false} />;
}

// ── Solar path dome ────────────────────────────────────────────────────────────
function SolarPathDome({
  lat,
  lng,
  dateStr,
  sunAz,
  sunAlt,
  meta,
  pivot,
  pathOpacity,
  pathSize,
  lineThick,
  showSunMarker,
}) {
  // Use pivot point as dome center if set, otherwise fall back to meta.center
  const cx = pivot?.[0] ?? meta?.center?.[0] ?? 0;
  const cy = pivot?.[1] ?? meta?.center?.[1] ?? 0;
  const cz = pivot?.[2] ?? meta?.center?.[2] ?? 0;
  const radius = Math.max(15, (meta?.extent ?? 20) * 0.75) * (pathSize ?? 1);

  // Helper: sun position → 3D point on dome surface
  const sunPt = useCallback(
    (pos) =>
      new THREE.Vector3(
        cx + radius * Math.cos(pos.altitude) * Math.sin(pos.azimuth),
        cy + radius * Math.sin(pos.altitude),
        cz + radius * Math.cos(pos.altitude) * Math.cos(pos.azimuth),
      ),
    [cx, cy, cz, radius],
  );

  // Parse year directly from string — avoids UTC-vs-local midnight ambiguity
  const year = useMemo(() => parseInt(dateStr.slice(0, 4), 10), [dateStr]);

  // Annual arcs — 21st of each month, fine time step
  const annualArcs = useMemo(
    () =>
      Array.from({ length: 12 }, (_, m) => {
        const base = new Date(year, m, 21).getTime();
        const pts = [];
        for (let h = 0; h < 24; h += 0.12) {
          const pos = SunCalc.getPosition(
            new Date(base + h * 3600000),
            lat,
            lng,
          );
          if (pos.altitude > 0.005) pts.push(sunPt(pos));
        }
        return { pts, month: m };
      }),
    [year, lat, lng, sunPt],
  );

  // Today's arc — finer step, highlighted
  const todayPts = useMemo(() => {
    // Use T00:00:00 (no timezone suffix) so the browser treats it as LOCAL midnight,
    // matching how sunPos is computed with `new Date(`${dateStr}T${hh}:${mm}:00`)`.
    const base = new Date(`${dateStr}T00:00:00`).getTime();
    const pts = [];
    for (let h = 0; h < 24; h += 0.06) {
      const pos = SunCalc.getPosition(new Date(base + h * 3600000), lat, lng);
      if (pos.altitude > 0.005) pts.push(sunPt(pos));
    }
    return pts;
  }, [dateStr, lat, lng, sunPt]);

  // Hour/analemma curves — sample EVERY day at the same clock hour to get true figure-8
  const hourCurves = useMemo(() => {
    const curves = [];
    const yearStartMs = new Date(year, 0, 1).getTime();
    for (let h = 5; h <= 19; h++) {
      const pts = [];
      for (let day = 0; day < 366; day += 2) {
        // every 2 days ≈ 183 pts per curve
        const pos = SunCalc.getPosition(
          new Date(yearStartMs + day * 86400000 + h * 3600000),
          lat,
          lng,
        );
        if (pos.altitude > 0.005) pts.push(sunPt(pos));
      }
      // Close loop: append first valid point so curve returns to start
      if (pts.length >= 3) {
        pts.push(pts[0].clone());
        curves.push({ pts, hour: h });
      }
    }
    return curves;
  }, [year, lat, lng, sunPt]);

  // Dome hemisphere geometry
  const domeGeom = useMemo(
    () =>
      new THREE.SphereGeometry(radius, 48, 24, 0, Math.PI * 2, 0, Math.PI / 2),
    [radius],
  );

  // Base ring geometry
  const baseRingGeom = useMemo(() => {
    const pts = [];
    for (let i = 0; i <= 128; i++) {
      const a = (i / 128) * Math.PI * 2;
      pts.push(
        new THREE.Vector3(
          cx + radius * Math.sin(a),
          cy,
          cz + radius * Math.cos(a),
        ),
      );
    }
    return new THREE.BufferGeometry().setFromPoints(pts);
  }, [radius, cx, cy, cz]);

  // Current sun position + ray
  const sunVisible = sunAlt > 0;
  const { sunVec, sunRayGeom } = useMemo(() => {
    const v = new THREE.Vector3(
      cx + radius * Math.cos(sunAlt) * Math.sin(sunAz),
      cy + radius * Math.sin(sunAlt),
      cz + radius * Math.cos(sunAlt) * Math.cos(sunAz),
    );
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(cx, cy, cz),
      v,
    ]);
    return { sunVec: v, sunRayGeom: g };
  }, [sunAz, sunAlt, radius, cx, cy, cz]);

  // Compass directions (SunCalc az: 0=South)
  const compass = [
    { label: "N", az: Math.PI },
    { label: "NE", az: (5 * Math.PI) / 4 },
    { label: "E", az: (3 * Math.PI) / 2 },
    { label: "SE", az: (7 * Math.PI) / 4 },
    { label: "S", az: 0 },
    { label: "SW", az: Math.PI / 4 },
    { label: "W", az: Math.PI / 2 },
    { label: "NW", az: (3 * Math.PI) / 4 },
  ];

  const op = pathOpacity ?? 1;
  const lw = lineThick ?? 1;

  // Monthly arc style — opacity scaled by op, tube scaled by lw
  const arcStyle = (month) => {
    if (month === 5)
      return { color: "#ff6622", opacity: 0.85 * op, tube: 0.09 * lw };
    if (month === 11)
      return { color: "#4499ff", opacity: 0.85 * op, tube: 0.09 * lw };
    if (month === 2 || month === 8)
      return { color: "#ffcc44", opacity: 0.65 * op, tube: 0.06 * lw };
    return { color: "#7799cc", opacity: 0.3 * op, tube: 0.04 * lw };
  };

  return (
    <group>
      {/* Dome shell */}
      <mesh geometry={domeGeom} position={[cx, cy, cz]} frustumCulled={false}>
        <meshBasicMaterial
          color="#5577ff"
          transparent
          opacity={0.04 * op}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      {/* Base ring */}
      <line geometry={baseRingGeom} frustumCulled={false}>
        <lineBasicMaterial color="#3366aa" transparent opacity={0.5 * op} />
      </line>

      {/* Hour curves (analemma loops) — subtle, drawn first */}
      {hourCurves.map(({ pts, hour }) => (
        <HourCurve key={hour} pts={pts} color="#3355aa" opacity={0.22 * op} />
      ))}

      {/* Annual sun path arcs */}
      {annualArcs.map(({ pts, month }) => {
        const s = arcStyle(month);
        return (
          <SunArc
            key={month}
            pts={pts}
            color={s.color}
            opacity={s.opacity}
            tube={s.tube}
          />
        );
      })}

      {/* Today's arc — bright yellow, slightly thicker */}
      <SunArc
        pts={todayPts}
        color="#ffee00"
        opacity={0.95 * op}
        tube={0.14 * lw}
      />

      {/* Current sun sphere + ray */}
      {sunVisible && showSunMarker !== false && (
        <>
          <mesh position={[sunVec.x, sunVec.y, sunVec.z]} frustumCulled={false}>
            <sphereGeometry args={[radius * 0.022, 14, 14]} />
            <meshBasicMaterial color="#ffee00" />
          </mesh>
          <line geometry={sunRayGeom} frustumCulled={false}>
            <lineBasicMaterial color="#ffee66" transparent opacity={0.4 * op} />
          </line>
        </>
      )}

      {/* Compass labels */}
      {compass.map(({ label, az }) => (
        <Html
          key={label}
          position={[
            cx + radius * 1.07 * Math.sin(az),
            cy + 0.5,
            cz + radius * 1.07 * Math.cos(az),
          ]}
          center
          style={{ zIndex: 1 }}
        >
          <div
            style={{
              color:
                label === "N"
                  ? "#ff8888"
                  : label === "S"
                    ? "#88ccff"
                    : "#aabbdd",
              fontSize: "12px",
              fontWeight: 700,
              textShadow:
                "0 0 0 1px rgba(0,0,0,0.8), 0 1px 2px rgba(0,0,0,0.5)",
              pointerEvents: "none",
              userSelect: "none",
            }}
          >
            {label}
          </div>
        </Html>
      ))}
    </group>
  );
}

// ── Ground plane with grid ────────────────────────────────────────────────────
function GroundPlane({ y, visible }) {
  if (!visible) return null;
  return (
    <group position={[0, y, 0]}>
      <gridHelper args={[800, 80, "#2255bb", "#132244"]} />
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[800, 800]} />
        <meshBasicMaterial
          color="#1a3a99"
          transparent
          opacity={0.07}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

// ── Pivot crosshair marker ─────────────────────────────────────────────────────
function PivotMarker({ position, extent }) {
  const len = Math.max(1, (extent ?? 20) * 0.06);
  const [xLine, yLine, zLine] = useMemo(() => {
    const mk = (a, b, color) => {
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(...a),
        new THREE.Vector3(...b),
      ]);
      return new THREE.Line(
        g,
        new THREE.LineBasicMaterial({ color, depthTest: false }),
      );
    };
    return [
      mk([-len, 0, 0], [len, 0, 0], "#ff4444"),
      mk([0, -len, 0], [0, len, 0], "#44ee44"),
      mk([0, 0, -len], [0, 0, len], "#4488ff"),
    ];
  }, [len]);
  if (!position) return null;
  return (
    <group position={position}>
      <mesh renderOrder={999}>
        <sphereGeometry args={[len * 0.18, 10, 10]} />
        <meshBasicMaterial color="#ffaa00" depthTest={false} />
      </mesh>
      <primitive object={xLine} />
      <primitive object={yLine} />
      <primitive object={zLine} />
    </group>
  );
}

// ── Leaflet location picker ────────────────────────────────────────────────────
// Fix the default Leaflet marker icon path that breaks under webpack/CRA
const LEAFLET_ICON = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

function LeafletMap({ lat, lng, onChange }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const markerRef = useRef(null);

  // Initialise the Leaflet map exactly once when this component mounts
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      center: [lat, lng],
      zoom: 3,
      zoomControl: true,
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution:
        '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors',
      maxZoom: 19,
    }).addTo(map);

    // Draggable marker — user can drag it to fine-tune position
    const marker = L.marker([lat, lng], {
      draggable: true,
      icon: LEAFLET_ICON,
    }).addTo(map);

    marker.on("dragend", () => {
      const { lat: la, lng: ln } = marker.getLatLng();
      onChange(la, ln);
    });

    // Click anywhere on the map to move the marker
    map.on("click", (e) => {
      marker.setLatLng(e.latlng);
      onChange(e.latlng.lat, e.latlng.lng);
    });

    mapRef.current = map;
    markerRef.current = marker;

    return () => {
      map.remove();
      mapRef.current = null;
      markerRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Keep marker in sync when parent lat/lng updates (e.g. after reverse geocode)
  useEffect(() => {
    if (markerRef.current) {
      markerRef.current.setLatLng([lat, lng]);
    }
  }, [lat, lng]);

  return <div ref={containerRef} className="leaflet-map-container" />;
}

// ── Keeps OrbitControls target in sync with pivot point ───────────────────────
function PivotSync({ pivotPoint, meta, orbitRef }) {
  useEffect(() => {
    if (!orbitRef?.current) return;
    const t = pivotPoint ?? meta?.center ?? [0, 0, 0];
    orbitRef.current.target.set(t[0], t[1], t[2]);
    orbitRef.current.update();
  }, [pivotPoint, meta, orbitRef]);
  return null;
}

// ── Sync background and FOV to scene ──────────────────────────────────────────
function SceneEnv({ bgColor, fov }) {
  const { scene, camera } = useThree();
  useEffect(() => {
    scene.background = new THREE.Color(bgColor);
  }, [scene, bgColor]);
  useEffect(() => {
    if (camera && fov != null) {
      camera.fov = fov;
      camera.updateProjectionMatrix();
    }
  }, [camera, fov]);
  return null;
}

// ── Scene ─────────────────────────────────────────────────────────────────────
function Scene(props) {
  const {
    viewMode,
    pointSize,
    pointOpacity,
    clipY,
    sunAz,
    sunAlt,
    shadowsEnabled,
    meshWireframe,
    meshOpacity,
    meshMaterialMode,
    meshDoubleSided,
    ambientIntensity,
    fov,
    cameraPreset,
    onCameraPresetConsumed,
    meta,
    modelRotation,
    pivotMode,
    onPivotPick,
    pivotPoint,
    showPivotMarker,
    showGroundPlane,
    groundY,
    orbitRef,
    overrideCenter,
    heatmapMode,
    heatmapSampleType,
    heatmapTrigger,
    heatmapTimeStep,
    savedHeatmap,
    heatmapRangeMinHrs,
    heatmapRangeMaxHrs,
    heatmapColormap,
    lat,
    lng,
    dateStr,
    onHeatmapStatus,
    showServerModels,
    objMeshUrl,
  } = props;
  const center = meta?.center ?? [0, 0, 0];
  const extent = meta?.extent ?? 20;

  const handleModelClick = useCallback(
    (e) => {
      if (!pivotMode) return;
      e.stopPropagation();
      onPivotPick([e.point.x, e.point.y, e.point.z]);
    },
    [pivotMode, onPivotPick],
  );

  return (
    <>
      <SceneEnv bgColor={props.bgColor} fov={fov} />
      <ambientLight intensity={ambientIntensity} />
      <SolarLight
        sunAz={sunAz}
        sunAlt={sunAlt}
        shadowsEnabled={shadowsEnabled}
      />
      {shadowsEnabled && <ShadowGround center={center} />}
      <CameraFit
        target={meta}
        preset={cameraPreset}
        onConsumed={onCameraPresetConsumed}
        orbitRef={orbitRef}
        overrideCenter={overrideCenter}
      />
      <GroundPlane y={groundY} visible={showGroundPlane} />
      <PivotMarker
        position={showPivotMarker ? pivotPoint : null}
        extent={extent}
      />
      <group
        rotation={modelRotation}
        onClick={handleModelClick}
        frustumCulled={false}
      >
        <Suspense fallback={<Loader />}>
          {showServerModels && (viewMode === "cloud" || viewMode === "both") && (
            <PointCloud
              visible
              pointSize={pointSize}
              opacity={pointOpacity}
              clipY={clipY}
              meta={meta}
            />
          )}
          {showServerModels &&
            (!heatmapMode || viewMode !== "solar") &&
            (viewMode === "mesh" ||
              viewMode === "solar" ||
              viewMode === "both") && (
              <BuildingMeshSimple
                visible
                shadowsEnabled={shadowsEnabled}
                wireframe={meshWireframe}
                opacity={meshOpacity}
                materialMode={meshMaterialMode}
                doubleSided={meshDoubleSided}
                clipY={clipY}
                meta={meta}
              />
            )}
          {heatmapMode && viewMode === "solar" && (
            <SolarHeatmapMesh
              visible
              dateStr={dateStr}
              lat={lat}
              lng={lng}
              modelRotation={modelRotation}
              timeStep={heatmapTimeStep}
              sampleType={heatmapSampleType}
              trigger={heatmapTrigger}
              savedHeatmap={savedHeatmap}
              rangeMinHrs={heatmapRangeMinHrs}
              rangeMaxHrs={heatmapRangeMaxHrs}
              colormap={heatmapColormap}
              onStatus={onHeatmapStatus}
              clipY={clipY}
              meta={meta}
              meshUrl={objMeshUrl ?? undefined}
              meshType={objMeshUrl ? "obj" : "glb"}
            />
          )}
        </Suspense>
      </group>
    </>
  );
}

// ── Heatmap Analysis Modal ────────────────────────────────────────────────────
// ── Colormap dropdown with gradient swatches ──────────────────────────────────
function ColormapDropdown({ value, onChange }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const selected = COLORMAPS[value] ?? COLORMAPS.thermal;

  return (
    <div className="colormap-dropdown" ref={ref}>
      <span className="colormap-dropdown-label">Gradient</span>
      <button
        type="button"
        className="colormap-dropdown-trigger"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span
          className="colormap-dropdown-preview"
          style={{ background: colormapCssGradient(value) }}
        />
        <span className="colormap-dropdown-name">{selected.label}</span>
        <span className="colormap-dropdown-chevron">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <ul className="colormap-dropdown-menu" role="listbox">
          {Object.entries(COLORMAPS).map(([key, cm]) => (
            <li
              key={key}
              role="option"
              aria-selected={key === value}
              className={`colormap-dropdown-option${key === value ? " selected" : ""}`}
              onClick={() => {
                onChange(key);
                setOpen(false);
              }}
            >
              <span
                className="colormap-option-bar"
                style={{ background: colormapCssGradient(key) }}
              />
              <span className="colormap-option-name">{cm.label}</span>
              {key === value && (
                <span className="colormap-option-check">✓</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function HeatmapAnalysisModal({
  heatArrays,
  maxHours,
  timeStep,
  rangeEnabled,
  rangeMinHrs,
  rangeMaxHrs,
  colormap = "thermal",
  onClose,
}) {
  const NUM_BINS = 20;

  const { allVals, totalCount } = useMemo(() => {
    if (!heatArrays) return { allVals: [], totalCount: 0 };
    const out = [];
    for (const arr of heatArrays) {
      for (const v of arr) out.push((v * timeStep) / 60);
    }
    return { allVals: out, totalCount: out.length };
  }, [heatArrays, timeStep]);

  const bins = useMemo(() => {
    if (!allVals.length || !maxHours) return [];
    const rMin = rangeEnabled ? (rangeMinHrs ?? 0) : 0;
    const rMax = rangeEnabled ? (rangeMaxHrs ?? maxHours) : maxHours;
    const binStep = maxHours / NUM_BINS;
    const counts = new Array(NUM_BINS).fill(0);
    for (const v of allVals) {
      const idx = Math.min(NUM_BINS - 1, Math.max(0, Math.floor(v / binStep)));
      counts[idx]++;
    }
    return counts.map((count, i) => {
      const binLo = i * binStep;
      const binHi = binLo + binStep;
      const midHrs = (binLo + binHi) / 2;
      const span = rMax - rMin;
      const t = span > 0 ? Math.max(0, Math.min(1, (midHrs - rMin) / span)) : 0;
      const [r, g, b] = heatmapColor(t, colormap);
      const color = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
      return {
        lo: binLo,
        hi: binHi,
        count,
        pct: totalCount > 0 ? (count / totalCount) * 100 : 0,
        color,
      };
    });
  }, [
    allVals,
    totalCount,
    maxHours,
    rangeEnabled,
    rangeMinHrs,
    rangeMaxHrs,
    colormap,
  ]);

  const svgW = 380,
    svgH = 170,
    padL = 36,
    padR = 8,
    padT = 12,
    padB = 28;
  const chartW = svgW - padL - padR;
  const chartH = svgH - padT - padB;
  const maxPct = Math.max(...bins.map((b) => b.pct), 1);
  const yTicks = [0, 25, 50, 75, 100].filter((p) => p <= Math.ceil(maxPct) + 5);
  const fmtH = (h) => (h < 10 ? h.toFixed(1) : Math.round(h).toString());

  return (
    <div className="heatmap-analysis-overlay" onClick={onClose}>
      <div
        className="heatmap-analysis-card"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="heatmap-analysis-header">
          <span className="heatmap-analysis-title">Exposure Distribution</span>
          <button
            type="button"
            className="heatmap-analysis-close"
            onClick={onClose}
          >
            ✕
          </button>
        </div>
        <div className="heatmap-analysis-body">
          <p className="heatmap-analysis-subtitle">
            {totalCount.toLocaleString()} vertices · {NUM_BINS} bins · max{" "}
            {fmtH(maxHours)} h
          </p>

          {/* SVG Bar Chart */}
          <svg viewBox={`0 0 ${svgW} ${svgH}`} className="heatmap-analysis-svg">
            {/* Y grid lines + labels */}
            {yTicks.map((p) => {
              const y = padT + chartH - (p / maxPct) * chartH;
              return (
                <g key={p}>
                  <line
                    x1={padL}
                    x2={padL + chartW}
                    y1={y}
                    y2={y}
                    stroke="var(--border)"
                    strokeWidth={p === 0 ? 1 : 0.5}
                  />
                  <text
                    x={padL - 4}
                    y={y + 3.5}
                    textAnchor="end"
                    fontSize="8"
                    fill="var(--muted-foreground)"
                  >
                    {p}%
                  </text>
                </g>
              );
            })}
            {/* Bars */}
            {bins.map((b, i) => {
              const bw = chartW / NUM_BINS;
              const x = padL + i * bw;
              const bh = (b.pct / maxPct) * chartH;
              const by = padT + chartH - bh;
              return (
                <g key={i}>
                  <rect
                    x={x + 0.5}
                    y={by}
                    width={bw - 1}
                    height={bh}
                    fill={b.color}
                    rx="1.5"
                  />
                </g>
              );
            })}
            {/* X axis */}
            <line
              x1={padL}
              x2={padL + chartW}
              y1={padT + chartH}
              y2={padT + chartH}
              stroke="var(--border)"
              strokeWidth="1"
            />
            {/* X tick labels */}
            {[0, 0.25, 0.5, 0.75, 1].map((f) => {
              const x = padL + f * chartW;
              return (
                <g key={f}>
                  <line
                    x1={x}
                    x2={x}
                    y1={padT + chartH}
                    y2={padT + chartH + 3}
                    stroke="var(--border)"
                    strokeWidth="1"
                  />
                  <text
                    x={x}
                    y={padT + chartH + 11}
                    textAnchor="middle"
                    fontSize="8"
                    fill="var(--muted-foreground)"
                  >
                    {fmtH(maxHours * f)}h
                  </text>
                </g>
              );
            })}
            {/* Y-axis label */}
            <text
              x={8}
              y={padT + chartH / 2}
              textAnchor="middle"
              fontSize="7.5"
              fill="var(--muted-foreground)"
              transform={`rotate(-90,8,${padT + chartH / 2})`}
            >
              % of vertices
            </text>
          </svg>

          {/* Table */}
          <div className="heatmap-analysis-table-wrap">
            <table className="heatmap-analysis-table">
              <thead>
                <tr>
                  <th></th>
                  <th>Range (hrs)</th>
                  <th>Vertices</th>
                  <th>%</th>
                </tr>
              </thead>
              <tbody>
                {bins.map((b, i) => (
                  <tr key={i} style={{ opacity: b.count === 0 ? 0.4 : 1 }}>
                    <td>
                      <span
                        className="heatmap-analysis-swatch"
                        style={{ background: b.color }}
                      />
                    </td>
                    <td>
                      {fmtH(b.lo)} – {fmtH(b.hi)}
                    </td>
                    <td>{b.count.toLocaleString()}</td>
                    <td>{b.pct.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [viewMode, setViewMode] = useState(() =>
    loadStored("viewMode", "cloud"),
  );
  const [solarMode, setSolarMode] = useState(false);
  const [shadowsEnabled, setShadows] = useState(() =>
    loadStored("shadowsEnabled", false),
  );
  const [pointSize, setPointSize] = useState(() =>
    loadStored("pointSize", 0.12),
  );
  const [pointOpacity, setPointOpacity] = useState(() =>
    loadStored("pointOpacity", 1),
  );
  const [meshWireframe, setMeshWireframe] = useState(() =>
    loadStored("meshWireframe", false),
  );
  const [meshOpacity, setMeshOpacity] = useState(() =>
    loadStored("meshOpacity", 1),
  );
  const [meshMaterialMode, setMeshMaterialMode] = useState(() =>
    loadStored("meshMaterialMode", "vertex"),
  );
  const [meshDoubleSided, setMeshDoubleSided] = useState(() =>
    loadStored("meshDoubleSided", true),
  );
  const [ambientIntensity, setAmbientIntensity] = useState(() =>
    loadStored("ambientIntensity", 0.4),
  );
  const [backgroundMode, setBackgroundMode] = useState(() =>
    loadStored("backgroundMode", "dark"),
  );
  const [uiTheme, setUiTheme] = useState(() => {
    const t = loadStored("theme", "dark");
    return t === "light" ? "light" : "dark";
  });
  const [fov, setFov] = useState(() => loadStored("fov", 45));
  const [clipY, setClipY] = useState(() => loadStored("clipY", null));
  // modelRotation stored as [xDeg, yDeg, zDeg] in degrees for readability
  const [rotDeg, setRotDeg] = useState(() => {
    const stored = loadStored("rotDeg", null);
    if (Array.isArray(stored) && stored.length === 3) return stored;
    // migrate old modelRotation (radians) if present
    const oldRad = loadStored("modelRotation", null);
    if (Array.isArray(oldRad) && oldRad.length === 3) {
      return oldRad.map((r) => Math.round((r * 180) / Math.PI));
    }
    return [0, 0, 0];
  });
  // baseRot: leveling locked by the user. null = free rotation mode.
  const [baseRot, setBaseRot] = useState(() => loadStored("baseRot", null));
  // heading: yaw applied in world space on top of baseRot (-180..180 deg)
  const [heading, setHeading] = useState(() => loadStored("heading", 0));

  // When baseRot is set: apply heading (world-Y rotation) THEN base leveling.
  // This lets the user spin the scan around the true vertical without tilting it.
  const modelRotation = useMemo(() => {
    if (baseRot !== null) {
      const bx = (baseRot[0] * Math.PI) / 180;
      const by = (baseRot[1] * Math.PI) / 180;
      const bz = (baseRot[2] * Math.PI) / 180;
      const h = (heading * Math.PI) / 180;
      const baseMatrix = new THREE.Matrix4().makeRotationFromEuler(
        new THREE.Euler(bx, by, bz, "XYZ"),
      );
      const headingMatrix = new THREE.Matrix4().makeRotationY(h);
      const combined = new THREE.Matrix4().multiplyMatrices(
        headingMatrix,
        baseMatrix,
      );
      const e = new THREE.Euler().setFromRotationMatrix(combined, "XYZ");
      return [e.x, e.y, e.z];
    }
    return rotDeg.map((d) => (d * Math.PI) / 180);
  }, [baseRot, heading, rotDeg]);
  const [cameraPreset, setCameraPreset] = useState(null);
  const [cameraOverrideCenter, setCameraOverrideCenter] = useState(null);
  const orbitRef = useRef();
  // Pivot point: world-space [x,y,z] clicked by user, null = use model center
  const [pivotMode, setPivotMode] = useState(false);
  const [pivotPoint, setPivotPoint] = useState(() =>
    loadStored("pivotPoint", null),
  );
  const [showPivotMarker, setShowPivotMarker] = useState(() =>
    loadStored("showPivotMarker", true),
  );
  // Ground plane
  const [showGroundPlane, setShowGroundPlane] = useState(() =>
    loadStored("showGroundPlane", false),
  );
  const [groundY, setGroundY] = useState(() => loadStored("groundY", 0));
  const [meta, setMeta] = useState(null);
  const [projectsPanelOpen, setProjectsPanelOpen] = useState(false);
  const [modelsInfo, setModelsInfo] = useState(null);
  const [projects, setProjects] = useState(() =>
    loadStored("projects", [
      { id: "default", name: "Default Project", createdAt: Date.now() },
    ]),
  );
  const [activeProjectId, setActiveProjectId] = useState(() =>
    loadStored("activeProjectId", "default"),
  );
  // Per-project OBJ files: { [projectId]: { url: blobUrl, name: fileName } }
  // Blob URLs are not serialisable, so this lives only in memory.
  const [objsByProject, setObjsByProject] = useState({});
  const objInputRef = useRef(null);
  // Always-current ref so handleObjUpload never captures a stale activeProjectId
  const activeProjectIdRef = useRef(activeProjectId);
  const [renamingProjectId, setRenamingProjectId] = useState(null);
  const [renameValue, setRenameValue] = useState("");
  const [processPanelOpen, setProcessPanelOpen] = useState(false);
  const [pipelineParams, setPipelineParams] = useState(() => ({
    video_path: "",
    fps: 5,
    max_features: 4096,
    poisson_depth: 9,
    min_density: 0.02,
    smooth_iterations: 2,
  }));
  const [videoFileInfo, setVideoFileInfo] = useState(null);
  const [videoUploading, setVideoUploading] = useState(false);
  const [videoUploadError, setVideoUploadError] = useState(null);
  const fileInputRef = useRef(null);
  const [pipelineStatus, setPipelineStatus] = useState({
    status: "idle",
    error: null,
  });
  const [logLines, setLogLines] = useState([]);
  const [solarPathOpacity, setSolarPathOpacity] = useState(() =>
    loadStored("solarPathOpacity", 1.0),
  );
  const [solarPathSize, setSolarPathSize] = useState(() =>
    loadStored("solarPathSize", 1.0),
  );
  const [solarLineThick, setSolarLineThick] = useState(() =>
    loadStored("solarLineThick", 1.0),
  );
  const [showSolarDome, setShowSolarDome] = useState(() =>
    loadStored("showSolarDome", true),
  );
  const [showSunMarker, setShowSunMarker] = useState(() =>
    loadStored("showSunMarker", true),
  );
  const [heatmapMode, setHeatmapMode] = useState(false);
  const [heatmapSampleType, setHeatmapSampleType] = useState(() =>
    loadStored("heatmapSampleType", "day"),
  ); // "day" | "month" | "year"
  const [heatmapTimeStep, setHeatmapTimeStep] = useState(() =>
    loadStored("heatmapTimeStep", 15),
  );
  // Auto-clamp sample interval when switching type so it stays within valid options
  useEffect(() => {
    const minStep =
      heatmapSampleType === "year"
        ? 30
        : heatmapSampleType === "month"
          ? 15
          : 5;
    if (heatmapTimeStep < minStep) setHeatmapTimeStep(minStep);
  }, [heatmapSampleType]); // eslint-disable-line react-hooks/exhaustive-deps
  const [heatmapTrigger, setHeatmapTrigger] = useState(0);
  const [heatmapStatus, setHeatmapStatus] = useState("idle");
  const [heatmapProgress, setHeatmapProgress] = useState(0);
  const [savedHeatmap, setSavedHeatmap] = useState(null);
  const [allSavedHeatmaps, setAllSavedHeatmaps] = useState(() =>
    listStoredHeatmaps(),
  );
  const [heatmapSavedFlash, setHeatmapSavedFlash] = useState(false);
  const [heatmapSaveError, setHeatmapSaveError] = useState(false);
  const [showHeatmapAnalysis, setShowHeatmapAnalysis] = useState(false);
  const [heatmapMaxHours, setHeatmapMaxHours] = useState(null);
  const [heatmapRangeEnabled, setHeatmapRangeEnabled] = useState(() =>
    loadStored("heatmapRangeEnabled", false),
  );
  const [heatmapRangeMin, setHeatmapRangeMin] = useState(() =>
    loadStored("heatmapRangeMin", 0),
  );
  const [heatmapRangeMax, setHeatmapRangeMax] = useState(() =>
    loadStored("heatmapRangeMax", null),
  );
  const [heatmapColormap, setHeatmapColormap] = useState(() =>
    loadStored("heatmapColormap", "thermal"),
  );
  // When the user first enables the custom range, seed max from the latest computed max
  useEffect(() => {
    if (
      heatmapRangeEnabled &&
      heatmapRangeMax === null &&
      heatmapMaxHours != null
    ) {
      setHeatmapRangeMax(parseFloat(heatmapMaxHours.toFixed(1)));
    }
  }, [heatmapRangeEnabled]); // eslint-disable-line react-hooks/exhaustive-deps
  const [activeSavedStorageKey, setActiveSavedStorageKey] = useState(null);
  const savedFlashTimerRef = useRef(null);
  const saveErrorTimerRef = useRef(null);
  const skipAutoLoadRef = useRef(false); // set by loadSavedHeatmap to prevent auto-load override
  // Keep latest modelRotation in a ref so handleHeatmapStatus always captures the current value
  const modelRotationRef = useRef(modelRotation);
  useEffect(() => {
    modelRotationRef.current = modelRotation;
  }, [modelRotation]);
  // Refresh saved list whenever the heatmap panel is opened
  useEffect(() => {
    if (heatmapMode) setAllSavedHeatmaps(listStoredHeatmaps());
  }, [heatmapMode]);

  const canvasRef = useRef(null);
  const logEndRef = useRef(null);
  const autoFitDoneRef = useRef(false);

  useEffect(() => {
    saveStored("viewMode", viewMode);
  }, [viewMode]);
  useEffect(() => {
    saveStored("pointSize", pointSize);
  }, [pointSize]);
  useEffect(() => {
    saveStored("pointOpacity", pointOpacity);
  }, [pointOpacity]);
  useEffect(() => {
    saveStored("meshWireframe", meshWireframe);
  }, [meshWireframe]);
  useEffect(() => {
    saveStored("meshOpacity", meshOpacity);
  }, [meshOpacity]);
  useEffect(() => {
    saveStored("meshMaterialMode", meshMaterialMode);
  }, [meshMaterialMode]);
  useEffect(() => {
    saveStored("meshDoubleSided", meshDoubleSided);
  }, [meshDoubleSided]);
  useEffect(() => {
    saveStored("ambientIntensity", ambientIntensity);
  }, [ambientIntensity]);
  useEffect(() => {
    saveStored("backgroundMode", backgroundMode);
  }, [backgroundMode]);
  useEffect(() => {
    saveStored("fov", fov);
  }, [fov]);
  useEffect(() => {
    saveStored("clipY", clipY);
  }, [clipY]);
  useEffect(() => {
    saveStored("rotDeg", rotDeg);
  }, [rotDeg]);
  useEffect(() => {
    saveStored("baseRot", baseRot);
  }, [baseRot]);
  useEffect(() => {
    saveStored("heading", heading);
  }, [heading]);
  useEffect(() => {
    saveStored("pivotPoint", pivotPoint);
  }, [pivotPoint]);
  useEffect(() => {
    saveStored("solarLineThick", solarLineThick);
  }, [solarLineThick]);
  useEffect(() => {
    saveStored("groundY", groundY);
  }, [groundY]);
  useEffect(() => {
    saveStored("theme", uiTheme);
    document.documentElement.classList.toggle("dark", uiTheme === "dark");
  }, [uiTheme]);
  useEffect(() => {
    saveStored("shadowsEnabled", shadowsEnabled);
  }, [shadowsEnabled]);
  useEffect(() => {
    saveStored("showPivotMarker", showPivotMarker);
  }, [showPivotMarker]);
  useEffect(() => {
    saveStored("showGroundPlane", showGroundPlane);
  }, [showGroundPlane]);
  useEffect(() => {
    saveStored("solarPathOpacity", solarPathOpacity);
  }, [solarPathOpacity]);
  useEffect(() => {
    saveStored("solarPathSize", solarPathSize);
  }, [solarPathSize]);
  useEffect(() => {
    saveStored("showSolarDome", showSolarDome);
  }, [showSolarDome]);
  useEffect(() => {
    saveStored("showSunMarker", showSunMarker);
  }, [showSunMarker]);
  useEffect(() => {
    saveStored("heatmapSampleType", heatmapSampleType);
  }, [heatmapSampleType]);
  useEffect(() => {
    saveStored("heatmapTimeStep", heatmapTimeStep);
  }, [heatmapTimeStep]);
  useEffect(() => {
    saveStored("heatmapRangeEnabled", heatmapRangeEnabled);
  }, [heatmapRangeEnabled]);
  useEffect(() => {
    saveStored("heatmapRangeMin", heatmapRangeMin);
  }, [heatmapRangeMin]);
  useEffect(() => {
    saveStored("heatmapRangeMax", heatmapRangeMax);
  }, [heatmapRangeMax]);
  useEffect(() => {
    saveStored("heatmapColormap", heatmapColormap);
  }, [heatmapColormap]);

  // Cancel pivot mode with Escape
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") setPivotMode(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    fetch("/models/point_cloud_meta.json")
      .then((r) => r.json())
      .then((data) => {
        if (
          data &&
          typeof data === "object" &&
          (data.center != null || data.extent != null)
        ) {
          setMeta(data);
        }
      })
      .catch(() => {});
  }, [pipelineStatus.status]);

  // Check whether the static model files are actually present by HEAD-requesting them.
  // This works even without the Python API server running.
  useEffect(() => {
    Promise.all([
      fetch("/models/point_cloud.ply", { method: "HEAD" })
        .then((r) => (r.ok ? { exists: true } : null))
        .catch(() => null),
      fetch("/models/drone_mesh.glb", { method: "HEAD" })
        .then((r) => (r.ok ? { exists: true } : null))
        .catch(() => null),
    ]).then(([ply, glb]) => {
      setModelsInfo((prev) => ({
        ...prev,
        point_cloud: prev?.point_cloud ?? ply,
        mesh: prev?.mesh ?? glb,
      }));
    });
  }, [pipelineStatus.status]); // re-check after pipeline finishes

  // Auto-fit camera to model center the first time meta loads
  useEffect(() => {
    if (meta?.center && !autoFitDoneRef.current) {
      autoFitDoneRef.current = true;
      setCameraPreset("auto");
    }
  }, [meta]);

  const statusPollFailuresRef = useRef(0);
  const [statusPollIntervalMs, setStatusPollIntervalMs] = useState(2000);

  const pollStatus = useCallback(() => {
    fetch(API_BASE + "/api/status")
      .then((r) => r.json())
      .then((d) => {
        statusPollFailuresRef.current = 0;
        setStatusPollIntervalMs(2000);
        if (d && typeof d.status !== "undefined") {
          setPipelineStatus({ status: d.status, error: d.error ?? null });
        }
        if (d?.models_info) setModelsInfo(d.models_info);
        const m = d?.models_info?.meta;
        if (m && typeof m === "object" && (m.center || m.extent != null)) {
          setMeta(m);
        }
      })
      .catch(() => {
        statusPollFailuresRef.current += 1;
        if (statusPollFailuresRef.current >= 2) {
          setStatusPollIntervalMs(15000);
        }
      });
  }, []);

  useEffect(() => {
    const t = setInterval(pollStatus, statusPollIntervalMs);
    return () => clearInterval(t);
  }, [pollStatus, statusPollIntervalMs]);

  useEffect(() => {
    if (pipelineStatus.status !== "running") return;
    const es = new EventSource(API_BASE + "/api/pipeline/log");
    const lines = [];
    es.onmessage = (e) => {
      if (e.data) {
        lines.push(e.data);
        setLogLines((prev) => [...prev.slice(-400), e.data]);
      }
    };
    es.onerror = () => es.close();
    return () => es.close();
  }, [pipelineStatus.status]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logLines]);

  const runPipeline = useCallback(() => {
    setLogLines([]);
    fetch(API_BASE + "/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_path: pipelineParams.video_path || undefined,
        fps: pipelineParams.fps,
        max_features: pipelineParams.max_features,
        poisson_depth: pipelineParams.poisson_depth,
        min_density: pipelineParams.min_density,
        smooth_iterations: pipelineParams.smooth_iterations,
      }),
    })
      .then((r) => r.json())
      .then((d) => {
        if (!d.ok) setPipelineStatus((s) => ({ ...s, error: d.error }));
      })
      .catch((e) => setPipelineStatus((s) => ({ ...s, error: e.message })));
  }, [pipelineParams]);

  const rebuildMesh = useCallback(() => {
    setLogLines([]);
    fetch(API_BASE + "/api/mesh/rebuild", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        poisson_depth: pipelineParams.poisson_depth,
        min_density: pipelineParams.min_density,
        smooth_iterations: pipelineParams.smooth_iterations,
      }),
    })
      .then((r) => r.json())
      .then((d) => {
        if (!d.ok) setPipelineStatus((s) => ({ ...s, error: d.error }));
      })
      .catch((e) => setPipelineStatus((s) => ({ ...s, error: e.message })));
  }, [pipelineParams]);

  const handleVideoFileSelect = useCallback(async (e) => {
    const file = e.target?.files?.[0];
    if (!file) return;
    setVideoUploadError(null);
    setVideoUploading(true);
    setVideoFileInfo(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const r = await fetch(API_BASE + "/api/upload/video", {
        method: "POST",
        body: formData,
      });
      const data = await r.json();
      if (!data.ok) {
        throw new Error(data.error || "Upload failed");
      }
      setPipelineParams((p) => ({ ...p, video_path: data.path }));
      setVideoFileInfo({
        filename: data.filename,
        size_mb: data.size_mb,
        ...data.metadata,
      });
    } catch (err) {
      setVideoUploadError(err.message);
    } finally {
      setVideoUploading(false);
      e.target.value = "";
    }
  }, []);

  const handleProbeVideo = useCallback(async () => {
    const path = pipelineParams.video_path?.trim();
    if (!path) return;
    setVideoUploadError(null);
    try {
      const r = await fetch(API_BASE + "/api/video/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_path: path }),
      });
      const data = await r.json();
      if (!data.ok) throw new Error(data.error || "Probe failed");
      setVideoFileInfo((prev) => ({ ...prev, ...data.metadata }));
    } catch (err) {
      setVideoUploadError(err.message);
    }
  }, [pipelineParams.video_path]);

  const applyPreset = useCallback((name) => {
    if (name === "fast") {
      setPipelineParams((p) => ({
        ...p,
        fps: 3,
        max_features: 2048,
        poisson_depth: 7,
      }));
    } else if (name === "balanced") {
      setPipelineParams((p) => ({
        ...p,
        fps: 5,
        max_features: 4096,
        poisson_depth: 9,
      }));
    } else if (name === "high") {
      setPipelineParams((p) => ({
        ...p,
        fps: 8,
        max_features: 8192,
        poisson_depth: 11,
      }));
    }
  }, []);

  const switchToProject = useCallback((id) => {
    activeProjectIdRef.current = id;
    setActiveProjectId(id);
    saveStored("activeProjectId", id);
    // OBJ projects don't have point-cloud or "both" — force mesh view
    if (id !== "default") {
      setViewMode("mesh");
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const createNewProject = useCallback(() => {
    const id = `proj_${Date.now()}`;
    const newProj = {
      id,
      name: `Project ${projects.length + 1}`,
      createdAt: Date.now(),
    };
    const updated = [...projects, newProj];
    setProjects(updated);
    saveStored("projects", updated);
    switchToProject(id);
  }, [projects, switchToProject]);

  const handleObjUpload = useCallback(
    (e) => {
      const file = e.target?.files?.[0];
      if (!file) return;
      // Read from ref — always the current project even if React hasn't re-rendered yet
      const projectId = activeProjectIdRef.current;
      setObjsByProject((prev) => {
        if (prev[projectId]?.url) URL.revokeObjectURL(prev[projectId].url);
        return {
          ...prev,
          [projectId]: { url: URL.createObjectURL(file), name: file.name },
        };
      });
      e.target.value = "";
    },
    [], // no dependency — always reads live value from ref
  );

  const removeProjectObj = useCallback(() => {
    const projectId = activeProjectIdRef.current;
    setObjsByProject((prev) => {
      if (prev[projectId]?.url) URL.revokeObjectURL(prev[projectId].url);
      const next = { ...prev };
      delete next[projectId];
      return next;
    });
  }, []);

  const commitRename = useCallback(() => {
    const trimmed = renameValue.trim();
    if (!trimmed || !renamingProjectId) {
      setRenamingProjectId(null);
      return;
    }
    setProjects((prev) => {
      const updated = prev.map((p) =>
        p.id === renamingProjectId ? { ...p, name: trimmed } : p,
      );
      saveStored("projects", updated);
      return updated;
    });
    setRenamingProjectId(null);
  }, [renamingProjectId, renameValue]);

  const bgColor = useMemo(() => {
    if (backgroundMode === "sky") return "#87ceeb";
    return uiTheme === "dark" ? "#0a0a0a" : "#f4f4f5";
  }, [backgroundMode, uiTheme]);

  const groundYRange = useMemo(() => {
    if (!meta || meta.bbox_min == null || meta.bbox_max == null)
      return [-100, 100];
    const e = meta.extent ?? 50;
    const y1 = meta.bbox_min[1];
    const y2 = meta.bbox_max[1];
    if (typeof y1 !== "number" || typeof y2 !== "number") return [-100, 100];
    return [
      parseFloat((y1 - e * 0.4).toFixed(1)),
      parseFloat((y2 + e * 0.4).toFixed(1)),
    ];
  }, [meta]);

  const today = new Date();
  const [dateStr, setDateStr] = useState(() =>
    loadStored("dateStr", today.toISOString().slice(0, 10)),
  );
  // Day-of-year derived from dateStr (1–365)
  const dayOfYear = useMemo(() => {
    // Parse as LOCAL date to avoid UTC-offset day-boundary errors
    const [y, m, d] = dateStr.split("-").map(Number);
    return Math.floor((new Date(y, m - 1, d) - new Date(y, 0, 0)) / 86400000);
  }, [dateStr]);
  const setDayOfYear = useCallback(
    (doy) => {
      const year = parseInt(dateStr.slice(0, 4), 10);
      const d = new Date(year, 0, Number(doy)); // LOCAL midnight
      // Format manually so there's no UTC conversion
      const yy = d.getFullYear();
      const mm = String(d.getMonth() + 1).padStart(2, "0");
      const dd = String(d.getDate()).padStart(2, "0");
      setDateStr(`${yy}-${mm}-${dd}`);
    },
    [dateStr],
  );
  const [timeMin, setTimeMin] = useState(() => loadStored("timeMin", 12 * 60));
  const [lat, setLat] = useState(() => loadStored("lat", 42.3314));
  const [lng, setLng] = useState(() => loadStored("lng", -83.0458));
  const [showMapModal, setShowMapModal] = useState(false);
  const [locationLabel, setLocationLabel] = useState(() =>
    loadStored("locationLabel", ""),
  );
  useEffect(() => {
    saveStored("dateStr", dateStr);
  }, [dateStr]);
  useEffect(() => {
    saveStored("timeMin", timeMin);
  }, [timeMin]);
  useEffect(() => {
    saveStored("lat", lat);
  }, [lat]);
  useEffect(() => {
    saveStored("lng", lng);
  }, [lng]);
  useEffect(() => {
    saveStored("locationLabel", locationLabel);
  }, [locationLabel]);

  // Base settings key (no timestamp) — identifies a unique set of compute parameters
  const heatmapSettingsKey = useMemo(() => {
    const normDate =
      heatmapSampleType === "year"
        ? dateStr.slice(0, 4)
        : heatmapSampleType === "month"
          ? dateStr.slice(0, 7)
          : dateStr;
    return [
      normDate,
      heatmapTimeStep,
      heatmapSampleType,
      Number(lat).toFixed(4),
      Number(lng).toFixed(4),
    ].join("_");
  }, [dateStr, heatmapTimeStep, heatmapSampleType, lat, lng]);
  useEffect(() => {
    // loadSavedHeatmap sets this flag to prevent the auto-load from overriding its choice
    if (skipAutoLoadRef.current) {
      skipAutoLoadRef.current = false;
      return;
    }
    // Find the most recently computed entry for the current settings
    const all = listStoredHeatmaps();
    const match = all.find((h) => h.settingsKey === heatmapSettingsKey);
    const cached = match
      ? loadStored("heatmapCache_" + match.cacheKey, null)
      : null;
    const rawArrays =
      cached?.heatArraysB64 && typeof cached.maxHeat === "number"
        ? decodeHeatArrays(cached.heatArraysB64, cached.maxHeat)
        : (cached?.heatArrays ?? null);
    const valid =
      cached &&
      typeof cached === "object" &&
      typeof cached.dateStr === "string" &&
      typeof cached.timeStep === "number" &&
      typeof cached.maxHeat === "number" &&
      Array.isArray(rawArrays) &&
      rawArrays.every((a) => Array.isArray(a));
    const sh = valid ? { ...cached, heatArrays: rawArrays } : null;
    setSavedHeatmap(sh);
    setHeatmapMaxHours(sh ? (sh.maxHeat * sh.timeStep) / 60 : null);
    setActiveSavedStorageKey(sh ? match.storageKey : null);
  }, [heatmapSettingsKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleHeatmapStatus = useCallback(
    (s, payload) => {
      const [base, pct] = String(s).split(":");
      setHeatmapStatus(base);
      if (pct !== undefined) setHeatmapProgress(Number(pct));
      else if (base === "done") {
        setHeatmapProgress(100);
        if (payload?.heatArrays != null && payload?.maxHeat != null) {
          const computedAt = Date.now();
          const savedModelRotation = modelRotationRef.current
            ? [...modelRotationRef.current]
            : null;
          // State object keeps plain arrays so existing rendering code is unchanged
          const toSave = {
            dateStr,
            timeStep: heatmapTimeStep,
            sampleType: heatmapSampleType,
            lat,
            lng,
            modelRotation: savedModelRotation,
            heatArrays: payload.heatArrays,
            maxHeat: payload.maxHeat,
          };
          // Each compute gets a unique storage key (settingsKey + timestamp) → never overwrites
          const uniqueCacheKey = heatmapSettingsKey + "_" + computedAt;
          // Storage object uses compact base64 Uint16 encoding (~6× smaller than JSON floats)
          const toStore = {
            dateStr,
            timeStep: heatmapTimeStep,
            sampleType: heatmapSampleType,
            settingsKey: heatmapSettingsKey,
            computedAt,
            lat,
            lng,
            modelRotation: savedModelRotation,
            heatArraysB64: encodeHeatArrays(
              payload.heatArrays,
              payload.maxHeat,
            ),
            maxHeat: payload.maxHeat,
          };
          const newStorageKey = STORAGE_KEY + "heatmapCache_" + uniqueCacheKey;
          // Always apply to scene regardless of whether storage succeeds
          setSavedHeatmap(toSave);
          const newMax = (payload.maxHeat * heatmapTimeStep) / 60;
          setHeatmapMaxHours(newMax);
          setActiveSavedStorageKey(newStorageKey);
          try {
            saveStored("heatmapCache_" + uniqueCacheKey, toStore);
            setAllSavedHeatmaps(listStoredHeatmaps());
            if (savedFlashTimerRef.current)
              clearTimeout(savedFlashTimerRef.current);
            setHeatmapSavedFlash(true);
            savedFlashTimerRef.current = setTimeout(
              () => setHeatmapSavedFlash(false),
              2500,
            );
          } catch (_) {
            // Storage quota exceeded or other error — notify user
            if (saveErrorTimerRef.current)
              clearTimeout(saveErrorTimerRef.current);
            setHeatmapSaveError(true);
            saveErrorTimerRef.current = setTimeout(
              () => setHeatmapSaveError(false),
              4000,
            );
          }
        }
      } else if (base !== "computing") setHeatmapProgress(0);
    },
    [dateStr, heatmapTimeStep, heatmapSampleType, lat, lng, heatmapSettingsKey],
  );

  const deleteHeatmap = useCallback((storageKey) => {
    try {
      localStorage.removeItem(storageKey);
    } catch (_) {}
    setAllSavedHeatmaps(listStoredHeatmaps());
    // Clear display only if we deleted the currently active entry
    setActiveSavedStorageKey((prev) => {
      if (prev === storageKey) {
        setSavedHeatmap(null);
        setHeatmapMaxHours(null);
        return null;
      }
      return prev;
    });
  }, []);

  const loadSavedHeatmap = useCallback((saved) => {
    // Prevent the auto-load useEffect from overriding our explicit choice
    skipAutoLoadRef.current = true;
    setDateStr(saved.dateStr);
    setHeatmapTimeStep(saved.timeStep);
    setHeatmapSampleType(saved.sampleType ?? "day");
    setLat(saved.lat);
    setLng(saved.lng);
    setHeatmapTrigger(0);
    setHeatmapStatus("idle");
    setSolarMode(false);
    setHeatmapMode(true);
    setActiveSavedStorageKey(saved.storageKey);
    // Load the full data for this specific entry directly
    const full = loadStored("heatmapCache_" + saved.cacheKey, null);
    const rawArrays =
      full?.heatArraysB64 && typeof full.maxHeat === "number"
        ? decodeHeatArrays(full.heatArraysB64, full.maxHeat)
        : (full?.heatArrays ?? null);
    if (
      rawArrays &&
      Array.isArray(rawArrays) &&
      rawArrays.every((a) => Array.isArray(a))
    ) {
      setSavedHeatmap({ ...full, heatArrays: rawArrays });
      setHeatmapMaxHours((full.maxHeat * full.timeStep) / 60);
    }
  }, []);

  const reverseGeocode = useCallback(async (la, ln) => {
    try {
      const r = await fetch(
        `https://nominatim.openstreetmap.org/reverse?lat=${la}&lon=${ln}&format=json`,
      );
      const d = await r.json();
      const a = d.address ?? {};
      setLocationLabel(
        [a.city || a.town || a.village || a.county, a.state, a.country]
          .filter(Boolean)
          .join(", "),
      );
    } catch {
      setLocationLabel("");
    }
  }, []);

  // Geocode on first load
  useEffect(() => {
    reverseGeocode(lat, lng);
  }, []); // eslint-disable-line
  const hour = Math.floor(timeMin / 60);
  const min = timeMin % 60;
  const sunPos = useMemo(() => {
    const d = new Date(
      `${dateStr}T${String(hour).padStart(2, "0")}:${String(min).padStart(2, "0")}:00`,
    );
    const pos = SunCalc.getPosition(d, lat, lng);
    const times = SunCalc.getTimes(d, lat, lng);
    const fmt = (t) =>
      t instanceof Date && !isNaN(t)
        ? t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
        : "--";
    return {
      azimuth: pos.azimuth,
      altitude: pos.altitude,
      altDeg: ((pos.altitude * 180) / Math.PI).toFixed(1),
      azDeg: (((pos.azimuth * 180) / Math.PI + 360) % 360).toFixed(1),
      isDay: pos.altitude > 0,
      sunrise: fmt(times.sunrise),
      sunset: fmt(times.sunset),
    };
  }, [dateStr, hour, min, lat, lng]);

  const nudgeRot = useCallback((axis, delta) => {
    setRotDeg((prev) => {
      const next = [...prev];
      let v = next[axis] + delta;
      while (v > 180) v -= 360;
      while (v <= -180) v += 360;
      next[axis] = Math.round(v);
      return next;
    });
  }, []);

  const nudgeHeading = useCallback((delta) => {
    setHeading((prev) => {
      let v = prev + delta;
      while (v > 180) v -= 360;
      while (v <= -180) v += 360;
      return Math.round(v);
    });
  }, []);

  const handleCameraPresetConsumed = useCallback(() => {
    setCameraPreset(null);
    setCameraOverrideCenter(null);
  }, []);

  return (
    <div
      className={`App${pivotMode ? " pivot-mode" : ""}`}
      style={{ background: bgColor }}
    >
      <div
        style={{
          position: "relative",
          zIndex: 0,
          width: "100%",
          height: "100%",
        }}
      >
        <Canvas
          ref={canvasRef}
          shadows={shadowsEnabled}
          camera={{ position: [0, 50, 120], fov: fov, near: 0.1, far: 5000 }}
          gl={{ antialias: true, logarithmicDepthBuffer: true }}
        >
          <ShadowMapType />
          <Scene
            viewMode={solarMode || heatmapMode ? "solar" : viewMode}
            pointSize={pointSize}
            pointOpacity={pointOpacity}
            clipY={clipY}
            sunAz={sunPos.azimuth}
            sunAlt={sunPos.altitude}
            shadowsEnabled={shadowsEnabled}
            meshWireframe={meshWireframe}
            meshOpacity={meshOpacity}
            meshMaterialMode={meshMaterialMode}
            meshDoubleSided={meshDoubleSided}
            ambientIntensity={ambientIntensity}
            bgColor={bgColor}
            fov={fov}
            cameraPreset={cameraPreset}
            onCameraPresetConsumed={handleCameraPresetConsumed}
            meta={meta}
            solarMode={solarMode}
            modelRotation={modelRotation}
            pivotMode={pivotMode}
            onPivotPick={(pt) => {
              setPivotPoint(pt);
              setPivotMode(false);
              setGroundY(parseFloat(pt[1].toFixed(2)));
            }}
            pivotPoint={pivotPoint}
            showPivotMarker={showPivotMarker}
            showGroundPlane={showGroundPlane}
            groundY={groundY}
            orbitRef={orbitRef}
            overrideCenter={cameraOverrideCenter}
            heatmapMode={heatmapMode}
            heatmapSampleType={heatmapSampleType}
            heatmapTrigger={heatmapTrigger}
            heatmapTimeStep={heatmapTimeStep}
            savedHeatmap={savedHeatmap}
            heatmapRangeMinHrs={heatmapRangeEnabled ? heatmapRangeMin : null}
            heatmapRangeMaxHrs={heatmapRangeEnabled ? heatmapRangeMax : null}
            heatmapColormap={heatmapColormap}
            lat={lat}
            lng={lng}
            dateStr={dateStr}
            onHeatmapStatus={handleHeatmapStatus}
            showServerModels={activeProjectId === "default"}
            objMeshUrl={activeProjectId !== "default" ? (objsByProject[activeProjectId]?.url ?? null) : null}
          />
          {(solarMode || heatmapMode) && showSolarDome && (
            <SolarPathDome
              lat={lat}
              lng={lng}
              dateStr={dateStr}
              sunAz={sunPos.azimuth}
              sunAlt={sunPos.altitude}
              meta={meta}
              pivot={pivotPoint}
              pathOpacity={solarPathOpacity}
              pathSize={solarPathSize}
              lineThick={solarLineThick}
              showSunMarker={showSunMarker}
            />
          )}
          {objsByProject[activeProjectId]?.url && (
            <group
              onClick={(e) => {
                if (!pivotMode) return;
                e.stopPropagation();
                setPivotPoint([e.point.x, e.point.y, e.point.z]);
                setPivotMode(false);
                setGroundY(parseFloat(e.point.y.toFixed(2)));
              }}
            >
              <ObjModel
                url={objsByProject[activeProjectId].url}
                modelRotation={modelRotation}
                wireframe={meshWireframe}
                opacity={meshOpacity}
                doubleSided={meshDoubleSided}
                materialMode={meshMaterialMode}
              />
            </group>
          )}
          <PivotSync pivotPoint={pivotPoint} meta={meta} orbitRef={orbitRef} />
          <OrbitControls
            ref={orbitRef}
            enableDamping
            dampingFactor={0.06}
            minDistance={1}
            maxDistance={2000}
          />
        </Canvas>
      </div>

      <div className="top-bar">
        <div className="mode-tabs">
          <button
            className={`tab projects-tab ${projectsPanelOpen ? "active" : ""}`}
            onClick={() => setProjectsPanelOpen((o) => !o)}
          >
            Projects
          </button>
          {activeProjectId === "default" ? (
            <>
              {[
                { id: "cloud", label: "Point Cloud" },
                { id: "mesh", label: "Mesh" },
                { id: "both", label: "Both" },
              ].map(({ id, label }) => (
                <button
                  key={id}
                  className={`tab ${viewMode === id && !solarMode && !heatmapMode ? "active" : ""}`}
                  onClick={() => {
                    setViewMode(id);
                    setSolarMode(false);
                    setShadows(false);
                    if (heatmapMode) {
                      setHeatmapMode(false);
                      setHeatmapTrigger(0);
                      setHeatmapStatus("idle");
                    }
                  }}
                >
                  {label}
                </button>
              ))}
              <button
                className={`tab solar-tab ${solarMode ? "active" : ""}`}
                onClick={() => {
                  const next = !solarMode;
                  setSolarMode(next);
                  setShadows(next);
                  if (heatmapMode) {
                    setHeatmapMode(false);
                    setHeatmapTrigger(0);
                    setHeatmapStatus("idle");
                  }
                }}
              >
                Solar Study
              </button>
              <button
                className={`tab heatmap-tab ${heatmapMode ? "active" : ""}`}
                onClick={() => {
                  const next = !heatmapMode;
                  setHeatmapMode(next);
                  if (!next) {
                    setHeatmapTrigger(0);
                    setHeatmapStatus("idle");
                  } else {
                    setSolarMode(false);
                  }
                }}
              >
                Sun Heatmap
                {allSavedHeatmaps.length > 0 && (
                  <span className="heatmap-saved-badge">{allSavedHeatmaps.length}</span>
                )}
              </button>
            </>
          ) : (
            /* OBJ project — mesh + solar study + sun heatmap */
            <>
              <button className="tab active obj-mesh-tab" disabled>
                Mesh
                <span className="obj-mesh-tab-badge">OBJ</span>
              </button>
              <button
                className={`tab solar-tab ${solarMode ? "active" : ""}`}
                onClick={() => {
                  const next = !solarMode;
                  setSolarMode(next);
                  setShadows(next);
                  if (heatmapMode) {
                    setHeatmapMode(false);
                    setHeatmapTrigger(0);
                    setHeatmapStatus("idle");
                  }
                }}
              >
                Solar Study
              </button>
              <button
                className={`tab heatmap-tab ${heatmapMode ? "active" : ""}`}
                onClick={() => {
                  const next = !heatmapMode;
                  setHeatmapMode(next);
                  if (!next) {
                    setHeatmapTrigger(0);
                    setHeatmapStatus("idle");
                  } else {
                    setSolarMode(false);
                  }
                }}
              >
                Sun Heatmap
                {allSavedHeatmaps.length > 0 && (
                  <span className="heatmap-saved-badge">{allSavedHeatmaps.length}</span>
                )}
              </button>
            </>
          )}
          <button
            className={`tab process-tab ${processPanelOpen ? "active" : ""}`}
            onClick={() => setProcessPanelOpen((o) => !o)}
          >
            Process
          </button>
        </div>
        <button
          type="button"
          className="theme-toggle"
          onClick={() => setUiTheme((t) => (t === "dark" ? "light" : "dark"))}
          title={
            uiTheme === "dark" ? "Switch to light mode" : "Switch to dark mode"
          }
          aria-label={
            uiTheme === "dark" ? "Switch to light mode" : "Switch to dark mode"
          }
        >
          {uiTheme === "dark" ? (
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="5" />
              <line x1="12" y1="1" x2="12" y2="3" />
              <line x1="12" y1="21" x2="12" y2="23" />
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
              <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
              <line x1="1" y1="12" x2="3" y2="12" />
              <line x1="21" y1="12" x2="23" y2="12" />
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
              <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
            </svg>
          ) : (
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            </svg>
          )}
        </button>
      </div>

      {/* Left panel: mode-specific controls + shared controls */}
      <div className="float-panel left-panel">
        <div className="panel-scroll">
          {/* Point Cloud controls — shown in cloud and both modes */}
          {!solarMode && (viewMode === "cloud" || viewMode === "both") && (
            <section className="panel-section">
              <h3 className="panel-title">Point Cloud</h3>
              <div className="panel-row">
                <label>
                  Size <strong>{pointSize.toFixed(2)}</strong>
                </label>
                <input
                  type="range"
                  min={0.02}
                  max={0.6}
                  step={0.01}
                  value={pointSize}
                  onChange={(e) => setPointSize(Number(e.target.value))}
                />
              </div>
              <div className="panel-row">
                <label>
                  Opacity <strong>{(pointOpacity * 100).toFixed(0)}%</strong>
                </label>
                <input
                  type="range"
                  min={0.1}
                  max={1}
                  step={0.05}
                  value={pointOpacity}
                  onChange={(e) => setPointOpacity(Number(e.target.value))}
                />
              </div>
              {meta?.num_points && (
                <div className="stat-pill">
                  {(meta.num_points / 1000).toFixed(0)}K points
                </div>
              )}
              {meta?.extent && (
                <div className="stat-pill">
                  Extent {meta.extent.toFixed(0)}m
                </div>
              )}
            </section>
          )}

          {/* Mesh controls — shown in mesh/both modes and in solar mode */}
          {(solarMode || viewMode === "mesh" || viewMode === "both") && (
            <section className="panel-section">
              <h3 className="panel-title">Mesh</h3>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={meshWireframe}
                  onChange={(e) => setMeshWireframe(e.target.checked)}
                />
                Wireframe
              </label>
              <div className="panel-row">
                <label>
                  Opacity <strong>{(meshOpacity * 100).toFixed(0)}%</strong>
                </label>
                <input
                  type="range"
                  min={0.1}
                  max={1}
                  step={0.05}
                  value={meshOpacity}
                  onChange={(e) => setMeshOpacity(Number(e.target.value))}
                />
              </div>
              <div className="panel-row">
                <label>Material</label>
                <select
                  value={meshMaterialMode}
                  onChange={(e) => setMeshMaterialMode(e.target.value)}
                >
                  <option value="vertex">Vertex Color</option>
                  <option value="solid">Solid Gray</option>
                  <option value="normal">Normal Map</option>
                </select>
              </div>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={meshDoubleSided}
                  onChange={(e) => setMeshDoubleSided(e.target.checked)}
                />
                Double-sided
              </label>
            </section>
          )}

          <section className="panel-section">
            <h3 className="panel-title">Orientation</h3>

            {baseRot === null ? (
              /* ── FREE MODE: level the scan ─────────────────────────────── */
              <>
                <p className="panel-hint" style={{ marginBottom: "0.4rem" }}>
                  Step 1 — tilt the scan until the floor is flat
                </p>
                {[
                  { axis: "X", color: "#ff5555" },
                  { axis: "Y", color: "#44ee66" },
                  { axis: "Z", color: "#4499ff" },
                ].map(({ axis, color }, i) => (
                  <div key={axis} className="orient-row">
                    <div className="orient-controls">
                      <span
                        className="orient-axis-badge"
                        style={{ color, borderColor: color }}
                      >
                        {axis}
                      </span>
                      {[-90, -10, -1].map((d) => (
                        <button
                          key={d}
                          type="button"
                          className="orient-step-btn"
                          onClick={() => nudgeRot(i, d)}
                        >
                          {d}
                        </button>
                      ))}
                      <input
                        type="number"
                        className="orient-num-input"
                        value={rotDeg[i]}
                        step={1}
                        onWheel={(e) => e.stopPropagation()}
                        onChange={(e) => {
                          const v = e.target.valueAsNumber;
                          if (!isNaN(v)) {
                            const next = [...rotDeg];
                            next[i] = Math.max(
                              -180,
                              Math.min(180, Math.round(v)),
                            );
                            setRotDeg(next);
                          }
                        }}
                      />
                      <span className="orient-deg">°</span>
                      {[1, 10, 90].map((d) => (
                        <button
                          key={d}
                          type="button"
                          className="orient-step-btn"
                          onClick={() => nudgeRot(i, d)}
                        >
                          +{d}
                        </button>
                      ))}
                    </div>
                    <input
                      type="range"
                      min={-180}
                      max={180}
                      step={1}
                      value={rotDeg[i]}
                      className="orient-slider"
                      style={{ "--orient-accent": color }}
                      onChange={(e) => {
                        const next = [...rotDeg];
                        next[i] = Number(e.target.value);
                        setRotDeg(next);
                      }}
                    />
                  </div>
                ))}
                <div className="orient-presets">
                  <button type="button" onClick={() => setRotDeg([0, 0, 0])}>
                    Reset
                  </button>
                  <button type="button" onClick={() => setRotDeg([-90, 0, 0])}>
                    −90° X
                  </button>
                  <button type="button" onClick={() => setRotDeg([90, 0, 0])}>
                    +90° X
                  </button>
                  <button type="button" onClick={() => setRotDeg([180, 0, 0])}>
                    Flip X
                  </button>
                  <button type="button" onClick={() => setRotDeg([0, 180, 0])}>
                    Flip Y
                  </button>
                  <button type="button" onClick={() => setRotDeg([0, 0, 180])}>
                    Flip Z
                  </button>
                </div>
                <button
                  type="button"
                  className="lock-level-btn"
                  onClick={() => {
                    setBaseRot(rotDeg);
                    setHeading(0);
                  }}
                >
                  Step 2 — Lock as Floor Level
                </button>
              </>
            ) : (
              /* ── LOCKED MODE: control heading only ─────────────────────── */
              <>
                <div className="level-locked-bar">
                  <div className="level-locked-label">
                    <span>Floor level locked</span>
                    <span className="level-locked-vals">
                      X{baseRot[0]}° Y{baseRot[1]}° Z{baseRot[2]}°
                    </span>
                  </div>
                  <button
                    type="button"
                    className="edit-level-btn"
                    onClick={() => {
                      setRotDeg(baseRot);
                      setBaseRot(null);
                    }}
                  >
                    Edit Level
                  </button>
                </div>

                <p className="panel-hint" style={{ margin: "0.4rem 0" }}>
                  Rotate the scan to face the sun — floor stays flat
                </p>

                <div className="orient-row">
                  <div className="orient-controls">
                    <span
                      className="orient-axis-badge"
                      style={{
                        color: "var(--primary)",
                        borderColor: "var(--primary)",
                      }}
                    >
                      ↻
                    </span>
                    {[-90, -15, -1].map((d) => (
                      <button
                        key={d}
                        type="button"
                        className="orient-step-btn"
                        onClick={() => nudgeHeading(d)}
                      >
                        {d}
                      </button>
                    ))}
                    <input
                      type="number"
                      className="orient-num-input"
                      value={heading}
                      step={1}
                      onWheel={(e) => e.stopPropagation()}
                      onChange={(e) => {
                        const v = e.target.valueAsNumber;
                        if (!isNaN(v)) {
                          let h = Math.round(v);
                          while (h > 180) h -= 360;
                          while (h <= -180) h += 360;
                          setHeading(h);
                        }
                      }}
                    />
                    <span className="orient-deg">°</span>
                    {[1, 15, 90].map((d) => (
                      <button
                        key={d}
                        type="button"
                        className="orient-step-btn"
                        onClick={() => nudgeHeading(d)}
                      >
                        +{d}
                      </button>
                    ))}
                  </div>
                  <input
                    type="range"
                    min={-180}
                    max={180}
                    step={1}
                    value={heading}
                    className="orient-slider"
                    style={{ "--orient-accent": "var(--primary)" }}
                    onChange={(e) => setHeading(Number(e.target.value))}
                  />
                </div>
                <div className="heading-compass-row">
                  <button
                    type="button"
                    className="compass-btn"
                    onClick={() => setHeading(0)}
                  >
                    Reset
                  </button>
                </div>
              </>
            )}

            <div className="panel-divider" />
            <p className="panel-hint">
              Pivot — orbit rotates around this point
            </p>
            <div className="pivot-controls">
              <button
                type="button"
                className={`pivot-pick-btn${pivotMode ? " active" : ""}`}
                onClick={() => setPivotMode((m) => !m)}
              >
                {pivotMode ? "▶ Click model…" : "Pick Pivot"}
              </button>
              <button
                type="button"
                disabled={!pivotPoint}
                onClick={() => setPivotPoint(null)}
              >
                Clear
              </button>
            </div>
            {pivotPoint && (
              <div className="pivot-info-row">
                <label className="toggle-row" style={{ marginBottom: 0 }}>
                  <input
                    type="checkbox"
                    checked={showPivotMarker}
                    onChange={(e) => setShowPivotMarker(e.target.checked)}
                  />
                  Show Pivot Point
                </label>
              </div>
            )}
            <div className="pivot-xyz-row">
              <label>Position</label>
              <div className="pivot-xyz-inputs">
                {["X", "Y", "Z"].map((axis, i) => {
                  const base = pivotPoint ?? meta?.center ?? [0, 0, 0];
                  const val = base[i];
                  return (
                    <label key={axis} className="pivot-xyz-label">
                      <span>{axis}</span>
                      <input
                        type="number"
                        step="any"
                        value={val}
                        onWheel={(e) => e.target.blur()}
                        onChange={(e) => {
                          const v = parseFloat(e.target.value);
                          const next = [...(pivotPoint ?? base)];
                          next[i] = Number.isFinite(v) ? v : next[i];
                          setPivotPoint(next);
                        }}
                      />
                    </label>
                  );
                })}
              </div>
            </div>
          </section>
          <section className="panel-section">
            <h3 className="panel-title">Ground Plane</h3>
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={showGroundPlane}
                onChange={(e) => setShowGroundPlane(e.target.checked)}
              />
              Show ground grid
            </label>
            {showGroundPlane && (
              <>
                <div className="panel-row">
                  <label>
                    Height Y <strong>{groundY.toFixed(1)}</strong>
                  </label>
                  <input
                    type="range"
                    min={groundYRange[0]}
                    max={groundYRange[1]}
                    step={0.1}
                    value={groundY}
                    onChange={(e) => setGroundY(Number(e.target.value))}
                  />
                </div>
                <div className="orientation-buttons">
                  {meta?.bbox_min != null && (
                    <button
                      type="button"
                      onClick={() =>
                        setGroundY(parseFloat(meta.bbox_min[1].toFixed(2)))
                      }
                    >
                      Model bottom
                    </button>
                  )}
                  {meta?.bbox_max != null && (
                    <button
                      type="button"
                      onClick={() =>
                        setGroundY(parseFloat(meta.bbox_max[1].toFixed(2)))
                      }
                    >
                      Model top
                    </button>
                  )}
                  <button type="button" onClick={() => setGroundY(0)}>
                    Y = 0
                  </button>
                </div>
              </>
            )}
          </section>
          <section className="panel-section">
            <h3 className="panel-title">Environment</h3>
            <div className="panel-row">
              <label>
                Ambient <strong>{(ambientIntensity * 100).toFixed(0)}%</strong>
              </label>
              <input
                type="range"
                min={0}
                max={3}
                step={0.05}
                value={ambientIntensity}
                onChange={(e) => setAmbientIntensity(Number(e.target.value))}
              />
            </div>
            <div className="panel-row">
              <label>Background</label>
              <select
                value={backgroundMode}
                onChange={(e) => setBackgroundMode(e.target.value)}
              >
                <option value="dark">Dark</option>
                <option value="light">Light</option>
                <option value="sky">Sky</option>
              </select>
            </div>
          </section>
          <section className="panel-section">
            <h3 className="panel-title">Clipping</h3>
            <div className="panel-row">
              <label>Height (Y) slice</label>
              <input
                type="text"
                placeholder="Off"
                value={clipY != null ? String(clipY) : ""}
                onChange={(e) => {
                  const v = e.target.value.trim();
                  setClipY(v === "" ? null : parseFloat(v));
                }}
              />
            </div>
          </section>
        </div>
        {/* panel-scroll */}
      </div>

      {/* Right panel: Camera (hidden when Solar or Heatmap panel open) */}
      {!solarMode && !heatmapMode && (
        <div className="float-panel right-panel">
          <div className="panel-scroll">
            <section className="panel-section">
              <h3 className="panel-title">Camera</h3>
              <div className="panel-row">
                <label>
                  FOV <strong>{fov}°</strong>
                </label>
                <input
                  type="range"
                  min={30}
                  max={90}
                  value={fov}
                  onChange={(e) => setFov(Number(e.target.value))}
                />
              </div>
              <div className="preset-buttons">
                <button
                  type="button"
                  onClick={() => {
                    setCameraOverrideCenter(pivotPoint);
                    setCameraPreset("top");
                  }}
                >
                  Top
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setCameraOverrideCenter(pivotPoint);
                    setCameraPreset("front");
                  }}
                >
                  Front
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setCameraOverrideCenter(pivotPoint);
                    setCameraPreset("left");
                  }}
                >
                  Left
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setCameraOverrideCenter(pivotPoint);
                    setCameraPreset("right");
                  }}
                >
                  Right
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setCameraOverrideCenter(null);
                    setCameraPreset("auto");
                  }}
                >
                  Auto-fit
                </button>
              </div>
              <button
                type="button"
                className="screenshot-btn"
                onClick={() => {
                  const canvas = document.querySelector("canvas");
                  if (canvas) {
                    const a = document.createElement("a");
                    a.download = "viewer-screenshot.png";
                    a.href = canvas.toDataURL("image/png");
                    a.click();
                  }
                }}
              >
                Screenshot
              </button>
            </section>
          </div>
          {/* panel-scroll */}
        </div>
      )}

      {/* Solar panel */}
      {solarMode && (
        <div className="solar-panel">
          <div className="solar-panel-scroll">
            <h2 className="solar-title">Solar Study</h2>

            {/* Location — moved to top */}
            <div className="solar-location-block">
              <div className="solar-location-coords">
                <span className="location-coords">
                  {lat.toFixed(4)}, {lng.toFixed(4)}
                </span>
                {locationLabel && (
                  <span className="location-addr">{locationLabel}</span>
                )}
              </div>
              <button
                type="button"
                className="set-location-btn set-location-btn--accent"
                onClick={() => setShowMapModal(true)}
              >
                Set on Map
              </button>
            </div>

            <div className="solar-row">
              <label>
                Date{" "}
                <strong>
                  {new Date(`${dateStr}T00:00:00`).toLocaleDateString(
                    undefined,
                    {
                      month: "short",
                      day: "numeric",
                    },
                  )}
                </strong>
              </label>
              <input
                type="range"
                min={1}
                max={365}
                value={dayOfYear}
                onChange={(e) => setDayOfYear(e.target.value)}
              />
            </div>
            <div className="solar-row">
              <label>
                Time{" "}
                <strong>
                  {String(hour).padStart(2, "0")}:{String(min).padStart(2, "0")}
                </strong>
              </label>
              <input
                type="range"
                min={0}
                max={1439}
                value={timeMin}
                onChange={(e) => setTimeMin(Number(e.target.value))}
              />
            </div>
            <div className="solar-stats">
              <div className={`stat-pill ${sunPos.isDay ? "day" : "night"}`}>
                {sunPos.isDay ? "Daytime" : "Night"}
              </div>
              <div className="stat-pill">Elev {sunPos.altDeg}°</div>
              <div className="stat-pill">Az {sunPos.azDeg}°</div>
            </div>
            <div className="solar-times">
              <span>Sunrise {sunPos.sunrise}</span>
              <span>Sunset {sunPos.sunset}</span>
            </div>
            <div className="solar-toggles-row">
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={showSolarDome}
                  onChange={(e) => setShowSolarDome(e.target.checked)}
                />
                Show dome &amp; paths
              </label>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={showSunMarker}
                  onChange={(e) => setShowSunMarker(e.target.checked)}
                />
                Show sun marker
              </label>
            </div>

            {showSolarDome && (
              <>
                <div className="solar-row">
                  <label>
                    Path Opacity{" "}
                    <strong>{(solarPathOpacity * 100).toFixed(0)}%</strong>
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={solarPathOpacity}
                    onChange={(e) =>
                      setSolarPathOpacity(Number(e.target.value))
                    }
                  />
                </div>
                <div className="solar-row">
                  <label>
                    Dome Size <strong>{solarPathSize.toFixed(1)}×</strong>
                  </label>
                  <input
                    type="range"
                    min={0.2}
                    max={3}
                    step={0.1}
                    value={solarPathSize}
                    onChange={(e) => setSolarPathSize(Number(e.target.value))}
                  />
                </div>
                <div className="solar-row">
                  <label>
                    Line Thickness <strong>{solarLineThick.toFixed(1)}×</strong>
                  </label>
                  <input
                    type="range"
                    min={0.25}
                    max={4}
                    step={0.25}
                    value={solarLineThick}
                    onChange={(e) => setSolarLineThick(Number(e.target.value))}
                  />
                </div>
              </>
            )}
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={shadowsEnabled}
                onChange={(e) => setShadows(e.target.checked)}
              />
              Shadow casting
            </label>
          </div>
          {/* solar-panel-scroll */}
        </div>
      )}

      {/* Sun Heatmap panel */}
      {heatmapMode && (
        <div className="heatmap-panel">
          <div className="heatmap-panel-scroll">
            <h2 className="heatmap-panel-title">Sun Exposure Heatmap</h2>

            {/* Orientation mismatch warning */}
            {savedHeatmap?.modelRotation &&
              (() => {
                const saved = savedHeatmap.modelRotation;
                const cur = modelRotation;
                const mismatch =
                  Array.isArray(saved) &&
                  Array.isArray(cur) &&
                  saved.some((v, i) => Math.abs(v - (cur[i] ?? 0)) > 0.017); // >1° threshold
                if (!mismatch) return null;
                const toDeg = (r) => Math.round((r * 180) / Math.PI);
                return (
                  <div className="heatmap-orient-warning">
                    <span className="heatmap-orient-warning-icon">⚠</span>
                    <span>
                      Building orientation has changed since this result was
                      computed. Saved: [{saved.map(toDeg).join("°, ")}°] ·
                      Current: [{cur.map(toDeg).join("°, ")}°]
                    </span>
                  </div>
                );
              })()}

            {/* Sample type selector */}
            <div className="heatmap-sample-type-row">
              {["day", "month", "year"].map((t) => (
                <button
                  key={t}
                  type="button"
                  className={`heatmap-sample-type-btn${heatmapSampleType === t ? " active" : ""}`}
                  onClick={() => setHeatmapSampleType(t)}
                >
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>

            {/* Date / Month / Year picker */}
            {heatmapSampleType === "day" && (
              <div className="solar-row">
                <label>
                  Date{" "}
                  <strong>
                    {new Date(`${dateStr}T00:00:00`).toLocaleDateString(
                      undefined,
                      {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                      },
                    )}
                  </strong>
                </label>
                <input
                  type="range"
                  min={1}
                  max={365}
                  value={dayOfYear}
                  onChange={(e) => setDayOfYear(e.target.value)}
                />
              </div>
            )}

            {heatmapSampleType === "month" &&
              (() => {
                const [curY, curMo] = dateStr.split("-").map(Number);
                const MONTHS = [
                  "January",
                  "February",
                  "March",
                  "April",
                  "May",
                  "June",
                  "July",
                  "August",
                  "September",
                  "October",
                  "November",
                  "December",
                ];
                return (
                  <div className="solar-row">
                    <label>
                      Month{" "}
                      <strong>
                        {MONTHS[curMo - 1]} {curY}
                      </strong>
                    </label>
                    <input
                      type="range"
                      min={1}
                      max={12}
                      value={curMo}
                      onChange={(e) => {
                        const mo = Number(e.target.value);
                        const dd = String(mo).padStart(2, "0");
                        setDateStr(`${curY}-${dd}-01`);
                      }}
                    />
                  </div>
                );
              })()}

            {heatmapSampleType === "year" &&
              (() => {
                const curY = parseInt(dateStr.slice(0, 4), 10);
                return (
                  <div className="solar-row">
                    <label>
                      Year <strong>{curY}</strong>
                    </label>
                    <div className="heatmap-year-row">
                      <button
                        type="button"
                        className="heatmap-year-btn"
                        onClick={() =>
                          setDateStr(`${curY - 1}${dateStr.slice(4)}`)
                        }
                      >
                        −
                      </button>
                      <input
                        type="number"
                        className="heatmap-year-input"
                        value={curY}
                        min={2000}
                        max={2100}
                        step={1}
                        onChange={(e) => {
                          const y = parseInt(e.target.value, 10);
                          if (y >= 2000 && y <= 2100)
                            setDateStr(`${y}${dateStr.slice(4)}`);
                        }}
                      />
                      <button
                        type="button"
                        className="heatmap-year-btn"
                        onClick={() =>
                          setDateStr(`${curY + 1}${dateStr.slice(4)}`)
                        }
                      >
                        +
                      </button>
                    </div>
                  </div>
                );
              })()}

            <div className="solar-row">
              <label>Location</label>
              <div className="location-display">
                <span className="location-coords">
                  {lat.toFixed(4)}, {lng.toFixed(4)}
                </span>
                {locationLabel && (
                  <span className="location-addr">{locationLabel}</span>
                )}
                <button
                  type="button"
                  className="set-location-btn"
                  onClick={() => setShowMapModal(true)}
                >
                  Set on Map
                </button>
              </div>
            </div>

            <div className="heatmap-divider" />

            <div className="solar-toggles-row">
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={showSolarDome}
                  onChange={(e) => setShowSolarDome(e.target.checked)}
                />
                Show solar dome &amp; paths
              </label>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={showSunMarker}
                  onChange={(e) => setShowSunMarker(e.target.checked)}
                />
                Show sun marker
              </label>
            </div>

            <div className="heatmap-divider" />

            {/* Custom scale range — always visible, persists across computes */}
            {(() => {
              const rangeMin = heatmapRangeMin;
              const rangeMax = heatmapRangeMax ?? heatmapMaxHours ?? 12;
              const dis = !heatmapRangeEnabled;
              const nudgeMin = (d) =>
                setHeatmapRangeMin(
                  parseFloat(
                    Math.max(0, Math.min(rangeMin + d, rangeMax - 0.1)).toFixed(
                      1,
                    ),
                  ),
                );
              const nudgeMax = (d) =>
                setHeatmapRangeMax(
                  parseFloat(Math.max(rangeMax + d, rangeMin + 0.1).toFixed(1)),
                );
              return (
                <div className="heatmap-range-section">
                  <div className="heatmap-range-header">
                    <span className="heatmap-range-title">
                      Custom scale range
                    </span>
                    <label className="heatmap-range-toggle-label">
                      <input
                        type="checkbox"
                        className="heatmap-range-toggle"
                        checked={heatmapRangeEnabled}
                        onChange={(e) =>
                          setHeatmapRangeEnabled(e.target.checked)
                        }
                      />
                      <span className="heatmap-range-toggle-track" />
                    </label>
                  </div>
                  <div
                    className={`heatmap-range-inputs${dis ? " disabled" : ""}`}
                  >
                    {/* Min row */}
                    <div className="orient-row">
                      <div className="orient-controls">
                        <span
                          className="orient-axis-badge"
                          style={{
                            color: "var(--heatmap-accent)",
                            borderColor: "var(--heatmap-accent)",
                          }}
                        >
                          Min
                        </span>
                        {[-1, -0.1].map((d) => (
                          <button
                            key={d}
                            type="button"
                            className="orient-step-btn"
                            disabled={dis}
                            onClick={() => nudgeMin(d)}
                          >
                            {d}
                          </button>
                        ))}
                        <input
                          type="number"
                          className="orient-num-input"
                          disabled={dis}
                          min={0}
                          max={parseFloat((rangeMax - 0.1).toFixed(1))}
                          step={0.1}
                          value={parseFloat(rangeMin.toFixed(1))}
                          onWheel={(e) => e.stopPropagation()}
                          onChange={(e) => {
                            const v = Math.max(
                              0,
                              Math.min(Number(e.target.value), rangeMax - 0.1),
                            );
                            setHeatmapRangeMin(parseFloat(v.toFixed(1)));
                          }}
                        />
                        <span className="orient-deg">h</span>
                        {[0.1, 1].map((d) => (
                          <button
                            key={d}
                            type="button"
                            className="orient-step-btn"
                            disabled={dis}
                            onClick={() => nudgeMin(d)}
                          >
                            +{d}
                          </button>
                        ))}
                      </div>
                    </div>
                    {/* Max row */}
                    <div className="orient-row">
                      <div className="orient-controls">
                        <span
                          className="orient-axis-badge"
                          style={{
                            color: "var(--heatmap-accent)",
                            borderColor: "var(--heatmap-accent)",
                          }}
                        >
                          Max
                        </span>
                        {[-1, -0.1].map((d) => (
                          <button
                            key={d}
                            type="button"
                            className="orient-step-btn"
                            disabled={dis}
                            onClick={() => nudgeMax(d)}
                          >
                            {d}
                          </button>
                        ))}
                        <input
                          type="number"
                          className="orient-num-input"
                          disabled={dis}
                          min={parseFloat((rangeMin + 0.1).toFixed(1))}
                          step={0.1}
                          value={parseFloat(rangeMax.toFixed(1))}
                          onWheel={(e) => e.stopPropagation()}
                          onChange={(e) => {
                            const v = Math.max(
                              Number(e.target.value),
                              rangeMin + 0.1,
                            );
                            setHeatmapRangeMax(parseFloat(v.toFixed(1)));
                          }}
                        />
                        <span className="orient-deg">h</span>
                        {[0.1, 1].map((d) => (
                          <button
                            key={d}
                            type="button"
                            className="orient-step-btn"
                            disabled={dis}
                            onClick={() => nudgeMax(d)}
                          >
                            +{d}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })()}

            <div className="heatmap-divider" />

            <div className="solar-row">
              <label>Sample interval</label>
              {(() => {
                const intervalOptions =
                  heatmapSampleType === "year"
                    ? [
                        { value: 30, label: "30 min (fine)" },
                        { value: 60, label: "1 hour" },
                        { value: 120, label: "2 hours" },
                        { value: 240, label: "4 hours (fast)" },
                      ]
                    : heatmapSampleType === "month"
                      ? [
                          { value: 15, label: "15 min (fine)" },
                          { value: 30, label: "30 min" },
                          { value: 60, label: "1 hour" },
                          { value: 120, label: "2 hours (fast)" },
                        ]
                      : [
                          { value: 5, label: "5 min (fine)" },
                          { value: 15, label: "15 min" },
                          { value: 30, label: "30 min" },
                          { value: 60, label: "1 hour (fast)" },
                        ];
                const clampedStep =
                  intervalOptions.find((o) => o.value >= heatmapTimeStep)
                    ?.value ??
                  intervalOptions[intervalOptions.length - 1].value;
                return (
                  <select
                    value={clampedStep}
                    onChange={(e) => setHeatmapTimeStep(Number(e.target.value))}
                    className="heatmap-select"
                  >
                    {intervalOptions.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                );
              })()}
            </div>

            <button
              type="button"
              className="heatmap-compute-btn"
              disabled={
                heatmapStatus === "computing" || heatmapStatus === "loading"
              }
              onClick={() => {
                setHeatmapProgress(0);
                setHeatmapTrigger((t) => t + 1);
              }}
            >
              {heatmapStatus === "loading"
                ? "Loading mesh…"
                : heatmapStatus === "computing"
                  ? `Computing… ${heatmapProgress}%`
                  : "Compute Heatmap"}
            </button>

            {heatmapStatus === "computing" && (
              <div className="heatmap-progress-track">
                <div
                  className="heatmap-progress-fill"
                  style={{ width: `${heatmapProgress}%` }}
                />
              </div>
            )}

            {heatmapSavedFlash && (
              <div className="heatmap-saved-flash">✓ Saved</div>
            )}
            {heatmapSaveError && (
              <div className="heatmap-save-error">
                ⚠ Could not save — delete old results to free space
              </div>
            )}

            <p className="panel-hint" style={{ fontSize: "0.68rem" }}>
              {heatmapSampleType === "year"
                ? "Cosine-weighted sun exposure — all daylight hours across the full year"
                : heatmapSampleType === "month"
                  ? "Cosine-weighted sun exposure — all daylight hours across the selected month"
                  : "Cosine-weighted sun exposure — all daylight hours on selected date"}
            </p>

            {/* Colormap picker — custom gradient dropdown */}
            <ColormapDropdown
              value={heatmapColormap}
              onChange={setHeatmapColormap}
            />

            {heatmapStatus === "done" &&
              heatmapMaxHours != null &&
              (() => {
                const dispMin = heatmapRangeEnabled ? heatmapRangeMin : 0;
                const dispMax = heatmapRangeEnabled
                  ? (heatmapRangeMax ?? heatmapMaxHours)
                  : heatmapMaxHours;
                const fmtH = (h) =>
                  h < 0.05
                    ? "0 h"
                    : h < 10
                      ? `${h.toFixed(1)} h`
                      : `${Math.round(h)} h`;
                return (
                  <>
                    <div className="heatmap-legend-wrap">
                      <div
                        className="heatmap-legend-bar"
                        style={{
                          background: colormapCssGradient(heatmapColormap),
                        }}
                      />
                      <div className="heatmap-legend-ticks">
                        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
                          <span key={f}>
                            {fmtH(dispMin + (dispMax - dispMin) * f)}
                          </span>
                        ))}
                      </div>
                      <div className="heatmap-legend-note">
                        {heatmapSampleType === "year"
                          ? "Effective annual sun-weighted exposure hours"
                          : heatmapSampleType === "month"
                            ? "Effective monthly sun-weighted exposure hours"
                            : "Effective sun-weighted exposure hours"}
                      </div>
                    </div>
                    <button
                      type="button"
                      className="heatmap-analysis-btn"
                      onClick={() => setShowHeatmapAnalysis(true)}
                    >
                      View Distribution
                    </button>
                  </>
                );
              })()}

            {showHeatmapAnalysis && savedHeatmap?.heatArrays && (
              <HeatmapAnalysisModal
                heatArrays={savedHeatmap.heatArrays}
                maxHours={heatmapMaxHours}
                timeStep={heatmapTimeStep}
                rangeEnabled={heatmapRangeEnabled}
                rangeMinHrs={heatmapRangeMin}
                rangeMaxHrs={heatmapRangeMax ?? heatmapMaxHours}
                colormap={heatmapColormap}
                onClose={() => setShowHeatmapAnalysis(false)}
              />
            )}

            {allSavedHeatmaps.length > 0 && (
              <div className="heatmap-saved-list">
                <div className="heatmap-saved-list-title">Saved Results</div>
                {allSavedHeatmaps.map((h) => {
                  const isActive = h.storageKey === activeSavedStorageKey;
                  const d = new Date(`${h.dateStr}T00:00:00`);
                  const pad = (n) => String(n).padStart(2, "0");
                  const mm = pad(d.getMonth() + 1);
                  const dd = pad(d.getDate());
                  const yyyy = d.getFullYear();
                  const typeLabel =
                    h.sampleType === "year"
                      ? "Year"
                      : h.sampleType === "month"
                        ? "Month"
                        : "Day";
                  const dateLabel =
                    h.sampleType === "year"
                      ? `${yyyy}`
                      : h.sampleType === "month"
                        ? `${mm}/${yyyy}`
                        : `${mm}/${dd}/${yyyy}`;
                  const label = `${typeLabel} · ${dateLabel}`;
                  return (
                    <div
                      key={h.cacheKey}
                      className={`heatmap-saved-item${isActive ? " active" : ""}`}
                    >
                      <div className="heatmap-saved-info">
                        <span className="heatmap-saved-date">{label}</span>
                        <span className="heatmap-saved-meta">
                          {h.sampleType !== "day" && (
                            <span className="heatmap-saved-type-badge">
                              {h.sampleType}
                            </span>
                          )}
                          {h.timeStep} min · {h.lat.toFixed(2)},{" "}
                          {h.lng.toFixed(2)}
                        </span>
                      </div>
                      <div className="heatmap-saved-actions">
                        {!isActive && (
                          <button
                            type="button"
                            className="heatmap-saved-load-btn"
                            onClick={() => loadSavedHeatmap(h)}
                          >
                            Load
                          </button>
                        )}
                        <button
                          type="button"
                          className="heatmap-saved-delete-btn"
                          title="Delete"
                          onClick={() => deleteHeatmap(h.storageKey)}
                        >
                          ✕
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
          {/* heatmap-panel-scroll */}
        </div>
      )}

      {/* Processing panel (collapsible bottom) */}
      {projectsPanelOpen &&
        (() => {
          const activeProject = projects.find((p) => p.id === activeProjectId);
          const activeObj = objsByProject[activeProjectId] ?? null;
          // "Default Project" shows the server-side files; other projects only show uploaded OBJ
          const isDefault = activeProjectId === "default";
          return (
            <div className="projects-panel">
              {/* Left sidebar */}
              <div className="projects-sidebar">
                <div className="projects-sidebar-header">
                  <span className="projects-sidebar-title">Projects</span>
                  <button
                    type="button"
                    className="projects-panel-close"
                    onClick={() => setProjectsPanelOpen(false)}
                    aria-label="Close"
                  >
                    ✕
                  </button>
                </div>
                <button
                  type="button"
                  className="projects-new-btn"
                  onClick={createNewProject}
                >
                  + New Project
                </button>
                <div className="projects-list">
                  {projects.map((p) => {
                    const pObj = objsByProject[p.id];
                    const isActive = activeProjectId === p.id;
                    const isRenaming = renamingProjectId === p.id;
                    return (
                      <div
                        key={p.id}
                        className={`projects-list-item ${isActive ? "active" : ""}`}
                        onClick={() => {
                          if (!isRenaming) switchToProject(p.id);
                        }}
                      >
                        {isRenaming ? (
                          <input
                            className="projects-rename-input"
                            autoFocus
                            value={renameValue}
                            onChange={(e) => setRenameValue(e.target.value)}
                            onBlur={commitRename}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") commitRename();
                              if (e.key === "Escape")
                                setRenamingProjectId(null);
                            }}
                            onClick={(e) => e.stopPropagation()}
                          />
                        ) : (
                          <span
                            className="projects-list-name"
                            title="Double-click to rename"
                            onDoubleClick={(e) => {
                              e.stopPropagation();
                              setRenamingProjectId(p.id);
                              setRenameValue(p.name);
                            }}
                          >
                            {p.name}
                          </span>
                        )}
                        <div className="projects-list-meta">
                          <span className="projects-list-date">
                            {new Date(p.createdAt).toLocaleDateString()}
                          </span>
                          {pObj && (
                            <span className="projects-list-obj-badge">OBJ</span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Main content */}
              <div className="projects-main">
                <div className="projects-main-header">
                  {renamingProjectId === activeProjectId ? (
                    <input
                      className="projects-title-rename-input"
                      autoFocus
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      onBlur={commitRename}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") commitRename();
                        if (e.key === "Escape") setRenamingProjectId(null);
                      }}
                    />
                  ) : (
                    <span
                      className="projects-main-title"
                      title="Double-click to rename"
                      onDoubleClick={() => {
                        setRenamingProjectId(activeProjectId);
                        setRenameValue(activeProject?.name ?? "");
                      }}
                    >
                      {activeProject?.name ?? "Project"}
                      <span className="projects-rename-hint">
                        double-click to rename
                      </span>
                    </span>
                  )}
                </div>

                {/* Server model files — only shown for Default Project */}
                {isDefault && (
                  <div className="projects-files-section">
                    <h3 className="projects-section-label">
                      Pipeline Model Files
                    </h3>
                    <div className="projects-file-list">
                      <div className="projects-file-row">
                        <span className="projects-file-icon">☁</span>
                        <div className="projects-file-info">
                          <span className="projects-file-name">
                            point_cloud.ply
                          </span>
                          <span className="projects-file-sub">
                            Point Cloud · /models/
                          </span>
                        </div>
                        {modelsInfo?.point_cloud?.size_mb != null && (
                          <span className="projects-file-size">
                            {modelsInfo.point_cloud.size_mb} MB
                          </span>
                        )}
                        <span
                          className={`projects-file-status ${modelsInfo?.point_cloud ? "ok" : "missing"}`}
                        >
                          {modelsInfo?.point_cloud ? "Loaded" : "Not found"}
                        </span>
                      </div>
                      <div className="projects-file-row">
                        <span className="projects-file-icon">⬡</span>
                        <div className="projects-file-info">
                          <span className="projects-file-name">
                            drone_mesh.glb
                          </span>
                          <span className="projects-file-sub">
                            3D Mesh · /models/
                          </span>
                        </div>
                        {modelsInfo?.mesh?.size_mb != null && (
                          <span className="projects-file-size">
                            {modelsInfo.mesh.size_mb} MB
                          </span>
                        )}
                        <span
                          className={`projects-file-status ${modelsInfo?.mesh ? "ok" : "missing"}`}
                        >
                          {modelsInfo?.mesh ? "Loaded" : "Not found"}
                        </span>
                      </div>
                      <div className="projects-file-row">
                        <span className="projects-file-icon">&#123;&#125;</span>
                        <div className="projects-file-info">
                          <span className="projects-file-name">
                            point_cloud_meta.json
                          </span>
                          <span className="projects-file-sub">
                            Spatial Metadata · /models/
                          </span>
                        </div>
                        <span
                          className={`projects-file-status ${meta ? "ok" : "missing"}`}
                        >
                          {meta ? "Loaded" : "Not found"}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {!isDefault && !activeObj && (
                  <div className="projects-empty-state">
                    <div className="projects-empty-icon">📂</div>
                    <p className="projects-empty-title">No model loaded</p>
                    <p className="projects-empty-desc">
                      Upload an OBJ file below to add a 3D model to this
                      project.
                    </p>
                  </div>
                )}

                {/* Per-project OBJ file */}
                {activeObj && (
                  <div className="projects-files-section">
                    <h3 className="projects-section-label">Uploaded Model</h3>
                    <div className="projects-file-list">
                      <div className="projects-file-row">
                        <span className="projects-file-icon">◻</span>
                        <div className="projects-file-info">
                          <span className="projects-file-name">
                            {activeObj.name}
                          </span>
                          <span className="projects-file-sub">
                            OBJ Model · browser upload
                          </span>
                        </div>
                        <span className="projects-file-status ok">
                          Active in viewer
                        </span>
                        <button
                          type="button"
                          className="projects-file-remove"
                          title="Remove OBJ from this project"
                          onClick={removeProjectObj}
                        >
                          ✕
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Heatmaps */}
                {isDefault && (
                  <div className="projects-files-section">
                    <h3 className="projects-section-label">Sun Heatmaps</h3>
                    <div className="projects-heatmap-count">
                      <span className="projects-heatmap-num">
                        {allSavedHeatmaps.length}
                      </span>
                      <span className="projects-heatmap-label">
                        saved heatmap{allSavedHeatmaps.length !== 1 ? "s" : ""}
                      </span>
                    </div>
                  </div>
                )}

                {/* Upload OBJ */}
                <div className="projects-files-section">
                  <h3 className="projects-section-label">Upload OBJ Model</h3>
                  <p className="projects-section-desc">
                    Load a <code>.obj</code> file directly in the browser — no
                    server required. The model appears in the 3D viewport and is
                    stored for this project only.
                  </p>
                  <input
                    ref={objInputRef}
                    type="file"
                    accept=".obj"
                    onChange={handleObjUpload}
                    style={{ display: "none" }}
                  />
                  <button
                    type="button"
                    className="projects-upload-btn"
                    onClick={() => objInputRef.current?.click()}
                  >
                    {activeObj ? "Replace OBJ…" : "Choose OBJ File…"}
                  </button>
                  {activeObj && (
                    <p className="projects-upload-loaded">
                      ✓ {activeObj.name} is active in the viewer
                    </p>
                  )}
                </div>
              </div>
            </div>
          );
        })()}

      {processPanelOpen && (
        <div className="process-panel">
          <div className="process-toolbar">
            <span className="process-title">Pipeline</span>
            <div className="process-presets">
              <button type="button" onClick={() => applyPreset("fast")}>
                Fast
              </button>
              <button type="button" onClick={() => applyPreset("balanced")}>
                Balanced
              </button>
              <button type="button" onClick={() => applyPreset("high")}>
                High
              </button>
            </div>
            <div className="pipeline-video-row">
              <label className="pipeline-video-label">Video</label>
              <div className="pipeline-video-controls">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,.mp4,.mov,.avi,.mkv,.webm"
                  onChange={handleVideoFileSelect}
                  style={{ display: "none" }}
                />
                <button
                  type="button"
                  className="pipeline-file-btn"
                  disabled={
                    videoUploading || pipelineStatus.status === "running"
                  }
                  onClick={() => fileInputRef.current?.click()}
                >
                  {videoUploading ? "Uploading…" : "Choose File"}
                </button>
                <input
                  type="text"
                  className="pipeline-path-input"
                  placeholder="or path: nerf_data/DJI_0957.MP4"
                  value={pipelineParams.video_path}
                  onChange={(e) => {
                    setPipelineParams((p) => ({
                      ...p,
                      video_path: e.target.value,
                    }));
                    setVideoFileInfo(null);
                  }}
                />
                {pipelineParams.video_path && (
                  <button
                    type="button"
                    className="pipeline-probe-btn"
                    disabled={pipelineStatus.status === "running"}
                    onClick={handleProbeVideo}
                    title="Get video metadata"
                  >
                    Probe
                  </button>
                )}
              </div>
              {videoUploadError && (
                <span className="pipeline-video-error">{videoUploadError}</span>
              )}
              {videoFileInfo && (
                <div className="pipeline-video-meta">
                  {videoFileInfo.filename && (
                    <span>{videoFileInfo.filename}</span>
                  )}
                  {videoFileInfo.size_mb != null && (
                    <span>{videoFileInfo.size_mb} MB</span>
                  )}
                  {videoFileInfo.duration_sec != null && (
                    <span>{videoFileInfo.duration_sec}s</span>
                  )}
                  {videoFileInfo.width != null &&
                    videoFileInfo.height != null && (
                      <span>
                        {videoFileInfo.width}×{videoFileInfo.height}
                      </span>
                    )}
                  {videoFileInfo.duration_sec != null && (
                    <span>
                      ~
                      {Math.round(
                        videoFileInfo.duration_sec * pipelineParams.fps,
                      )}{" "}
                      frames @ {pipelineParams.fps} fps
                    </span>
                  )}
                  {videoFileInfo.duration_sec != null &&
                    videoFileInfo.duration_sec > 120 && (
                      <span className="pipeline-video-hint">
                        Long video — consider 3 fps for faster run
                      </span>
                    )}
                  {videoFileInfo.width != null &&
                    videoFileInfo.width >= 3840 && (
                      <span className="pipeline-video-hint">
                        4K — 4096+ features recommended
                      </span>
                    )}
                </div>
              )}
            </div>
            <label>
              FPS{" "}
              <input
                type="number"
                min={1}
                max={10}
                value={pipelineParams.fps}
                onChange={(e) =>
                  setPipelineParams((p) => ({
                    ...p,
                    fps: Number(e.target.value),
                  }))
                }
              />
            </label>
            <label>
              Features{" "}
              <select
                value={pipelineParams.max_features}
                onChange={(e) =>
                  setPipelineParams((p) => ({
                    ...p,
                    max_features: Number(e.target.value),
                  }))
                }
              >
                <option value={2048}>2048</option>
                <option value={4096}>4096</option>
                <option value={8192}>8192</option>
              </select>
            </label>
            <label>
              Mesh depth{" "}
              <input
                type="number"
                min={7}
                max={12}
                value={pipelineParams.poisson_depth}
                onChange={(e) =>
                  setPipelineParams((p) => ({
                    ...p,
                    poisson_depth: Number(e.target.value),
                  }))
                }
              />
            </label>
            <label>
              Smooth{" "}
              <input
                type="number"
                min={0}
                max={5}
                value={pipelineParams.smooth_iterations}
                onChange={(e) =>
                  setPipelineParams((p) => ({
                    ...p,
                    smooth_iterations: Number(e.target.value),
                  }))
                }
              />
            </label>
            <button
              type="button"
              className="run-btn"
              disabled={pipelineStatus.status === "running"}
              onClick={runPipeline}
            >
              Run Pipeline
            </button>
            <button
              type="button"
              className="run-btn secondary"
              disabled={pipelineStatus.status === "running"}
              onClick={rebuildMesh}
            >
              Rebuild Mesh
            </button>
            <span className={`status-badge ${pipelineStatus.status}`}>
              {pipelineStatus.status}
            </span>
            {pipelineStatus.error && (
              <span className="status-error">{pipelineStatus.error}</span>
            )}
          </div>
          <div className="process-log">
            <pre>{logLines.join("\n")}</pre>
            <div ref={logEndRef} />
          </div>
        </div>
      )}

      {/* Floating view navigation panel */}
      <div className="view-nav-panel">
        <button
          type="button"
          className="view-nav-zoom"
          title={
            pivotPoint
              ? "Zoom camera to pivot point"
              : "Zoom camera to model center"
          }
          onClick={() => {
            setCameraOverrideCenter(pivotPoint);
            setCameraPreset("auto");
          }}
        >
          <svg viewBox="0 0 20 20" fill="none" width="13" height="13">
            <circle
              cx="10"
              cy="10"
              r="3.5"
              stroke="currentColor"
              strokeWidth="1.5"
            />
            <path
              d="M10 1v3M10 16v3M1 10h3M16 10h3"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
          {pivotPoint ? "Zoom to Pivot" : "Zoom to Object"}
        </button>
        <div className="view-nav-divider" />
        <div className="view-nav-elev">
          <button
            type="button"
            title="Top view"
            onClick={() => {
              setCameraOverrideCenter(pivotPoint);
              setCameraPreset("top");
            }}
          >
            Top
          </button>
          <button
            type="button"
            title="Bottom view"
            onClick={() => {
              setCameraOverrideCenter(pivotPoint);
              setCameraPreset("bottom");
            }}
          >
            Bot
          </button>
        </div>
        <div className="view-nav-compass">
          <button
            type="button"
            className="compass-n"
            title="View from North"
            onClick={() => {
              setCameraOverrideCenter(pivotPoint);
              setCameraPreset("north");
            }}
          >
            N
          </button>
          <button
            type="button"
            className="compass-w"
            title="View from West"
            onClick={() => {
              setCameraOverrideCenter(pivotPoint);
              setCameraPreset("west");
            }}
          >
            W
          </button>
          <div className="compass-center" />
          <button
            type="button"
            className="compass-e"
            title="View from East"
            onClick={() => {
              setCameraOverrideCenter(pivotPoint);
              setCameraPreset("east");
            }}
          >
            E
          </button>
          <button
            type="button"
            className="compass-s"
            title="View from South"
            onClick={() => {
              setCameraOverrideCenter(pivotPoint);
              setCameraPreset("south");
            }}
          >
            S
          </button>
        </div>
      </div>

      <div className="hint">
        {pivotMode
          ? "Click on model to set pivot point — Esc to cancel"
          : "Drag to orbit · Scroll to zoom · Right-drag to pan"}
      </div>

      {/* Location map modal */}
      {showMapModal && (
        <div
          className="map-modal-overlay"
          onClick={() => setShowMapModal(false)}
        >
          <div className="map-modal" onClick={(e) => e.stopPropagation()}>
            <div className="map-modal-header">
              <span>Set Location — click or drag marker to place pin</span>
              <button type="button" onClick={() => setShowMapModal(false)}>
                ✕
              </button>
            </div>
            <div className="map-modal-body">
              <LeafletMap
                lat={lat}
                lng={lng}
                onChange={(la, ln) => {
                  setLat(+la.toFixed(6));
                  setLng(+ln.toFixed(6));
                  reverseGeocode(la, ln);
                }}
              />
            </div>
            <div className="map-modal-info">
              <div className="map-modal-coords">
                {lat.toFixed(4)}, {lng.toFixed(4)}
              </div>
              {locationLabel && (
                <div className="map-modal-addr">{locationLabel}</div>
              )}
            </div>
            <button
              type="button"
              className="map-modal-confirm"
              onClick={() => setShowMapModal(false)}
            >
              Confirm Location
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
