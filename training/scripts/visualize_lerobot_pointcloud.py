#!/usr/bin/env python3
"""Point-cloud reconstruction and crop-preview helpers.

This is a preview/crop-bound tuning tool. It does not convert the complete
dataset to the fixed-size Zarr format expected by DP3 training.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np


ORBBEC_RGB_KEY = "observation.images.orbbec"
ORBBEC_DEPTH_KEY = "observation.images.orbbec_depth"
ORBBEC_INTRINSICS_KEY = "observation.camera.orbbec_intrinsics"
ORBBEC_DEPTH_SCALE_KEY = "observation.camera.orbbec_depth_scale_m"


def to_numpy(value) -> np.ndarray:
    """Convert a torch tensor or array-like value to a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def image_to_hwc(value: np.ndarray) -> np.ndarray:
    """Convert a CHW LeRobot image to HWC while accepting HWC as well."""
    image = to_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    if image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    return image


def reconstruct_point_cloud(
    sample: dict,
    pixel_stride: int,
    min_depth_m: float,
    max_depth_m: float,
    include_rgb: bool,
) -> np.ndarray:
    """Back-project registered Orbbec depth into camera optical coordinates."""
    depth = image_to_hwc(sample[ORBBEC_DEPTH_KEY])[..., 0].astype(np.float32)
    intrinsics = to_numpy(sample[ORBBEC_INTRINSICS_KEY]).reshape(3, 3)
    depth_scale = float(to_numpy(sample[ORBBEC_DEPTH_SCALE_KEY]).reshape(-1)[0])

    rows = np.arange(0, depth.shape[0], pixel_stride)
    cols = np.arange(0, depth.shape[1], pixel_stride)
    u, v = np.meshgrid(cols, rows)
    depth_m = depth[::pixel_stride, ::pixel_stride] * depth_scale

    valid = np.isfinite(depth_m)
    valid &= depth_m >= min_depth_m
    valid &= depth_m <= max_depth_m

    z = depth_m[valid]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    xyz = np.stack((x, y, z), axis=-1).astype(np.float32)

    if not include_rgb:
        return xyz

    rgb = image_to_hwc(sample[ORBBEC_RGB_KEY])[::pixel_stride, ::pixel_stride, :3]
    rgb = rgb[valid].astype(np.float32)
    if rgb.size and rgb.max() <= 1.0 + 1e-6:
        rgb *= 255.0
    rgb = np.clip(rgb, 0.0, 255.0)
    return np.concatenate((xyz, rgb), axis=-1)


def load_gravity_direction(path: Path) -> np.ndarray:
    """Read the simple unit_vector entry without requiring PyYAML."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"unit_vector:\s*\[([^\]]+)\]", text)
    if match is None:
        raise ValueError(f"No gravity_direction.unit_vector found in {path}")
    gravity = np.fromstring(match.group(1), sep=",", dtype=np.float64)
    if gravity.shape != (3,):
        raise ValueError(f"Expected three gravity values in {path}, got {gravity}")
    return gravity / np.linalg.norm(gravity)


def align_with_gravity(point_cloud: np.ndarray, gravity_down: np.ndarray) -> np.ndarray:
    """Rotate camera points to a frame whose positive Z axis points upward."""
    z_up = -gravity_down
    camera_x = np.array([1.0, 0.0, 0.0])
    x_axis = camera_x - np.dot(camera_x, z_up) * z_up
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_up, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    camera_to_gravity = np.stack((x_axis, y_axis, z_up), axis=0)

    result = point_cloud.copy()
    result[:, :3] = result[:, :3] @ camera_to_gravity.T
    return result


def crop_point_cloud(
    point_cloud: np.ndarray,
    crop_min: np.ndarray,
    crop_max: np.ndarray,
    planes: list[dict] | None = None,
) -> np.ndarray:
    if np.any(crop_min >= crop_max):
        raise ValueError(f"Each crop-min value must be smaller than crop-max: {crop_min}, {crop_max}")
    xyz = point_cloud[:, :3]
    mask = np.all(xyz >= crop_min, axis=1)
    mask &= np.all(xyz <= crop_max, axis=1)
    for plane in normalize_planes(planes or []):
        mask &= xyz @ plane["normal"] + plane["offset"] >= plane["margin"]
    return point_cloud[mask]


def normalize_planes(planes: list[dict]) -> list[dict]:
    """Normalize plane equations; kept points satisfy normal·xyz + offset >= margin."""
    normalized = []
    for index, plane in enumerate(planes):
        normal = np.asarray(plane.get("normal"), dtype=np.float64)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            raise ValueError(f"Plane {index} needs a finite three-value normal")
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            raise ValueError(f"Plane {index} normal must not be zero")
        offset = float(plane.get("offset", 0.0)) / length
        margin = float(plane.get("margin", 0.0))
        if not np.isfinite(offset) or not np.isfinite(margin) or margin < 0:
            raise ValueError(f"Plane {index} offset must be finite and margin must be non-negative")
        anchor = np.asarray(plane.get("anchor", -offset * normal / length), dtype=np.float64)
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            raise ValueError(f"Plane {index} anchor must contain three finite values")
        normalized.append({
            "name": str(plane.get("name", f"plane_{index + 1}")),
            "normal": (normal / length).tolist(),
            "offset": offset,
            "margin": margin,
            "anchor": anchor.tolist(),
        })
    return normalized


def load_crop_config(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    config = json.loads(path.expanduser().read_text(encoding="utf-8"))
    crop_min = np.asarray(config.get("crop_min"), dtype=np.float32)
    crop_max = np.asarray(config.get("crop_max"), dtype=np.float32)
    if crop_min.shape != (3,) or crop_max.shape != (3,) or np.any(crop_min >= crop_max):
        raise ValueError(f"Invalid crop_min/crop_max in {path}")
    return crop_min, crop_max, normalize_planes(config.get("planes", []))


def preview_subset(point_cloud: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(point_cloud) <= max_points:
        return point_cloud
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(point_cloud), max_points, replace=False)
    return point_cloud[indices]


def describe(name: str, point_cloud: np.ndarray) -> None:
    print(f"{name}: {len(point_cloud):,} points, shape={point_cloud.shape}")
    if len(point_cloud):
        xyz = point_cloud[:, :3]
        print(f"  xyz min: {xyz.min(axis=0)}")
        print(f"  xyz max: {xyz.max(axis=0)}")


def save_crop_selector_html(
    point_cloud: np.ndarray,
    file_path: Path,
    initial_min: np.ndarray | None = None,
    initial_max: np.ndarray | None = None,
    frame_ids: np.ndarray | None = None,
    frame_labels: list[str] | None = None,
    initial_planes: list[dict] | None = None,
    config_path_hint: str = "crop_config.json",
    camera_origin: bool = False,
) -> None:
    """Save an offline HTML editor for AABB plus arbitrary clipping planes."""
    from plotly.offline import get_plotlyjs

    xyz = point_cloud[:, :3]
    if frame_ids is None:
        frame_ids = np.zeros(len(xyz), dtype=np.int32)
    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    if frame_ids.shape != (len(xyz),):
        raise ValueError("frame_ids must contain one integer per point")
    unique_frames = np.unique(frame_ids)
    if not np.array_equal(unique_frames, np.arange(len(unique_frames))):
        raise ValueError("frame_ids must be contiguous and start at zero")
    if frame_labels is None:
        frame_labels = [f"frame {index}" for index in unique_frames]
    if len(frame_labels) != len(unique_frames):
        raise ValueError("frame_labels must contain one label per frame")
    data_min = xyz.min(axis=0)
    data_max = xyz.max(axis=0)
    if initial_min is None:
        initial_min = data_min
    if initial_max is None:
        initial_max = data_max
    initial_planes = normalize_planes(initial_planes or [])

    if point_cloud.shape[1] >= 6:
        rgb = np.clip(point_cloud[:, 3:6], 0, 255).astype(np.uint8)
    else:
        span = np.maximum(data_max - data_min, 1e-8)
        rgb = np.clip((xyz - data_min) / span * 255, 0, 255).astype(np.uint8)
    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]

    payload = json.dumps(
        {
            "x": xyz[:, 0].tolist(),
            "y": xyz[:, 1].tolist(),
            "z": xyz[:, 2].tolist(),
            "colors": colors,
            "frameIds": frame_ids.tolist(),
            "frameLabels": frame_labels,
            "dataMin": data_min.tolist(),
            "dataMax": data_max.tolist(),
            "initialMin": np.asarray(initial_min).tolist(),
            "initialMax": np.asarray(initial_max).tolist(),
            "initialPlanes": initial_planes,
            "configPathHint": config_path_hint,
            "cameraOrigin": camera_origin,
        },
        separators=(",", ":"),
    ).replace("</", "<\\/")

    frame_control = ""
    if len(frame_labels) > 1:
        options = "".join(
            f'<option value="{index}">{escape(label)}</option>'
            for index, label in enumerate(frame_labels)
        )
        frame_control = f"""
        <div id="frame-control">
          <label for="frameSelect">显示点云帧</label>
          <select id="frameSelect">{options}<option value="-1">全部代表帧（检查整体范围）</option></select>
        </div>
        """

    controls = []
    axis_names = ("X", "Y", "Z")
    for axis_index, axis_name in enumerate(axis_names):
        axis_min = float(data_min[axis_index])
        axis_max = float(data_max[axis_index])
        step = max((axis_max - axis_min) / 1000.0, 0.0001)
        controls.append(
            f"""
            <div class="axis-control">
              <div class="axis-title">{axis_name}</div>
              <label>min
                <input id="{axis_name.lower()}minNumber" type="number" step="{step:.7f}" value="{initial_min[axis_index]:.7f}">
              </label>
              <input id="{axis_name.lower()}minRange" type="range" min="{axis_min:.7f}" max="{axis_max:.7f}"
                     step="{step:.7f}" value="{initial_min[axis_index]:.7f}">
              <label>max
                <input id="{axis_name.lower()}maxNumber" type="number" step="{step:.7f}" value="{initial_max[axis_index]:.7f}">
              </label>
              <input id="{axis_name.lower()}maxRange" type="range" min="{axis_min:.7f}" max="{axis_max:.7f}"
                     step="{step:.7f}" value="{initial_max[axis_index]:.7f}">
            </div>
            """
        )

    html_template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>DP3 point-cloud crop selector</title>
  <style>
    body { margin: 0; font-family: sans-serif; color: #222; background: #f5f5f5; }
    #toolbar { padding: 12px 16px; background: white; box-shadow: 0 1px 4px #bbb; }
    #controls { display: grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap: 16px; }
    .axis-control { padding: 10px; border: 1px solid #ddd; border-radius: 6px; }
    .axis-title { font-weight: 700; margin-bottom: 5px; }
    label { display: inline-flex; width: 49%; align-items: center; gap: 5px; }
    input[type=number] { width: 120px; }
    input[type=range] { width: 100%; }
    #status { margin-top: 10px; font-weight: 600; }
    #frame-control { margin: 0 0 10px; }
    #frameSelect { min-width: 520px; max-width: 100%; padding: 5px; }
    #plane-controls { margin-top: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 6px; }
    #plane-controls input { width: 110px; padding: 4px; }
    #plane-list { margin-top: 6px; font: 13px monospace; }
    .plane-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 3px 0; }
    .plane-row input[type=number] { width: 75px; }
    .plane-row input[type=range] { width: 180px; }
    .plane-row label { width: auto; }
    #command { width: calc(100% - 24px); margin-top: 8px; padding: 8px; font-family: monospace; }
    button { margin: 8px 8px 0 0; padding: 7px 12px; cursor: pointer; }
    #cropPlot { width: 100%; height: calc(100vh - 325px); min-height: 520px; }
    .hint { color: #555; font-size: 13px; }
    @media (max-width: 900px) { #controls { grid-template-columns: 1fr; } #cropPlot { height: 650px; } }
  </style>
  <script>__PLOTLY_JS__</script>
</head>
<body>
  <div id="toolbar">
    <div class="hint">先调 AABB；要加斜切面时，在点云上依次点击同一平面的三个点，再添加平面。彩色点保留，灰色点删除。</div>
    __FRAME_CONTROL__
    <div id="controls">__CONTROLS__</div>
    <div id="plane-controls">
      <b>自定义裁剪平面</b>
      <input id="planeName" value="plane_1" aria-label="平面名称">
      <label>边界内缩(m) <input id="planeMargin" type="number" min="0" step="0.005" value="0.030"></label>
      <button id="addPlaneButton">用已选3点添加</button>
      <button id="clearPickedButton">清除选点</button>
      <span id="pickStatus">已选 0/3 点</span>
      <div id="plane-list"></div>
    </div>
    <div id="status"></div>
    <input id="command" readonly>
    <button id="copyButton">复制使用参数</button>
    <button id="downloadButton">下载 crop_config.json</button>
    <span id="copyStatus"></span>
  </div>
  <div id="cropPlot"></div>
  <script>
    const cloud = __PAYLOAD__;
    const axes = ['x', 'y', 'z'];
    const frameSelect = document.getElementById('frameSelect');
    let planes = cloud.initialPlanes.map(p => ({...p, normal:[...p.normal], anchor:[...p.anchor]}));
    let picked = [];
    let cachedFrame = null, cachedIndices = [];

    const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    const degrees = radians => radians * 180 / Math.PI;
    const radians = degrees => degrees * Math.PI / 180;
    function normalAngles(n) {
      return [degrees(Math.atan2(n[1], n[0])), degrees(Math.asin(Math.max(-1, Math.min(1, n[2]))))];
    }
    function normalFromAngles(yaw, pitch) {
      const y=radians(yaw), p=radians(pitch), cp=Math.cos(p);
      return [cp*Math.cos(y), cp*Math.sin(y), Math.sin(p)];
    }
    function updatePicked() {
      document.getElementById('pickStatus').textContent = `已选 ${picked.length}/3 点`;
      Plotly.restyle('cropPlot', {x:[picked.map(p=>p[0])], y:[picked.map(p=>p[1])], z:[picked.map(p=>p[2])]}, [3]);
    }

    function value(id) { return Number(document.getElementById(id).value); }
    function bounds() {
      return {
        min: axes.map(a => value(a + 'minNumber')),
        max: axes.map(a => value(a + 'maxNumber'))
      };
    }
    function visibleIndices(selectedFrame) {
      if (selectedFrame === cachedFrame) return cachedIndices;
      cachedFrame = selectedFrame;
      cachedIndices = [];
      for (let i=0; i<cloud.x.length; i++) {
        if (selectedFrame < 0 || cloud.frameIds[i] === selectedFrame) cachedIndices.push(i);
      }
      return cachedIndices;
    }
    function boxCoordinates(lo, hi) {
      const c = [
        [lo[0],lo[1],lo[2]], [hi[0],lo[1],lo[2]], [hi[0],hi[1],lo[2]], [lo[0],hi[1],lo[2]],
        [lo[0],lo[1],hi[2]], [hi[0],lo[1],hi[2]], [hi[0],hi[1],hi[2]], [lo[0],hi[1],hi[2]]
      ];
      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      const out = {x: [], y: [], z: []};
      for (const [a,b] of edges) {
        out.x.push(c[a][0], c[b][0], null);
        out.y.push(c[a][1], c[b][1], null);
        out.z.push(c[a][2], c[b][2], null);
      }
      return out;
    }
    function planeCoordinates(plane) {
      const n = plane.normal;
      const center = cloud.dataMin.map((v,i) => (v + cloud.dataMax[i]) / 2);
      const signed = dot(n, center) + plane.offset - plane.margin;
      const c = center.map((v,i) => v - signed*n[i]);
      const helper = Math.abs(n[2]) < 0.9 ? [0,0,1] : [0,1,0];
      let u = [n[1]*helper[2]-n[2]*helper[1], n[2]*helper[0]-n[0]*helper[2], n[0]*helper[1]-n[1]*helper[0]];
      const ul = Math.sqrt(dot(u,u)); u = u.map(v => v/ul);
      const v = [n[1]*u[2]-n[2]*u[1], n[2]*u[0]-n[0]*u[2], n[0]*u[1]-n[1]*u[0]];
      const size = Math.max(...cloud.dataMax.map((x,i)=>x-cloud.dataMin[i])) * 0.65;
      const corners = [[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]].map(([a,b]) => c.map((x,i)=>x+size*(a*u[i]+b*v[i])));
      return {x:corners.map(p=>p[0]), y:corners.map(p=>p[1]), z:corners.map(p=>p[2])};
    }
    function allPlaneCoordinates() {
      const out = {x:[],y:[],z:[]};
      for (const plane of planes) {
        const p = planeCoordinates(plane);
        out.x.push(...p.x, null); out.y.push(...p.y, null); out.z.push(...p.z, null);
      }
      return out;
    }
    function renderPlaneList() {
      const list = document.getElementById('plane-list');
      list.replaceChildren();
      planes.forEach((plane, index) => {
        const row = document.createElement('div'); row.className = 'plane-row';
        const text = document.createElement('span');
        const refreshText = () => text.textContent = `${index+1}. ${plane.name} n=[${plane.normal.map(v=>v.toFixed(4)).join(', ')}] d=${plane.offset.toFixed(4)}`;
        refreshText();
        const [initialYaw, initialPitch] = normalAngles(plane.normal);
        const yaw = document.createElement('input'); yaw.type='range'; yaw.min='-180'; yaw.max='180'; yaw.step='0.1'; yaw.value=initialYaw;
        const yawValue = document.createElement('span'); yawValue.textContent=`${initialYaw.toFixed(1)}°`;
        const pitch = document.createElement('input'); pitch.type='range'; pitch.min='-89.9'; pitch.max='89.9'; pitch.step='0.1'; pitch.value=initialPitch;
        const pitchValue = document.createElement('span'); pitchValue.textContent=`${initialPitch.toFixed(1)}°`;
        const rotate = () => {
          plane.normal=normalFromAngles(Number(yaw.value), Number(pitch.value));
          plane.offset=-dot(plane.normal, plane.anchor);
          yawValue.textContent=`${Number(yaw.value).toFixed(1)}°`;
          pitchValue.textContent=`${Number(pitch.value).toFixed(1)}°`;
          refreshText(); updateCrop();
        };
        yaw.addEventListener('input', rotate); pitch.addEventListener('input', rotate);
        const margin = document.createElement('input'); margin.type='range'; margin.min='0'; margin.max='0.20'; margin.step='0.001'; margin.value=plane.margin;
        const marginValue = document.createElement('span'); marginValue.textContent=`${Number(plane.margin).toFixed(3)}m`;
        margin.title='沿保留侧法向内缩距离（米）';
        margin.addEventListener('input', () => { plane.margin=Number(margin.value); marginValue.textContent=`${plane.margin.toFixed(3)}m`; updateCrop(); });
        const flip = document.createElement('button'); flip.textContent='翻转保留侧';
        flip.addEventListener('click', () => { plane.normal=plane.normal.map(v=>-v); plane.offset=-plane.offset; renderPlaneList(); updateCrop(); });
        const remove = document.createElement('button'); remove.textContent='删除';
        remove.addEventListener('click', () => { planes.splice(index,1); renderPlaneList(); updateCrop(); });
        row.append(text, document.createTextNode(' 方位:'), yaw, yawValue,
                   document.createTextNode(' 俯仰:'), pitch, pitchValue,
                   document.createTextNode(' 内缩:'), margin, marginValue, flip, remove);
        list.append(row);
      });
    }
    function updateCrop() {
      const b = bounds();
      const selectedFrame = frameSelect ? Number(frameSelect.value) : 0;
      const inside = {x: [], y: [], z: [], colors: []};
      const outside = {x: [], y: [], z: []};
      for (const i of visibleIndices(selectedFrame)) {
        const point = [cloud.x[i], cloud.y[i], cloud.z[i]];
        const keep = cloud.x[i] >= b.min[0] && cloud.x[i] <= b.max[0] &&
                     cloud.y[i] >= b.min[1] && cloud.y[i] <= b.max[1] &&
                     cloud.z[i] >= b.min[2] && cloud.z[i] <= b.max[2] &&
                     planes.every(p => dot(p.normal, point) + p.offset >= p.margin);
        const target = keep ? inside : outside;
        target.x.push(cloud.x[i]); target.y.push(cloud.y[i]); target.z.push(cloud.z[i]);
        if (keep) inside.colors.push(cloud.colors[i]);
      }
      const box = boxCoordinates(b.min, b.max);
      Plotly.restyle('cropPlot', {x:[outside.x], y:[outside.y], z:[outside.z]}, [0]);
      Plotly.restyle('cropPlot', {x:[inside.x], y:[inside.y], z:[inside.z], 'marker.color':[inside.colors]}, [1]);
      Plotly.restyle('cropPlot', {x:[box.x], y:[box.y], z:[box.z]}, [2]);
      const planeLines = allPlaneCoordinates();
      Plotly.restyle('cropPlot', {x:[planeLines.x], y:[planeLines.y], z:[planeLines.z]}, [4]);
      const validBounds = b.min.every((v, i) => v < b.max[i]);
      const label = selectedFrame < 0 ? '全部代表帧' : cloud.frameLabels[selectedFrame];
      document.getElementById('status').textContent = validBounds
        ? `${label}：保留 ${inside.x.length.toLocaleString()} / ${(inside.x.length + outside.x.length).toLocaleString()} 个预览点；${planes.length} 个自定义平面`
        : '边界无效：每个 min 必须小于 max';
      const fmt = values => values.map(v => v.toFixed(6)).join(' ');
      document.getElementById('command').value = `--crop-config ${cloud.configPathHint}`;
    }
    function bindPair(axis, side) {
      const range = document.getElementById(axis + side + 'Range');
      const number = document.getElementById(axis + side + 'Number');
      range.addEventListener('input', () => { number.value = range.value; updateCrop(); });
      number.addEventListener('input', () => { range.value = number.value; updateCrop(); });
    }
    for (const axis of axes) { bindPair(axis, 'min'); bindPair(axis, 'max'); }
    if (frameSelect) frameSelect.addEventListener('change', updateCrop);

    const initialBox = boxCoordinates(cloud.initialMin, cloud.initialMax);
    const traces = [
      {type:'scatter3d', mode:'markers', name:'框外背景', x:[], y:[], z:[],
       marker:{size:2, color:'#aaaaaa', opacity:0.10}, hoverinfo:'skip'},
      {type:'scatter3d', mode:'markers', name:'保留区域', x:cloud.x, y:cloud.y, z:cloud.z,
       marker:{size:3, color:cloud.colors, opacity:0.9},
       hovertemplate:'x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>'},
      {type:'scatter3d', mode:'lines', name:'裁剪框', x:initialBox.x, y:initialBox.y, z:initialBox.z,
       line:{color:'#ff3030', width:7}, hoverinfo:'skip'},
      {type:'scatter3d', mode:'markers', name:'已选平面点', x:[], y:[], z:[],
       marker:{size:8, color:'#ff00ff'}, hoverinfo:'skip'},
      {type:'scatter3d', mode:'lines', name:'自定义平面边界', x:[], y:[], z:[],
       line:{color:'#ff8c00', width:8}, hoverinfo:'skip'}
      ,...(cloud.cameraOrigin ? [
        {type:'scatter3d', mode:'markers+text', name:'CAM0 相机', x:[0], y:[0], z:[0],
         text:['CAM0'], textposition:'top center', marker:{size:10, color:'#00d8ff', symbol:'diamond'},
         hovertemplate:'CAM0 光心<br>(0, 0, 0)<extra></extra>'},
        {type:'scatter3d', mode:'lines+text', name:'CAM0 光轴 +Z', x:[0,0], y:[0,0], z:[0,0.35],
         text:['','+Z'], textposition:'top center', line:{color:'#00d8ff', width:9}, hoverinfo:'skip'},
        {type:'scatter3d', mode:'lines', name:'CAM0 视锥',
         x:[0,-0.18,null,0,0.18,null,0,0.18,null,0,-0.18,null,-0.18,0.18,0.18,-0.18,-0.18],
         y:[0,-0.12,null,0,-0.12,null,0,0.12,null,0,0.12,null,-0.12,-0.12,0.12,0.12,-0.12],
         z:[0,0.30,null,0,0.30,null,0,0.30,null,0,0.30,null,0.30,0.30,0.30,0.30,0.30],
         line:{color:'#00d8ff', width:5}, hoverinfo:'skip'}
      ] : [])
    ];
    Plotly.newPlot('cropPlot', traces, {
      margin:{l:0,r:0,b:0,t:0},
      scene:{aspectmode:'data', xaxis:{title:'X (m)'}, yaxis:{title:'Y (m)'}, zaxis:{title:'Z (m)'}},
      legend:{x:0.01,y:0.99}
    }, {responsive:true}).then(plot => {
      let pickArmed = false;
      plot.addEventListener('pointerdown', () => { pickArmed=true; }, true);
      plot.on('plotly_click', event => {
        const point = event.points && event.points[0];
        if (!pickArmed || !point || point.curveNumber > 1 || picked.length >= 3) return;
        pickArmed=false;
        const selected=[Number(point.x), Number(point.y), Number(point.z)];
        if (!picked.some(p => Math.sqrt(p.reduce((sum,v,i)=>sum+(v-selected[i])**2,0)) < 1e-6)) {
          picked.push(selected); updatePicked();
        }
      });
      renderPlaneList(); updateCrop();
    });
    document.getElementById('clearPickedButton').addEventListener('click', () => { picked=[]; updatePicked(); });
    document.getElementById('addPlaneButton').addEventListener('click', () => {
      if (picked.length !== 3) { alert('请先在幕布或其他边界平面上点选三个分散的点。'); return; }
      const a=picked[0], b=picked[1], c=picked[2];
      const u=b.map((v,i)=>v-a[i]), v=c.map((x,i)=>x-a[i]);
      let n=[u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]];
      const length=Math.sqrt(dot(n,n));
      if (length < 1e-8) { alert('三个点太近或近似共线，请重新选点。'); return; }
      n=n.map(x=>x/length);
      planes.push({name:document.getElementById('planeName').value || `plane_${planes.length+1}`,
                   normal:n, offset:-dot(n,a), margin:Math.max(0,Number(document.getElementById('planeMargin').value)||0),
                   anchor:[...a]});
      picked=[]; document.getElementById('planeName').value=`plane_${planes.length+1}`;
      updatePicked(); renderPlaneList(); updateCrop();
    });

    document.getElementById('copyButton').addEventListener('click', async () => {
      const text = document.getElementById('command').value;
      await navigator.clipboard.writeText(text);
      document.getElementById('copyStatus').textContent = '已复制，把这行参数发给 Codex 即可。';
    });
    document.getElementById('downloadButton').addEventListener('click', () => {
      const b = bounds();
      const config = {schema:'point_cloud_crop/v1', coordinate_frame:'source_point_frame',
                      keep_rule:'normal_dot_xyz_plus_offset_gte_margin', crop_min:b.min, crop_max:b.max, planes:planes};
      const blob = new Blob([JSON.stringify(config, null, 2)], {type:'application/json'});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob); link.download = 'crop_config.json'; link.click();
      URL.revokeObjectURL(link.href);
    });
  </script>
</body>
</html>
"""
    html = (
        html_template.replace("__PLOTLY_JS__", get_plotlyjs())
        .replace("__CONTROLS__", "".join(controls))
        .replace("__FRAME_CONTROL__", frame_control)
        .replace("__PAYLOAD__", payload)
    )
    file_path.write_text(html, encoding="utf-8")


def get_visualizer_class():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_project = repo_root / "visualizer"
    sys.path.insert(0, str(visualizer_project))
    try:
        from visualizer import Visualizer
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import DP3 visualizer. Install its dependencies and package with:\n"
            "  pip install flask plotly matplotlib termcolor\n"
            "  pip install -e visualizer"
        ) from exc
    return Visualizer


def make_clean_dataset_view(dataset_root: Path):
    """Hide macOS AppleDouble files without modifying the source dataset."""
    ignored_names = {".DS_Store"}
    has_sidecars = any(
        path.name.startswith("._") or path.name in ignored_names
        for path in dataset_root.rglob("*")
    )
    if not has_sidecars:
        return dataset_root, None

    temporary_dir = tempfile.TemporaryDirectory(prefix="lerobot-clean-")
    clean_root = Path(temporary_dir.name)
    for source in dataset_root.rglob("*"):
        if source.name.startswith("._") or source.name in ignored_names:
            continue
        relative = source.relative_to(dataset_root)
        destination = clean_root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source.resolve())

    print(f"Using a temporary clean dataset view that ignores macOS sidecar files: {clean_root}")
    return clean_root, temporary_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/example_rgbd")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.8)
    parser.add_argument("--xyz-only", action="store_true", help="Do not attach registered RGB colors")
    parser.add_argument(
        "--gravity-align",
        action="store_true",
        help="Rotate points so positive Z points upward using meta/orbbec_gravity.yaml",
    )
    parser.add_argument("--crop-min", type=float, nargs=3, metavar=("XMIN", "YMIN", "ZMIN"))
    parser.add_argument("--crop-max", type=float, nargs=3, metavar=("XMAX", "YMAX", "ZMAX"))
    parser.add_argument("--preview-points", type=int, default=30_000)
    parser.add_argument(
        "--crop-selector",
        action="store_true",
        help="Save an interactive HTML crop-bound selector with live point counts",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("pointcloud_preview"))
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Also serve the selected cloud at http://127.0.0.1:5000 (blocks until stopped)",
    )
    args = parser.parse_args()

    if args.pixel_stride < 1:
        parser.error("--pixel-stride must be at least 1")
    if args.frame_index < 0:
        parser.error("--frame-index must be non-negative")
    if (args.crop_min is None) != (args.crop_max is None):
        parser.error("--crop-min and --crop-max must be provided together")
    return args


def main() -> None:
    args = parse_args()
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is required to decode this dataset. Install the version documented by the dataset:\n"
            "  pip install 'lerobot[dataset]==0.6.0' 'av==15.1.0'"
        ) from exc

    dataset_root = args.dataset_root.expanduser().resolve()
    loader_root, clean_view = make_clean_dataset_view(dataset_root)
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=loader_root,
        video_backend="pyav",
    )
    if args.frame_index >= len(dataset):
        raise IndexError(f"frame-index {args.frame_index} is outside dataset length {len(dataset)}")

    sample = dataset[args.frame_index]
    point_cloud = reconstruct_point_cloud(
        sample=sample,
        pixel_stride=args.pixel_stride,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        include_rgb=not args.xyz_only,
    )

    if args.gravity_align:
        gravity_path = dataset_root / "meta" / "orbbec_gravity.yaml"
        point_cloud = align_with_gravity(point_cloud, load_gravity_direction(gravity_path))

    describe("reconstructed", point_cloud)
    selected = point_cloud
    if args.crop_min is not None:
        selected = crop_point_cloud(
            point_cloud,
            np.asarray(args.crop_min, dtype=np.float32),
            np.asarray(args.crop_max, dtype=np.float32),
        )
        describe("cropped", selected)
        if not len(selected):
            raise ValueError("The crop is empty; widen or correct the crop bounds")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"frame_{args.frame_index:06d}"
    Visualizer = get_visualizer_class()

    raw_preview = preview_subset(point_cloud, args.preview_points, args.seed)
    raw_html = args.output_dir / f"{prefix}_reconstructed.html"
    Visualizer().save_visualization_to_file(raw_preview, str(raw_html))
    print(f"Saved reconstructed preview: {raw_html.resolve()}")

    if args.crop_selector:
        selector_html = args.output_dir / f"{prefix}_crop_selector.html"
        selector_min = None if args.crop_min is None else np.asarray(args.crop_min, dtype=np.float32)
        selector_max = None if args.crop_max is None else np.asarray(args.crop_max, dtype=np.float32)
        save_crop_selector_html(raw_preview, selector_html, selector_min, selector_max)
        print(f"Saved interactive crop selector: {selector_html.resolve()}")

    if args.crop_min is not None:
        cropped_preview = preview_subset(selected, args.preview_points, args.seed)
        cropped_html = args.output_dir / f"{prefix}_cropped.html"
        Visualizer().save_visualization_to_file(cropped_preview, str(cropped_html))
        print(f"Saved cropped preview: {cropped_html.resolve()}")

    npy_path = args.output_dir / f"{prefix}_selected.npy"
    np.save(npy_path, selected)
    print(f"Saved selected point cloud: {npy_path.resolve()}")

    if args.serve:
        served = preview_subset(selected, args.preview_points, args.seed)
        print("Serving point cloud at http://127.0.0.1:5000 (Ctrl+C to stop)")
        Visualizer().visualize_pointcloud(served)

    # Keep the temporary clean dataset view alive until all decoding is done.
    del clean_view


if __name__ == "__main__":
    main()
