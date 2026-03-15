from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate.config import CFL
from generate.test_data import (
    TEST_ITERS,
    LevelSetReinitializer,
    build_flower_phi0,
    build_grid,
    interface_band_mask,
)


FIELD_COLORS = {
    "phi0": "#808080",
    "iter5": "#00b894",
    "iter10": "#ff8c00",
    "iter20": "#ff007f",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive single-case viewer with step slider and mode switch."
    )
    parser.add_argument("exp_id_pos", nargs="?", default=None, help="Experiment folder name under data/")
    parser.add_argument("--exp-id", dest="exp_id_opt", default=None, help="Experiment folder name under data/")
    parser.add_argument(
        "steps_pos",
        type=int,
        nargs="*",
        default=None,
        help=f"Optional positional reinitialization steps. Default: {TEST_ITERS}",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="*",
        default=None,
        help=f"Reinitialization steps to include. Default: {TEST_ITERS}",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "data"),
        help="Project data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output html path. Defaults to output_plots/<exp>_interactive.html",
    )
    parser.add_argument(
        "--grid-step",
        type=int,
        default=1,
        help="Draw every k-th grid line. Default 1 means full grid.",
    )
    return parser


def resolve_args(args: argparse.Namespace) -> tuple[str, list[int]]:
    exp_id = args.exp_id_opt or args.exp_id_pos or "smooth_276"
    steps = args.steps if args.steps is not None else args.steps_pos
    if not steps:
        steps = list(TEST_ITERS)
    return exp_id, [int(step) for step in steps]


def generate_interactive_case_plot(
    *,
    exp_id: str,
    steps: list[int],
    data_dir: str | Path,
    output: str | Path = "",
    grid_step: int = 1,
) -> Path:
    project_root = Path(__file__).resolve().parent
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    exp_dir = data_dir / exp_id
    meta_path = exp_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    x_grid, y_grid, h = build_grid(float(meta["L"]), int(meta["N"]))
    phi0 = build_flower_phi0(x_grid, y_grid, float(meta["a"]), float(meta["b"]), float(meta["p"]))

    fields: dict[str, np.ndarray] = {"phi0": phi0}
    reinitializer = LevelSetReinitializer(cfl=CFL)
    for step in steps:
        fields[f"iter{step}"] = reinitializer.reinitialize(phi0, h, int(step))

    payload = build_payload(fields, x_grid, y_grid, h, grid_step=max(1, int(grid_step)))
    payload["meta"] = {
        "exp_id": exp_id,
        "grid_n": int(meta["N"]),
        "h": float(h),
        "steps": [int(step) for step in steps],
        "domain": f"[{x_grid[0, 0]:.6f}, {x_grid[0, -1]:.6f}] x [{y_grid[0, 0]:.6f}, {y_grid[-1, 0]:.6f}]",
    }

    output_path = Path(output) if output else default_output_path(project_root, exp_id)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(payload), encoding="utf-8")
    return output_path


def build_payload(
    fields: dict[str, np.ndarray],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    h: float,
    *,
    grid_step: int,
) -> dict[str, object]:
    x_coords = x_grid[0, :]
    y_coords = y_grid[:, 0]
    phi0 = fields["phi0"]

    order: list[tuple[str, int]] = [("phi0", 0)]
    order.extend((f"iter{step}", int(step)) for step in sorted(int(name[4:]) for name in fields if name.startswith("iter")))

    contour_cache: dict[str, list[list[list[float]]]] = {}
    for name, _step in order:
        if name in fields:
            contour_cache[name] = extract_zero_contours(x_grid, y_grid, fields[name])

    baseline_points = flatten_contour_points(contour_cache["phi0"])
    shift_summary: list[dict[str, float | int | str]] = []
    eikonal_summary: list[dict[str, float | int]] = []
    field_entries: list[dict[str, object]] = []
    for name, step in order:
        if name not in fields:
            continue
        phi = fields[name]
        grad_norm = compute_grad_norm(phi, h)
        eikonal_abs = np.abs(grad_norm - 1.0)
        delta = phi - phi0
        indices = np.argwhere(interface_band_mask(phi))
        contours = contour_cache[name]
        contour_points = flatten_contour_points(contours)
        shift_stats = compute_shift_summary_stats(baseline_points, contour_points, h)
        shift_summary.append(
            {
                "name": name,
                "step": step,
                **shift_stats,
            }
        )
        eikonal_summary.append(
            {
                "name": name,
                "step": step,
                "mean": float(np.mean(eikonal_abs)),
                "p95": float(np.quantile(eikonal_abs, 0.95)),
                "max": float(np.max(eikonal_abs)),
            }
        )
        field_entries.append(
            {
                "name": name,
                "step": step,
                "color": FIELD_COLORS.get(name, "#444444"),
                "phi": phi.tolist(),
                "grad_norm": grad_norm.tolist(),
                "eikonal_abs": eikonal_abs.tolist(),
                "delta_to_phi0": delta.tolist(),
                "sample_indices": indices.tolist(),
                "contours": contours,
                "stats": {
                    "delta_abs_max": float(np.max(np.abs(delta))),
                    "eikonal_abs_max": float(np.max(eikonal_abs)),
                },
            }
        )

    return {
        "x_coords": x_coords.tolist(),
        "y_coords": y_coords.tolist(),
        "h": float(h),
        "grid_step": grid_step,
        "shift_summary": shift_summary,
        "eikonal_summary": eikonal_summary,
        "fields": field_entries,
    }


def compute_grad_norm(phi: np.ndarray, h: float) -> np.ndarray:
    phi_x = np.zeros_like(phi, dtype=np.float64)
    phi_y = np.zeros_like(phi, dtype=np.float64)
    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * h)
    phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * h)
    return np.sqrt(phi_x**2 + phi_y**2)


def extract_zero_contours(x_grid: np.ndarray, y_grid: np.ndarray, phi: np.ndarray) -> list[list[list[float]]]:
    fig, ax = plt.subplots(figsize=(4, 4))
    contour = ax.contour(x_grid, y_grid, phi, levels=[0.0])
    segments: list[list[list[float]]] = []
    for segment in contour.allsegs[0]:
        if len(segment) >= 2:
            segments.append(segment.astype(float).tolist())
    plt.close(fig)
    return segments


def flatten_contour_points(contours: list[list[list[float]]]) -> np.ndarray:
    points: list[list[float]] = []
    for segment in contours:
        points.extend(segment)
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def compute_shift_summary_stats(base_points: np.ndarray, other_points: np.ndarray, h: float) -> dict[str, float | str]:
    if base_points.size == 0 or other_points.size == 0:
        return {
            "mean_h": 0.0,
            "p95_h": 0.0,
            "max_h": 0.0,
            "verdict": "no contour",
        }

    distances = symmetric_nearest_distances(base_points, other_points)
    distances_h = distances / float(h)
    max_h = float(np.max(distances_h))
    mean_h = float(np.mean(distances_h))
    p95_h = float(np.quantile(distances_h, 0.95))
    return {
        "mean_h": mean_h,
        "p95_h": p95_h,
        "max_h": max_h,
        "verdict": shift_verdict(max_h),
    }


def symmetric_nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([nearest_distances(a, b), nearest_distances(b, a)])


def nearest_distances(source: np.ndarray, target: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    result = np.empty(source.shape[0], dtype=np.float64)
    for start in range(0, source.shape[0], chunk_size):
        chunk = source[start:start + chunk_size]
        diff = chunk[:, None, :] - target[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        result[start:start + chunk.shape[0]] = np.sqrt(np.min(dist_sq, axis=1))
    return result


def shift_verdict(max_shift_h: float) -> str:
    if max_shift_h < 0.25:
        return "visually almost unchanged"
    if max_shift_h < 1.0:
        return "sub-grid motion"
    return "grid-scale shift"


def build_html(payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Reinit Viewer</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background: #f6f4ef;
      color: #1f1f1f;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 330px 1fr;
      min-height: 100vh;
    }}
    .panel {{
      padding: 18px;
      border-right: 1px solid #d7d2c5;
      background: #faf8f1;
      overflow-y: auto;
    }}
    .panel h1 {{
      margin: 0 0 12px 0;
      font-size: 24px;
    }}
    .panel h2 {{
      margin: 18px 0 8px 0;
      font-size: 15px;
    }}
    .hint {{
      color: #555;
      font-size: 13px;
      line-height: 1.55;
    }}
    .control-block {{
      margin: 14px 0;
    }}
    .status-chip {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      background: #ece6d5;
      font-size: 12px;
      margin-top: 6px;
    }}
    .sparkline-wrap {{
      margin-top: 10px;
      background: #fffdf7;
      border: 1px solid #dad3c3;
      border-radius: 8px;
      padding: 8px;
    }}
    .sparkline-note {{
      margin-top: 6px;
      font-size: 12px;
      color: #555;
      line-height: 1.45;
    }}
    .sparkline-explain {{
      margin: 4px 0 8px 0;
      font-size: 12px;
      color: #666;
      line-height: 1.45;
    }}
    .radio-group label,
    .check-group label {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 8px 0;
      font-size: 14px;
    }}
    input[type="range"] {{
      width: 100%;
      margin-top: 8px;
    }}
    button {{
      width: 100%;
      padding: 8px 10px;
      border: 1px solid #c9c2b0;
      border-radius: 8px;
      background: #f2eee2;
      font-size: 14px;
      cursor: pointer;
      margin-top: 8px;
    }}
    .legend-line {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      margin: 6px 0;
    }}
    .line-sample {{
      width: 28px;
      height: 0;
      border-top: 3px solid;
    }}
    .line-dashed {{
      border-top-style: dashed;
    }}
    .info-box {{
      padding: 10px 12px;
      background: #fffdf7;
      border: 1px solid #dad3c3;
      border-radius: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      white-space: pre-wrap;
      line-height: 1.55;
    }}
    .canvas-wrap {{
      padding: 16px;
      overflow: auto;
    }}
    .workspace {{
      display: flex;
      align-items: flex-start;
      gap: 18px;
      width: fit-content;
    }}
    .figure-card {{
      background: #fff;
      border: 1px solid #d8d3c8;
      border-radius: 10px;
      padding: 14px;
      box-shadow: 0 4px 18px rgba(0,0,0,0.06);
      width: fit-content;
    }}
    .side-stack {{
      width: 320px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .side-card {{
      background: #fffdf7;
      border: 1px solid #dad3c3;
      border-radius: 10px;
      padding: 12px;
      box-shadow: 0 4px 18px rgba(0,0,0,0.04);
    }}
    .side-card h2 {{
      margin: 0 0 8px 0;
      font-size: 15px;
    }}
    .figure-title {{
      font-size: 15px;
      font-weight: 600;
      margin-bottom: 6px;
    }}
    .subtle {{
      color: #666;
      font-size: 12px;
      line-height: 1.5;
      margin-bottom: 8px;
    }}
    canvas {{
      background: #fff;
      border: 1px solid #cfc7b5;
      box-shadow: 0 4px 18px rgba(0,0,0,0.08);
      display: block;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="panel">
      <h1>Reinit Viewer</h1>
      <div id="meta" class="hint"></div>

      <div class="control-block">
        <h2>Step</h2>
        <div id="stepLabel" class="status-chip"></div>
        <input type="range" id="stepSlider" min="0" max="0" value="0" step="1">
        <button id="playButton" type="button">Auto Play</button>
      </div>

      <div class="control-block">
        <h2>Mode</h2>
        <div class="radio-group">
          <label><input type="radio" name="mode" value="interface" checked> Interface Fidelity</label>
          <label><input type="radio" name="mode" value="delta"> Field Difference: phi_k - phi0</label>
          <label><input type="radio" name="mode" value="eikonal"> Eikonal Error: ||grad phi| - 1|</label>
        </div>
      </div>

      <div class="control-block">
        <h2>Options</h2>
        <div class="check-group">
          <label><input type="checkbox" id="toggleGrid" checked> Show grid</label>
          <label><input type="checkbox" id="toggleSamples"> Show sample points</label>
          <label><input type="checkbox" id="toggleBand" checked> Show narrow band only (|phi0| &lt; 3h)</label>
        </div>
        <button id="blinkButton" type="button">Blink phi0 / phi_k</button>
      </div>
    </aside>

    <main class="canvas-wrap">
      <div class="workspace">
        <section class="figure-card">
          <div id="figureTitle" class="figure-title">Interface Fidelity</div>
          <div id="figureSubtitle" class="subtle"></div>
          <canvas id="mainPlot" width="1000" height="1000"></canvas>
          <canvas id="colorbar" width="1000" height="72"></canvas>
        </section>
        <aside class="side-stack">
          <section class="side-card">
            <h2>Shift Trend</h2>
            <div class="sparkline-explain">
              <code>d_i = dist_sym(Gamma_0, Gamma_k) / h</code><br>
              <code>mean = (1/N) sum d_i</code>, <code>p95 = Q_0.95(d_i)</code>, <code>max = max(d_i)</code>
            </div>
            <canvas id="shiftTrend" width="280" height="160"></canvas>
            <div id="shiftSummaryText" class="sparkline-note"></div>
          </section>
          <section class="side-card">
            <h2>Eikonal Trend</h2>
            <div class="sparkline-explain">
              <code>e_ij = ||grad phi(i,j)| - 1|</code><br>
              <code>mean = (1/N) sum e_ij</code>, <code>p95 = Q_0.95(e_ij)</code>, <code>max = max(e_ij)</code>
            </div>
            <canvas id="eikonalTrend" width="280" height="160"></canvas>
            <div id="eikonalSummaryText" class="sparkline-note"></div>
          </section>
          <section class="side-card">
            <h2>Legend</h2>
            <div class="legend-line"><span class="line-sample line-dashed" style="border-top-color:#808080"></span><span>phi0 = 0 baseline</span></div>
            <div class="legend-line"><span class="line-sample" style="border-top-color:#ff007f"></span><span>phi_k = 0 current step</span></div>
          </section>
          <section class="side-card">
            <h2>Hovered / Pinned Node</h2>
            <div id="nodeInfo" class="info-box">Move the mouse near the plot to inspect the nearest grid node.</div>
          </section>
        </aside>
      </div>
    </main>
  </div>

  <script>
    const data = {payload_json};
    const fields = data.fields;
    const baseline = fields[0];
    const xCoords = data.x_coords;
    const yCoords = data.y_coords;
    const h = data.h;
    const bandWidth = 3.0 * h;

    const mainCanvas = document.getElementById("mainPlot");
    const mainCtx = mainCanvas.getContext("2d");
    const colorbarCanvas = document.getElementById("colorbar");
    const colorbarCtx = colorbarCanvas.getContext("2d");
    const infoBox = document.getElementById("nodeInfo");
    const metaBox = document.getElementById("meta");
    const figureTitle = document.getElementById("figureTitle");
    const figureSubtitle = document.getElementById("figureSubtitle");
    const stepSlider = document.getElementById("stepSlider");
    const stepLabel = document.getElementById("stepLabel");
    const shiftTrendCanvas = document.getElementById("shiftTrend");
    const shiftTrendCtx = shiftTrendCanvas.getContext("2d");
    const shiftSummaryText = document.getElementById("shiftSummaryText");
    const eikonalTrendCanvas = document.getElementById("eikonalTrend");
    const eikonalTrendCtx = eikonalTrendCanvas.getContext("2d");
    const eikonalSummaryText = document.getElementById("eikonalSummaryText");
    const playButton = document.getElementById("playButton");
    const blinkButton = document.getElementById("blinkButton");

    const xMin = xCoords[0];
    const xMax = xCoords[xCoords.length - 1];
    const yMin = yCoords[0];
    const yMax = yCoords[yCoords.length - 1];
    const dx = xCoords.length > 1 ? xCoords[1] - xCoords[0] : 1;
    const dy = yCoords.length > 1 ? yCoords[1] - yCoords[0] : 1;
    const gridStep = data.grid_step;

    const state = {{
      mode: "interface",
      stepIndex: fields.length > 1 ? 1 : 0,
      showGrid: true,
      showSamples: false,
      showBandOnly: true,
      hovered: null,
      pinned: null,
      autoplayId: null,
      blink: false,
    }};

    metaBox.textContent =
      `exp_id=${{data.meta.exp_id}}\\nOmega=${{data.meta.domain}}\\nN=${{data.meta.grid_n}}\\nh=${{data.meta.h.toExponential(6)}}\\nsteps=${{data.meta.steps.join(", ")}}`;

    function makeTransform(canvas, margin) {{
      const width = canvas.width - 2 * margin;
      const height = canvas.height - 2 * margin;
      return {{
        canvas,
        margin,
        width,
        height,
        toX(x) {{
          return margin + ((x - xMin) / (xMax - xMin)) * width;
        }},
        toY(y) {{
          return canvas.height - margin - ((y - yMin) / (yMax - yMin)) * height;
        }},
        fromX(px) {{
          return xMin + ((px - margin) / width) * (xMax - xMin);
        }},
        fromY(py) {{
          return yMin + ((canvas.height - margin - py) / height) * (yMax - yMin);
        }},
        inside(px, py) {{
          return px >= margin && px <= canvas.width - margin && py >= margin && py <= canvas.height - margin;
        }},
      }};
    }}

    const tf = makeTransform(mainCanvas, 60);

    function currentField() {{
      return fields[state.stepIndex];
    }}

    function activeNode() {{
      return state.pinned || state.hovered;
    }}

    function nearestIndex(values, target) {{
      const spacing = values.length > 1 ? values[1] - values[0] : 1;
      const idx = Math.round((target - values[0]) / spacing);
      return Math.max(0, Math.min(values.length - 1, idx));
    }}

    function otherFieldForBlink() {{
      if (!state.blink || state.stepIndex === 0) return currentField();
      return (Math.floor(performance.now() / 450) % 2 === 0) ? baseline : currentField();
    }}

    function syncStepLabel() {{
      const field = currentField();
      stepLabel.textContent = field.step === 0 ? "Initial (0)" : `Iter ${{field.step}}`;
      stepSlider.value = state.stepIndex;
      const shift = data.shift_summary[state.stepIndex];
      shiftSummaryText.textContent =
        `mean=${{shift.mean_h.toFixed(3)}} h | p95=${{shift.p95_h.toFixed(3)}} h | ` +
        `max=${{shift.max_h.toFixed(3)}} h | ${{shift.verdict}}`;
      const eikonal = data.eikonal_summary[state.stepIndex];
      eikonalSummaryText.textContent =
        `mean=${{eikonal.mean.toExponential(3)}} | p95=${{eikonal.p95.toExponential(3)}} | ` +
        `max=${{eikonal.max.toExponential(3)}}`;
    }}

    function hexToRgba(hex, alpha) {{
      const clean = hex.replace("#", "");
      const bigint = parseInt(clean, 16);
      const r = (bigint >> 16) & 255;
      const g = (bigint >> 8) & 255;
      const b = bigint & 255;
      return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
    }}

    function deltaColor(value, limit) {{
      const t = Math.max(-1, Math.min(1, value / Math.max(limit, 1e-12)));
      if (t >= 0) {{
        const g = Math.round(255 * (1 - t));
        return `rgb(255,${{g}},${{g}})`;
      }}
      const r = Math.round(255 * (1 + t));
      return `rgb(${{r}},${{r}},255)`;
    }}

    function eikonalColor(value, limit) {{
      const t = Math.max(0, Math.min(1, value / Math.max(limit, 1e-12)));
      const r = Math.round(255 * t);
      const g = Math.round(245 - 165 * t);
      const b = Math.round(255 - 255 * t);
      return `rgb(${{r}},${{g}},${{b}})`;
    }}

    function updateFigureText() {{
      if (state.mode === "interface") {{
        figureTitle.textContent = "Interface Fidelity";
        figureSubtitle.textContent = "Gray dashed line is phi0 = 0. Bright line is the current phi_k = 0.";
      }} else if (state.mode === "delta") {{
        figureTitle.textContent = "Field Difference: phi_k - phi0";
        figureSubtitle.textContent = "Blue-white-red diverging map with the phi0 interface shown as a black reference curve.";
      }} else {{
        figureTitle.textContent = "Eikonal Error: ||grad phi| - 1|";
        figureSubtitle.textContent = "Sequential error map with optional narrow-band masking.";
      }}
    }}

    function drawShiftTrend() {{
      const ctx = shiftTrendCtx;
      const canvas = shiftTrendCanvas;
      const margin = {{ left: 34, right: 10, top: 12, bottom: 28 }};
      const width = canvas.width - margin.left - margin.right;
      const height = canvas.height - margin.top - margin.bottom;
      const rows = data.shift_summary;
      const xs = rows.map((row, idx) => idx);
      const yMax = Math.max(
        1e-6,
        ...rows.map(row => Math.max(Number(row.mean_h), Number(row.p95_h), Number(row.max_h)))
      );

      const toX = (idx) => margin.left + (idx / Math.max(1, xs.length - 1)) * width;
      const toY = (value) => margin.top + (1 - value / yMax) * height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#fffdf7";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.strokeStyle = "rgba(180,180,180,0.5)";
      ctx.lineWidth = 0.8;
      for (let k = 0; k <= 4; k++) {{
        const y = margin.top + (k / 4) * height;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(canvas.width - margin.right, y);
        ctx.stroke();
      }}

      ctx.strokeStyle = "#222";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(margin.left, margin.top);
      ctx.lineTo(margin.left, canvas.height - margin.bottom);
      ctx.lineTo(canvas.width - margin.right, canvas.height - margin.bottom);
      ctx.stroke();

      drawSeries("mean_h", "#2d6cdf");
      drawSeries("p95_h", "#f39c12");
      drawSeries("max_h", "#d64545");

      ctx.fillStyle = "#333";
      ctx.font = "11px sans-serif";
      rows.forEach((row, idx) => {{
        const label = String(row.step);
        ctx.fillText(label, toX(idx) - 4, canvas.height - 10);
      }});
      ctx.save();
      ctx.translate(12, margin.top + height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("shift / h", 0, 0);
      ctx.restore();

      const legend = [
        ["mean", "#2d6cdf"],
        ["p95", "#f39c12"],
        ["max", "#d64545"],
      ];
      let lx = margin.left;
      for (const [label, color] of legend) {{
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(lx, 10);
        ctx.lineTo(lx + 14, 10);
        ctx.stroke();
        ctx.fillStyle = "#333";
        ctx.fillText(label, lx + 18, 14);
        lx += 60;
      }}

      function drawSeries(key, color) {{
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        rows.forEach((row, idx) => {{
          const x = toX(idx);
          const y = toY(Number(row[key]));
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.stroke();

        rows.forEach((row, idx) => {{
          const x = toX(idx);
          const y = toY(Number(row[key]));
          ctx.beginPath();
          ctx.fillStyle = idx === state.stepIndex ? color : "#fff";
          ctx.strokeStyle = color;
          ctx.lineWidth = idx === state.stepIndex ? 2.5 : 1.5;
          ctx.arc(x, y, idx === state.stepIndex ? 4.5 : 3.2, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }});
      }}
    }}

    function drawEikonalTrend() {{
      const ctx = eikonalTrendCtx;
      const canvas = eikonalTrendCanvas;
      const margin = {{ left: 42, right: 10, top: 12, bottom: 28 }};
      const width = canvas.width - margin.left - margin.right;
      const height = canvas.height - margin.top - margin.bottom;
      const rows = data.eikonal_summary;
      const xs = rows.map((row, idx) => idx);
      const yMax = Math.max(1e-12, ...rows.map(row => Math.max(Number(row.mean), Number(row.p95), Number(row.max))));

      const toX = (idx) => margin.left + (idx / Math.max(1, xs.length - 1)) * width;
      const toY = (value) => margin.top + (1 - value / yMax) * height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#fffdf7";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.strokeStyle = "rgba(180,180,180,0.5)";
      ctx.lineWidth = 0.8;
      for (let k = 0; k <= 4; k++) {{
        const y = margin.top + (k / 4) * height;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(canvas.width - margin.right, y);
        ctx.stroke();
      }}

      ctx.strokeStyle = "#222";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(margin.left, margin.top);
      ctx.lineTo(margin.left, canvas.height - margin.bottom);
      ctx.lineTo(canvas.width - margin.right, canvas.height - margin.bottom);
      ctx.stroke();

      drawSeries("mean", "#2d6cdf");
      drawSeries("p95", "#f39c12");
      drawSeries("max", "#d64545");

      ctx.fillStyle = "#333";
      ctx.font = "11px sans-serif";
      rows.forEach((row, idx) => {{
        ctx.fillText(String(row.step), toX(idx) - 4, canvas.height - 10);
      }});
      ctx.save();
      ctx.translate(12, margin.top + height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("||grad phi|-1|", 0, 0);
      ctx.restore();

      const legend = [
        ["mean", "#2d6cdf"],
        ["p95", "#f39c12"],
        ["max", "#d64545"],
      ];
      let lx = margin.left;
      for (const [label, color] of legend) {{
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(lx, 10);
        ctx.lineTo(lx + 14, 10);
        ctx.stroke();
        ctx.fillStyle = "#333";
        ctx.fillText(label, lx + 18, 14);
        lx += 60;
      }}

      function drawSeries(key, color) {{
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        rows.forEach((row, idx) => {{
          const x = toX(idx);
          const y = toY(Number(row[key]));
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.stroke();

        rows.forEach((row, idx) => {{
          const x = toX(idx);
          const y = toY(Number(row[key]));
          ctx.beginPath();
          ctx.fillStyle = idx === state.stepIndex ? color : "#fff";
          ctx.strokeStyle = color;
          ctx.lineWidth = idx === state.stepIndex ? 2.5 : 1.5;
          ctx.arc(x, y, idx === state.stepIndex ? 4.5 : 3.2, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }});
      }}
    }}

    function drawGrid(ctx, alpha = 0.45) {{
      ctx.save();
      ctx.strokeStyle = `rgba(180,180,180,${{alpha}})`;
      ctx.lineWidth = 0.6;
      for (let i = 0; i < xCoords.length; i += gridStep) {{
        const x = tf.toX(xCoords[i]);
        ctx.beginPath();
        ctx.moveTo(x, tf.margin);
        ctx.lineTo(x, tf.canvas.height - tf.margin);
        ctx.stroke();
      }}
      for (let j = 0; j < yCoords.length; j += gridStep) {{
        const y = tf.toY(yCoords[j]);
        ctx.beginPath();
        ctx.moveTo(tf.margin, y);
        ctx.lineTo(tf.canvas.width - tf.margin, y);
        ctx.stroke();
      }}
      ctx.restore();
    }}

    function drawAxes(ctx) {{
      ctx.save();
      ctx.strokeStyle = "#222";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(tf.margin, tf.margin);
      ctx.lineTo(tf.margin, tf.canvas.height - tf.margin);
      ctx.lineTo(tf.canvas.width - tf.margin, tf.canvas.height - tf.margin);
      ctx.stroke();
      ctx.fillStyle = "#222";
      ctx.font = "12px sans-serif";
      ctx.fillText("x", tf.canvas.width - tf.margin + 10, tf.canvas.height - tf.margin + 4);
      ctx.fillText("y", tf.margin - 14, tf.margin - 8);
      ctx.fillStyle = "#555";
      ctx.font = "11px sans-serif";
      ctx.fillText(xMin.toFixed(3), tf.margin - 10, tf.canvas.height - tf.margin + 18);
      ctx.fillText(xMax.toFixed(3), tf.canvas.width - tf.margin - 18, tf.canvas.height - tf.margin + 18);
      ctx.fillText(yMin.toFixed(3), tf.margin - 36, tf.canvas.height - tf.margin + 4);
      ctx.fillText(yMax.toFixed(3), tf.margin - 36, tf.margin + 4);
      ctx.restore();
    }}

    function drawContourLines(ctx, field, color, dashed = false, width = 2.8, alpha = 1.0) {{
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.setLineDash(dashed ? [10, 8] : []);
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = width;
      for (const segment of field.contours) {{
        if (!segment.length) continue;
        ctx.beginPath();
        ctx.moveTo(tf.toX(segment[0][0]), tf.toY(segment[0][1]));
        for (let k = 1; k < segment.length; k++) {{
          ctx.lineTo(tf.toX(segment[k][0]), tf.toY(segment[k][1]));
        }}
        ctx.stroke();
      }}
      ctx.restore();
    }}

    function drawSamplePoints(ctx, field, color) {{
      ctx.save();
      ctx.fillStyle = hexToRgba(color, 0.55);
      for (const ij of field.sample_indices) {{
        const i = ij[0];
        const j = ij[1];
        ctx.beginPath();
        ctx.arc(tf.toX(xCoords[j]), tf.toY(yCoords[i]), 2.0, 0, Math.PI * 2);
        ctx.fill();
      }}
      ctx.restore();
    }}

    function drawFocusNode(ctx) {{
      const focus = activeNode();
      if (!focus) return;
      const x = tf.toX(xCoords[focus.j]);
      const y = tf.toY(yCoords[focus.i]);
      ctx.save();
      ctx.strokeStyle = state.pinned ? "#000" : "#444";
      ctx.lineWidth = state.pinned ? 1.8 : 1.4;
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - 10, y);
      ctx.lineTo(x + 10, y);
      ctx.moveTo(x, y - 10);
      ctx.lineTo(x, y + 10);
      ctx.stroke();
      ctx.restore();
    }}

    function drawHeatmap(ctx, values, mode, limit) {{
      const cellW = tf.width / Math.max(1, xCoords.length - 1);
      const cellH = tf.height / Math.max(1, yCoords.length - 1);
      for (let i = 0; i < yCoords.length; i++) {{
        for (let j = 0; j < xCoords.length; j++) {{
          const phi0 = Number(baseline.phi[i][j]);
          if (state.showBandOnly && Math.abs(phi0) > bandWidth) {{
            ctx.fillStyle = "rgba(35,35,35,0.78)";
          }} else {{
            const value = Number(values[i][j]);
            ctx.fillStyle = mode === "delta" ? deltaColor(value, limit) : eikonalColor(value, limit);
          }}
          ctx.fillRect(
            tf.toX(xCoords[j]) - 0.5 * cellW,
            tf.toY(yCoords[i]) - 0.5 * cellH,
            cellW + 1,
            cellH + 1
          );
        }}
      }}
    }}

    function drawMagnifier() {{
      const focus = activeNode();
      if (!focus) return;

      const insetSize = 240;
      const insetX = mainCanvas.width - insetSize - 24;
      const insetY = 24;
      const halfWorld = 10 * Math.max(dx, dy);
      const cx = xCoords[focus.j];
      const cy = yCoords[focus.i];
      const field = currentField();
      const values = state.mode === "delta" ? field.delta_to_phi0 : field.eikonal_abs;
      const limit = state.mode === "delta"
        ? Math.max(field.stats.delta_abs_max, 1e-12)
        : Math.max(field.stats.eikonal_abs_max, 1e-12);

      const toInsetX = (x) => insetX + ((x - (cx - halfWorld)) / (2 * halfWorld)) * insetSize;
      const toInsetY = (y) => insetY + insetSize - ((y - (cy - halfWorld)) / (2 * halfWorld)) * insetSize;
      const cell = insetSize / 20;

      mainCtx.save();
      mainCtx.fillStyle = "rgba(255,255,255,0.96)";
      mainCtx.strokeStyle = "#222";
      mainCtx.lineWidth = 1.1;
      mainCtx.fillRect(insetX, insetY, insetSize, insetSize);
      mainCtx.strokeRect(insetX, insetY, insetSize, insetSize);
      mainCtx.fillStyle = "#222";
      mainCtx.font = "12px sans-serif";
      mainCtx.fillText("Magnifier", insetX + 8, insetY + 16);

      mainCtx.strokeStyle = "rgba(180,180,180,0.35)";
      mainCtx.lineWidth = 0.6;
      for (let x = cx - halfWorld; x <= cx + halfWorld + 0.5 * dx; x += dx) {{
        const px = toInsetX(x);
        mainCtx.beginPath();
        mainCtx.moveTo(px, insetY);
        mainCtx.lineTo(px, insetY + insetSize);
        mainCtx.stroke();
      }}
      for (let y = cy - halfWorld; y <= cy + halfWorld + 0.5 * dy; y += dy) {{
        const py = toInsetY(y);
        mainCtx.beginPath();
        mainCtx.moveTo(insetX, py);
        mainCtx.lineTo(insetX + insetSize, py);
        mainCtx.stroke();
      }}

      if (state.mode === "interface") {{
        const activeField = otherFieldForBlink();
        drawInsetContour(baseline, "#808080", true, 2.0);
        if (activeField.step !== 0) {{
          drawInsetContour(activeField, activeField.color || "#ff007f", false, 2.8);
          if (state.showSamples) drawInsetSamples(activeField, activeField.color || "#ff007f");
        }}
      }} else {{
        for (let i = 0; i < yCoords.length; i++) {{
          for (let j = 0; j < xCoords.length; j++) {{
            const x = xCoords[j];
            const y = yCoords[i];
            if (Math.abs(x - cx) > halfWorld || Math.abs(y - cy) > halfWorld) continue;
            const phi0 = Number(baseline.phi[i][j]);
            if (state.showBandOnly && Math.abs(phi0) > bandWidth) {{
              mainCtx.fillStyle = "rgba(35,35,35,0.78)";
            }} else {{
              const value = Number(values[i][j]);
              mainCtx.fillStyle = state.mode === "delta" ? deltaColor(value, limit) : eikonalColor(value, limit);
            }}
            mainCtx.fillRect(toInsetX(x) - 0.5 * cell, toInsetY(y) - 0.5 * cell, cell + 1, cell + 1);
          }}
        }}
        drawInsetContour(baseline, "#111111", false, 1.5);
      }}

      const centerX = toInsetX(cx);
      const centerY = toInsetY(cy);
      mainCtx.strokeStyle = state.pinned ? "#000" : "#444";
      mainCtx.lineWidth = 1.2;
      mainCtx.beginPath();
      mainCtx.arc(centerX, centerY, 6, 0, Math.PI * 2);
      mainCtx.stroke();
      mainCtx.beginPath();
      mainCtx.moveTo(centerX - 9, centerY);
      mainCtx.lineTo(centerX + 9, centerY);
      mainCtx.moveTo(centerX, centerY - 9);
      mainCtx.lineTo(centerX, centerY + 9);
      mainCtx.stroke();
      mainCtx.restore();

      function drawInsetContour(insetField, color, dashed, width) {{
        mainCtx.save();
        mainCtx.setLineDash(dashed ? [8, 7] : []);
        mainCtx.strokeStyle = color;
        mainCtx.lineWidth = width;
        mainCtx.lineCap = "round";
        mainCtx.lineJoin = "round";
        for (const segment of insetField.contours) {{
          let drawing = false;
          mainCtx.beginPath();
          for (const point of segment) {{
            const x = point[0];
            const y = point[1];
            if (Math.abs(x - cx) > halfWorld || Math.abs(y - cy) > halfWorld) {{
              drawing = false;
              continue;
            }}
            const px = toInsetX(x);
            const py = toInsetY(y);
            if (!drawing) {{
              mainCtx.moveTo(px, py);
              drawing = true;
            }} else {{
              mainCtx.lineTo(px, py);
            }}
          }}
          if (drawing) mainCtx.stroke();
        }}
        mainCtx.restore();
      }}

      function drawInsetSamples(insetField, color) {{
        mainCtx.save();
        mainCtx.fillStyle = hexToRgba(color, 0.55);
        for (const ij of insetField.sample_indices) {{
          const i = ij[0];
          const j = ij[1];
          const x = xCoords[j];
          const y = yCoords[i];
          if (Math.abs(x - cx) > halfWorld || Math.abs(y - cy) > halfWorld) continue;
          mainCtx.beginPath();
          mainCtx.arc(toInsetX(x), toInsetY(y), 2.4, 0, Math.PI * 2);
          mainCtx.fill();
        }}
        mainCtx.restore();
      }}
    }}

    function drawColorbar(mode, limit) {{
      colorbarCtx.clearRect(0, 0, colorbarCanvas.width, colorbarCanvas.height);
      const x0 = 60;
      const y0 = 16;
      const barW = colorbarCanvas.width - 120;
      const barH = 18;

      if (mode === "interface") {{
        colorbarCtx.fillStyle = "#222";
        colorbarCtx.font = "12px sans-serif";
        colorbarCtx.fillText("Interface mode uses line comparison, blink, and the magnifier instead of a heatmap.", x0, y0 + 16);
        return;
      }}

      for (let k = 0; k < barW; k++) {{
        const t = k / Math.max(1, barW - 1);
        const value = mode === "delta" ? (2 * t - 1) * limit : t * limit;
        colorbarCtx.fillStyle = mode === "delta" ? deltaColor(value, limit) : eikonalColor(value, limit);
        colorbarCtx.fillRect(x0 + k, y0, 1, barH);
      }}
      colorbarCtx.strokeStyle = "#444";
      colorbarCtx.strokeRect(x0, y0, barW, barH);
      colorbarCtx.fillStyle = "#222";
      colorbarCtx.font = "12px sans-serif";
      if (mode === "delta") {{
        colorbarCtx.fillText((-limit).toExponential(3), x0 - 6, y0 + 34);
        colorbarCtx.fillText("0", x0 + 0.5 * barW - 4, y0 + 34);
        colorbarCtx.fillText(limit.toExponential(3), x0 + barW - 42, y0 + 34);
        colorbarCtx.fillText("Blue: phi_k < phi0 | White: unchanged | Red: phi_k > phi0", x0, y0 + 54);
      }} else {{
        colorbarCtx.fillText("0", x0 - 4, y0 + 34);
        colorbarCtx.fillText(limit.toExponential(3), x0 + barW - 42, y0 + 34);
        colorbarCtx.fillText("White: low error | Yellow/Red: high ||grad phi| - 1|", x0, y0 + 54);
      }}
    }}

    function drawInterfaceMode() {{
      mainCtx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
      mainCtx.fillStyle = "#fff";
      mainCtx.fillRect(0, 0, mainCanvas.width, mainCanvas.height);
      if (state.showGrid) drawGrid(mainCtx);
      drawAxes(mainCtx);

      const activeField = otherFieldForBlink();
      drawContourLines(mainCtx, baseline, "#808080", true, 2.2, 0.8);
      if (activeField.step !== 0) {{
        drawContourLines(mainCtx, activeField, activeField.color || "#ff007f", false, 3.0, 1.0);
        if (state.showSamples) drawSamplePoints(mainCtx, activeField, activeField.color || "#ff007f");
      }}

      drawFocusNode(mainCtx);
      drawMagnifier();
      drawColorbar("interface", 1.0);
    }}

    function drawCurrentHeatmap(mode) {{
      const field = currentField();
      const values = mode === "delta" ? field.delta_to_phi0 : field.eikonal_abs;
      const limit = mode === "delta"
        ? Math.max(field.stats.delta_abs_max, 1e-12)
        : Math.max(field.stats.eikonal_abs_max, 1e-12);

      mainCtx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
      mainCtx.fillStyle = "#fff";
      mainCtx.fillRect(0, 0, mainCanvas.width, mainCanvas.height);
      if (state.showGrid) drawGrid(mainCtx, 0.18);
      drawHeatmap(mainCtx, values, mode, limit);
      drawContourLines(mainCtx, baseline, "#111111", false, 1.4, 0.95);
      drawAxes(mainCtx);
      drawFocusNode(mainCtx);
      drawMagnifier();
      drawColorbar(mode, limit);
    }}

    function draw() {{
      updateFigureText();
      syncStepLabel();
      drawShiftTrend();
      drawEikonalTrend();
      if (state.mode === "interface") {{
        drawInterfaceMode();
      }} else if (state.mode === "delta") {{
        drawCurrentHeatmap("delta");
      }} else {{
        drawCurrentHeatmap("eikonal");
      }}
      if (state.blink && state.mode === "interface") {{
        requestAnimationFrame(draw);
      }}
    }}

    function updateInfo(i, j) {{
      const field = currentField();
      const lines = [];
      lines.push(state.pinned ? "mode: pinned" : "mode: hover");
      lines.push(`step: ${{field.step === 0 ? "Initial (0)" : "Iter " + field.step}}`);
      lines.push(`grid index: (i=${{i}}, j=${{j}})`);
      lines.push(`x=${{xCoords[j].toExponential(6)}}`);
      lines.push(`y=${{yCoords[i].toExponential(6)}}`);
      lines.push("");
      lines.push(`phi0=${{Number(baseline.phi[i][j]).toExponential(6)}}`);
      lines.push(`phi_k=${{Number(field.phi[i][j]).toExponential(6)}}`);
      lines.push(`delta phi=${{Number(field.delta_to_phi0[i][j]).toExponential(6)}}`);
      lines.push(`|grad phi_k|=${{Number(field.grad_norm[i][j]).toExponential(6)}}`);
      lines.push(`||grad phi_k| - 1|=${{Number(field.eikonal_abs[i][j]).toExponential(6)}}`);
      lines.push("");
      const sample0 = baseline.sample_indices.some(pair => pair[0] === i && pair[1] === j);
      const samplek = field.sample_indices.some(pair => pair[0] === i && pair[1] === j);
      lines.push(`sample in phi0 band=${{sample0}}`);
      lines.push(`sample in current field=${{samplek}}`);
      infoBox.textContent = lines.join("\\n");
    }}

    function bindPointer() {{
      mainCanvas.addEventListener("click", (event) => {{
        const rect = mainCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        if (!tf.inside(x, y)) return;
        const worldX = tf.fromX(x);
        const worldY = tf.fromY(y);
        const j = nearestIndex(xCoords, worldX);
        const i = nearestIndex(yCoords, worldY);
        if (state.pinned && state.pinned.i === i && state.pinned.j === j) {{
          state.pinned = null;
        }} else {{
          state.pinned = {{ i, j }};
        }}
        updateInfo(i, j);
        draw();
      }});

      mainCanvas.addEventListener("mousemove", (event) => {{
        const rect = mainCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        if (!tf.inside(x, y)) {{
          if (!state.pinned) {{
            state.hovered = null;
            infoBox.textContent = "Move the mouse near the plot to inspect the nearest grid node.";
            draw();
          }}
          return;
        }}
        const worldX = tf.fromX(x);
        const worldY = tf.fromY(y);
        const j = nearestIndex(xCoords, worldX);
        const i = nearestIndex(yCoords, worldY);
        state.hovered = {{ i, j }};
        updateInfo(i, j);
        draw();
      }});

      mainCanvas.addEventListener("mouseleave", () => {{
        if (!state.pinned) {{
          state.hovered = null;
          infoBox.textContent = "Move the mouse near the plot to inspect the nearest grid node.";
          draw();
        }}
      }});
    }}

    function buildControls() {{
      stepSlider.max = String(Math.max(0, fields.length - 1));
      stepSlider.value = String(state.stepIndex);

      stepSlider.addEventListener("input", (event) => {{
        state.stepIndex = Number(event.target.value);
        const focus = activeNode();
        if (focus) updateInfo(focus.i, focus.j);
        draw();
      }});

      playButton.addEventListener("click", () => {{
        if (state.autoplayId !== null) {{
          clearInterval(state.autoplayId);
          state.autoplayId = null;
          playButton.textContent = "Auto Play";
          return;
        }}
        state.autoplayId = window.setInterval(() => {{
          state.stepIndex = (state.stepIndex + 1) % fields.length;
          const focus = activeNode();
          if (focus) updateInfo(focus.i, focus.j);
          draw();
        }}, 700);
        playButton.textContent = "Stop";
      }});

      document.querySelectorAll("input[name='mode']").forEach((radio) => {{
        radio.addEventListener("change", (event) => {{
          state.mode = event.target.value;
          draw();
        }});
      }});

      document.getElementById("toggleGrid").addEventListener("change", (event) => {{
        state.showGrid = event.target.checked;
        draw();
      }});
      document.getElementById("toggleSamples").addEventListener("change", (event) => {{
        state.showSamples = event.target.checked;
        draw();
      }});
      document.getElementById("toggleBand").addEventListener("change", (event) => {{
        state.showBandOnly = event.target.checked;
        draw();
      }});

      blinkButton.addEventListener("click", () => {{
        state.blink = !state.blink;
        blinkButton.textContent = state.blink ? "Stop Blink" : "Blink phi0 / phi_k";
        draw();
      }});

      window.addEventListener("keydown", (event) => {{
        if (event.code === "Space") {{
          event.preventDefault();
          state.blink = true;
          blinkButton.textContent = "Stop Blink";
          draw();
        }}
      }});
      window.addEventListener("keyup", (event) => {{
        if (event.code === "Space") {{
          state.blink = false;
          blinkButton.textContent = "Blink phi0 / phi_k";
          draw();
        }}
      }});
    }}

    buildControls();
    bindPointer();
    draw();
  </script>
</body>
</html>
"""


def default_output_path(project_root: Path, exp_id: str) -> Path:
    return project_root / "output_plots" / f"{exp_id}_interactive.html"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    exp_id, steps = resolve_args(args)

    output_path = generate_interactive_case_plot(
        exp_id=exp_id,
        steps=steps,
        data_dir=args.data_dir,
        output=args.output,
        grid_step=args.grid_step,
    )

    print(f"Output format : HTML")
    print(f"Saved to      : {output_path}")
    print(f"Case / steps  : {exp_id} / {steps}")


if __name__ == "__main__":
    main()
