#!/usr/bin/env python
"""Build a self-contained interactive protein/batch throughput explorer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "submission/data/protein_size_scaling"
OUTPUT = ROOT / "submission/figures/protein_batch_scaling_interactive.html"
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)

COLORS = {
    "baseline": "#C46686",
    "flash": "#689BCC",
    "dark": "#25282B",
    "grid": "#D8DADC",
    "accent": "#D87756",
}

PANELS = {
    "schnet": {
        "title": "CGSchNet (37-1,851 beads)",
        "xlabel": "Protein size (beads)",
        "xticks": [32, 128, 512, 2048],
    },
    "mace": {
        "title": "MACE (166-5,807 atoms)",
        "xlabel": "Protein size (atoms)",
        "xticks": [256, 1024, 4096],
    },
    "bioemu": {
        "title": "BioEmu (8-377 aa)",
        "xlabel": "Protein size (residues)",
        "xticks": [8, 32, 128, 512],
    },
}


def load_rows() -> pd.DataFrame:
    frames = []
    for batch_size in BATCH_SIZES:
        path = DATA / f"protein_size_scaling_bs{batch_size}.csv"
        frame = pd.read_csv(path)
        if len(frame) != 96:
            raise ValueError(f"Expected 96 rows in {path}, found {len(frame)}")
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    if len(rows) != 672:
        raise ValueError(f"Expected 672 rows, found {len(rows)}")
    return rows


def aggregate_surface(
    rows: pd.DataFrame, family: str, series: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = rows[(rows["family"] == family) & (rows["series"] == series)]
    sizes = np.array(sorted(selected["size"].unique()), dtype=float)
    successful = selected[selected["status"] == "ok"]
    values = successful.pivot_table(
        index="batch_size", columns="size", values="throughput", aggfunc="median"
    ).reindex(index=BATCH_SIZES, columns=sizes)
    return sizes, np.array(BATCH_SIZES, dtype=float), values.to_numpy(dtype=float)


def text_or_dash(value: object, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def point_customdata(frame: pd.DataFrame) -> list[list[object]]:
    values = []
    for row in frame.itertuples(index=False):
        values.append(
            [
                row.system,
                int(row.size),
                row.size_unit,
                int(row.batch_size),
                float(row.throughput) if pd.notna(row.throughput) else None,
                row.throughput_unit,
                text_or_dash(row.peak_allocated_gb),
                text_or_dash(row.peak_reserved_gb),
                row.source,
                row.embedding if pd.notna(row.embedding) else "-",
                row.series,
                row.status,
                row.error if pd.notna(row.error) else "",
            ]
        )
    return values


def build_figure(rows: pd.DataFrame) -> tuple[go.Figure, dict, dict]:
    figure = go.Figure()
    highlight_payload: dict[str, dict] = {}
    trace_map = {
        "surface_baseline": [],
        "surface_flash": [],
        "points": [],
        "highlights": {},
        "failures": [],
    }
    scenes = ("scene", "scene2", "scene3")
    families = ("schnet", "mace", "bioemu")

    for family_index, (family, scene) in enumerate(zip(families, scenes)):
        family_rows = rows[rows["family"] == family]
        successful_family = family_rows[family_rows["status"] == "ok"]
        z_logs = np.log10(successful_family["throughput"].astype(float).to_numpy())
        z_floor = float(np.floor(z_logs.min()))
        z_ceiling = float(np.ceil(z_logs.max()))

        for series in ("baseline", "flash"):
            sizes, batches, values = aggregate_surface(rows, family, series)
            x_grid, y_grid = np.meshgrid(np.log2(sizes), np.log2(batches))
            z_grid = np.log10(values)
            customdata = np.empty((len(batches), len(sizes), 3), dtype=object)
            for row_index, batch_size in enumerate(batches):
                for column_index, size in enumerate(sizes):
                    customdata[row_index, column_index] = [
                        int(size),
                        int(batch_size),
                        values[row_index, column_index],
                    ]
            surface_index = len(figure.data)
            figure.add_trace(
                go.Surface(
                    x=x_grid,
                    y=y_grid,
                    z=z_grid,
                    customdata=customdata,
                    surfacecolor=np.zeros_like(z_grid),
                    colorscale=[[0, COLORS[series]], [1, COLORS[series]]],
                    cmin=0,
                    cmax=1,
                    showscale=False,
                    opacity=0.42 if series == "baseline" else 0.64,
                    name="Baseline" if series == "baseline" else "FlashMD (ours)",
                    legendgroup=series,
                    showlegend=False,
                    scene=scene,
                    hovertemplate=(
                        "Protein size: %{customdata[0]:,}<br>"
                        "Batch size: %{customdata[1]}<br>"
                        "Median throughput: %{customdata[2]:.3g}<extra></extra>"
                    ),
                )
            )
            trace_map[f"surface_{series}"].append(surface_index)

            successful = family_rows[
                (family_rows["series"] == series) & (family_rows["status"] == "ok")
            ].copy()
            points_index = len(figure.data)
            figure.add_trace(
                go.Scatter3d(
                    x=np.log2(successful["size"].astype(float)),
                    y=np.log2(successful["batch_size"].astype(float)),
                    z=np.log10(successful["throughput"].astype(float)),
                    mode="markers",
                    customdata=point_customdata(successful),
                    marker={
                        "size": 3.4,
                        "color": COLORS[series],
                        "symbol": "circle" if series == "baseline" else "square",
                        "opacity": 0.78 if series == "baseline" else 0.96,
                        "line": {"width": 0},
                    },
                    opacity=1.0,
                    name="Baseline" if series == "baseline" else "FlashMD (ours)",
                    legendgroup=series,
                    showlegend=family_index == 0,
                    scene=scene,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Protein size: %{customdata[1]:,} %{customdata[2]}<br>"
                        "Batch size: %{customdata[3]}<br>"
                        "Throughput: %{customdata[4]:.4g} %{customdata[5]}<br>"
                        "Peak allocated: %{customdata[6]} GB<br>"
                        "Peak reserved: %{customdata[7]} GB<br>"
                        "Source: %{customdata[8]}<br>"
                        "Embedding: %{customdata[9]}<extra></extra>"
                    ),
                )
            )
            trace_map["points"].append(points_index)

            initial_batch = 64
            batch_rows = successful[successful["batch_size"] == initial_batch]
            grouped = (
                batch_rows.groupby("size", as_index=False)
                .agg(throughput=("throughput", "median"), system=("system", ", ".join))
                .sort_values("size")
            )
            highlight_index = len(figure.data)
            figure.add_trace(
                go.Scatter3d(
                    x=np.log2(grouped["size"].astype(float)),
                    y=[np.log2(initial_batch)] * len(grouped),
                    z=np.log10(grouped["throughput"].astype(float)),
                    mode="lines+markers",
                    customdata=np.column_stack(
                        [
                            grouped["system"],
                            grouped["size"].astype(int),
                            grouped["throughput"],
                        ]
                    ),
                    line={"color": COLORS[series], "width": 8},
                    marker={
                        "color": COLORS[series],
                        "size": 5.2,
                        "symbol": "circle" if series == "baseline" else "square",
                        "line": {"color": "white", "width": 1.2},
                    },
                    opacity=1.0,
                    name=f"Selected batch: {series}",
                    legendgroup=series,
                    showlegend=False,
                    scene=scene,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Protein size: %{customdata[1]:,}<br>"
                        "Selected-batch throughput: %{customdata[2]:.4g}<extra></extra>"
                    ),
                )
            )
            trace_map["highlights"][f"{family}:{series}"] = highlight_index

        failures = family_rows[family_rows["status"] != "ok"].copy()
        if failures.empty:
            failure_x: list[float] = []
            failure_y: list[float] = []
            failure_z: list[float] = []
            failure_colors: list[str] = []
        else:
            failure_x = np.log2(failures["size"].astype(float)).tolist()
            failure_y = np.log2(failures["batch_size"].astype(float)).tolist()
            failure_z = [z_floor + 0.03] * len(failures)
            failure_colors = [COLORS[value] for value in failures["series"]]
        failure_index = len(figure.data)
        figure.add_trace(
            go.Scatter3d(
                x=failure_x,
                y=failure_y,
                z=failure_z,
                mode="markers",
                customdata=point_customdata(failures),
                marker={
                    "size": 6.2,
                    "color": failure_colors,
                    "symbol": "x",
                    "line": {"width": 1.5},
                },
                name="Did not complete",
                legendgroup="failure",
                showlegend=family_index == 0,
                scene=scene,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Protein size: %{customdata[1]:,} %{customdata[2]}<br>"
                    "Batch size: %{customdata[3]}<br>"
                    "Implementation: %{customdata[10]}<br>"
                    "Status: %{customdata[11]}<extra></extra>"
                ),
            )
        )
        trace_map["failures"].append(failure_index)

        for batch_size in BATCH_SIZES:
            batch_key = str(batch_size)
            highlight_payload.setdefault(batch_key, {})
            highlight_payload[batch_key].setdefault(family, {})
            for series in ("baseline", "flash"):
                successful = family_rows[
                    (family_rows["series"] == series)
                    & (family_rows["status"] == "ok")
                    & (family_rows["batch_size"] == batch_size)
                ]
                grouped = (
                    successful.groupby("size", as_index=False)
                    .agg(
                        throughput=("throughput", "median"),
                        system=("system", ", ".join),
                    )
                    .sort_values("size")
                )
                highlight_payload[batch_key][family][series] = {
                    "x": np.log2(grouped["size"].astype(float)).tolist(),
                    "y": [float(np.log2(batch_size))] * len(grouped),
                    "z": np.log10(grouped["throughput"].astype(float)).tolist(),
                    "customdata": [
                        [system, int(size), float(throughput)]
                        for system, size, throughput in zip(
                            grouped["system"], grouped["size"], grouped["throughput"]
                        )
                    ],
                }

        all_sizes = sorted(family_rows["size"].unique())
        x_ticks = [
            value for value in PANELS[family]["xticks"] if value <= max(all_sizes) * 1.2
        ]
        z_ticks = list(range(int(z_floor), int(z_ceiling) + 1))
        z_tick_text = []
        for value in z_ticks:
            throughput = 10**value
            z_tick_text.append(
                f"{throughput / 1000:g}k" if throughput >= 1000 else f"{throughput:g}"
            )
        scene_layout = {
            "domain": {
                "x": [family_index / 3 + 0.005, (family_index + 1) / 3 - 0.005],
                "y": [0.0, 1.0],
            },
            "xaxis": {
                "title": {"text": PANELS[family]["xlabel"], "font": {"size": 13}},
                "tickvals": np.log2(x_ticks).tolist(),
                "ticktext": [f"{value:,}" for value in x_ticks],
                "range": [np.log2(min(all_sizes) / 1.15), np.log2(max(all_sizes) * 1.3)],
                "gridcolor": COLORS["grid"],
                "showbackground": False,
                "showspikes": False,
                "tickfont": {"size": 11},
            },
            "yaxis": {
                "title": {"text": "Batch size", "font": {"size": 13}},
                "tickvals": np.log2(BATCH_SIZES).tolist(),
                "ticktext": [str(value) for value in BATCH_SIZES],
                "range": [-0.2, np.log2(64) + 0.3],
                "gridcolor": COLORS["grid"],
                "showbackground": False,
                "showspikes": False,
                "tickfont": {"size": 11},
            },
            "zaxis": {
                "title": {"text": "Throughput", "font": {"size": 13}},
                "tickvals": z_ticks,
                "ticktext": z_tick_text,
                "range": [z_floor, z_ceiling + 0.08],
                "gridcolor": COLORS["grid"],
                "showbackground": False,
                "showspikes": False,
                "tickfont": {"size": 11},
            },
            "camera": {
                "eye": {"x": 1.48, "y": -1.75, "z": 0.76},
                "projection": {"type": "orthographic"},
            },
            "aspectmode": "manual",
            "aspectratio": {"x": 1.22, "y": 1.0, "z": 0.82},
            "dragmode": "orbit",
        }
        figure.update_layout(**{scene: scene_layout})

    annotations = []
    for x_position, family in zip((1 / 6, 0.5, 5 / 6), families):
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": x_position,
                "y": 1.04,
                "text": f"<b>{PANELS[family]['title']}</b>",
                "showarrow": False,
                "xanchor": "center",
                "yanchor": "bottom",
                "font": {"family": "Times New Roman, Times, serif", "size": 18},
            }
        )
    figure.update_layout(
        autosize=True,
        height=670,
        margin={"l": 8, "r": 8, "t": 72, "b": 8},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Times New Roman, Times, serif", "color": COLORS["dark"]},
        annotations=annotations,
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.115,
            "yanchor": "top",
            "font": {"size": 13},
            "groupclick": "togglegroup",
            "itemsizing": "constant",
        },
        hoverlabel={"bgcolor": "white", "font": {"family": "Arial", "size": 12}},
        uirevision="protein-batch-scaling-v1",
    )
    return figure, highlight_payload, trace_map


def safe_script(value: str) -> str:
    return value.replace("</script>", "<\\/script>")


def build_html(rows: pd.DataFrame) -> str:
    figure, highlight_payload, trace_map = build_figure(rows)
    plotly_js = safe_script(get_plotlyjs())
    figure_json = safe_script(figure.to_json())
    raw_json = safe_script(rows.to_json(orient="records"))
    highlights_json = safe_script(json.dumps(highlight_payload, separators=(",", ":")))
    trace_map_json = safe_script(json.dumps(trace_map, separators=(",", ":")))

    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FlashMD protein and batch scaling</title>
<style>
:root {
  --ink: #25282b;
  --muted: #666b70;
  --line: #d8dadc;
  --soft: #f5f6f7;
  --baseline: #c46686;
  --flash: #689bcc;
  --accent: #d87756;
}
* { box-sizing: border-box; }
html, body { margin: 0; min-height: 100%; background: #fff; color: var(--ink); }
body { font-family: Arial, Helvetica, sans-serif; }
header {
  min-height: 64px;
  padding: 12px 22px 10px;
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
h1 { margin: 0; font-family: "Times New Roman", Times, serif; font-size: 23px; letter-spacing: 0; }
.meta { color: var(--muted); font-size: 12px; white-space: nowrap; }
.toolbar {
  min-height: 58px;
  padding: 9px 22px;
  border-bottom: 1px solid var(--line);
  background: var(--soft);
  display: flex;
  align-items: center;
  gap: 15px;
  flex-wrap: wrap;
}
.control { display: flex; align-items: center; gap: 7px; font-size: 12px; }
.control label { font-weight: 600; white-space: nowrap; }
button, select, input[type="search"] {
  min-height: 31px;
  border: 1px solid #bfc3c7;
  border-radius: 4px;
  background: #fff;
  color: var(--ink);
  font: inherit;
}
button { padding: 5px 11px; cursor: pointer; }
button:hover { border-color: #7c8288; background: #fafafa; }
button:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid var(--flash); outline-offset: 1px; }
input[type="range"] { width: 230px; accent-color: var(--flash); }
.switch { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; white-space: nowrap; }
.switch input { accent-color: var(--flash); }
#batch-status { color: var(--accent); font-weight: 700; min-width: 290px; }
.plot-viewport { width: 100%; overflow-x: auto; border-bottom: 1px solid var(--line); }
#plot { width: 100%; min-width: 1120px; height: 680px; }
.plot-note { padding: 7px 22px 10px; color: var(--muted); font-size: 11px; text-align: center; }
.data-section { border-top: 1px solid var(--line); }
.data-section > summary {
  cursor: pointer;
  padding: 14px 22px;
  font-weight: 700;
  font-size: 14px;
  list-style-position: inside;
}
.data-controls {
  padding: 10px 22px;
  background: var(--soft);
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 9px;
}
.data-controls select, .data-controls input[type="search"] { padding: 5px 8px; }
.data-controls input[type="search"] { width: 220px; }
#row-count { margin-left: auto; color: var(--muted); font-size: 12px; }
.table-wrap { overflow: auto; max-height: 610px; }
table { width: 100%; min-width: 1280px; border-collapse: collapse; font-size: 11px; }
th, td { padding: 7px 8px; border-bottom: 1px solid #e5e7e9; text-align: left; vertical-align: top; }
th { position: sticky; top: 0; background: #fff; z-index: 1; font-size: 10px; text-transform: uppercase; color: #555b60; }
td.numeric { font-variant-numeric: tabular-nums; text-align: right; }
.status-ok { color: #25704a; font-weight: 700; }
.status-oom, .status-failed { color: #a13f52; font-weight: 700; }
details.row-details summary { cursor: pointer; color: #3d668d; }
details.row-details pre { max-width: 620px; white-space: pre-wrap; overflow-wrap: anywhere; color: #454a4f; font-size: 10px; }
.pager { padding: 10px 22px 16px; display: flex; align-items: center; justify-content: center; gap: 10px; }
#page-status { min-width: 130px; text-align: center; font-size: 12px; color: var(--muted); }
@media (max-width: 700px) {
  header { align-items: flex-start; padding: 10px 12px; }
  h1 { font-size: 19px; }
  .meta { white-space: normal; text-align: right; }
  .toolbar, .data-controls { padding-left: 12px; padding-right: 12px; }
  input[type="range"] { width: 160px; }
  #batch-status { width: 100%; min-width: 0; }
  #row-count { width: 100%; margin-left: 0; }
}
</style>
<script>__PLOTLY_JS__</script>
</head>
<body>
<header>
  <h1>FlashMD: joint protein-size and batch-size scaling</h1>
  <div class="meta">672 configurations &middot; NVIDIA RTX PRO 6000 &middot; batch 1-64</div>
</header>
<section class="toolbar" aria-label="Visualization controls">
  <div class="control">
    <label for="batch-slider">Highlighted batch</label>
    <input id="batch-slider" type="range" min="0" max="6" value="6" step="1" list="batch-ticks">
    <datalist id="batch-ticks"><option value="0" label="1"></option><option value="1" label="2"></option><option value="2" label="4"></option><option value="3" label="8"></option><option value="4" label="16"></option><option value="5" label="32"></option><option value="6" label="64"></option></datalist>
  </div>
  <button id="play-batches" type="button">Play batches</button>
  <label class="switch"><input id="focus-slice" type="checkbox">Focus slice</label>
  <label class="switch"><input id="sync-cameras" type="checkbox" checked>Sync cameras</label>
  <button id="reset-view" type="button">Reset view</button>
  <div id="batch-status" aria-live="polite"></div>
</section>
<div class="plot-viewport"><div id="plot" role="img" aria-label="Interactive three-dimensional throughput scaling surfaces"></div></div>
<div class="plot-note">Protein size and throughput use logarithmic coordinates. Surface vertices are measured; crosses mark runs that did not complete.</div>
<details class="data-section" id="data-section">
  <summary>Embedded benchmark data (672 rows)</summary>
  <div class="data-controls">
    <select id="family-filter" aria-label="Filter by model family"><option value="all">All families</option><option value="schnet">CGSchNet</option><option value="mace">MACE</option><option value="bioemu">BioEmu</option></select>
    <select id="series-filter" aria-label="Filter by implementation"><option value="all">Both implementations</option><option value="baseline">Baseline</option><option value="flash">FlashMD</option></select>
    <select id="status-filter" aria-label="Filter by status"><option value="all">All statuses</option><option value="ok">Successful</option><option value="oom">OOM</option><option value="failed">Failed</option></select>
    <select id="batch-filter" aria-label="Filter by batch"><option value="all">All batches</option><option value="1">Batch 1</option><option value="2">Batch 2</option><option value="4">Batch 4</option><option value="8">Batch 8</option><option value="16">Batch 16</option><option value="32">Batch 32</option><option value="64">Batch 64</option></select>
    <input id="search-filter" type="search" placeholder="Search system, source, path" aria-label="Search rows">
    <button id="export-csv" type="button">Export embedded CSV</button>
    <span id="row-count"></span>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Family</th><th>Implementation</th><th>System</th><th>Protein size</th><th>Batch</th><th>Throughput</th><th>Peak allocated</th><th>Peak reserved</th><th>Status</th><th>Source</th><th>Result</th><th>All fields</th></tr></thead>
      <tbody id="data-body"></tbody>
    </table>
  </div>
  <div class="pager"><button id="prev-page" type="button">Previous</button><span id="page-status"></span><button id="next-page" type="button">Next</button></div>
</details>
<script type="application/json" id="benchmark-data">__RAW_JSON__</script>
<script>
const FIGURE = __FIGURE_JSON__;
const HIGHLIGHTS = __HIGHLIGHTS_JSON__;
const TRACE_MAP = __TRACE_MAP_JSON__;
const RAW_DATA = JSON.parse(document.getElementById('benchmark-data').textContent);
const BATCHES = [1, 2, 4, 8, 16, 32, 64];
const FAMILIES = ['schnet', 'mace', 'bioemu'];
const SERIES = ['baseline', 'flash'];
const FAMILY_LABELS = {schnet: 'CGSchNet', mace: 'MACE', bioemu: 'BioEmu'};
const SERIES_LABELS = {baseline: 'Baseline', flash: 'FlashMD'};
const INITIAL_CAMERA = {eye: {x: 1.48, y: -1.75, z: 0.76}, projection: {type: 'orthographic'}};
const plot = document.getElementById('plot');
const config = {responsive: true, displaylogo: false, scrollZoom: true, modeBarButtonsToRemove: ['toImage']};
let selectedBatch = 64;
let playTimer = null;
let syncingCamera = false;
let filteredRows = RAW_DATA.slice();
let currentPage = 1;
const pageSize = 50;

Plotly.newPlot(plot, FIGURE.data, FIGURE.layout, config).then(() => {
  updateResponsiveLegend();
  updateBatchStatus();
  installCameraSync();
  renderTable();
});

function updateResponsiveLegend() {
  const narrow = window.matchMedia('(max-width: 700px)').matches;
  Plotly.relayout(plot, {
    'legend.x': narrow ? 0.01 : 0.5,
    'legend.xanchor': narrow ? 'left' : 'center'
  });
}

window.addEventListener('resize', updateResponsiveLegend);

function updateHighlight(batch) {
  selectedBatch = batch;
  const updates = [];
  FAMILIES.forEach((family) => {
    SERIES.forEach((series) => {
      const index = TRACE_MAP.highlights[`${family}:${series}`];
      const payload = HIGHLIGHTS[String(batch)][family][series];
      updates.push(Plotly.restyle(plot, {
        x: [payload.x], y: [payload.y], z: [payload.z], customdata: [payload.customdata]
      }, [index]));
    });
  });
  Promise.all(updates).then(updateBatchStatus);
}

function updateBatchStatus() {
  const rows = RAW_DATA.filter((row) => Number(row.batch_size) === selectedBatch);
  const ok = rows.filter((row) => row.status === 'ok').length;
  const incomplete = rows.length - ok;
  document.getElementById('batch-status').textContent = `Batch ${selectedBatch}: ${ok} successful, ${incomplete} did not complete`;
}

document.getElementById('batch-slider').addEventListener('input', (event) => {
  updateHighlight(BATCHES[Number(event.target.value)]);
});

document.getElementById('play-batches').addEventListener('click', (event) => {
  if (playTimer !== null) {
    clearInterval(playTimer);
    playTimer = null;
    event.target.textContent = 'Play batches';
    return;
  }
  event.target.textContent = 'Stop';
  playTimer = setInterval(() => {
    const slider = document.getElementById('batch-slider');
    slider.value = (Number(slider.value) + 1) % BATCHES.length;
    updateHighlight(BATCHES[Number(slider.value)]);
  }, 950);
});

document.getElementById('focus-slice').addEventListener('change', (event) => {
  const focused = event.target.checked;
  Plotly.restyle(plot, {opacity: focused ? 0.10 : 0.42}, TRACE_MAP.surface_baseline);
  Plotly.restyle(plot, {opacity: focused ? 0.13 : 0.64}, TRACE_MAP.surface_flash);
  Plotly.restyle(plot, {opacity: focused ? 0.10 : 1.0}, TRACE_MAP.points);
});

function installCameraSync() {
  plot.on('plotly_relayout', (event) => {
    if (syncingCamera || !document.getElementById('sync-cameras').checked) return;
    const source = ['scene', 'scene2', 'scene3'].find((scene) => event[`${scene}.camera`]);
    if (!source) return;
    const camera = event[`${source}.camera`];
    const update = {};
    ['scene', 'scene2', 'scene3'].filter((scene) => scene !== source).forEach((scene) => {
      update[`${scene}.camera`] = camera;
    });
    syncingCamera = true;
    Plotly.relayout(plot, update).finally(() => { syncingCamera = false; });
  });
}

document.getElementById('reset-view').addEventListener('click', () => {
  Plotly.relayout(plot, {'scene.camera': INITIAL_CAMERA, 'scene2.camera': INITIAL_CAMERA, 'scene3.camera': INITIAL_CAMERA});
});

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (character) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[character]));
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || value === '') return '-';
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return number.toLocaleString(undefined, {maximumFractionDigits: digits});
}

function applyFilters() {
  const family = document.getElementById('family-filter').value;
  const series = document.getElementById('series-filter').value;
  const status = document.getElementById('status-filter').value;
  const batch = document.getElementById('batch-filter').value;
  const search = document.getElementById('search-filter').value.trim().toLowerCase();
  filteredRows = RAW_DATA.filter((row) => {
    if (family !== 'all' && row.family !== family) return false;
    if (series !== 'all' && row.series !== series) return false;
    if (status !== 'all' && row.status !== status) return false;
    if (batch !== 'all' && Number(row.batch_size) !== Number(batch)) return false;
    if (search) {
      const haystack = `${row.system ?? ''} ${row.source ?? ''} ${row.result_path ?? ''} ${row.embedding ?? ''}`.toLowerCase();
      if (!haystack.includes(search)) return false;
    }
    return true;
  });
  currentPage = 1;
  renderTable();
}

['family-filter', 'series-filter', 'status-filter', 'batch-filter'].forEach((id) => {
  document.getElementById(id).addEventListener('change', applyFilters);
});
document.getElementById('search-filter').addEventListener('input', applyFilters);

function renderTable() {
  const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  currentPage = Math.min(currentPage, totalPages);
  const start = (currentPage - 1) * pageSize;
  const pageRows = filteredRows.slice(start, start + pageSize);
  document.getElementById('data-body').innerHTML = pageRows.map((row) => {
    const details = escapeHtml(JSON.stringify(row, null, 2));
    const implementation = SERIES_LABELS[row.series] ?? row.series;
    return `<tr>
      <td>${escapeHtml(FAMILY_LABELS[row.family] ?? row.family)}</td>
      <td>${escapeHtml(implementation)}</td>
      <td>${escapeHtml(row.system)}</td>
      <td class="numeric">${formatNumber(row.size, 0)} ${escapeHtml(row.size_unit)}</td>
      <td class="numeric">${formatNumber(row.batch_size, 0)}</td>
      <td class="numeric">${formatNumber(row.throughput, 4)} ${escapeHtml(row.throughput_unit ?? '')}</td>
      <td class="numeric">${formatNumber(row.peak_allocated_gb, 2)} GB</td>
      <td class="numeric">${formatNumber(row.peak_reserved_gb, 2)} GB</td>
      <td class="status-${escapeHtml(row.status)}">${escapeHtml(row.status)}</td>
      <td>${escapeHtml(row.source)}</td>
      <td>${escapeHtml(row.result_path)}</td>
      <td><details class="row-details"><summary>Details</summary><pre>${details}</pre></details></td>
    </tr>`;
  }).join('');
  document.getElementById('row-count').textContent = `${filteredRows.length.toLocaleString()} rows`;
  document.getElementById('page-status').textContent = `Page ${currentPage} of ${totalPages}`;
  document.getElementById('prev-page').disabled = currentPage <= 1;
  document.getElementById('next-page').disabled = currentPage >= totalPages;
}

document.getElementById('prev-page').addEventListener('click', () => { if (currentPage > 1) { currentPage -= 1; renderTable(); } });
document.getElementById('next-page').addEventListener('click', () => { const max = Math.ceil(filteredRows.length / pageSize); if (currentPage < max) { currentPage += 1; renderTable(); } });

document.getElementById('export-csv').addEventListener('click', () => {
  const columns = Object.keys(RAW_DATA[0]);
  const quote = (value) => {
    if (value === null || value === undefined) return '';
    const text = String(value).replace(/"/g, '""');
    return /[",\n\r]/.test(text) ? `"${text}"` : text;
  };
  const csv = [columns.join(','), ...RAW_DATA.map((row) => columns.map((column) => quote(row[column])).join(','))].join('\n');
  const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'protein_batch_scaling.csv';
  link.click();
  URL.revokeObjectURL(url);
});
</script>
</body>
</html>
'''
    return (
        template.replace("__PLOTLY_JS__", plotly_js)
        .replace("__RAW_JSON__", raw_json)
        .replace("__FIGURE_JSON__", figure_json)
        .replace("__HIGHLIGHTS_JSON__", highlights_json)
        .replace("__TRACE_MAP_JSON__", trace_map_json)
    )


def main() -> None:
    rows = load_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_html(rows), encoding="utf-8")
    print(OUTPUT)
    print(f"{OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
