"""Chart and map rendering."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import folium
import matplotlib.pyplot as plt
import pandas as pd
from folium.plugins import HeatMap

from .analysis import GridResult

PRIMARY = "#4269d0"
# Sequential single-hue ramp for ordinal severity (light -> dark = minor -> fatal).
SEVERITY_COLORS = {"minor": "#fb923c", "major": "#ea580c", "fatal": "#7c2d12"}
INK = "#374151"
GRID_COLOR = "#e5e7eb"


def _bar(ax, series: pd.Series, color, title: str, direct_labels: bool = False) -> None:
    ax.bar(series.index.astype(str), series.values, color=color, width=0.72)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    if direct_labels:
        for i, v in enumerate(series.values):
            ax.text(i, v, f" {int(v)}", ha="center", va="bottom", fontsize=9, color=INK)


def render_charts(stats: dict[str, pd.Series], out_dir: str | Path) -> list[Path]:
    """Write temporal/severity PNG charts, return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    specs = [
        ("by_hour", "Accidents by hour of day", PRIMARY, False),
        ("by_weekday", "Accidents by weekday", PRIMARY, True),
        ("by_month", "Accidents by month", PRIMARY, True),
    ]
    for key, title, color, labels in specs:
        fig, ax = plt.subplots(figsize=(8, 3.5), dpi=150)
        _bar(ax, stats[key], color, title, direct_labels=labels)
        if key == "by_month":
            ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        p = out / f"{key}.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    sev = stats["by_severity"]
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    _bar(ax, sev, [SEVERITY_COLORS[s] for s in sev.index], "Accidents by severity",
         direct_labels=True)
    fig.tight_layout()
    p = out / "by_severity.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)
    return paths


def _cell_polygon(lat: float, lng: float, grid: GridResult) -> list[list[float]]:
    """Cell corner ring [lng, lat] for GeoJSON, from center + cell size."""
    import numpy as np

    half = grid.cell_size_m / 2
    dlat = half / 110_540.0
    dlng = half / (111_320.0 * np.cos(np.radians(grid.lat0)))
    return [
        [lng - dlng, lat - dlat], [lng + dlng, lat - dlat],
        [lng + dlng, lat + dlat], [lng - dlng, lat + dlat],
        [lng - dlng, lat - dlat],
    ]


def hotspots_geojson(hotspots: pd.DataFrame, grid: GridResult) -> dict:
    features = []
    for _, row in hotspots.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [_cell_polygon(row["lat"], row["lng"], grid)],
            },
            "properties": {"count": int(row["count"]), "z_score": float(row["z_score"])},
        })
    return {"type": "FeatureCollection", "features": features}


def render_map(df: pd.DataFrame, hotspots: pd.DataFrame, grid: GridResult,
               out_path: str | Path) -> Path:
    """Folium map: accident heat layer + statistically significant hotspot cells."""
    m = folium.Map(location=[grid.lat0, grid.lng0], zoom_start=12, tiles="OpenStreetMap")

    HeatMap(
        df[["lat", "lng"]].values.tolist(), radius=14, blur=18, name="Accident density",
    ).add_to(m)

    fg = folium.FeatureGroup(name="Hotspots (Gi* z ≥ 1.96)")
    z_max = float(hotspots["z_score"].max()) if len(hotspots) else 1.0
    for _, row in hotspots.iterrows():
        ring = _cell_polygon(row["lat"], row["lng"], grid)
        folium.Polygon(
            locations=[(p[1], p[0]) for p in ring],
            color="#7c2d12",
            weight=1,
            fill=True,
            fill_color="#ea580c",
            fill_opacity=0.25 + 0.5 * float(row["z_score"]) / z_max,
            tooltip=f"accidents: {int(row['count'])}, Gi* z = {row['z_score']}",
        ).add_to(fg)
    fg.add_to(m)

    folium.LayerControl().add_to(m)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    return out_path


def write_geojson(geojson: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(geojson, ensure_ascii=False, indent=2))
    return out_path
