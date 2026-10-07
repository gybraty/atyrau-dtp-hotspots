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
            "properties": {
                "count": int(row["count"]),
                "z_score": float(row["z_score"]),
                **({"camera_dist_m": float(row["camera_dist_m"]),
                    "covered": bool(row["covered"])} if "covered" in row else {}),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _cell_breakdown_html(cell_df: pd.DataFrame, row, has_coverage: bool) -> str:
    """Popup HTML: accidents inside one hotspot cell, by severity and type."""
    sev = cell_df["severity"].value_counts()
    sev_line = ", ".join(f"{k}: {v}" for k, v in sev.items())
    types = cell_df["type"].value_counts().head(4)
    type_rows = "".join(
        f"<tr><td style='padding-right:8px'>{t}</td><td align='right'>{c}</td></tr>"
        for t, c in types.items()
    )
    cam = ""
    if has_coverage:
        state = ("<b style='color:#b91c1c'>not covered</b>"
                 if not row["covered"] else "covered")
        cam = f"<div>Nearest camera: {int(row['camera_dist_m'])} m — {state}</div>"
    return (
        f"<div style='font: 12px sans-serif; min-width: 220px'>"
        f"<b>{len(cell_df)} accidents</b> · Gi* z = {row['z_score']}<br>"
        f"Severity: {sev_line}<br>"
        f"Dead: {int(cell_df['dead'].sum())}, injured: {int(cell_df['injured'].sum())}"
        f"{cam}"
        f"<table style='margin-top:4px'>{type_rows}</table>"
        f"</div>"
    )


def _legend_html(df: pd.DataFrame, hotspots: pd.DataFrame, has_coverage: bool,
                 share_covered: float | None, radius_m: float) -> str:
    uncovered = int((~hotspots["covered"]).sum()) if has_coverage else None
    stats = [
        (
            f"<b>{len(df)}</b> accidents, "
            f"{df['datetime'].min():%d.%m.%Y} – {df['datetime'].max():%d.%m.%Y}"
        ),
        f"dead {int(df['dead'].sum())}, injured {int(df['injured'].sum())}",
        f"<b>{len(hotspots)}</b> significant hotspot cells (Gi* z ≥ 1.96)",
    ]
    if share_covered is not None:
        stats.append(f"{share_covered:.0%} of accidents within {radius_m:.0f} m of a camera")
    if uncovered is not None:
        stats.append(f"<b style='color:#b91c1c'>{uncovered}</b> hotspot cells not covered")
    items = [
        (
            "<span style='display:inline-block;width:12px;height:12px;"
            "background:#ea580c;opacity:.55;border:1px solid #7c2d12'></span> "
            "hotspot cell (opacity ~ Gi* z)"
        ),
    ]
    if has_coverage:
        items.append(
            "<span style='display:inline-block;width:12px;height:12px;"
            "background:#ea580c;opacity:.55;border:2px dashed #b91c1c'></span> "
            "hotspot without camera nearby"
        )
        items.append(
            "<span style='display:inline-block;width:12px;height:12px;"
            "border-radius:50%;background:#6baed6;border:1px solid #3182bd'></span> "
            "cameras (clustered)"
        )
    return (
        "<div style=\"position:fixed; bottom:24px; left:12px; z-index:9999;"
        " background:rgba(255,255,255,.95); color:#374151; padding:10px 12px;"
        " border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,.3);"
        " font:12px/1.6 sans-serif; max-width:300px\">"
        "<b>Accident hotspot analysis</b><br>"
        + "<br>".join(stats)
        + "<hr style='margin:6px 0;border:none;border-top:1px solid #e5e7eb'>"
        + "<br>".join(items)
        + "<br><i>Click a hotspot cell for its breakdown</i>"
        "</div>"
    )


def render_map(df: pd.DataFrame, hotspots: pd.DataFrame, grid: GridResult,
               out_path: str | Path, cameras: pd.DataFrame | None = None,
               share_covered: float | None = None, radius_m: float = 150.0) -> Path:
    """Folium map: accident heat layer + significant hotspot cells (+ camera layer)."""
    # Esri World Street Map: keyless, and reachable from networks where
    # tile.openstreetmap.org is blocked.
    m = folium.Map(
        location=[grid.lat0, grid.lng0],
        zoom_start=12,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map"
              "/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri, HERE, Garmin, OpenStreetMap contributors",
    )

    HeatMap(
        df[["lat", "lng"]].values.tolist(), radius=14, blur=18, name="Accident density",
    ).add_to(m)

    fg = folium.FeatureGroup(name="Hotspots (Gi* z ≥ 1.96)")
    z_max = float(hotspots["z_score"].max()) if len(hotspots) else 1.0
    has_coverage = "covered" in hotspots.columns
    half_lat = grid.cell_size_m / 2 / 110_540.0
    import numpy as np

    half_lng = grid.cell_size_m / 2 / (111_320.0 * np.cos(np.radians(grid.lat0)))
    for _, row in hotspots.iterrows():
        ring = _cell_polygon(row["lat"], row["lng"], grid)
        uncovered = has_coverage and not row["covered"]
        tooltip = f"accidents: {int(row['count'])}, Gi* z = {row['z_score']}"
        if has_coverage:
            tooltip += (f", nearest camera {int(row['camera_dist_m'])} m"
                        + (" — NOT COVERED" if uncovered else ""))
        cell_df = df[
            df["lat"].between(row["lat"] - half_lat, row["lat"] + half_lat)
            & df["lng"].between(row["lng"] - half_lng, row["lng"] + half_lng)
        ]
        folium.Polygon(
            locations=[(p[1], p[0]) for p in ring],
            color="#b91c1c" if uncovered else "#7c2d12",
            weight=2.5 if uncovered else 1,
            dash_array="4" if uncovered else None,
            fill=True,
            fill_color="#ea580c",
            fill_opacity=0.25 + 0.5 * float(row["z_score"]) / z_max,
            tooltip=tooltip,
            popup=folium.Popup(_cell_breakdown_html(cell_df, row, has_coverage),
                               max_width=320) if len(cell_df) else None,
        ).add_to(fg)
    fg.add_to(m)

    m.get_root().html.add_child(folium.Element(
        _legend_html(df, hotspots, has_coverage, share_covered, radius_m)
    ))

    if cameras is not None and len(cameras):
        from folium.plugins import FastMarkerCluster

        FastMarkerCluster(
            cameras[["lat", "lng"]].values.tolist(), name="Cameras",
        ).add_to(m)

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
