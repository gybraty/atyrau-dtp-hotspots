"""Spatial (KDE, Getis-Ord Gi*) and temporal analysis of accident points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from scipy.stats import gaussian_kde

EARTH_M_PER_DEG_LAT = 110_540.0
EARTH_M_PER_DEG_LNG_EQ = 111_320.0


def project_local(lat: np.ndarray, lng: np.ndarray, lat0: float, lng0: float):
    """Project lat/lng to local meters around (lat0, lng0).

    ponytail: equirectangular approximation — error is negligible at city scale
    (<0.1% across Atyrau); switch to pyproj UTM 42N if region-wide precision matters.
    """
    x = (np.asarray(lng) - lng0) * EARTH_M_PER_DEG_LNG_EQ * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * EARTH_M_PER_DEG_LAT
    return x, y


def unproject_local(x: np.ndarray, y: np.ndarray, lat0: float, lng0: float):
    lat = np.asarray(y) / EARTH_M_PER_DEG_LAT + lat0
    lng = np.asarray(x) / (EARTH_M_PER_DEG_LNG_EQ * np.cos(np.radians(lat0))) + lng0
    return lat, lng


@dataclass
class GridResult:
    """Cell-based spatial analysis result. Arrays are shape (ny, nx)."""

    counts: np.ndarray
    density: np.ndarray  # KDE, normalized to [0, 1]
    z_scores: np.ndarray  # Getis-Ord Gi*
    cell_lat: np.ndarray  # cell-center latitudes, shape (ny, nx)
    cell_lng: np.ndarray
    cell_size_m: float
    lat0: float
    lng0: float


def spatial_grid(df: pd.DataFrame, cell_size_m: float = 250.0, pad_cells: int = 2) -> GridResult:
    """Bin points into a square grid, compute KDE density and Gi* z-score per cell."""
    lat0, lng0 = float(df["lat"].mean()), float(df["lng"].mean())
    x, y = project_local(df["lat"].to_numpy(), df["lng"].to_numpy(), lat0, lng0)

    pad = pad_cells * cell_size_m
    x_edges = np.arange(x.min() - pad, x.max() + pad + cell_size_m, cell_size_m)
    y_edges = np.arange(y.min() - pad, y.max() + pad + cell_size_m, cell_size_m)

    counts, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])

    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(xc, yc)

    kde = gaussian_kde(np.vstack([x, y]))
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(counts.shape)
    if density.max() > 0:
        density = density / density.max()

    z = getis_ord_gi_star(counts)

    cell_lat, cell_lng = unproject_local(xx, yy, lat0, lng0)
    return GridResult(counts, density, z, cell_lat, cell_lng, cell_size_m, lat0, lng0)


def getis_ord_gi_star(values: np.ndarray) -> np.ndarray:
    """Getis-Ord Gi* z-scores with binary 3x3 queen-contiguity weights (incl. self).

    Standard formulation (Getis & Ord 1992); edge cells use their actual
    neighbor counts via boundary-aware convolution.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    mean = v.mean()
    s = np.sqrt((v**2).mean() - mean**2)
    if s == 0 or n < 2:
        return np.zeros_like(v)

    kernel = np.ones((3, 3))
    w_sum = convolve2d(np.ones_like(v), kernel, mode="same")  # neighbors per cell
    local_sum = convolve2d(v, kernel, mode="same")

    num = local_sum - mean * w_sum
    den = s * np.sqrt((n * w_sum - w_sum**2) / (n - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(den > 0, num / den, 0.0)
    return z


def hotspot_cells(grid: GridResult, z_threshold: float = 1.96) -> pd.DataFrame:
    """Cells with Gi* z >= threshold (95% confidence by default), sorted by z."""
    mask = (grid.z_scores >= z_threshold) & (grid.counts > 0)
    iy, ix = np.where(mask)
    out = pd.DataFrame({
        "lat": grid.cell_lat[iy, ix],
        "lng": grid.cell_lng[iy, ix],
        "count": grid.counts[iy, ix].astype(int),
        "z_score": grid.z_scores[iy, ix].round(2),
    })
    return out.sort_values("z_score", ascending=False).reset_index(drop=True)


def temporal_stats(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Counts by hour, weekday, month; severity breakdown; casualty totals."""
    dt = df["datetime"]
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {
        "by_hour": dt.dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index(),
        "by_weekday": (
            dt.dt.day_name().str[:3].value_counts().reindex(weekday_order, fill_value=0)
        ),
        "by_month": dt.dt.to_period("M").astype(str).value_counts().sort_index(),
        "by_severity": df["severity"].value_counts().reindex(
            ["minor", "major", "fatal"], fill_value=0
        ),
        "casualties": pd.Series(
            {"dead": int(df["dead"].sum()), "injured": int(df["injured"].sum())}
        ),
    }
