"""Generate a synthetic sample accident dataset (no real data).

Points cluster around three fictitious hotspots in Atyrau city plus regional noise,
mimicking the schema of the real police export. Deterministic (fixed seed).

Usage:
    python scripts/make_sample.py > data/sample.csv
    python scripts/make_sample.py cameras > data/sample_cameras.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

CLUSTERS = [  # (lat, lng, sigma_deg, n) — city-centre, bridge, highway junction
    (47.106, 51.882, 0.004, 150),
    (47.117, 51.916, 0.005, 110),
    (47.093, 51.845, 0.006, 80),
]
NOISE_N = 160  # scattered across the city and region

TYPES = ["столкновение", "наезд на пешехода", "опрокидывание", "наезд на препятствие"]


def make_cameras() -> None:
    """Synthetic cameras: dense around the first two clusters, none at the third,
    so the demo shows both covered and uncovered hotspots."""
    rows = []
    for lat0, lng0, sigma, _ in CLUSTERS[:2]:
        n = 25
        rows.append(pd.DataFrame({
            "lat": RNG.normal(lat0, sigma * 2, n).round(6),
            "lng": RNG.normal(lng0, sigma * 2.8, n).round(6),
        }))
    rows.append(pd.DataFrame({
        "lat": RNG.uniform(46.95, 47.3, 30).round(6),
        "lng": RNG.uniform(51.65, 52.15, 30).round(6),
    }))
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "id", np.arange(1, len(df) + 1))
    df.insert(1, "type", RNG.choice(["overview", "intersection", "speed"], len(df),
                                    p=[0.5, 0.35, 0.15]))
    print(df.to_csv(index=False), end="")


def main() -> None:
    rows = []
    for lat0, lng0, sigma, n in CLUSTERS:
        lats = RNG.normal(lat0, sigma, n)
        lngs = RNG.normal(lng0, sigma * 1.4, n)
        rows.append(pd.DataFrame({"lat": lats, "lng": lngs}))
    rows.append(pd.DataFrame({
        "lat": RNG.uniform(46.9, 47.35, NOISE_N),
        "lng": RNG.uniform(51.6, 52.2, NOISE_N),
    }))
    df = pd.concat(rows, ignore_index=True)
    n = len(df)

    df.insert(0, "id", np.arange(1, n + 1))
    df["type"] = RNG.choice(TYPES, n, p=[0.45, 0.3, 0.1, 0.15])
    severity = RNG.choice(["minor", "major", "fatal"], n, p=[0.62, 0.31, 0.07])
    df["severity"] = severity
    # Rush-hour-weighted times across Jan–Jul 2026.
    days = RNG.integers(0, 211, n)
    hours = RNG.choice(24, n, p=_hour_weights())
    minutes = RNG.integers(0, 60, n)
    df["datetime"] = (
        pd.Timestamp("2026-01-01")
        + pd.to_timedelta(days, "D") + pd.to_timedelta(hours, "h")
        + pd.to_timedelta(minutes, "m")
    ).strftime("%Y-%m-%d %H:%M:%S")
    df["dead"] = np.where(severity == "fatal", RNG.integers(1, 3, n), 0)
    df["injured"] = np.where(severity == "minor", 0, RNG.integers(0, 4, n))
    df["lat"] = df["lat"].round(6)
    df["lng"] = df["lng"].round(6)

    print(df.to_csv(index=False), end="")


def _hour_weights() -> np.ndarray:
    w = np.ones(24)
    w[[8, 9, 17, 18, 19]] = 3.0
    w[[0, 1, 2, 3, 4, 5]] = 0.3
    return w / w.sum()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cameras":
        make_cameras()
    else:
        main()
