import numpy as np
import pandas as pd

from dtp_hotspots.analysis import (
    getis_ord_gi_star,
    hotspot_cells,
    project_local,
    spatial_grid,
    temporal_stats,
    unproject_local,
)


def clustered_df(n_cluster=120, n_noise=60, seed=1):
    """Synthetic frame: dense cluster at (47.10, 51.90) + uniform noise."""
    rng = np.random.default_rng(seed)
    lat = np.concatenate([
        rng.normal(47.10, 0.002, n_cluster),
        rng.uniform(47.0, 47.3, n_noise),
    ])
    lng = np.concatenate([
        rng.normal(51.90, 0.002, n_cluster),
        rng.uniform(51.7, 52.1, n_noise),
    ])
    n = n_cluster + n_noise
    return pd.DataFrame({
        "lat": lat, "lng": lng,
        "severity": ["minor"] * n,
        "datetime": pd.to_datetime(["2026-02-03 08:30:00"] * n),
        "dead": [0] * n, "injured": [0] * n,
    })


def test_projection_roundtrip():
    lat, lng = np.array([47.1, 47.2]), np.array([51.8, 52.0])
    x, y = project_local(lat, lng, 47.11, 51.88)
    lat2, lng2 = unproject_local(x, y, 47.11, 51.88)
    assert np.allclose(lat, lat2) and np.allclose(lng, lng2)


def test_gi_star_flags_hot_cell():
    grid = np.zeros((9, 9))
    grid[4, 4] = 30
    grid[4, 5] = 25
    z = getis_ord_gi_star(grid)
    assert z[4, 4] > 1.96
    assert z[0, 0] < 1.0


def test_gi_star_uniform_is_flat():
    z = getis_ord_gi_star(np.ones((5, 5)))
    assert np.allclose(z, 0.0)


def test_spatial_grid_finds_known_cluster():
    df = clustered_df()
    grid = spatial_grid(df, cell_size_m=250)
    hs = hotspot_cells(grid)
    assert len(hs) > 0
    top = hs.iloc[0]
    # Top hotspot lands within ~500 m of the planted cluster centre.
    assert abs(top["lat"] - 47.10) < 0.005
    assert abs(top["lng"] - 51.90) < 0.007


def test_temporal_stats_counts():
    df = clustered_df(n_cluster=3, n_noise=0)
    stats = temporal_stats(df)
    assert stats["by_hour"].sum() == 3
    assert stats["by_hour"][8] == 3  # all at 08:30
    assert stats["by_weekday"]["Tue"] == 3  # 2026-02-03 is a Tuesday
    assert stats["by_severity"]["minor"] == 3
    assert stats["casualties"]["dead"] == 0
