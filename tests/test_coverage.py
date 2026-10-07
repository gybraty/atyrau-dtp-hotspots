import numpy as np
import pandas as pd

from dtp_hotspots.analysis import camera_coverage

LAT_PER_M = 1 / 110_540.0


def test_camera_coverage_distances_and_flags():
    # Two accidents: one at the camera, one ~1000 m north of it.
    df = pd.DataFrame({
        "lat": [47.10, 47.10 + 1000 * LAT_PER_M],
        "lng": [51.90, 51.90],
    })
    cameras = pd.DataFrame({"id": [1], "type": ["overview"], "lat": [47.10], "lng": [51.90]})
    hotspots = pd.DataFrame({
        "lat": [47.10, 47.10 + 1000 * LAT_PER_M],
        "lng": [51.90, 51.90],
        "count": [5, 5],
        "z_score": [3.0, 3.0],
    })

    cov = camera_coverage(df, cameras, hotspots, radius_m=150.0)

    assert cov["accident_dist_m"][0] < 1.0
    assert abs(cov["accident_dist_m"][1] - 1000) < 20
    assert cov["share_covered"] == 0.5
    assert cov["hotspots"]["covered"].tolist() == [True, False]


def test_coverage_empty_hotspots():
    df = pd.DataFrame({"lat": [47.1], "lng": [51.9]})
    cameras = pd.DataFrame({"id": [1], "type": ["speed"], "lat": [47.1], "lng": [51.9]})
    hotspots = pd.DataFrame({"lat": [], "lng": [], "count": [], "z_score": []})
    cov = camera_coverage(df, cameras, hotspots)
    assert cov["share_covered"] == 1.0
    assert len(cov["hotspots"]) == 0
    assert np.isnan(cov["hotspots"]["camera_dist_m"]).sum() == 0
