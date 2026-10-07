import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_report_end_to_end(tmp_path):
    sample = tmp_path / "sample.csv"
    gen = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_sample.py")],
        capture_output=True, text=True, check=True,
    )
    sample.write_text(gen.stdout)

    out = tmp_path / "out"
    res = subprocess.run(
        [sys.executable, "-m", "dtp_hotspots.cli", "report", str(sample), "-o", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr

    assert (out / "map.html").exists()
    assert (out / "summary.md").exists()
    for chart in ("by_hour", "by_weekday", "by_month", "by_severity"):
        assert (out / "charts" / f"{chart}.png").exists()

    geo = json.loads((out / "hotspots.geojson").read_text())
    assert geo["type"] == "FeatureCollection"
    assert len(geo["features"]) > 0  # sample has planted clusters
    props = geo["features"][0]["properties"]
    assert props["z_score"] >= 1.96
