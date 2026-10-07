import pandas as pd
import pytest

from dtp_hotspots.io import load_crashes


def write_csv(tmp_path, rows):
    df = pd.DataFrame(rows)
    p = tmp_path / "in.csv"
    df.to_csv(p, index=False)
    return p


GOOD = {
    "id": 1, "type": "столкновение", "severity": "minor",
    "lat": 47.11, "lng": 51.88, "datetime": "2026-03-01 12:00:00",
    "dead": 0, "injured": 1,
}


def test_valid_row_passes(tmp_path):
    df, report = load_crashes(write_csv(tmp_path, [GOOD]))
    assert len(df) == 1
    assert report.total_dropped == 0
    assert df.loc[0, "dead"] == 0


def test_missing_column_raises(tmp_path):
    row = {k: v for k, v in GOOD.items() if k != "severity"}
    with pytest.raises(ValueError, match="severity"):
        load_crashes(write_csv(tmp_path, [row]))


@pytest.mark.parametrize("patch,reason", [
    ({"datetime": "not-a-date"}, "invalid datetime"),
    ({"lat": 55.0}, "coordinates outside Atyrau region bbox"),
    ({"lng": 10.0}, "coordinates outside Atyrau region bbox"),
    ({"severity": "huge"}, "unknown severity"),
    ({"dead": -1}, "negative or missing casualty counts"),
])
def test_bad_rows_dropped_with_reason(tmp_path, patch, reason):
    bad = {**GOOD, **patch, "id": 2}
    df, report = load_crashes(write_csv(tmp_path, [GOOD, bad]))
    assert len(df) == 1
    assert report.dropped == {reason: 1}
