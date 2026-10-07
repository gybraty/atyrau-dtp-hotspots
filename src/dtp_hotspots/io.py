"""Loading and validation of accident CSV data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["id", "type", "severity", "lat", "lng", "datetime", "dead", "injured"]

# Atyrau region bounding box (approximate).
LAT_MIN, LAT_MAX = 46.0, 49.5
LNG_MIN, LNG_MAX = 48.5, 54.5

SEVERITIES = {"minor", "major", "fatal"}


class ValidationReport:
    """Counts of rows dropped per reason during validation."""

    def __init__(self) -> None:
        self.dropped: dict[str, int] = {}

    def add(self, reason: str, count: int) -> None:
        if count:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(count)

    @property
    def total_dropped(self) -> int:
        return sum(self.dropped.values())

    def __repr__(self) -> str:  # pragma: no cover
        return f"ValidationReport({self.dropped})"


def load_crashes(path: str | Path) -> tuple[pd.DataFrame, ValidationReport]:
    """Load an accident CSV, drop invalid rows, return (clean dataframe, report).

    Raises ValueError if required columns are missing.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    report = ValidationReport()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df["dead"] = pd.to_numeric(df["dead"], errors="coerce")
    df["injured"] = pd.to_numeric(df["injured"], errors="coerce")

    checks = [
        ("invalid datetime", df["datetime"].isna()),
        ("coordinates outside Atyrau region bbox",
         ~(df["lat"].between(LAT_MIN, LAT_MAX) & df["lng"].between(LNG_MIN, LNG_MAX))),
        ("unknown severity", ~df["severity"].isin(SEVERITIES)),
        ("negative or missing casualty counts",
         df["dead"].isna() | df["injured"].isna() | (df["dead"] < 0) | (df["injured"] < 0)),
    ]

    bad = pd.Series(False, index=df.index)
    for reason, mask in checks:
        # Count each row once, under the first failed check.
        new_bad = mask & ~bad
        report.add(reason, new_bad.sum())
        bad |= mask

    clean = df[~bad].reset_index(drop=True)
    clean["dead"] = clean["dead"].astype(int)
    clean["injured"] = clean["injured"].astype(int)
    return clean, report
