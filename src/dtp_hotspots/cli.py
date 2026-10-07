"""Command-line interface: dtp-hotspots report <csv> -o <dir>."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import analysis, io, viz


def write_summary(df, report, stats, hotspots, out_path: Path,
                  coverage=None, radius_m: float = 150.0) -> None:
    peak_hour = int(stats["by_hour"].idxmax())
    peak_day = str(stats["by_weekday"].idxmax())
    lines = [
        "# Accident hotspot report",
        "",
        (
            f"- Valid records analysed: **{len(df)}** "
            f"(dropped {report.total_dropped}: "
            f"{', '.join(f'{k}: {v}' for k, v in report.dropped.items()) or 'none'})"
        ),
        f"- Period: {df['datetime'].min():%Y-%m-%d} — {df['datetime'].max():%Y-%m-%d}",
        (
            f"- Casualties: {stats['casualties']['dead']} dead, "
            f"{stats['casualties']['injured']} injured"
        ),
        "- Severity: " + ", ".join(f"{k}: {v}" for k, v in stats["by_severity"].items()),
        f"- Peak hour: {peak_hour}:00, peak weekday: {peak_day}",
        f"- Statistically significant hotspot cells (Gi* z ≥ 1.96): **{len(hotspots)}**",
    ]
    if coverage is not None:
        uncovered = coverage["hotspots"][~coverage["hotspots"]["covered"]]
        lines += [
            "",
            "## Camera coverage",
            "",
            (
                f"- Accidents within {radius_m:.0f} m of a camera: "
                f"**{coverage['share_covered']:.1%}**"
            ),
            (
                "- Median distance to nearest camera: "
                f"{coverage['accident_dist_m'].median():.0f} m"
            ),
            (
                f"- Hotspot cells without a camera within {radius_m:.0f} m: "
                f"**{len(uncovered)}** of {len(coverage['hotspots'])}"
            ),
        ]
    lines += [
        "",
        "## Top hotspots",
        "",
        "| # | lat | lng | accidents | Gi* z |",
        "|---|-----|-----|-----------|-------|",
    ]
    for i, row in hotspots.head(10).iterrows():
        lines.append(
            f"| {i + 1} | {row['lat']:.5f} | {row['lng']:.5f} "
            f"| {int(row['count'])} | {row['z_score']} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dtp-hotspots",
                                     description="Road accident hotspot analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="full analysis report from an accident CSV")
    rep.add_argument("csv", help="input CSV (id,type,severity,lat,lng,datetime,dead,injured)")
    rep.add_argument("-o", "--out", default="out", help="output directory (default: out)")
    rep.add_argument("--cell", type=float, default=250.0, help="grid cell size, meters")
    rep.add_argument("--z", type=float, default=1.96, help="Gi* z-score threshold")
    rep.add_argument("--cameras", help="optional camera CSV (id,type,lat,lng)")
    rep.add_argument("--radius", type=float, default=150.0,
                     help="camera coverage radius, meters (default: 150)")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, report = io.load_crashes(args.csv)
    if df.empty:
        parser.error("no valid rows in input")
    print(f"loaded {len(df)} valid rows ({report.total_dropped} dropped)")

    grid = analysis.spatial_grid(df, cell_size_m=args.cell)
    hotspots = analysis.hotspot_cells(grid, z_threshold=args.z)
    stats = analysis.temporal_stats(df)

    cameras, coverage = None, None
    if args.cameras:
        cameras, cam_report = io.load_cameras(args.cameras)
        print(f"loaded {len(cameras)} cameras ({cam_report.total_dropped} dropped)")
        coverage = analysis.camera_coverage(df, cameras, hotspots, radius_m=args.radius)
        hotspots = coverage["hotspots"]

    viz.render_charts(stats, out / "charts")
    viz.render_map(df, hotspots, grid, out / "map.html", cameras=cameras,
                   share_covered=coverage["share_covered"] if coverage else None,
                   radius_m=args.radius)
    viz.write_geojson(viz.hotspots_geojson(hotspots, grid), out / "hotspots.geojson")
    write_summary(df, report, stats, hotspots, out / "summary.md",
                  coverage=coverage, radius_m=args.radius)

    print(f"report written to {out}/ ({len(hotspots)} hotspot cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
