from dataclasses import dataclass
import re
import csv
from pathlib import Path
from typing import Optional
import numpy as np

from models import GeotechPoint, SheetPileLine
from utils import parse_float

# ===========================================================================
# Elevation data  (read from elev.csv)
# ===========================================================================

def read_elev_csv(csv_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Parse elev.csv into per-transect (distances, elevations) arrays.

    Relevant columns
    ----------------
    line_no         : transect id  (e.g. '0220' or '220')
    dist_first_gp_m : raw distance along transect (m)
    geop_id         : integer geophone id; the row where geop_id == 0
                      defines x = 0 on the plot axis.  All other x values
                      are shifted by the same offset so that the geop_id==0
                      row lands exactly at x = 0.
    z               : elevation (m ASL)

    Rows with x < ELEV_X_MIN or x > ELEV_X_MAX are kept so that
    np.interp can extrapolate (clamp) the surface elevation at the
    plot boundaries when needed.

    Returns
    -------
    dict  transect_id → (dist_array, elev_array)  sorted by x
    """
    # raw storage: tid → list of (raw_dist, geop_id_or_None, z)
    raw: dict[int, list[tuple[float, Optional[float], float]]] = {}

    if not csv_path.exists():
        print(f"  ⚠  elev.csv not found at {csv_path} – no elevation data.")
        return {}

    with open(csv_path, newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_line = row.get('line_no', '').strip()
            if not raw_line:
                continue
            try:
                tid = int(raw_line)
            except ValueError:
                continue

            dist = parse_float(row.get('dist_first_gp_m', ''))
            z    = parse_float(row.get('z', ''))
            if dist is None or z is None:
                continue

            geop_id = parse_float(row.get('geop_id', ''))
            raw.setdefault(tid, []).append((dist, geop_id, z))

    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for tid, pts in raw.items():
        # Find the raw dist value for the row where geop_id == 0
        ref_dist: Optional[float] = None
        for dist, geop_id, _ in pts:
            if geop_id is not None and geop_id == 0.0:
                ref_dist = dist
                break

        if ref_dist is None:
            print(f"  ⚠  Transect {tid}: no geop_id==0 row found in "
                  f"elev.csv – using raw distances unchanged.")
            ref_dist = 0.0

        # Apply shift so that geop_id==0 lands at x=0
        shifted = sorted(
            [(dist - ref_dist, z) for dist, _, z in pts],
            key=lambda p: p[0],
        )

        result[tid] = (
            np.array([p[0] for p in shifted]),
            np.array([p[1] for p in shifted]),
        )

        x_min_data = result[tid][0].min()
        x_max_data = result[tid][0].max()
        print(f"  Transect {tid}: {len(shifted)} elevation points, "
              f"x = {x_min_data:.1f} … {x_max_data:.1f} m "
              f"(ref shift = {-ref_dist:+.2f} m)")

    print(f"  elev.csv: elevation data loaded for "
          f"{len(result)} transect(s).")
    return result


# ===========================================================================
# Geotech data
# ===========================================================================

def read_geotech_csv(csv_path: Path) -> tuple[dict[int, list[GeotechPoint]], dict[int, SheetPileLine]]:
    """
    Parse geotech.csv.

    Returns
    -------
    geotech_by_tid   : transect_id → list[GeotechPoint]
    sheetpile_by_tid : transect_id → SheetPileLine
    """
    geotech:   dict[int, list[GeotechPoint]] = {}
    sheetpile: dict[int, SheetPileLine]      = {}

    if not csv_path.exists():
        print(f"  ⚠  geotech.csv not found at {csv_path} – skipping.")
        return geotech, sheetpile

    with open(csv_path, newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_line = row.get('line_no', '').strip()
            if not raw_line:
                continue
            try:
                tid = int(raw_line)
            except ValueError:
                continue

            dist_sheet = parse_float(row.get('dist_sheet_pile_m', ''))
            dist_x     = parse_float(row.get('dist_first_gp_m', ''))
            if dist_x is None:
                continue

            # Sheet-pile reference (dist_sheet_pile_m == 0)
            if dist_sheet is not None and dist_sheet == 0.0:
                if tid not in sheetpile:
                    sheetpile[tid] = SheetPileLine(line_no=tid, dist_x=dist_x)

            # Actual test row
            test_type = row.get('test_type', '').strip()
            if not test_type:
                continue

            tested = parse_float(row.get('tested_depth_m', ''))
            rock   = parse_float(row.get('depth_of_rock_m', ''))
            if tested is None:
                continue

            geotech.setdefault(tid, []).append(GeotechPoint(
                test_type     = test_type,
                line_no       = tid,
                dist_x        = dist_x,
                tested_depth  = tested,
                depth_of_rock = rock,
            ))

    print(f"  Geotech CSV: {sum(len(v) for v in geotech.values())} tests "
          f"across {len(geotech)} transect(s); "
          f"{len(sheetpile)} sheet-pile position(s) loaded.")
    return geotech, sheetpile


# ===========================================================================
# Pick data from .vs files
# ===========================================================================

def read_vs_file(file_path: Path) -> tuple[Optional[float], np.ndarray, np.ndarray]:
    """
    Parse one .vs pick file.

    Lines
    -----
    1-2  : header (ignored)
    3    : shot position in column 0
    4-27 : geophone_distance  travel_time_ms  [weight]
    """
    shot_pos:  Optional[float] = None
    distances: list[float] = []
    times_ms:  list[float] = []

    with open(file_path, 'r') as fh:
        for line_num, line in enumerate(fh, start=1):
            row = line.strip().split()
            if line_num < 3:
                continue
            if line_num == 3:
                shot_pos = float(row[0])
                continue
            if line_num > 27:
                break
            if len(row) >= 2:
                try:
                    distances.append(float(row[0]))
                    times_ms.append(float(row[1]))
                except ValueError:
                    print(f"  Warning: non-numeric data on line {line_num} "
                          f"of {file_path.name} – skipped.")

    return shot_pos, np.array(distances), np.array(times_ms)

def group_by_transect(vs_files: list[Path]) -> dict[int, list[Path]]:
    """Group .vs files by transect id from filenames ``<tid>-<sid>.vs``."""
    pattern = re.compile(r'^(\d+)-(\d+)\.vs$', re.IGNORECASE)
    groups: dict[int, list[Path]] = {}
    for path in vs_files:
        m = pattern.match(path.name)
        if not m:
            print(f"  Skipping unrecognised filename: {path.name}")
            continue
        groups.setdefault(int(m.group(1)), []).append(path)
    return dict(sorted(groups.items()))