import csv
from dataclasses import dataclass
from pathlib import Path

from models import SheetPileLine, GeotechPoint
from utils import parse_float

def read_geotech_csv(csv_path: Path
                     ) -> tuple[dict[int, list[GeotechPoint]],
                                dict[int, SheetPileLine]]:
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