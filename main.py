"""
refra__hybrid/
│
├── __init__.py
├── main.py
├── config.py
├── io_helpers.py
├── elevation.py
├── geotech.py
├── analysis_itm.py
├── analysis_plusminus.py
├── plotting.py
└── results.py
```

### File Descriptions

1. **`__init__.py`**: This file can be empty or can contain package-level documentation.

2. **`main.py`**: This file contains the main execution logic of the script, including the main function and the overall workflow.

3. **`config.py`**: This file contains all the configuration constants and parameters used throughout the code.

4. **`io_helpers.py`**: This module handles all input/output operations, such as reading CSV files and writing results to Excel or PDF.

5. **`elevation.py`**: This module contains functions related to elevation data processing, including reading elevation data and interpolating elevations.

6. **`geotech.py`**: This module handles geotechnical data processing, including reading geotechnical data and managing geotech points.

7. **`analysis_itm.py`**: This module contains functions related to seismic ITM analysis, including shot analysis and breakpoint detection.

8. **`analysis_plusminus.py`**: (not yet implemented) This module will implement the Plus-Minus (Reciprocal) Method for deriving 2D bedrock topography from seismic refraction arrivals.

9. **`plotting.py`**: This module handles all plotting functions, including travel-time plots and elevation profiles.

10. **`results.py`**: This module handles the collection and formatting of results, including saving results to Excel and PDF.
"""
 
import sys
from pathlib import Path
import config
from io_helpers import read_elev_csv, read_geotech_csv, read_vs_file, group_by_transect
from analysis_itm import analyse_shot_itm
from analysis_plusminus import analyse_transect_pm, collect_pm_rock_points
from plotting import save_traveltime_pdf, save_elevation_pdf, save_pm_traveltime_pdf
from results import save_excel, save_pm_excel
from config import DATA_DIR

def main() -> int:
    script_dir = Path(__file__).parent
    work_dir = DATA_DIR if DATA_DIR.is_dir() else script_dir

    vs_files = sorted(work_dir.glob('*.vs'))
    if not vs_files:
        print("No .vs files found.")
        return 1

    elev_data = read_elev_csv(script_dir / config.ELEV_FILE_NAME)
    geotech_by_tid, sheetpile_by_tid = read_geotech_csv(script_dir / config.GEOTECH_FILE_NAME)

    all_records = []
    transects = group_by_transect(vs_files)

    for tid, files in transects.items():
        for path in sorted(files):
            try:
                shot_pos, distances, times_ms = read_vs_file(path)
                res = analyse_shot_itm(distances, times_ms, shot_pos=shot_pos)
                all_records.append({
                    'transect_id': tid,
                    'file_name': path.name,
                    'geophone_locs': distances,
                    'times_ms': times_ms,
                    'res': res,
                })
            except Exception as exc:
                print(f"Error processing {path.name}: {exc}")

    if not all_records:
        print("No files processed successfully.")
        return 1

    # ── Plus-Minus analysis (per transect) ─────────────────────────
    print("\n" + "=" * 60)
    print("  PLUS-MINUS (HAGEDOORN) ANALYSIS")
    print("=" * 60)

    pm_results: dict[int, list] = {}
    for tid, files in transects.items():
        try:
            pairs = analyse_transect_pm(files, tid)
            if pairs:
                pm_results[tid] = pairs
        except Exception as exc:
            print(f"  ⚠  PM error on transect {tid}: {exc}")

    pm_rock_by_tid = collect_pm_rock_points(pm_results, elev_data)

    # ── Output ─────────────────────────────────────────────────────
    save_traveltime_pdf(all_records, script_dir / "refra_itm_traveltimes.pdf")
    save_excel(all_records, script_dir / "refra_itm_results.xlsx")
    save_elevation_pdf(all_records, elev_data, script_dir / "sections.pdf",
                       geotech_by_tid, sheetpile_by_tid,
                       pm_rock_by_tid=pm_rock_by_tid)

    if pm_results:
        save_pm_traveltime_pdf(pm_results,
                               script_dir / "refra_pm_traveltimes.pdf")
        save_pm_excel(pm_results, elev_data,
                      script_dir / "refra_pm_results.xlsx")

    return 0

if __name__ == "__main__":
    sys.exit(main())