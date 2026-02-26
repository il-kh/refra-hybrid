"""
refra__hybrid/
│
├── __init__.py
├── main.py
├── config.py
├── io_helpers.py
├── elevation.py
├── geotech.py
├── analysis.py
├── plotting.py
└── results.py
```

### File Descriptions

1. **`__init__.py`**: This file can be empty or can contain package-level documentation.

2. **`main.py`**: This file will contain the main execution logic of the script, including the main function and the overall workflow.

3. **`config.py`**: This file will contain all the configuration constants and parameters used throughout the code.

4. **`io_helpers.py`**: This module will handle all input/output operations, such as reading CSV files and writing results to Excel or PDF.

5. **`elevation.py`**: This module will contain functions related to elevation data processing, including reading elevation data and interpolating elevations.

6. **`geotech.py`**: This module will handle geotechnical data processing, including reading geotechnical data and managing geotech points.

7. **`analysis.py`**: This module will contain functions related to seismic analysis, including shot analysis and breakpoint detection.

8. **`plotting.py`**: This module will handle all plotting functions, including travel-time plots and elevation profiles.

9. **`results.py`**: This module will handle the collection and formatting of results, including saving results to Excel and PDF.
"""
 
import sys
from pathlib import Path
import config
from io_helpers import read_elev_csv, read_geotech_csv, read_vs_file, group_by_transect
from analysis import analyse_shot
from plotting import save_traveltime_pdf, save_elevation_pdf
from results import save_excel
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
                res = analyse_shot(distances, times_ms, shot_pos=shot_pos)
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

    save_traveltime_pdf(all_records, script_dir / "refra_itm_traveltimes.pdf")
    save_excel(all_records, script_dir / "refra_itm_results.xlsx")
    save_elevation_pdf(all_records, elev_data, script_dir / "refra_itm_elevation.pdf", geotech_by_tid, sheetpile_by_tid)

    return 0

if __name__ == "__main__":
    sys.exit(main())