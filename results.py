from pathlib import Path
from openpyxl import Workbook

from analysis_itm import WingResult

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


# ===========================================================================
# Excel writer
# ===========================================================================

_HEADERS = [
    "Transect", "File", "Shot pos (m)",
    "V1 right (m/s)", "V2 right (m/s)", "t_i right (ms)", "Depth right (m)",
    "V1 left (m/s)",  "V2 left (m/s)",  "t_i left (ms)",  "Depth left (m)",
    "Depth avg (m)", "Warnings",
]
_COL_WIDTHS = [10, 24, 13, 14, 14, 14, 14, 13, 13, 13, 13, 13, 50]


def _wing_val(wing: WingResult, key: str, decimals: int = 2):
    """Safe getter – returns 'N/A' for missing / NaN values."""
    if wing is None:
        return 'N/A'
    val = wing.get(key, float('nan'))
    if isinstance(val, float):
        return round(val, decimals) if not np.isnan(val) else 'N/A'
    return val


def save_excel(results: list[dict], xlsx_path: Path) -> None:
    """Write a formatted summary spreadsheet."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Refraction ITM Results"

    thin      = Side(style='thin')
    border    = Border(left=thin, right=thin, top=thin, bottom=thin)
    centre    = Alignment(horizontal="center")
    warn_fill = PatternFill("solid", fgColor="FCE4D6")

    for col, (header, width) in enumerate(
            zip(_HEADERS, _COL_WIDTHS), start=1):
        cell           = ws.cell(row=1, column=col, value=header)
        cell.font      = Font(bold=True, color="FFFFFF")
        cell.fill      = PatternFill("solid", fgColor="2F5496")
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                   wrap_text=True)
        cell.border    = border
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.row_dimensions[1].height = 30

    for row_num, r in enumerate(results, start=2):
        res   = r['res']
        right = res.get('right')
        left  = res.get('left')
        depths = [w['depth'] for w in filter(None, (right, left))]
        d_avg  = float(np.nanmean(depths)) if depths else float('nan')

        row_values = [
            r['transect_id'], r['file_name'],
            round(res['true_shot_loc'], 3),
            _wing_val(right, 'v1', 1), _wing_val(right, 'v2', 1),
            _wing_val(right, 't_i_ms', 3), _wing_val(right, 'depth', 3),
            _wing_val(left,  'v1', 1), _wing_val(left,  'v2', 1),
            _wing_val(left,  't_i_ms', 3), _wing_val(left,  'depth', 3),
            round(d_avg, 3) if not np.isnan(d_avg) else 'N/A',
            ' | '.join(res.get('warnings', [])),
        ]
        has_warnings = bool(res.get('warnings'))
        for col, value in enumerate(row_values, start=1):
            cell           = ws.cell(row=row_num, column=col, value=value)
            cell.border    = border
            cell.alignment = centre
            if has_warnings:
                cell.fill = warn_fill

    ws.auto_filter.ref = ws.dimensions
    ws.freeze_panes    = "A2"
    wb.save(xlsx_path)
    print(f"  Excel saved → {xlsx_path}")
