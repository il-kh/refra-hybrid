import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from scipy.stats import linregress
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------------------
# Site-specific plausibility bounds (m/s)
# ---------------------------------------------------------------------------
V1_MIN, V1_MAX = 150,  800
V2_MIN, V2_MAX = 800, 3000


def read_vs_file(file_path):
    shot_pos  = None
    distances = np.array([])
    times_ms  = np.array([])
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, start=1):
            row = line.strip().split()
            if line_num < 3:
                continue
            if line_num == 3:
                shot_pos = float(row[0])
                continue
            if line_num > 3 and line_num < 28:
                if len(row) >= 2:
                    try:
                        distance = float(row[0])
                        time_ms  = float(row[1])
                        distances = np.append(distances, distance)
                        times_ms  = np.append(times_ms,  time_ms)
                    except ValueError:
                        print(f"  Warning: Non-numeric data on line {line_num}"
                              f" – '{line.strip()}'")
                        continue
            else:
                break
    return shot_pos, distances, times_ms


def compute_offsets(geophone_locs, times_ms, shot_pos=None):
    shot_idx    = int(np.argmin(times_ms))
    nearest_geo = geophone_locs[shot_idx]
    if shot_pos is not None and shot_pos != 0.0:
        true_shot_loc = float(shot_pos)
    else:
        true_shot_loc = nearest_geo
    offsets = np.abs(geophone_locs - true_shot_loc)
    return offsets, true_shot_loc, shot_idx


def find_breakpoint_on_wing(wing_offsets, wing_times_sec, min_points=3):
    """
    Detect the V1/V2 breakpoint on a single wing sorted by ascending
    absolute offset (closest to shot first, farthest last).

    Strategy
    --------
    Start with a window of exactly `min_points` points at the far end.
    Only extend the window toward the shot if ALL of the following hold:

        1. The regression slope of the extended window is still positive
           (positive slope = positive velocity = physically meaningful).
        2. The slope does not increase by more than `tolerance` relative
           to the pure far-end seed slope — i.e. we are still on V2,
           not sliding back onto the steeper V1 segment.

    If the seed window itself has a non-positive slope we shrink it by
    one point at a time (down to 2 points) until a positive slope is
    found.  If no positive-slope window exists the function returns
    `n - 2` so that only the last two points are used (minimum fit).

    This replaces the old "take absolute value as fallback" approach:
    the correction is now made at source, before any velocity is computed.

    Parameters
    ----------
    wing_offsets   : 1-D array, ascending absolute offsets
    wing_times_sec : 1-D array, travel times in seconds
    min_points     : minimum number of points in the V2 segment

    Returns
    -------
    v2_local_start : index into wing arrays where V2 begins;
                     wing_offsets[v2_local_start:] are the V2 points.
    """
    n = len(wing_offsets)

    if n < 2 * min_points:
        return _ssr_breakpoint_wing(wing_offsets, wing_times_sec, min_points)

    # ── Find a valid (positive-slope) seed window at the far end ─────────────
    # Start with `min_points` far-end points and shrink toward 2 if needed.
    seed_start = n - min_points
    seed_slope = None

    for seed_size in range(min_points, 1, -1):       # min_points … 2
        seed_start = n - seed_size
        seg_off    = wing_offsets[seed_start:]
        seg_t      = wing_times_sec[seed_start:]
        if len(seg_off) < 2:
            continue
        sl, *_ = linregress(seg_off, seg_t)
        if sl > 0:
            seed_slope = sl
            break

    if seed_slope is None:
        # No positive slope found anywhere at the far end — nothing to fit
        print("    ⚠  No positive V2 slope found on wing; "
              "using last 2 points only.")
        return n - 2

    # ── Grow the window toward the shot while slope stays positive & stable ──
    tolerance      = 1.20
    best_v2_start  = seed_start     # start conservative (far end only)

    for extend_start in range(seed_start - 1, min_points - 1, -1):
        seg_off = wing_offsets[extend_start:]
        seg_t   = wing_times_sec[extend_start:]
        if len(seg_off) < 2:
            break
        sl, *_ = linregress(seg_off, seg_t)

        # Reject if slope is non-positive OR has jumped too far above seed
        if sl <= 0:
            break
        if sl > seed_slope * tolerance:
            break

        best_v2_start = extend_start   # this extension is still acceptable

    return best_v2_start


def _ssr_breakpoint_wing(wing_offsets, wing_times_sec, min_points):
    """
    Exhaustive SSR fallback for a very short wing.
    Only considers split points that yield a positive V2 slope.
    """
    n        = len(wing_offsets)
    best_ssr = np.inf
    best_idx = n - 2        # safe fallback: last two points

    for idx in range(min_points, n - min_points + 1):
        # Check V2 slope positivity before computing SSR
        if len(wing_offsets[idx:]) >= 2:
            sl, *_ = linregress(wing_offsets[idx:], wing_times_sec[idx:])
            if sl <= 0:
                continue    # skip splits that produce non-positive V2 slope

        p1  = np.polyval(np.polyfit(wing_offsets[:idx],
                                    wing_times_sec[:idx], 1),
                         wing_offsets[:idx])
        p2  = np.polyval(np.polyfit(wing_offsets[idx:],
                                    wing_times_sec[idx:], 1),
                         wing_offsets[idx:])
        ssr = (np.sum((wing_times_sec[:idx] - p1) ** 2) +
               np.sum((wing_times_sec[idx:] - p2) ** 2))
        if ssr < best_ssr:
            best_ssr = ssr
            best_idx = idx
    return best_idx


def _fit_wing(wing_geo, wing_offsets, wing_times_sec,
              full_geo, full_offsets, full_times_sec,
              side, min_points=3):
    """
    Run the full V1/V2 analysis on one wing.

    The wing arrays must arrive sorted by ASCENDING absolute offset
    (closest geophone to shot first, farthest last).  This means:

        index 0              → geophone nearest to shot  (always V1)
        index n-1            → geophone farthest from shot (always V2)

    This convention is identical for both wings because np.abs() was
    already applied to all offsets before this function is called.

    find_breakpoint_on_wing() starts its growing window from the FAR end
    (index n-1) and works toward the shot — so it naturally assigns the
    far geophones to V2 on both left and right wings.

    The v2_mask_full is built using the sorted wing indices directly,
    so the breakpoint index maps correctly to the full geophone array.

    Parameters
    ----------
    side : 'left' or 'right'  – used only for labels/debug
    """
    n_wing = len(wing_offsets)
    if n_wing < 2 * min_points:
        return None

    # ── Detect breakpoint on this wing ───────────────────────────────────────
    v2_local_start = find_breakpoint_on_wing(
        wing_offsets, wing_times_sec, min_points)

    # ── Map sorted wing positions back to the full geophone array ────────────
    wing_full_indices = np.array(
        [np.where(full_geo == g)[0][0] for g in wing_geo])

    v2_mask_full = np.zeros(len(full_geo), dtype=bool)
    v2_mask_full[wing_full_indices[v2_local_start:]] = True
    v1_mask_full = ~v2_mask_full

    # ── Regressions ──────────────────────────────────────────────────────────
    slope2, intercept2, *_ = linregress(
        full_offsets[v2_mask_full], full_times_sec[v2_mask_full])
    slope1, intercept1, *_ = linregress(
        full_offsets[v1_mask_full], full_times_sec[v1_mask_full])

    # Both slopes must be positive after the improved breakpoint detection.
    # Log a warning if one still slips through (e.g. extreme noise), but do
    # NOT silently force it — instead mark the result as invalid (NaN depth).
    if slope1 <= 0:
        print(f"    ⚠  [{side}] V1 slope still non-positive ({slope1:.6f}) "
              f"after breakpoint correction — result unreliable.")
    if slope2 <= 0:
        print(f"    ⚠  [{side}] V2 slope still non-positive ({slope2:.6f}) "
              f"after breakpoint correction — result unreliable.")

    # Use abs only as a last-resort guard so downstream code never divides
    # by a negative number, but the warning above flags it clearly.
    slope1 = abs(slope1) if slope1 <= 0 else slope1
    slope2 = abs(slope2) if slope2 <= 0 else slope2

    v1     = 1.0 / slope1
    v2     = 1.0 / slope2
    t_i    = intercept2
    t_i_ms = t_i * 1000.0

    if v2 ** 2 - v1 ** 2 > 0:
        depth = (t_i * v1 * v2) / (2.0 * np.sqrt(v2 ** 2 - v1 ** 2))
    else:
        depth = float('nan')

    bp_offset = full_offsets[v2_mask_full].min()

    return dict(
        v1=v1, v2=v2, t_i_ms=t_i_ms, depth=depth,
        slope1=slope1, intercept1=intercept1,
        slope2=slope2, intercept2=intercept2,
        bp_offset=bp_offset,
        v2_mask=v2_mask_full,
        side=side,
    )


def _check_wing(wing_res, label, warnings):
    """Append plausibility warnings for one wing's results."""
    if wing_res is None:
        return
    v1, v2 = wing_res['v1'], wing_res['v2']
    if not (V1_MIN <= v1 <= V1_MAX):
        msg = f"⚠ {label} V1 = {v1:.0f} m/s outside [{V1_MIN}–{V1_MAX}] m/s"
        print(f"    {msg}")
        warnings.append(msg)
    if not (V2_MIN <= v2 <= V2_MAX):
        msg = f"⚠ {label} V2 = {v2:.0f} m/s outside [{V2_MIN}–{V2_MAX}] m/s"
        print(f"    {msg}")
        warnings.append(msg)
    if v2 <= v1:
        msg = f"⚠ {label} V2 ≤ V1 – refraction condition not met"
        print(f"    {msg}")
        warnings.append(msg)


def analyse_shot(geophone_locs, times_ms, shot_pos=None):
    """
    Dual-wing split-spread analysis.

    LEFT wing  : geophones with position < shot — sorted ascending by
                 offset (closest first, farthest last).
    RIGHT wing : geophones with position > shot — sorted ascending by
                 offset (closest first, farthest last).

    In both cases the FAR-end geophones (highest offset) are assigned
    to V2 (head wave / rock).  The near-end geophones (low offset,
    below the crossover distance) are assigned to V1 (direct wave / sand).
    """
    times_sec = times_ms / 1000.0

    offsets, true_shot_loc, shot_idx = compute_offsets(
        geophone_locs, times_ms, shot_pos=shot_pos)

    nearest_geo = geophone_locs[shot_idx]
    print(f"  Shot pos (header)   : {shot_pos} m  |  "
          f"nearest geophone : {nearest_geo:.1f} m  |  "
          f"true shot loc : {true_shot_loc:.1f} m")
    if shot_pos and shot_pos != 0.0:
        print(f"  Sub-geophone correction : "
              f"{abs(true_shot_loc - nearest_geo):.2f} m")

    left_idx  = np.where(geophone_locs < true_shot_loc)[0]
    right_idx = np.where(geophone_locs > true_shot_loc)[0]

    left_sort  = np.argsort(offsets[left_idx])
    right_sort = np.argsort(offsets[right_idx])

    left_idx_sorted  = left_idx[left_sort]
    right_idx_sorted = right_idx[right_sort]

    print(f"  Wing sizes  →  left : {len(left_idx)} pts  |  "
          f"right : {len(right_idx)} pts")

    warnings = []

    print("  — Right wing (forward) —")
    right_res = _fit_wing(
        wing_geo       = geophone_locs[right_idx_sorted],
        wing_offsets   = offsets[right_idx_sorted],
        wing_times_sec = times_sec[right_idx_sorted],
        full_geo       = geophone_locs,
        full_offsets   = offsets,
        full_times_sec = times_sec,
        side           = 'right',
    )
    if right_res is not None:
        right_res['bp_geo'] = true_shot_loc + right_res['bp_offset']
        print(f"    V1={right_res['v1']:.1f}  V2={right_res['v2']:.1f} m/s  "
              f"t_i={right_res['t_i_ms']:.2f} ms  "
              f"z={right_res['depth']:.2f} m  "
              f"BP@{right_res['bp_geo']:.1f} m")
        _check_wing(right_res, 'Right', warnings)
    else:
        print("    Right wing too short – skipped.")

    print("  — Left wing (reverse) —")
    left_res = _fit_wing(
        wing_geo       = geophone_locs[left_idx_sorted],
        wing_offsets   = offsets[left_idx_sorted],
        wing_times_sec = times_sec[left_idx_sorted],
        full_geo       = geophone_locs,
        full_offsets   = offsets,
        full_times_sec = times_sec,
        side           = 'left',
    )
    if left_res is not None:
        left_res['bp_geo'] = true_shot_loc - left_res['bp_offset']
        print(f"    V1={left_res['v1']:.1f}  V2={left_res['v2']:.1f} m/s  "
              f"t_i={left_res['t_i_ms']:.2f} ms  "
              f"z={left_res['depth']:.2f} m  "
              f"BP@{left_res['bp_geo']:.1f} m")
        _check_wing(left_res, 'Left', warnings)
    else:
        print("    Left wing too short – skipped.")

    return dict(
        right         = right_res,
        left          = left_res,
        true_shot_loc = true_shot_loc,
        offsets       = offsets,
        warnings      = warnings,
    )


def _draw_plot(ax, geophone_locs, times_ms, res, title):
    true_shot_loc = res['true_shot_loc']
    warnings      = res.get('warnings', [])
    right         = res.get('right')
    left          = res.get('left')

    ref = right if right is not None else left
    if ref is None:
        ax.set_title(title + '  [no data]')
        return

    # ── V2 segment points (large diamonds, drawn first = below) ──────────────
    for wing, color in [(right, 'blue'), (left, 'darkgreen')]:
        if wing is None:
            continue
        mask = wing['v2_mask']
        ax.scatter(geophone_locs[mask], times_ms[mask],
                   color=color, zorder=5, s=70, marker='D',
                   label=f"V2 {wing['side']} ({mask.sum()} pts)")

    # ── All first-arrival picks (small red circles, on top) ──────────────────
    ax.scatter(geophone_locs, times_ms, color='red', zorder=6, s=20,
               label='First-arrival picks')

    # ── V1 – single symmetric V-shape ────────────────────────────────────────
    geo_full     = np.linspace(geophone_locs.min(), geophone_locs.max(), 500)
    abs_off_full = np.abs(geo_full - true_shot_loc)
    ax.plot(geo_full,
            (ref['slope1'] * abs_off_full + ref['intercept1']) * 1000,
            color='green', linestyle='--', linewidth=1.5,
            label=f"V1 = {ref['v1']:.0f} m/s  (sand)")

    # ── V2 lines – one per wing ───────────────────────────────────────────────
    if right is not None:
        geo_r  = np.linspace(true_shot_loc, geophone_locs.max(), 300)
        off_r  = geo_r - true_shot_loc
        ax.plot(geo_r,
                (right['slope2'] * off_r + right['intercept2']) * 1000,
                color='blue', linestyle='--', linewidth=1.5,
                label=f"V2 right = {right['v2']:.0f} m/s")

    if left is not None:
        geo_l  = np.linspace(geophone_locs.min(), true_shot_loc, 300)
        off_l  = true_shot_loc - geo_l
        ax.plot(geo_l,
                (left['slope2'] * off_l + left['intercept2']) * 1000,
                color='darkgreen', linestyle='--', linewidth=1.5,
                label=f"V2 left  = {left['v2']:.0f} m/s")

    # ── Span lines ────────────────────────────────────────────────────────────
    ax.axvline(x=true_shot_loc, color='purple', linestyle='-',
               linewidth=1.2, zorder=4,
               label=f'Shot @ {true_shot_loc:.1f} m')

    for wing, color in [(right, 'orange'), (left, 'saddlebrown')]:
        if wing is None:
            continue
        ax.axvline(x=wing['bp_geo'], color=color, linestyle='--',
                   linewidth=1.2, zorder=4,
                   label=f"BP {wing['side']} @ {wing['bp_geo']:.1f} m")

    for wing, color in [(right, 'steelblue'), (left, 'teal')]:
        if wing is None:
            continue
        ax.axhline(y=wing['t_i_ms'], color=color, linestyle=':',
                   linewidth=1.0, zorder=4,
                   label=f"t_i {wing['side']} = {wing['t_i_ms']:.1f} ms")
        ax.scatter([true_shot_loc], [wing['t_i_ms']],
                   color=color, zorder=7, s=60, marker='^')

    # ── Annotation box (top-left) ─────────────────────────────────────────────
    lines = []
    if right is not None:
        lines += [f"Right:  V1={right['v1']:.0f}  V2={right['v2']:.0f} m/s"
                  f"  t_i={right['t_i_ms']:.1f} ms  z={right['depth']:.2f} m"]
    if left is not None:
        lines += [f"Left:   V1={left['v1']:.0f}  V2={left['v2']:.0f} m/s"
                  f"  t_i={left['t_i_ms']:.1f} ms  z={left['depth']:.2f} m"]
    if right is not None and left is not None:
        z_avg = np.nanmean([right['depth'], left['depth']])
        lines += [f"z avg = {z_avg:.2f} m  |  Shot @ {true_shot_loc:.1f} m"]

    ax.annotate('\n'.join(lines),
                xy=(0.02, 0.97), xycoords='axes fraction', fontsize=7.5,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='wheat', alpha=0.9))

    # ── Warning box (bottom-left) ─────────────────────────────────────────────
    if warnings:
        ax.annotate('\n'.join(warnings),
                    xy=(0.02, 0.02), xycoords='axes fraction', fontsize=7,
                    verticalalignment='bottom', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE0E0',
                              edgecolor='red', linewidth=1.2, alpha=0.95))

    ax.set_xlabel('Geophone position (m)')
    ax.set_ylabel('Travel Time (ms)')
    ax.set_title(title, color='darkred' if warnings else 'black')
    ax.legend(fontsize=7, loc='lower right')   # ← moved to bottom-right
    ax.grid(True, alpha=0.4)


def make_figure(file_name, geophone_locs, times_ms, res):
    fig, ax = plt.subplots(figsize=(11, 5))
    _draw_plot(ax, geophone_locs, times_ms, res,
               title=f'Refraction ITM – {file_name}')
    fig.tight_layout()
    return fig


def save_pdf(records, pdf_path):
    A4_W, A4_H = 8.27, 11.69
    LEFT   = 0.10
    RIGHT  = 0.97
    BOTTOM = 0.04
    TOP    = 0.97
    HGAP   = 0.06

    plot_w  = RIGHT - LEFT
    plot_h  = (TOP - BOTTOM - HGAP) / 2.0
    bottoms = [BOTTOM + plot_h + HGAP, BOTTOM]

    with pdf_backend.PdfPages(pdf_path) as pdf:
        i = 0
        while i < len(records):
            fig = plt.figure(figsize=(A4_W, A4_H))
            fig.patch.set_facecolor('white')

            for slot in range(2):
                if i >= len(records):
                    break
                rec = records[i]
                ax  = fig.add_axes([LEFT, bottoms[slot], plot_w, plot_h])
                _draw_plot(ax,
                           rec['geophone_locs'],
                           rec['times_ms'],
                           rec['res'],
                           title=f"Refraction ITM – {rec['file_name']}")
                i += 1

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"  PDF saved  → {pdf_path}")


def save_excel(results, xlsx_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Refraction ITM Results"

    hdr_font  = Font(bold=True, color="FFFFFF")
    hdr_fill  = PatternFill("solid", fgColor="2F5496")
    hdr_align = Alignment(horizontal="center", vertical="center",
                          wrap_text=True)
    thin   = Side(style='thin')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    headers = ["File", "Shot pos (m)",
               "V1 right (m/s)", "V2 right (m/s)",
               "t_i right (ms)", "Depth right (m)",
               "V1 left (m/s)",  "V2 left (m/s)",
               "t_i left (ms)",  "Depth left (m)",
               "Depth avg (m)"]
    col_widths = [24, 13, 14, 14, 14, 14, 13, 13, 13, 13, 13]

    for col, (h, w) in enumerate(zip(headers, col_widths), start=1):
        cell           = ws.cell(row=1, column=col, value=h)
        cell.font      = hdr_font
        cell.fill      = hdr_fill
        cell.alignment = hdr_align
        cell.border    = border
        ws.column_dimensions[get_column_letter(col)].width = w
    ws.row_dimensions[1].height = 30

    warn_fill = PatternFill("solid", fgColor="FCE4D6")

    def _val(wing, key, decimals=2):
        if wing is None:
            return 'N/A'
        v = wing.get(key, float('nan'))
        return round(v, decimals) if not np.isnan(v) else 'N/A'

    for row_num, r in enumerate(results, start=2):
        right = r.get('right')
        left  = r.get('left')

        d_right = right['depth'] if right else float('nan')
        d_left  = left['depth']  if left  else float('nan')
        d_avg   = float(np.nanmean([d_right, d_left]))

        values = [
            r['filename'],
            round(r['shot_pos'], 3),
            _val(right, 'v1', 1), _val(right, 'v2', 1),
            _val(right, 't_i_ms', 3), _val(right, 'depth', 3),
            _val(left,  'v1', 1), _val(left,  'v2', 1),
            _val(left,  't_i_ms', 3), _val(left,  'depth', 3),
            round(d_avg, 3) if not np.isnan(d_avg) else 'N/A',
        ]

        for col, val in enumerate(values, start=1):
            cell           = ws.cell(row=row_num, column=col, value=val)
            cell.border    = border
            cell.alignment = Alignment(horizontal="center")

        if bool(r.get('warnings')):
            for col in range(1, len(headers) + 1):
                ws.cell(row=row_num, column=col).fill = warn_fill

    ws.auto_filter.ref = ws.dimensions
    ws.freeze_panes    = "A2"
    wb.save(xlsx_path)
    print(f"  Excel saved → {xlsx_path}")


def main():
    work_dir = Path('./pick_files')
    vs_files = sorted(work_dir.glob('*.vs'))

    if not vs_files:
        print("No .vs files found in the './pick_files' directory.")
        return 1

    print(f"Found {len(vs_files)} .vs file(s).\n")

    records    = []
    excel_rows = []

    for vs_path in vs_files:
        print(f"{'─'*50}")
        print(f"Processing : {vs_path.name}")

        try:
            shot_pos, distances, times_ms = read_vs_file(vs_path)
        except Exception as e:
            print(f"  ✗ Read error: {e}")
            continue

        if times_ms.size == 0:
            print(f"  ✗ No data rows found – skipping.")
            continue

        try:
            res = analyse_shot(distances, times_ms, shot_pos=shot_pos)
        except Exception as e:
            print(f"  ✗ Analysis error: {e}")
            continue

        records.append(dict(
            file_name     = vs_path.name,
            geophone_locs = distances,
            times_ms      = times_ms,
            res           = res,
        ))
        excel_rows.append(dict(
            filename = vs_path.name,
            shot_pos = res['true_shot_loc'],
            right    = res.get('right'),
            left     = res.get('left'),
            warnings = res.get('warnings', []),
        ))

    if not records:
        print("\nNo files could be processed.")
        return 1

    print(f"\n{'─'*50}")
    print(f"Processed {len(records)} / {len(vs_files)} file(s) successfully.\n")

    pdf_path  = work_dir / "refra_itm_results.pdf"
    xlsx_path = work_dir / "refra_itm_results.xlsx"

    save_pdf(records, pdf_path)
    save_excel(excel_rows, xlsx_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())