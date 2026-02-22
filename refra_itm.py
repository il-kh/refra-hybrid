import sys
import re
from pathlib import Path
from typing import Optional

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
V2_ROCK_MIN    = 800   # m/s – depth point only kept if v2 >= this


# ---------------------------------------------------------------------------
# Elevation helpers
# ---------------------------------------------------------------------------
def read_elevation_file(elev_path: Path):
    distances  = []
    elevations = []
    with open(elev_path, 'r') as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                distances.append(float(parts[0]))
                elevations.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(distances), np.array(elevations)


def find_elevation_file(elev_dir: Path, transect_id: int) -> Optional[Path]:
    for name in (f"elev_{transect_id:04d}.txt", f"elev_{transect_id}.txt"):
        p = elev_dir / name
        if p.exists():
            return p
    return None


def interpolate_elevation(elev_x: np.ndarray, elev_z: np.ndarray,
                          query_x: float) -> Optional[float]:
    if len(elev_x) == 0:
        return None
    return float(np.interp(query_x, elev_x, elev_z))


# ---------------------------------------------------------------------------
# Rock-depth point collector  (works with dict-based wing results)
# ---------------------------------------------------------------------------
def collect_rock_points(records: list,
                        elev_dir: Path) -> dict[int, list[dict]]:
    """
    Returns { transect_id : [ {x, z_surface, depth, z_rock, side, v2}, … ] }
    Only includes wings where depth is valid and v2 >= V2_ROCK_MIN.
    """
    rock: dict[int, list[dict]] = {}

    for rec in records:
        tid = rec['transect_id']
        res = rec['res']

        elev_path = find_elevation_file(elev_dir, tid)
        if elev_path is None:
            continue
        elev_x, elev_z = read_elevation_file(elev_path)

        for wing in (res.get('right'), res.get('left')):
            if wing is None:
                continue
            depth = wing.get('depth', float('nan'))
            v2    = wing.get('v2',    float('nan'))
            if np.isnan(depth) or depth <= 0:
                continue
            if v2 < V2_ROCK_MIN:
                continue

            x      = res['true_shot_loc']
            z_surf = interpolate_elevation(elev_x, elev_z, x)
            if z_surf is None:
                continue

            rock.setdefault(tid, []).append(dict(
                x_geo    = x,
                z_surface= z_surf,
                depth    = depth,
                z_rock   = z_surf - depth,
                side     = wing['side'],
                v2       = v2,
            ))

    return rock


# ---------------------------------------------------------------------------
# Elevation / rock-depth plot
# ---------------------------------------------------------------------------
def _draw_elevation_plot(ax, elev_x: np.ndarray, elev_z: np.ndarray,
                         rock_points: list[dict], title: str):
    if len(elev_x) == 0:
        ax.set_title(title + '  [no elevation data]')
        return

    # Ground surface
    ax.fill_between(elev_x, elev_z, elev_z.min() - 2,
                    color='#D2B48C', alpha=0.45, label='Ground surface')
    ax.plot(elev_x, elev_z,
            color='saddlebrown', linewidth=1.8, label='Elevation profile')

    # Rock depth points
    if rock_points:
        pts_sorted = sorted(rock_points, key=lambda p: p['x_geo'])

        x_rock = np.array([p['x_geo']  for p in pts_sorted])
        z_rock = np.array([p['z_rock'] for p in pts_sorted])

        # Connecting dashed line
        ax.plot(x_rock, z_rock,
                color='dimgray', linestyle='--', linewidth=0.9,
                alpha=0.6, zorder=3)

        # Markers per side
        for side, color, label in [
                ('right', 'steelblue', 'Rock (right wing)'),
                ('left',  'darkgreen', 'Rock (left wing)'),
        ]:
            xs = [p['x_geo']  for p in pts_sorted if p['side'] == side]
            zs = [p['z_rock'] for p in pts_sorted if p['side'] == side]
            if xs:
                ax.scatter(xs, zs, color=color, marker='D',
                           s=55, zorder=5, label=label)

        # Depth labels
        for p in pts_sorted:
            ax.annotate(f"{p['depth']:.1f} m",
                        xy=(p['x_geo'], p['z_rock']),
                        xytext=(4, -10), textcoords='offset points',
                        fontsize=6.5, color='navy')

    # Axis limits with margin
    all_z = list(elev_z)
    if rock_points:
        all_z += [p['z_rock'] for p in rock_points]
    z_min, z_max = min(all_z), max(all_z)
    margin = (z_max - z_min) * 0.15 or 0.5
    ax.set_ylim(z_min - margin, z_max + margin)
    ax.set_xlim(elev_x.min() - 0.5, elev_x.max() + 0.5)

    ax.set_xlabel('Distance along transect (m)', fontsize=8)
    ax.set_ylabel('Elevation (m ASL)', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=7)


def save_elevation_pdf(records: list, elev_dir: Path, pdf_path: Path):
    rock_by_tid = collect_rock_points(records, elev_dir)

    # One entry per transect, in encounter order
    seen = {}
    for rec in records:
        tid = rec['transect_id']
        if tid not in seen:
            seen[tid] = rec
    transect_ids = list(seen.keys())

    if not transect_ids:
        print("  No transect data to plot in elevation PDF.")
        return

    A4_W, A4_H = 8.27, 11.69
    LEFT, RIGHT, BOTTOM, TOP, HGAP = 0.10, 0.97, 0.05, 0.96, 0.07
    plot_w  = RIGHT - LEFT
    plot_h  = (TOP - BOTTOM - HGAP) / 2.0
    bottoms = [BOTTOM + plot_h + HGAP, BOTTOM]

    with pdf_backend.PdfPages(pdf_path) as pdf:
        i = 0
        while i < len(transect_ids):
            fig = plt.figure(figsize=(A4_W, A4_H))
            fig.patch.set_facecolor('white')

            for slot in range(2):
                if i >= len(transect_ids):
                    break
                tid = transect_ids[i]
                i  += 1

                elev_path = find_elevation_file(elev_dir, tid)
                if elev_path is None:
                    elev_x = np.array([])
                    elev_z = np.array([])
                    print(f"  ⚠  No elevation file for transect {tid}")
                else:
                    elev_x, elev_z = read_elevation_file(elev_path)

                ax = fig.add_axes([LEFT, bottoms[slot], plot_w, plot_h])
                _draw_elevation_plot(
                    ax, elev_x, elev_z,
                    rock_by_tid.get(tid, []),
                    title=f"Transect {tid} — Ground profile & rock depth")

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"  Elevation PDF saved → {pdf_path}")


# ---------------------------------------------------------------------------
# File I/O  (unchanged from working version)
# ---------------------------------------------------------------------------
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
                        distances = np.append(distances, float(row[0]))
                        times_ms  = np.append(times_ms,  float(row[1]))
                    except ValueError:
                        print(f"  Warning: Non-numeric data on line {line_num}"
                              f" – '{line.strip()}'")
            else:
                break
    return shot_pos, distances, times_ms


def group_by_transect(vs_files):
    groups: dict[int, list[Path]] = {}
    pattern = re.compile(r'^(\d+)-(\d+)\.vs$', re.IGNORECASE)
    for p in vs_files:
        m = pattern.match(p.name)
        if not m:
            print(f"  Skipping unrecognised filename: {p.name}")
            continue
        tid = int(m.group(1))
        groups.setdefault(tid, []).append(p)
    return dict(sorted(groups.items()))


# ---------------------------------------------------------------------------
# Offset / geometry helpers  (unchanged from working version)
# ---------------------------------------------------------------------------
def compute_offsets(geophone_locs, times_ms, shot_pos=None):
    shot_idx    = int(np.argmin(times_ms))
    nearest_geo = geophone_locs[shot_idx]
    if shot_pos is not None and shot_pos != 0.0:
        true_shot_loc = float(shot_pos)
    else:
        true_shot_loc = nearest_geo
    offsets = np.abs(geophone_locs - true_shot_loc)
    return offsets, true_shot_loc, shot_idx


# ---------------------------------------------------------------------------
# Breakpoint detection  (unchanged from working version)
# ---------------------------------------------------------------------------
def find_breakpoint_on_wing(wing_offsets, wing_times_sec, min_points=3):
    n = len(wing_offsets)
    if n < 2 * min_points:
        return _ssr_breakpoint_wing(wing_offsets, wing_times_sec, min_points)

    # ── Step 1: Find a valid positive-slope seed at the far end ───────────────
    # Start from the last `min_points` points and shrink inward until we get
    # a positive slope.  If even 2 points at the very end give negative slope,
    # we must grow leftward (Step 2b) until the slope becomes positive.
    seed_start = None
    seed_slope = None

    for seed_size in range(min_points, n + 1):          # grow seed if needed
        start   = n - seed_size
        seg_off = wing_offsets[start:]
        seg_t   = wing_times_sec[start:]
        if len(seg_off) < 2:
            continue
        sl, *_ = linregress(seg_off, seg_t)
        if sl > 0:
            seed_start = start
            seed_slope = sl
            break

    if seed_slope is None:
        # Every possible right-end segment has a non-positive slope –
        # the wing has no detectable refractor; fall back to SSR split.
        print("    ⚠  No positive V2 slope found anywhere on wing; "
              "falling back to SSR breakpoint.")
        return _ssr_breakpoint_wing(wing_offsets, wing_times_sec, min_points)

    # ── Step 2: Extend the V2 segment leftward as long as slope stays ─────────
    # positive and does not deviate more than `tolerance` from the seed slope.
    tolerance     = 1.20          # allow up to 20 % steeper than seed slope
    best_v2_start = seed_start

    for extend_start in range(seed_start - 1, min_points - 1, -1):
        seg_off = wing_offsets[extend_start:]
        seg_t   = wing_times_sec[extend_start:]
        if len(seg_off) < 2:
            break
        sl, *_ = linregress(seg_off, seg_t)
        if sl <= 0:
            break                           # slope went negative – stop
        if sl > seed_slope * tolerance:
            break                           # slope got too steep – stop
        best_v2_start = extend_start       # this start is still acceptable

    return best_v2_start


def _ssr_breakpoint_wing(wing_offsets, wing_times_sec, min_points):
    n        = len(wing_offsets)
    best_ssr = np.inf
    best_idx = n - 2

    for idx in range(min_points, n - min_points + 1):
        if len(wing_offsets[idx:]) >= 2:
            sl, *_ = linregress(wing_offsets[idx:], wing_times_sec[idx:])
            if sl <= 0:
                continue
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


# ---------------------------------------------------------------------------
# Per-wing fitting  (unchanged from working version, returns dict)
# ---------------------------------------------------------------------------
def _fit_wing(wing_geo, wing_offsets, wing_times_sec,
              full_geo, full_offsets, full_times_sec,
              side, min_points=3):
    n_wing = len(wing_offsets)
    if n_wing < 2 * min_points:
        return None

    v2_local_start = find_breakpoint_on_wing(
        wing_offsets, wing_times_sec, min_points)

    wing_full_indices = np.array(
        [np.where(full_geo == g)[0][0] for g in wing_geo])

    v2_mask_full = np.zeros(len(full_geo), dtype=bool)
    v2_mask_full[wing_full_indices[v2_local_start:]] = True
    v1_mask_full = ~v2_mask_full

    slope2, intercept2, *_ = linregress(
        full_offsets[v2_mask_full], full_times_sec[v2_mask_full])
    slope1, intercept1, *_ = linregress(
        full_offsets[v1_mask_full], full_times_sec[v1_mask_full])

    if slope1 <= 0:
        print(f"    ⚠  [{side}] V1 slope non-positive ({slope1:.6f}) "
              f"– result unreliable.")
    if slope2 <= 0:
        print(f"    ⚠  [{side}] V2 slope non-positive ({slope2:.6f}) "
              f"– result unreliable.")

    slope1 = abs(slope1) if slope1 <= 0 else slope1
    slope2 = abs(slope2) if slope2 <= 0 else slope2

    v1     = 1.0 / slope1
    v2     = 1.0 / slope2
    t_i    = intercept2
    t_i_ms = t_i * 1000.0

    if v2 >= V2_ROCK_MIN and v2 ** 2 - v1 ** 2 > 0:
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


# ---------------------------------------------------------------------------
# Plausibility checks  (unchanged from working version)
# ---------------------------------------------------------------------------
def _check_wing(wing_res, label, warnings):
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


# ---------------------------------------------------------------------------
# Shot analyser  (unchanged from working version)
# ---------------------------------------------------------------------------
def analyse_shot(geophone_locs, times_ms, shot_pos=None):
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


# ---------------------------------------------------------------------------
# Traveltime plot  (unchanged from working version)
# ---------------------------------------------------------------------------
def _draw_plot(ax, geophone_locs, times_ms, res, title):
    true_shot_loc = res['true_shot_loc']
    warnings      = res.get('warnings', [])
    right         = res.get('right')
    left          = res.get('left')

    ref = right if right is not None else left
    if ref is None:
        ax.set_title(title + '  [no data]')
        return

    for wing, color in [(right, 'blue'), (left, 'darkgreen')]:
        if wing is None:
            continue
        mask = wing['v2_mask']
        ax.scatter(geophone_locs[mask], times_ms[mask],
                   color=color, zorder=5, s=70, marker='D',
                   label=f"V2 {wing['side']} ({mask.sum()} pts)")

    ax.scatter(geophone_locs, times_ms, color='red', zorder=6, s=20,
               label='First-arrival picks')

    geo_full     = np.linspace(geophone_locs.min(), geophone_locs.max(), 500)
    abs_off_full = np.abs(geo_full - true_shot_loc)
    ax.plot(geo_full,
            (ref['slope1'] * abs_off_full + ref['intercept1']) * 1000,
            color='green', linestyle='--', linewidth=1.5,
            label=f"V1 = {ref['v1']:.0f} m/s  (sand)")

    if right is not None:
        geo_r = np.linspace(true_shot_loc, geophone_locs.max(), 300)
        off_r = geo_r - true_shot_loc
        ax.plot(geo_r,
                (right['slope2'] * off_r + right['intercept2']) * 1000,
                color='blue', linestyle='--', linewidth=1.5,
                label=f"V2 right = {right['v2']:.0f} m/s")

    if left is not None:
        geo_l = np.linspace(geophone_locs.min(), true_shot_loc, 300)
        off_l = true_shot_loc - geo_l
        ax.plot(geo_l,
                (left['slope2'] * off_l + left['intercept2']) * 1000,
                color='darkgreen', linestyle='--', linewidth=1.5,
                label=f"V2 left  = {left['v2']:.0f} m/s")

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

    if warnings:
        ax.annotate('\n'.join(warnings),
                    xy=(0.02, 0.02), xycoords='axes fraction', fontsize=7,
                    verticalalignment='bottom', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE0E0',
                              edgecolor='red', linewidth=1.2, alpha=0.95))

    ax.set_xlabel('Geophone position (m)')
    ax.set_ylabel('Travel Time (ms)')
    ax.set_title(title, color='darkred' if warnings else 'black')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.4)


# ---------------------------------------------------------------------------
# Traveltime PDF  (unchanged from working version)
# ---------------------------------------------------------------------------
def save_pdf(records, pdf_path):
    A4_W, A4_H = 8.27, 11.69
    LEFT, RIGHT, BOTTOM, TOP, HGAP = 0.10, 0.97, 0.04, 0.97, 0.06
    plot_w  = RIGHT - LEFT
    plot_h  = (TOP - BOTTOM - HGAP) / 2.0
    bottoms = [BOTTOM + plot_h + HGAP, BOTTOM]

    with pdf_backend.PdfPages(pdf_path) as pdf:
        prev_tid = None
        i = 0
        while i < len(records):
            rec = records[i]
            tid = rec['transect_id']

            if tid != prev_tid:
                sep = plt.figure(figsize=(A4_W, 0.8))
                sep.text(0.5, 0.5, f"Transect  {tid}",
                         ha='center', va='center',
                         fontsize=14, fontweight='bold',
                         transform=sep.transFigure)
                pdf.savefig(sep, bbox_inches='tight')
                plt.close(sep)
                prev_tid = tid

            fig = plt.figure(figsize=(A4_W, A4_H))
            fig.patch.set_facecolor('white')

            for slot in range(2):
                if i >= len(records) or records[i]['transect_id'] != tid:
                    break
                r  = records[i]
                ax = fig.add_axes([LEFT, bottoms[slot], plot_w, plot_h])
                _draw_plot(ax,
                           r['geophone_locs'], r['times_ms'], r['res'],
                           title=f"Refraction ITM – {r['file_name']}")
                i += 1

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"  Traveltime PDF saved → {pdf_path}")


# ---------------------------------------------------------------------------
# Excel output  (unchanged from working version)
# ---------------------------------------------------------------------------
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

    headers = ["Transect", "File", "Shot pos (m)",
               "V1 right (m/s)", "V2 right (m/s)",
               "t_i right (ms)", "Depth right (m)",
               "V1 left (m/s)",  "V2 left (m/s)",
               "t_i left (ms)",  "Depth left (m)",
               "Depth avg (m)",  "Warnings"]
    col_widths = [10, 24, 13, 14, 14, 14, 14, 13, 13, 13, 13, 13, 50]

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
        res   = r['res']
        right = res.get('right')
        left  = res.get('left')

        d_right = right['depth'] if right else float('nan')
        d_left  = left['depth']  if left  else float('nan')
        d_avg   = float(np.nanmean([d_right, d_left]))

        values = [
            r['transect_id'],
            r['file_name'],
            round(res['true_shot_loc'], 3),
            _val(right, 'v1', 1), _val(right, 'v2', 1),
            _val(right, 't_i_ms', 3), _val(right, 'depth', 3),
            _val(left,  'v1', 1), _val(left,  'v2', 1),
            _val(left,  't_i_ms', 3), _val(left,  'depth', 3),
            round(d_avg, 3) if not np.isnan(d_avg) else 'N/A',
            ' | '.join(res.get('warnings', [])),
        ]

        for col, val in enumerate(values, start=1):
            cell           = ws.cell(row=row_num, column=col, value=val)
            cell.border    = border
            cell.alignment = Alignment(horizontal="center")

        if res.get('warnings'):
            for col in range(1, len(headers) + 1):
                ws.cell(row=row_num, column=col).fill = warn_fill

    ws.auto_filter.ref = ws.dimensions
    ws.freeze_panes    = "A2"
    wb.save(xlsx_path)
    print(f"  Excel saved → {xlsx_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    script_dir = Path(__file__).parent
    pick_dir   = script_dir / 'pick_files'
    elev_dir   = script_dir / 'elevation_files'
    work_dir   = pick_dir if pick_dir.is_dir() else script_dir

    vs_files = sorted(work_dir.glob('*.vs'))
    if not vs_files:
        print("No .vs files found.")
        return 1

    print(f"Found {len(vs_files)} .vs file(s).\n")
    transects = group_by_transect(vs_files)
    print(f"Transects found: {list(transects.keys())}\n")

    all_records: list = []
    excel_rows:  list = []

    for tid, files in transects.items():
        print(f"\n{'═'*60}")
        print(f"  TRANSECT  {tid}  ({len(files)} shots)")
        print(f"{'═'*60}\n")

        for path in sorted(files):
            print(f"  {'─'*50}")
            print(f"  Processing : {path.name}")

            try:
                shot_pos, distances, times_ms = read_vs_file(path)
            except Exception as e:
                print(f"  ✗ Read error: {e}")
                continue

            if times_ms.size == 0:
                print("  ✗ No data – skipping.")
                continue

            try:
                res = analyse_shot(distances, times_ms, shot_pos=shot_pos)
            except Exception as e:
                print(f"  ✗ Analysis error: {e}")
                continue

            all_records.append(dict(
                transect_id   = tid,
                file_name     = path.name,
                geophone_locs = distances,
                times_ms      = times_ms,
                res           = res,
            ))
            excel_rows.append(dict(
                transect_id = tid,
                file_name   = path.name,
                res         = res,
            ))

    if not all_records:
        print("\nNo files processed.")
        return 1

    print(f"\n{'═'*60}")
    print(f"Processed {len(all_records)} / {len(vs_files)} shots.\n")

    tt_pdf    = script_dir / "refra_itm_traveltimes.pdf"
    elev_pdf  = script_dir / "refra_itm_elevation.pdf"
    xlsx_path = script_dir / "refra_itm_results.xlsx"

    save_pdf(all_records, tt_pdf)
    save_excel(excel_rows, xlsx_path)

    if elev_dir.is_dir():
        save_elevation_pdf(all_records, elev_dir, elev_pdf)
    else:
        print("  ⚠  elevation_files/ not found – skipping elevation PDF.")

    return 0


if __name__ == "__main__":
    sys.exit(main())