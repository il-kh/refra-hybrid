from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

import config
from analysis_itm import ShotResult, collect_rock_points
from analysis_plusminus import PMPairResult
from elevation import interpolate_elevation
from geotech import GeotechPoint
from io_helpers import SheetPileLine


# ===========================================================================
# Travel-time plot
# ===========================================================================

def _draw_traveltime_plot(ax: plt.Axes,
                          geophone_locs: np.ndarray,
                          times_ms: np.ndarray,
                          res: ShotResult,
                          title: str) -> None:
    """Travel-time vs. geophone-position plot for one shot."""
    true_shot_loc = res['true_shot_loc']
    warnings      = res.get('warnings', [])
    right         = res.get('right')
    left          = res.get('left')
    ref = right if right is not None else left
    if ref is None:
        ax.set_title(title + '  [no data]')
        return

    for wing, color in ((right, 'blue'), (left, 'darkgreen')):
        if wing is None:
            continue
        mask = wing['v2_mask']
        ax.scatter(geophone_locs[mask], times_ms[mask],
                   color=color, zorder=5, s=70, marker='D',
                   label=f"V2 {wing['side']} ({mask.sum()} pts)")

    ax.scatter(geophone_locs, times_ms,
               color='red', zorder=6, s=20, label='First-arrival picks')

    geo_full = np.linspace(geophone_locs.min(), geophone_locs.max(), 500)
    off_full = np.abs(geo_full - true_shot_loc)
    ax.plot(geo_full,
            (ref['slope1'] * off_full + ref['intercept1']) * 1000,
            color='green', linestyle='--', linewidth=1.5,
            label=f"V1 = {ref['v1']:.0f} m/s")

    for wing, color, geo_lo, geo_hi, sign in (
            (right, 'blue',      true_shot_loc,       geophone_locs.max(), +1),
            (left,  'darkgreen', geophone_locs.min(), true_shot_loc,       -1),
    ):
        if wing is None:
            continue
        geo_w = np.linspace(geo_lo, geo_hi, 300)
        ax.plot(geo_w,
                (wing['slope2'] * sign * (geo_w - true_shot_loc)
                 + wing['intercept2']) * 1000,
                color=color, linestyle='--', linewidth=1.5,
                label=f"V2 {wing['side']} = {wing['v2']:.0f} m/s")

    ax.axvline(x=true_shot_loc, color='purple', linestyle='-',
               linewidth=1.2, zorder=4,
               label=f'Shot @ {true_shot_loc:.1f} m')

    for wing, color in ((right, 'orange'), (left, 'saddlebrown')):
        if wing is None:
            continue
        ax.axvline(x=wing['bp_geo'], color=color, linestyle='--',
                   linewidth=1.2, zorder=4,
                   label=f"BP {wing['side']} @ {wing['bp_geo']:.1f} m")

    for wing, color in ((right, 'steelblue'), (left, 'teal')):
        if wing is None:
            continue
        ax.axhline(y=wing['t_i_ms'], color=color, linestyle=':',
                   linewidth=1.0, zorder=4,
                   label=f"t_i {wing['side']} = {wing['t_i_ms']:.1f} ms")
        ax.scatter([true_shot_loc], [wing['t_i_ms']],
                   color=color, zorder=7, s=60, marker='^')

    lines: list[str] = []
    for wing in filter(None, (right, left)):
        lines.append(
            f"{wing['side'].capitalize()}: "
            f"V1={wing['v1']:.0f}  V2={wing['v2']:.0f} m/s  "
            f"t_i={wing['t_i_ms']:.1f} ms  z={wing['depth']:.2f} m")
    if right is not None and left is not None:
        z_avg = np.nanmean([right['depth'], left['depth']])
        lines.append(f"z avg = {z_avg:.2f} m  |  "
                     f"Shot @ {true_shot_loc:.1f} m")

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
    ax.set_ylabel('Travel time (ms)')
    ax.set_title(title, color='darkred' if warnings else 'black')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.4)


# ===========================================================================
# Elevation + geotech plot
# ===========================================================================

def _draw_geotech_overlays(ax: plt.Axes,
                            elev_x: np.ndarray,
                            elev_z: np.ndarray,
                            geotech_pts: list[GeotechPoint],
                            sheetpile: Optional[SheetPileLine]) -> None:
    """
    Overlay sheet-pile line and geotech test lines on an elevation axes.

    Rules
    -----
    - Sheet-pile: dashed vertical line from surface to SHEET_PILE_DEPTH.
    - Each test: solid vertical line from surface to tested_depth.
    - Rock found: star marker at the rock depth with depth annotation.
    - No rock found: NO point symbol at the bottom of the line.
    """
    # ── Sheet-pile ─────────────────────────────────────────────────────────────
    if sheetpile is not None:
        x_sp  = sheetpile.dist_x
        z_top = interpolate_elevation(elev_x, elev_z, x_sp)
        if z_top is not None:
            z_bot = z_top - config.SHEET_PILE_DEPTH
            ax.plot([x_sp, x_sp], [z_top, z_bot],
                    color=config.SHEET_PILE_COLOR, linestyle='--',
                    linewidth=1.8, zorder=6,
                    label=f"Sheet pile @ x={x_sp:.1f} m "
                          f"(d={config.SHEET_PILE_DEPTH:.0f} m)")

    # ── Geotech tests ─────────────────────────────────────────────────────────
    # Colour map: test_type → (line_color, star_color, legend_label)
    _STYLE: dict[str, tuple[str, str, str]] = {
        'DPL':  (config.DPL_COLOR,  'goldenrod',  'DPL'),
        'CPTU': (config.CPTU_COLOR, 'steelblue',  'CPTu'),
    }

    legend_line_added:  set[str] = set()
    legend_rock_added:  set[str] = set()

    for pt in geotech_pts:
        key        = pt.test_type.upper()
        line_color, star_color, nice_label = _STYLE.get(
            key, (config.CPTU_COLOR, 'steelblue', pt.test_type))

        z_surf = interpolate_elevation(elev_x, elev_z, pt.dist_x)
        if z_surf is None:
            continue

        z_bot = z_surf - pt.tested_depth

        # ── Vertical test line ────────────────────────────────────────────────
        line_label = (nice_label
                      if key not in legend_line_added
                      else '_nolegend_')
        ax.plot([pt.dist_x, pt.dist_x], [z_surf, z_bot],
                color=line_color, linewidth=1.4, zorder=5,
                label=line_label)
        legend_line_added.add(key)

        # ── Rock marker: star at rock depth ───────────────────────────────────
        if pt.depth_of_rock is not None:
            z_rock     = z_surf - pt.depth_of_rock
            rock_key   = f'rock_{key}'
            rock_label = (f'Rock ({nice_label})'
                          if rock_key not in legend_rock_added
                          else '_nolegend_')
            ax.scatter([pt.dist_x], [z_rock],
                       color=star_color, marker='*',
                       s=config.ROCK_MARKER_SIZE ** 2,
                       zorder=7, label=rock_label)
            legend_rock_added.add(rock_key)
            ax.annotate(f"{pt.depth_of_rock:.1f} m",
                        xy=(pt.dist_x, z_rock),
                        xytext=(4, 3), textcoords='offset points',
                        fontsize=6, color=line_color, clip_on=True)


def _spline_segments(
        xs: np.ndarray,
        zs: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return a single smooth interpolating curve through all (xs, zs) points.

    Uses CubicSpline for ≥4 points, quadratic interp1d for exactly 3,
    and linear interpolation for 2 points.  Returns a list with one
    (x_dense, z_dense) tuple (or empty list if fewer than 2 points).
    """
    n = len(xs)
    if n < 2:
        return []
    x_dense = np.linspace(xs[0], xs[-1], 300)
    if n >= 4:
        z_dense = CubicSpline(xs, zs)(x_dense)
    elif n == 3:
        z_dense = interp1d(xs, zs, kind='quadratic')(x_dense)
    else:                               # n == 2 → linear
        z_dense = np.interp(x_dense, xs, zs)
    return [(x_dense, z_dense)]


def _draw_elevation_plot(ax: plt.Axes,
                         elev_x: np.ndarray,
                         elev_z: np.ndarray,
                         rock_points: list[dict],
                         title: str,
                         geotech_pts: Optional[list[GeotechPoint]] = None,
                         sheetpile: Optional[SheetPileLine] = None,
                         pm_rock_points: Optional[list[dict]] = None,
                         ) -> None:
    """
    Ground-surface profile with ITM rock-depth points and geotech overlays.

    The elevation data may not span the full plot x-range [ELEV_X_MIN,
    ELEV_X_MAX].  Where it falls short the surface line is extended
    horizontally (clamped) to the nearest known elevation so the fill and
    profile always reach both axis edges.

    Fixed axes  x: ELEV_X_MIN → ELEV_X_MAX
                y: ELEV_Y_MIN → ELEV_Y_MAX
    with 1:1 data-unit aspect ratio enforced by set_aspect('equal').
    """
    if len(elev_x) == 0:
        ax.set_title(title + '  [no elevation data]')
        return

    # Build a dense x-grid that covers the full plot range, then
    # use np.interp (which clamps at the boundary values) to get z.
    # This automatically extends the profile line to both edges.
    x_plot = np.linspace(config.ELEV_X_MIN, config.ELEV_X_MAX, 1000)
    z_plot = np.interp(x_plot, elev_x, elev_z)   # clamps outside data range

    # ── Ground surface ────────────────────────────────────────────────────────
    ax.fill_between(x_plot, z_plot, config.ELEV_Y_MIN,
                    color='#D2B48C', alpha=0.45, label='Ground surface')
    ax.plot(x_plot, z_plot,
            color='saddlebrown', linewidth=1.8, label='Elevation profile')

    # Mark the extent of actual measured data vs. extrapolated ends
    x_data_min, x_data_max = elev_x.min(), elev_x.max()
    for x_lo, x_hi in (
            (config.ELEV_X_MIN, min(x_data_min, config.ELEV_X_MAX)),
            (max(x_data_max, config.ELEV_X_MIN), config.ELEV_X_MAX),
    ):
        if x_lo < x_hi:
            mask = (x_plot >= x_lo) & (x_plot <= x_hi)
            ax.plot(x_plot[mask], z_plot[mask],
                    color='saddlebrown', linewidth=1.2,
                    linestyle=':', alpha=0.6)   # dotted = extrapolated

    # ── Barrier x-positions: NO LONGER USED ─────────────────────────────────
    # The spline is free to cross any geotech line.

    # ── Collect all rock depth points for the combined spline ───────────────
    # Geotech rock points (DPL / CPTu) converted to absolute z
    geo_rock_pts: list[tuple[float, float]] = []
    for pt in (geotech_pts or []):
        if pt.depth_of_rock is None:
            continue
        z_surf = interpolate_elevation(x_plot, z_plot, pt.dist_x)
        if z_surf is not None:
            geo_rock_pts.append((pt.dist_x, z_surf - pt.depth_of_rock))

    # ── ITM rock-depth points ─────────────────────────────────────────────────
    _itm_cfg = {
        'right': (config.SHOW_ITM_RIGHT, config.ITM_RIGHT_DEPTH_MIN,
                  config.ITM_RIGHT_X_MIN, config.ITM_RIGHT_X_MAX),
        'left':  (config.SHOW_ITM_LEFT,  config.ITM_LEFT_DEPTH_MIN,
                  config.ITM_LEFT_X_MIN,  config.ITM_LEFT_X_MAX),
    }

    def _itm_visible(p: dict) -> bool:
        show, dmin, xmin, xmax = _itm_cfg.get(p['side'], (False, 0.0, -np.inf, np.inf))
        return bool(show) and p['depth'] >= dmin and xmin <= p['x_geo'] <= xmax

    visible_itm = [p for p in sorted(rock_points or [], key=lambda p: p['x_geo'])
                   if _itm_visible(p)]

    if visible_itm:
        x_rock = np.array([p['x_geo']  for p in visible_itm])
        z_rock = np.array([p['z_rock'] for p in visible_itm])
        # Build combined point set: seismic ITM + geotech rock, sorted by x
        combined = sorted(
            [(p['x_geo'], p['z_rock']) for p in visible_itm] + geo_rock_pts
        )
        cx = np.array([p[0] for p in combined])
        cz = np.array([p[1] for p in combined])
        for x_seg, z_seg in _spline_segments(cx, cz):
            ax.plot(x_seg, z_seg,
                    color='dimgray', linestyle='--', linewidth=0.9,
                    alpha=0.6, zorder=3)

        for side, color, label in (
                ('right', 'steelblue', 'ITM rock (right wing)'),
                ('left',  'darkgreen', 'ITM rock (left wing)'),
        ):
            xs = [p['x_geo']  for p in visible_itm if p['side'] == side]
            zs = [p['z_rock'] for p in visible_itm if p['side'] == side]
            if xs:
                ax.scatter(xs, zs, color=color, marker='D',
                           s=55, zorder=5, label=label)

        for p in visible_itm:
            ax.annotate(f"{p['depth']:.1f} m",
                        xy=(p['x_geo'], p['z_rock']),
                        xytext=(4, -10), textcoords='offset points',
                        fontsize=6.5, color='navy', clip_on=True)

    # ── PM refractor points (triangles, median-aggregated) ──────────────────
    # NOTE: this is the shallow refractor (V₂ ≈ 870 m/s), NOT rock.
    # See analysis_plusminus module docstring for the three-layer explanation.
    if pm_rock_points and config.SHOW_PM_REFRACTOR:
        pm_sorted = [
            p for p in sorted(pm_rock_points, key=lambda p: p['x_geo'])
            if (p['depth'] >= config.PM_REFRACTOR_DEPTH_MIN
                and config.PM_REFRACTOR_X_MIN <= p['x_geo'] <= config.PM_REFRACTOR_X_MAX)
        ]
        if not pm_sorted:
            pm_sorted = None  # fall through to skip block
    else:
        pm_sorted = None

    if pm_sorted:
        x_pm = np.array([p['x_geo']  for p in pm_sorted])
        z_pm = np.array([p['z_rock'] for p in pm_sorted])

        # Compute median V₂ across all bins for the legend
        med_v2 = np.median([p.get('v2_median', 0) for p in pm_sorted])
        lbl = f'PM refractor (V\u2082\u2248{med_v2:.0f} m/s)'

        # Build combined point set: PM + geotech rock, sorted by x
        combined_pm = sorted(
            [(p['x_geo'], p['z_rock']) for p in pm_sorted] + geo_rock_pts
        )
        cpx = np.array([p[0] for p in combined_pm])
        cpz = np.array([p[1] for p in combined_pm])
        for x_seg, z_seg in _spline_segments(cpx, cpz):
            ax.plot(x_seg, z_seg,
                    color='purple', linestyle='-', linewidth=1.2,
                    alpha=0.7, zorder=3)
        ax.scatter(x_pm, z_pm, color='darkorchid', marker='^',
                   s=65, zorder=5, edgecolors='indigo', linewidths=0.5,
                   label=lbl)

        for p in pm_sorted:
            ax.annotate(f"{p['depth']:.1f}",
                        xy=(p['x_geo'], p['z_rock']),
                        xytext=(-4, -10), textcoords='offset points',
                        fontsize=5.5, color='purple', clip_on=True)

    # ── Geotech overlays ──────────────────────────────────────────────────────
    # Pass the full (possibly extrapolated) x/z arrays so that
    # interpolate_elevation works correctly for any x in [0, 28].
    _draw_geotech_overlays(ax, x_plot, z_plot, geotech_pts or [], sheetpile)

    # ── Ground-water level ───────────────────────────────────────────────────
    gwl_y = config.GWL_Y
    if config.ELEV_Y_MIN <= gwl_y <= config.ELEV_Y_MAX:
        ax.axhline(y=gwl_y, color='steelblue', linestyle='--',
                   linewidth=1.4, zorder=4,
                   label=f'GWL {gwl_y:.1f} m ASL ({config.GWL_DATE})')
        ax.annotate(
            f'GWL  {gwl_y:.1f} m ASL\n({config.GWL_DATE})',
            xy=(config.ELEV_X_MIN, gwl_y),
            xytext=(4, 3), textcoords='offset points',
            fontsize=6, color='steelblue', clip_on=True,
        )

    # ── Fixed 1:1 axes ────────────────────────────────────────────────────────
    ax.set_xlim(config.ELEV_X_MIN, config.ELEV_X_MAX)
    ax.set_ylim(config.ELEV_Y_MIN, config.ELEV_Y_MAX)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('Distance along transect (m)', fontsize=8)
    ax.set_ylabel('Elevation (m ASL)', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc='lower right')
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=7)


# ===========================================================================
# PDF page layout
# ===========================================================================

# A4 portrait in inches
_A4_W, _A4_H = 8.27, 11.69

# Travel-time page margins (figure-fraction)
_TT_L, _TT_R  = 0.10, 0.97
_TT_B, _TT_T  = 0.04, 0.97
_TT_GAP       = 0.06

# Elevation page layout constants (all in inches)
_ELEV_MARGIN_L_IN  = 0.70   # left margin  (room for y-axis label)
_ELEV_MARGIN_R_IN  = 0.30   # right margin
_ELEV_MARGIN_T_IN  = 0.45   # top margin   (room for page top)
_ELEV_MARGIN_B_IN  = 0.55   # bottom margin
_ELEV_GAP_IN       = 0.90   # gap between bottom of upper plot
                             # and top of lower plot  (title + y-label room)

# Maximum fraction of A4 height the two plots + gap may occupy
# (leaves _ELEV_MARGIN_T_IN + _ELEV_MARGIN_B_IN + _ELEV_GAP_IN around them)
_ELEV_MAX_CONTENT_H_IN = (_A4_H
                           - _ELEV_MARGIN_T_IN
                           - _ELEV_MARGIN_B_IN
                           - _ELEV_GAP_IN)


def _elev_page_layout() -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Create one A4 figure with two elevation axes that each have a true
    1:1 data-unit aspect ratio for the configured data window.

    Returns (fig, [ax_top, ax_bottom]).
    """
    data_w = config.ELEV_X_MAX - config.ELEV_X_MIN   # e.g. 28 m
    data_h = config.ELEV_Y_MAX - config.ELEV_Y_MIN   # e.g. 22 m
    aspect = data_w / data_h                          # ≈ 1.273

    usable_w_in = _A4_W - _ELEV_MARGIN_L_IN - _ELEV_MARGIN_R_IN
    plot_w_in   = usable_w_in
    plot_h_in   = plot_w_in / aspect   # 1:1 unconstrained height

    # Cap plot height so that two plots + gap fit between minimum margins.
    # Subtracting 2×GAP is intentionally conservative (leaves title room).
    max_single_h = (_A4_H - _ELEV_MARGIN_T_IN - _ELEV_MARGIN_B_IN
                    - 2.0 * _ELEV_GAP_IN) / 2.0
    if plot_h_in > max_single_h:
        plot_h_in = max_single_h
        plot_w_in = plot_h_in * aspect

    # Figure is exactly A4
    fig = plt.figure(figsize=(_A4_W, _A4_H))
    fig.patch.set_facecolor('white')

    # Convert inches → figure fractions
    def _fy(y_in: float) -> float:
        return y_in / _A4_H

    def _fx(x_in: float) -> float:
        return x_in / _A4_W

    h_f = _fy(plot_h_in)
    w_f = _fx(plot_w_in)

    # Horizontal centering: place (y-label margin + plot + right margin)
    # block in the centre of the A4 width.
    total_block_w = _ELEV_MARGIN_L_IN + plot_w_in + _ELEV_MARGIN_R_IN
    left_edge_in  = max(0.0, (_A4_W - total_block_w) / 2.0)
    left_f = _fx(left_edge_in + _ELEV_MARGIN_L_IN)

    # Vertical centering: distribute remaining space evenly above and below
    # the two-plots-plus-gap block, respecting minimum title/xlabel margins.
    content_h_in = 2.0 * plot_h_in + _ELEV_GAP_IN
    available_h  = _A4_H - _ELEV_MARGIN_T_IN - _ELEV_MARGIN_B_IN
    extra_v      = max(0.0, available_h - content_h_in)
    bot_in       = _ELEV_MARGIN_B_IN + extra_v / 2.0

    bot_bot_f = _fy(bot_in)
    top_bot_f = _fy(bot_in + plot_h_in + _ELEV_GAP_IN)

    ax_bot = fig.add_axes([left_f, bot_bot_f, w_f, h_f])
    ax_top = fig.add_axes([left_f, top_bot_f, w_f, h_f])

    return fig, [ax_top, ax_bot]


# ===========================================================================
# PDF writers
# ===========================================================================


def save_traveltime_pdf(records: list[dict], pdf_path: Path) -> None:
    """Write one travel-time plot per shot, two per A4 page."""
    plot_w = _TT_R - _TT_L
    plot_h = (_TT_T - _TT_B - _TT_GAP) / 2.0
    bottoms = [_TT_B + plot_h + _TT_GAP, _TT_B]

    with pdf_backend.PdfPages(pdf_path) as pdf:
        prev_tid = None
        i = 0
        while i < len(records):
            tid = records[i]['transect_id']
            prev_tid = tid

            fig = plt.figure(figsize=(_A4_W, _A4_H))
            fig.patch.set_facecolor('white')

            for slot in range(2):
                if i >= len(records) or records[i]['transect_id'] != tid:
                    break
                r  = records[i]
                ax = fig.add_axes([_TT_L, bottoms[slot], plot_w, plot_h])
                _draw_traveltime_plot(
                    ax, r['geophone_locs'], r['times_ms'], r['res'],
                    title=f"Refraction ITM – {r['file_name']}")
                i += 1

            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Travel-time PDF saved → {pdf_path}")


def save_elevation_pdf(records: list[dict],
                       elev_data: dict[int, tuple[np.ndarray, np.ndarray]],
                       pdf_path: Path,
                       geotech_by_tid: Optional[
                           dict[int, list[GeotechPoint]]] = None,
                       sheetpile_by_tid: Optional[
                           dict[int, SheetPileLine]] = None,
                       pm_rock_by_tid: Optional[
                           dict[int, list[dict]]] = None,
                       ) -> None:
    """
    Write one elevation + rock-depth profile per transect, two per A4 page.
    Each plot has a true 1:1 data-unit aspect ratio.
    """
    rock_by_tid   = collect_rock_points(records, elev_data)
    transect_ids  = list(dict.fromkeys(r['transect_id'] for r in records))

    # Also include transects that have geotech data but no seismic records
    if geotech_by_tid:
        for tid in geotech_by_tid:
            if tid not in transect_ids:
                transect_ids.append(tid)

    if not transect_ids:
        print("  No transect data – skipping elevation PDF.")
        return

    with pdf_backend.PdfPages(pdf_path) as pdf:
        i = 0
        while i < len(transect_ids):
            fig, axes = _elev_page_layout()

            for ax in axes:
                if i >= len(transect_ids):
                    ax.set_visible(False)
                    continue

                tid = transect_ids[i]
                i  += 1

                elev_x, elev_z = elev_data.get(tid,
                                               (np.array([]), np.array([])))
                if len(elev_x) == 0:
                    print(f"  ⚠  No elevation data for transect {tid}")

                _draw_elevation_plot(
                    ax, elev_x, elev_z,
                    rock_by_tid.get(tid, []),
                    title=f"Transect {tid} — Ground profile & rock depth",
                    geotech_pts    = (geotech_by_tid or {}).get(tid),
                    sheetpile      = (sheetpile_by_tid or {}).get(tid),
                    pm_rock_points = (pm_rock_by_tid or {}).get(tid),
                )

            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Elevation PDF saved → {pdf_path}")


# ===========================================================================
# Plus-Minus travel-time PDF
# ===========================================================================

def _draw_pm_traveltime_plot(ax: plt.Axes, pr: PMPairResult) -> None:
    """T⁺ / T⁻ curves and V2 regression for one PM pair."""
    geo = pr.geo_x

    # T⁺ (depth-related)
    ax.plot(geo, pr.t_plus_ms, 'o-', color='royalblue', markersize=4,
            linewidth=1.2, label='T⁺ (depth)')

    # T⁻ (velocity-related)
    ax.plot(geo, pr.t_minus_ms, 's-', color='darkorange', markersize=4,
            linewidth=1.2, label='T⁻ (V₂)')

    # V2 regression line
    if not np.isnan(pr.v2) and len(geo) >= 2:
        from scipy.stats import linregress as _lr
        slope, intercept, *_ = _lr(geo, pr.t_minus_ms / 1000.0)
        fit_x = np.linspace(geo.min(), geo.max(), 100)
        fit_t = (slope * fit_x + intercept) * 1000.0
        ax.plot(fit_x, fit_t, '--', color='red', linewidth=1.0,
                label=f'T⁻ fit  V₂={pr.v2:.0f} m/s  R²={pr.v2_r2:.3f}')

    # Info box
    valid_d = pr.depths[~np.isnan(pr.depths)]
    info = (f"Pair: {pr.file_a} ↔ {pr.file_b}\n"
            f"Shots: A={pr.shot_a:.1f} m  B={pr.shot_b:.1f} m  "
            f"T_AB={pr.t_ab_ms:.2f} ms\n"
            f"V₂ = {pr.v2:.0f} m/s  (R²={pr.v2_r2:.3f})")
    if len(valid_d):
        info += f"\nDepth: {valid_d.min():.2f}–{valid_d.max():.2f} m  "
        info += f"(mean {valid_d.mean():.2f} m)"
    ax.annotate(info, xy=(0.02, 0.97), xycoords='axes fraction',
                fontsize=7, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='lightyellow', alpha=0.9))

    if pr.warnings:
        ax.annotate('\n'.join(pr.warnings),
                    xy=(0.02, 0.02), xycoords='axes fraction',
                    fontsize=6.5, verticalalignment='bottom',
                    color='darkred',
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='#FFE0E0', edgecolor='red',
                              linewidth=1.2, alpha=0.95))

    ax.set_xlabel('Geophone position (m)', fontsize=8)
    ax.set_ylabel('Time (ms)', fontsize=8)
    ax.set_title(f"PM  T {pr.file_a} ↔ {pr.file_b}", fontsize=8)
    ax.legend(fontsize=6.5, loc='upper right')
    ax.grid(True, alpha=0.4)


def save_pm_traveltime_pdf(
        pm_results: dict[int, list[PMPairResult]],
        pdf_path: Path,
) -> None:
    """Write PM T⁺/T⁻ plots, two per A4 page, grouped by transect."""
    plot_w = _TT_R - _TT_L
    plot_h = (_TT_T - _TT_B - _TT_GAP) / 2.0
    bottoms = [_TT_B + plot_h + _TT_GAP, _TT_B]

    # Flatten to ordered list
    pairs: list[tuple[int, PMPairResult]] = []
    for tid in sorted(pm_results):
        for pr in pm_results[tid]:
            pairs.append((tid, pr))

    if not pairs:
        print("  No PM pairs — skipping PM traveltime PDF.")
        return

    with pdf_backend.PdfPages(pdf_path) as pdf:
        prev_tid = None
        i = 0
        while i < len(pairs):
            tid, _ = pairs[i]
            prev_tid = tid

            fig = plt.figure(figsize=(_A4_W, _A4_H))
            fig.patch.set_facecolor('white')

            for slot in range(2):
                if i >= len(pairs) or pairs[i][0] != tid:
                    break
                _, pr = pairs[i]
                ax = fig.add_axes([_TT_L, bottoms[slot], plot_w, plot_h])
                _draw_pm_traveltime_plot(ax, pr)
                i += 1

            pdf.savefig(fig)
            plt.close(fig)

    print(f"  PM travel-time PDF saved → {pdf_path}")