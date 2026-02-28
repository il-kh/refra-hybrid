"""
Seismic refraction analysis using the Intercept-Time Method (ITM).

Workflow
--------
1. Read first-arrival picks from .vs files grouped by transect.
2. For each shot, split the spread into left / right wings and fit a
   two-segment linear travel-time curve (V1 = direct wave, V2 = refracted).
3. Derive depth-to-refractor from the intercept-time formula.
4. Export results to:
   - PDF  : travel-time plots  (refra_itm_traveltimes.pdf)
   - PDF  : elevation + rock-depth profiles  (sections.pdf)
   - XLSX : tabular summary  (refra_itm_results.xlsx)
"""


from typing import Optional

import numpy as np
from scipy.stats import linregress

import config
from elevation import compute_offsets, interpolate_elevation

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
WingResult = Optional[dict]
ShotResult = dict


# ===========================================================================
# Breakpoint detection
# ===========================================================================

def _ssr_breakpoint(wing_offsets: np.ndarray,
                    wing_times_sec: np.ndarray,
                    min_points: int) -> int:
    """Fallback: minimum-SSR two-segment split."""
    n = len(wing_offsets)
    best_ssr, best_idx = np.inf, n - 2

    for idx in range(min_points, n - min_points + 1):
        v2_off, v2_t = wing_offsets[idx:], wing_times_sec[idx:]
        if len(v2_off) < 2:
            continue
        sl, *_ = linregress(v2_off, v2_t)
        if sl <= 0:
            continue
        v1_fit = np.polyval(
            np.polyfit(wing_offsets[:idx], wing_times_sec[:idx], 1),
            wing_offsets[:idx])
        v2_fit = np.polyval(np.polyfit(v2_off, v2_t, 1), v2_off)
        ssr = (np.sum((wing_times_sec[:idx] - v1_fit) ** 2) +
               np.sum((v2_t - v2_fit) ** 2))
        if ssr < best_ssr:
            best_ssr, best_idx = ssr, idx
    return best_idx


def find_breakpoint_on_wing(wing_offsets: np.ndarray,
                             wing_times_sec: np.ndarray,
                             min_points: int = config.MIN_SEGMENT_POINTS) -> int:
    """
    Locate the V1→V2 breakpoint on one wing.

    1. Grow a far-end seed until its regression slope is positive.
    2. Extend leftward while slope stays positive and ≤ seed × tolerance.
    3. Fall back to SSR split if no positive slope exists.
    """
    n = len(wing_offsets)
    if n < 2 * min_points:
        return _ssr_breakpoint(wing_offsets, wing_times_sec, min_points)

    seed_start: Optional[int]   = None
    seed_slope: Optional[float] = None

    for seed_size in range(min_points, n + 1):
        start   = n - seed_size
        seg_off = wing_offsets[start:]
        seg_t   = wing_times_sec[start:]
        if len(seg_off) < 2:
            continue
        sl, *_ = linregress(seg_off, seg_t)
        if sl > 0:
            seed_start, seed_slope = start, sl
            break

    if seed_slope is None:
        print("    ⚠  No positive V2 slope on wing – SSR fallback.")
        return _ssr_breakpoint(wing_offsets, wing_times_sec, min_points)

    best_start = seed_start
    for ext in range(seed_start - 1, min_points - 1, -1):
        sl, *_ = linregress(wing_offsets[ext:], wing_times_sec[ext:])
        if sl <= 0 or sl > seed_slope * config.BP_SLOPE_TOLERANCE:
            break
        best_start = ext
    return best_start


# ===========================================================================
# Per-wing refraction fit
# ===========================================================================

def _fit_wing(wing_geo: np.ndarray,
              wing_offsets: np.ndarray,
              wing_times_sec: np.ndarray,
              full_geo: np.ndarray,
              full_offsets: np.ndarray,
              full_times_sec: np.ndarray,
              side: str,
              min_points: int = config.MIN_SEGMENT_POINTS) -> WingResult:
    """
    Two-segment linear fit on one wing.

    V1 is regressed on **wing-local** near-offset points only (not the
    full spread), eliminating cross-wing contamination from the other
    wing's refracted arrivals.  When the wing-local V1 fit is unreliable
    (non-positive slope, out-of-range velocity, or too few points) the
    CPTu prior ``config.V1_PRIOR`` is used instead.

    Returns a parameter dict with quality metrics, or None if the wing
    is too short.
    """
    if len(wing_offsets) < 2 * min_points:
        return None

    v2_local_start = find_breakpoint_on_wing(
        wing_offsets, wing_times_sec, min_points)

    # ── Build full-array masks (for plotting) ──────────────────────────
    wing_full_indices = np.array(
        [np.where(full_geo == g)[0][0] for g in wing_geo])
    v2_mask = np.zeros(len(full_geo), dtype=bool)
    v2_mask[wing_full_indices[v2_local_start:]] = True
    v1_mask = np.zeros(len(full_geo), dtype=bool)
    v1_mask[wing_full_indices[:v2_local_start]] = True

    # ── V2 regression (wing-local far-offset segment) ─────────────────
    v2_off = wing_offsets[v2_local_start:]
    v2_t   = wing_times_sec[v2_local_start:]
    slope2, intercept2, r_v2, _, _ = linregress(v2_off, v2_t)

    if slope2 <= 0:
        print(f"    ⚠  [{side}] V2 slope non-positive "
              f"({slope2:.6f}) – unreliable.")
        slope2 = abs(slope2)

    # ── V1 regression (wing-local near-offset segment) ────────────────
    v1_off = wing_offsets[:v2_local_start]
    v1_t   = wing_times_sec[:v2_local_start]
    v1_constrained = False
    r_v1 = float('nan')

    if len(v1_off) >= 2:
        slope1, intercept1, r_v1, _, _ = linregress(v1_off, v1_t)
        v1_from_fit = 1.0 / slope1 if slope1 > 0 else float('inf')
        if slope1 <= 0 or not (config.V1_MIN <= v1_from_fit <= config.V1_MAX):
            print(f"    ⚠  [{side}] V1 = {v1_from_fit:.0f} m/s unreliable "
                  f"→ using CPTu prior {config.V1_PRIOR:.0f} m/s")
            slope1 = 1.0 / config.V1_PRIOR
            intercept1 = float(np.mean(v1_t - slope1 * v1_off))
            v1_constrained = True
            r_v1 = float('nan')
    else:
        print(f"    ⚠  [{side}] Only {len(v1_off)} V1 point(s) "
              f"→ using CPTu prior {config.V1_PRIOR:.0f} m/s")
        slope1 = 1.0 / config.V1_PRIOR
        if len(v1_off) > 0:
            intercept1 = float(np.mean(v1_t - slope1 * v1_off))
        else:
            intercept1 = 0.0
        v1_constrained = True

    v1, v2 = 1.0 / slope1, 1.0 / slope2
    t_i    = intercept2
    depth  = (
        (t_i * v1 * v2) / (2.0 * np.sqrt(v2 ** 2 - v1 ** 2))
        if v2 >= config.V2_ROCK_MIN and (v2 ** 2 - v1 ** 2) > 0 and t_i > 0
        else float('nan')
    )

    # ── Quality metrics ────────────────────────────────────────────────
    v2_count   = int(v2_mask.sum())
    v1_count   = int(v1_mask.sum())
    max_offset = float(wing_offsets.max())
    r2_v1      = float(r_v1 ** 2) if not np.isnan(r_v1) else float('nan')
    r2_v2      = float(r_v2 ** 2)

    if v2 > v1 and not np.isnan(t_i) and t_i > 0:
        xc_est   = t_i * v1 * v2 / (v2 - v1)
        xc_ratio = max_offset / xc_est if xc_est > 0 else 0.0
    else:
        xc_est   = float('nan')
        xc_ratio = 0.0

    return dict(
        side           = side,
        v1             = v1,
        v2             = v2,
        t_i_ms         = t_i * 1000.0,
        depth          = depth,
        slope1         = slope1,
        intercept1     = intercept1,
        slope2         = slope2,
        intercept2     = intercept2,
        bp_offset      = float(v2_off.min()),
        bp_geo         = 0.0,
        v2_mask        = v2_mask,
        v1_mask        = v1_mask,
        # Quality metrics
        v1_constrained = v1_constrained,
        v2_count       = v2_count,
        v1_count       = v1_count,
        max_offset     = max_offset,
        r2_v1          = r2_v1,
        r2_v2          = r2_v2,
        xc_est         = xc_est,
        xc_ratio       = xc_ratio,
    )


# ===========================================================================
# Plausibility checks
# ===========================================================================

def _check_wing(wing: WingResult, label: str, warnings: list[str]) -> None:
    """Append warning strings for out-of-range velocities and quality issues."""
    if wing is None:
        return
    v1, v2 = wing['v1'], wing['v2']
    checks = [
        (not (config.V1_MIN <= v1 <= config.V1_MAX),
         f"⚠ {label} V1 = {v1:.0f} m/s outside "
         f"[{config.V1_MIN:.0f}–{config.V1_MAX:.0f}]"),
        (not (config.V2_MIN <= v2 <= config.V2_MAX),
         f"⚠ {label} V2 = {v2:.0f} m/s outside "
         f"[{config.V2_MIN:.0f}–{config.V2_MAX:.0f}]"),
        (v2 <= v1,
         f"⚠ {label} V2 ≤ V1 – refraction condition not met"),
        (wing.get('t_i_ms', 0) <= 0,
         f"⚠ {label} t_i = {wing.get('t_i_ms', 0):.1f} ms ≤ 0 "
         f"– non-physical intercept time"),
        (wing.get('v2_count', 0) < config.MIN_V2_POINTS,
         f"⚠ {label} only {wing.get('v2_count', 0)} V2 points "
         f"(need ≥ {config.MIN_V2_POINTS})"),
        (0 < wing.get('xc_ratio', 0) < config.XC_COVERAGE_WARN,
         f"⚠ {label} max offset covers only "
         f"{wing.get('xc_ratio', 0):.0%} of crossover distance"),
        (wing.get('v1_constrained', False),
         f"ℹ {label} V1 constrained to CPTu prior "
         f"{config.V1_PRIOR:.0f} m/s"),
    ]
    for condition, msg in checks:
        if condition:
            print(f"    {msg}")
            warnings.append(msg)


# ===========================================================================
# Shot analyser (ITM = Intercept Time Method)
# ===========================================================================

def analyse_shot_itm(geophone_locs: np.ndarray,
                 times_ms: np.ndarray,
                 shot_pos: Optional[float] = None) -> ShotResult:
    """
    Full ITM analysis for one shot record.

    Returns dict with keys: right, left, true_shot_loc, offsets, warnings.
    """
    times_sec = times_ms / 1000.0
    offsets, true_shot_loc, shot_idx = compute_offsets(
        geophone_locs, times_ms, shot_pos=shot_pos)

    print(f"  Shot pos (header) : {shot_pos} m  |  "
          f"nearest geo : {geophone_locs[shot_idx]:.1f} m  |  "
          f"true shot : {true_shot_loc:.1f} m")

    left_idx  = np.where(geophone_locs < true_shot_loc)[0]
    right_idx = np.where(geophone_locs > true_shot_loc)[0]
    left_idx_s  = left_idx[np.argsort(offsets[left_idx])]
    right_idx_s = right_idx[np.argsort(offsets[right_idx])]

    print(f"  Wings → left : {len(left_idx)} pts  |  "
          f"right : {len(right_idx)} pts")

    warnings: list[str] = []

    def _process_wing(idx_sorted, side: str, sign: int) -> WingResult:
        if len(idx_sorted) == 0:
            print(f"  — {side.capitalize()} wing: no geophones – skipped.")
            return None
        print(f"  — {side.capitalize()} wing —")
        result = _fit_wing(
            wing_geo       = geophone_locs[idx_sorted],
            wing_offsets   = offsets[idx_sorted],
            wing_times_sec = times_sec[idx_sorted],
            full_geo       = geophone_locs,
            full_offsets   = offsets,
            full_times_sec = times_sec,
            side           = side,
        )
        if result is None:
            print("    Too few points – skipped.")
            return None
        result['bp_geo'] = true_shot_loc + sign * result['bp_offset']
        xc_str = (f"{result['xc_ratio']:.0%}"
                  if result['xc_ratio'] > 0 else "N/A")
        constrained_tag = " [V1=CPTu]" if result.get('v1_constrained') else ""
        print(f"    V1={result['v1']:.1f}  V2={result['v2']:.1f} m/s  "
              f"t_i={result['t_i_ms']:.2f} ms  "
              f"z={result['depth']:.2f} m  "
              f"BP@{result['bp_geo']:.1f} m  "
              f"V2pts={result['v2_count']}  "
              f"xc_cov={xc_str}"
              f"{constrained_tag}")
        _check_wing(result, side.capitalize(), warnings)
        return result

    return dict(
        right         = _process_wing(right_idx_s, 'right', +1),
        left          = _process_wing(left_idx_s,  'left',  -1),
        true_shot_loc = true_shot_loc,
        offsets       = offsets,
        warnings      = warnings,
    )


# ===========================================================================
# Rock-depth collector
# ===========================================================================

def collect_rock_points(records: list[dict],
                        elev_data: dict[int, tuple[np.ndarray, np.ndarray]]
                        ) -> dict[int, list[dict]]:
    """
    Convert ITM depths to absolute rock elevations per transect.

    Only wings with depth > 0 and v2 >= V2_ROCK_MIN are included.
    Uses the in-memory *elev_data* dict instead of on-disk files.
    """
    rock: dict[int, list[dict]] = {}

    for rec in records:
        tid = rec['transect_id']
        res = rec['res']

        if tid not in elev_data:
            continue
        elev_x, elev_z = elev_data[tid]

        for wing in filter(None, (res.get('right'), res.get('left'))):
            depth = wing.get('depth', float('nan'))
            v2    = wing.get('v2',    float('nan'))
            if np.isnan(depth) or depth <= 0 or v2 < config.V2_ROCK_MIN:
                continue

            x      = res['true_shot_loc']
            z_surf = interpolate_elevation(elev_x, elev_z, x)
            if z_surf is None:
                continue

            rock.setdefault(tid, []).append(dict(
                x_geo     = x,
                z_surface = z_surf,
                depth     = depth,
                z_rock    = z_surf - depth,
                side      = wing['side'],
                v2        = v2,
            ))

    return rock