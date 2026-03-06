"""
analysis_plusminus.py  –  Plus-Minus (Hagedoorn) Method
========================================================
Derives a **continuous depth-to-refractor profile** from reciprocal
shot pairs.  Unlike the ITM (single depth per shot), the Plus-Minus
method resolves z(x) at every geophone by combining forward and
reverse travel times.

Theory (Hagedoorn, 1959)
------------------------
Given shot A (left) and shot B (right) with reciprocal time T_AB:

    T⁺ᵢ  =  T_Ai  +  T_Bi  −  T_AB        (plus time → depth)
    T⁻ᵢ  =  T_Ai  −  T_Bi  +  T_AB        (minus time → V2)

In the refraction overlap zone (where both shots produce head-wave
first arrivals):

    V₂  =  2 / slope(T⁻ vs x)             (true refractor velocity)
    zᵢ  =  T⁺ᵢ · V₁ · V₂ / (2 √(V₂² − V₁²))   (depth at geophone i)
Three-layer limitation
----------------------
The standard PM two-layer analysis resolves only the SHALLOWEST
velocity contrast that produces head-wave first arrivals.  On this
site the subsurface has at least three layers:

    Layer 1  V₁ ≈ 350 m/s   (loose alluvium)            0–3 m
    Layer 2  V₂ ≈ 870 m/s   (stiff clay / dense sand)   3–7+ m
    Layer 3  V₃ > 2000 m/s  (boulders / bedrock)        7–19 m

For geophones between two shots on the array, the first arrivals
are refracted along the Layer-1/Layer-2 boundary.  The deeper
Layer-2/Layer-3 (rock) head waves arrive LATER and are invisible
to first-arrival picking.  Therefore **the PM depths reported here
represent the shallow refractor (compaction boundary), NOT the
rock/boulder layer**.  Geotech DPL/CPTu data should be used for
the rock surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import linregress

import config
from elevation import interpolate_elevation


# ═══════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ShotRecord:
    """One shot's travel-time data in the standard geophone frame."""
    file_name: str
    transect_id: int
    shot_pos: float                  # true shot position (m)
    geo_x: np.ndarray                # geophone positions (m)
    times_ms: np.ndarray             # first-arrival times (ms)


@dataclass
class PMPairResult:
    """Result for one reciprocal shot pair."""
    tid: int
    file_a: str                      # left shot filename
    file_b: str                      # right shot filename
    shot_a: float                    # position of shot A (m)
    shot_b: float                    # position of shot B (m)
    t_ab_ms: float                   # reciprocal time A→B (ms)

    # Shared geophone positions in the overlap zone
    geo_x: np.ndarray = field(default_factory=lambda: np.array([]))

    # Raw plus / minus times (ms) at each geo_x position
    t_plus_ms: np.ndarray = field(default_factory=lambda: np.array([]))
    t_minus_ms: np.ndarray = field(default_factory=lambda: np.array([]))

    # Derived quantities
    v2: float = float('nan')         # true refractor velocity (m/s)
    v2_r2: float = float('nan')      # R² of the T⁻ regression
    v1_used: float = float('nan')    # V1 used for depth conversion
    depths: np.ndarray = field(default_factory=lambda: np.array([]))
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _normalize_geo(shot: ShotRecord) -> ShotRecord:
    """
    If geophone coordinates are shifted (e.g. stored as shot_pos + x),
    shift them back to the standard [0, 23] frame.
    """
    geo = shot.geo_x
    if len(geo) < 2:
        return shot
    # If the first geophone is far from 0, apply shift
    if abs(geo.min() - 0.0) > config.PM_GEO_TOLERANCE:
        shift = geo.min()
        return ShotRecord(
            file_name=shot.file_name,
            transect_id=shot.transect_id,
            shot_pos=shot.shot_pos,
            geo_x=geo - shift,
            times_ms=shot.times_ms,
        )
    return shot


def _match_geophones(geo_a: np.ndarray, geo_b: np.ndarray,
                     tol: float = config.PM_GEO_TOLERANCE
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find common geophone positions between two shots.

    Returns (common_x, idx_a, idx_b) where idx_a[k] and idx_b[k]
    index into the original arrays for the k-th matched position.
    """
    common_x, idx_a, idx_b = [], [], []
    for i, xa in enumerate(geo_a):
        diffs = np.abs(geo_b - xa)
        j = int(np.argmin(diffs))
        if diffs[j] <= tol:
            common_x.append(0.5 * (xa + geo_b[j]))  # average position
            idx_a.append(i)
            idx_b.append(j)
    return np.array(common_x), np.array(idx_a), np.array(idx_b)


def _extrapolate_time(geo_x: np.ndarray, times_ms: np.ndarray,
                      target_x: float, n_fit: int = 5) -> float:
    """
    Extrapolate travel time to *target_x* using a linear fit on the
    nearest *n_fit* points on the correct side of the array.

    For targets beyond the array, uses the outermost points (which
    should be on the refracted branch for far targets).
    """
    if target_x <= geo_x.min():
        # Target is to the left → use leftmost points
        sel = np.argsort(geo_x)[:n_fit]
    elif target_x >= geo_x.max():
        # Target is to the right → use rightmost points
        sel = np.argsort(geo_x)[-n_fit:]
    else:
        # Target is inside the array → use nearest points
        dists = np.abs(geo_x - target_x)
        sel = np.argsort(dists)[:n_fit]

    slope, intercept, *_ = linregress(geo_x[sel], times_ms[sel])
    return float(slope * target_x + intercept)


def _estimate_t_ab(shot_a: ShotRecord, shot_b: ShotRecord
                   ) -> tuple[float, list[str]]:
    """
    Estimate the reciprocal travel time T_AB (ms) between shots A and B.

    Uses both directions (A→x_B and B→x_A) and averages for robustness.
    Returns (t_ab_ms, warnings).
    """
    warnings: list[str] = []

    t_a_to_b = _extrapolate_time(shot_a.geo_x, shot_a.times_ms,
                                 shot_b.shot_pos)
    t_b_to_a = _extrapolate_time(shot_b.geo_x, shot_b.times_ms,
                                 shot_a.shot_pos)

    # Both must be positive
    if t_a_to_b <= 0 or t_b_to_a <= 0:
        warnings.append(f"⚠ T_AB extrapolation non-positive "
                        f"(A→B={t_a_to_b:.1f}, B→A={t_b_to_a:.1f})")

    t_ab = 0.5 * (t_a_to_b + t_b_to_a)
    delta = abs(t_a_to_b - t_b_to_a)
    mean_t = 0.5 * (abs(t_a_to_b) + abs(t_b_to_a))

    # Reciprocity check
    if mean_t > 0 and delta / mean_t > 0.20:
        warnings.append(f"⚠ T_AB reciprocity violation: "
                        f"|Δ|={delta:.1f} ms ({delta/mean_t*100:.0f}%)")

    print(f"    T_AB estimate: A→B={t_a_to_b:.2f}  B→A={t_b_to_a:.2f}  "
          f"avg={t_ab:.2f} ms  (|Δ|={delta:.2f} ms)")
    return t_ab, warnings


# ═══════════════════════════════════════════════════════════════════
# Overlap zone detection
# ═══════════════════════════════════════════════════════════════════

def _find_refraction_zone(geo_x: np.ndarray,
                          t_minus_ms: np.ndarray,
                          trim: int = config.PM_EDGE_TRIM
                          ) -> tuple[np.ndarray, int, int]:
    """
    Identify the refraction overlap zone from T⁻ linearity.

    In the zone where both shots produce refracted first arrivals,
    T⁻ should be linear in x with slope = 2/V₂ (small slope, high V₂).
    In the direct-wave zone T⁻ is also linear but with slope ≈ 2/V₁
    (large slope, low V₂).

    We use a sliding-window approach: keep the widest zone where
    R² ≥ PM_TMINUS_R2_MIN **and** the derived V₂ > V₁ × PM_V2_FLOOR_FACTOR.

    Returns (zone_mask, start_idx, end_idx).
    """
    n = len(geo_x)
    if n < config.PM_MIN_OVERLAP:
        return np.ones(n, dtype=bool), 0, n

    v1 = config.PM_V1_FOR_DEPTH
    v2_floor = v1 * config.PM_V2_FLOOR_FACTOR

    best_span = 0
    best_start = 0
    best_end = n

    for start in range(n):
        for end in range(start + config.PM_MIN_OVERLAP, n + 1):
            seg_x = geo_x[start:end]
            seg_t = t_minus_ms[start:end]
            if len(seg_x) < 3:
                continue
            slope, _, r, _, _ = linregress(seg_x, seg_t / 1000.0)
            r2 = r ** 2
            v2_seg = 2.0 / slope if slope > 0 else float('inf')
            span = end - start
            if (r2 >= config.PM_TMINUS_R2_MIN
                    and v2_seg >= v2_floor
                    and span > best_span):
                best_span = span
                best_start = start
                best_end = end

    if best_span == 0:
        # No zone with V₂ above floor – fall back to full array
        mask = np.ones(n, dtype=bool)
        return mask, 0, n

    # Apply edge trim
    trimmed_start = min(best_start + trim, best_end - config.PM_MIN_OVERLAP)
    trimmed_end = max(best_end - trim, trimmed_start + config.PM_MIN_OVERLAP)

    mask = np.zeros(n, dtype=bool)
    mask[trimmed_start:trimmed_end] = True
    return mask, trimmed_start, trimmed_end


# ═══════════════════════════════════════════════════════════════════
# Core Plus-Minus analysis for one shot pair
# ═══════════════════════════════════════════════════════════════════

def analyse_pair(shot_a: ShotRecord, shot_b: ShotRecord) -> Optional[PMPairResult]:
    """
    Perform the Plus-Minus analysis for one reciprocal shot pair.

    Parameters
    ----------
    shot_a : ShotRecord   left shot  (shot_a.shot_pos < shot_b.shot_pos)
    shot_b : ShotRecord   right shot

    Returns
    -------
    PMPairResult or None if the pair is unusable.
    """
    tid = shot_a.transect_id
    warnings: list[str] = []

    # Ensure A is left, B is right
    if shot_a.shot_pos > shot_b.shot_pos:
        shot_a, shot_b = shot_b, shot_a

    print(f"\n  ── PM pair: {shot_a.file_name} (A={shot_a.shot_pos:.1f} m) ↔ "
          f"{shot_b.file_name} (B={shot_b.shot_pos:.1f} m) ──")

    sep = shot_b.shot_pos - shot_a.shot_pos
    if sep < config.PM_MIN_SHOT_SEP:
        print(f"    ⚠  Shot separation {sep:.1f} m < {config.PM_MIN_SHOT_SEP} m – skipped.")
        return None

    # Normalize geophone coordinates
    a = _normalize_geo(shot_a)
    b = _normalize_geo(shot_b)

    # Match common geophones
    common_x, idx_a, idx_b = _match_geophones(a.geo_x, b.geo_x)
    if len(common_x) < config.PM_MIN_OVERLAP:
        print(f"    ⚠  Only {len(common_x)} common geophones – skipped.")
        return None

    # Sort by position
    order = np.argsort(common_x)
    common_x = common_x[order]
    idx_a = idx_a[order]
    idx_b = idx_b[order]

    # ── Between-shots geometric filter ─────────────────────────────
    # The T⁺/T⁻ equations are only valid when the geophone sits
    # between the two shots, so that BOTH shots can illuminate it
    # with refracted (head-wave) energy.
    margin = config.PM_SHOT_MARGIN
    between = ((common_x > a.shot_pos + margin) &
               (common_x < b.shot_pos - margin))
    n_between = int(between.sum())
    if n_between < config.PM_MIN_OVERLAP:
        print(f"    ⚠  Only {n_between} geophones between shots "
              f"[{a.shot_pos + margin:.1f}, {b.shot_pos - margin:.1f}] – skipped.")
        return None
    if n_between < len(common_x):
        print(f"    Between-shots filter: {len(common_x)} → {n_between} geophones "
              f"(x ∈ [{a.shot_pos + margin:.1f}, {b.shot_pos - margin:.1f}])")
    common_x = common_x[between]
    idx_a = idx_a[between]
    idx_b = idx_b[between]

    t_a = a.times_ms[idx_a]   # T_Ai at common geophones (ms)
    t_b = b.times_ms[idx_b]   # T_Bi at common geophones (ms)

    # Reciprocal time
    t_ab, t_ab_warnings = _estimate_t_ab(a, b)
    warnings.extend(t_ab_warnings)

    # ── Plus and Minus times (ms) ──────────────────────────────────
    t_plus  = t_a + t_b - t_ab      # related to depth
    t_minus = t_a - t_b + t_ab      # related to V2

    print(f"    Common geophones: {len(common_x)}  "
          f"x = [{common_x.min():.1f}, {common_x.max():.1f}] m")
    print(f"    T⁺ range: [{t_plus.min():.2f}, {t_plus.max():.2f}] ms")
    print(f"    T⁻ range: [{t_minus.min():.2f}, {t_minus.max():.2f}] ms")

    # ── Find refraction overlap zone via T⁻ linearity ─────────────
    zone_mask, z_start, z_end = _find_refraction_zone(common_x, t_minus)
    n_zone = int(zone_mask.sum())

    # Check if an actual refracted zone was found (vs. full-array fallback)
    is_fallback = (z_start == 0 and z_end == len(common_x)
                   and n_zone == len(common_x))

    if n_zone < config.PM_MIN_OVERLAP:
        msg = (f"⚠ Only {n_zone} geophones in refraction zone "
               f"(need ≥ {config.PM_MIN_OVERLAP})")
        print(f"    {msg}")
        warnings.append(msg)
        # Use all geophones as fallback
        zone_mask[:] = True
        z_start, z_end = 0, len(common_x)
        n_zone = len(common_x)
        is_fallback = True

    zone_x = common_x[zone_mask]
    zone_t_minus = t_minus[zone_mask]
    zone_t_plus = t_plus[zone_mask]

    if is_fallback:
        v2_floor = config.PM_V1_FOR_DEPTH * config.PM_V2_FLOOR_FACTOR
        msg = (f"⚠ No refraction zone found (V₂ < {v2_floor:.0f} m/s "
               f"everywhere) – direct-wave dominated")
        print(f"    {msg}")
        warnings.append(msg)

    print(f"    Refraction zone: geos {z_start}–{z_end}  "
          f"x=[{zone_x.min():.1f}, {zone_x.max():.1f}] m  "
          f"({n_zone} pts){' [FALLBACK]' if is_fallback else ''}")

    # ── V2 from T⁻ regression ─────────────────────────────────────
    #   T⁻ᵢ = 2xᵢ/V₂ + const  →  slope = 2/V₂  →  V₂ = 2/slope
    slope_minus, intercept_minus, r_minus, _, _ = linregress(
        zone_x, zone_t_minus / 1000.0)   # convert to seconds for slope

    r2_minus = r_minus ** 2
    if slope_minus > 0:
        v2 = 2.0 / slope_minus
    else:
        v2 = float('nan')
        warnings.append(f"⚠ T⁻ slope non-positive ({slope_minus:.6f})")

    print(f"    T⁻ regression: slope={slope_minus:.6f} s/m  "
          f"R²={r2_minus:.4f}  → V₂ = {v2:.0f} m/s")

    if r2_minus < config.PM_TMINUS_R2_MIN:
        warnings.append(f"⚠ T⁻ R²={r2_minus:.3f} < {config.PM_TMINUS_R2_MIN}")

    if not np.isnan(v2) and not (config.V2_MIN <= v2 <= config.V2_MAX):
        warnings.append(f"⚠ V₂ = {v2:.0f} m/s outside "
                        f"[{config.V2_MIN:.0f}–{config.V2_MAX:.0f}]")

    # ── Depth at each geophone from T⁺ ────────────────────────────
    v1 = config.PM_V1_FOR_DEPTH
    t_plus_sec = zone_t_plus / 1000.0

    if not np.isnan(v2) and v2 > v1:
        cos_ic = np.sqrt(v2 ** 2 - v1 ** 2) / v2
        depths = t_plus_sec * v1 / (2.0 * cos_ic)
        # Mask out non-physical negative depths (from bad T⁺ or T_AB)
        neg_mask = depths < 0
        if neg_mask.any():
            n_neg = int(neg_mask.sum())
            warnings.append(f"⚠ {n_neg} negative depth(s) masked "
                            f"(T⁺ issue)")
            depths[neg_mask] = float('nan')
    else:
        depths = np.full(len(zone_x), float('nan'))
        if np.isnan(v2) or v2 <= v1:
            warnings.append("⚠ Cannot compute depths (V₂ ≤ V₁ or NaN)")

    # Log depth summary
    valid_depths = depths[~np.isnan(depths)]
    if len(valid_depths) > 0:
        print(f"    Depths: min={valid_depths.min():.2f}  "
              f"max={valid_depths.max():.2f}  "
              f"mean={valid_depths.mean():.2f} m  "
              f"(V₁={v1:.0f}, V₂={v2:.0f} m/s)")
    else:
        print("    Depths: all NaN")

    return PMPairResult(
        tid=tid,
        file_a=shot_a.file_name,
        file_b=shot_b.file_name,
        shot_a=shot_a.shot_pos,
        shot_b=shot_b.shot_pos,
        t_ab_ms=t_ab,
        geo_x=zone_x,
        t_plus_ms=zone_t_plus,
        t_minus_ms=zone_t_minus,
        v2=v2,
        v2_r2=r2_minus,
        v1_used=v1,
        depths=depths,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════
# Transect-level analysis
# ═══════════════════════════════════════════════════════════════════

def _build_shot_records(files: list, tid: int) -> list[ShotRecord]:
    """Read .vs files and build ShotRecord objects."""
    from io_helpers import read_vs_file
    from elevation import compute_offsets

    records = []
    for path in sorted(files):
        shot_pos, geo_x, times_ms = read_vs_file(path)
        # Use compute_offsets to resolve missing shot positions;
        # we only need true_shot_loc from it.
        _, true_shot, _ = compute_offsets(geo_x, times_ms, shot_pos=shot_pos)
        records.append(ShotRecord(
            file_name=path.name,
            transect_id=tid,
            shot_pos=true_shot,
            geo_x=geo_x,
            times_ms=times_ms,
        ))
    return records


def analyse_transect_pm(files: list, tid: int) -> list[PMPairResult]:
    """
    Run Plus-Minus analysis on all valid reciprocal shot pairs in a transect.

    Pairing strategy:
    - Every shot to the left of the midpoint is paired with every shot
      to the right, subject to PM_MIN_SHOT_SEP.
    - Pairs are sorted by separation (widest first → best overlap).
    """
    shots = _build_shot_records(files, tid)
    if len(shots) < 2:
        print(f"  Transect {tid}: fewer than 2 shots – PM analysis skipped.")
        return []

    # Sort by shot position
    shots.sort(key=lambda s: s.shot_pos)

    # Form all left-right pairs with sufficient separation
    pairs: list[tuple[ShotRecord, ShotRecord]] = []
    for i, a in enumerate(shots):
        for b in shots[i + 1:]:
            if b.shot_pos - a.shot_pos >= config.PM_MIN_SHOT_SEP:
                pairs.append((a, b))

    # Sort by separation (widest first)
    pairs.sort(key=lambda p: p[1].shot_pos - p[0].shot_pos, reverse=True)

    print(f"\n{'=' * 60}")
    print(f"  Transect {tid}: {len(shots)} shots, "
          f"{len(pairs)} valid PM pairs")
    print(f"{'=' * 60}")

    results: list[PMPairResult] = []
    for a, b in pairs:
        try:
            res = analyse_pair(a, b)
            if res is not None:
                results.append(res)
        except Exception as exc:
            print(f"    ⚠  Error in pair {a.file_name}↔{b.file_name}: {exc}")

    print(f"  Transect {tid}: {len(results)} PM pair(s) produced results.")
    return results


# ═══════════════════════════════════════════════════════════════════
# Refractor-depth collector
# ═══════════════════════════════════════════════════════════════════

def collect_pm_rock_points(
        pm_results: dict[int, list[PMPairResult]],
        elev_data: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[int, list[dict]]:
    """
    Convert PM depth profiles to absolute refractor elevations, applying
    strict quality filters and aggregating overlapping estimates.

    **Important**: In a three-layer subsurface the PM two-layer analysis
    detects the SHALLOWEST velocity contrast (V₂ ≈ 800–1000 m/s), which
    is the compaction boundary or the base of loose fill — NOT the
    rock/boulder layer (see module docstring).  The returned 'z_rock'
    key is kept for backward compatibility but represents the
    refractor elevation, not rock.

    Filtering pipeline
    ------------------
    1. **Pair-level**: V₂ in [PM_V2_ACCEPT_MIN, PM_V2_ACCEPT_MAX],
       T⁻ R² ≥ threshold, no refraction-zone fallback.
    2. **Point-level**: depth in [PM_DEPTH_MIN, PM_DEPTH_MAX]
       (relaxed near river).
    3. **Aggregation**: median depth per x-bin (PM_AGG_BIN_WIDTH)
       collapses the cloud of overlapping pairs into a single profile.

    Returns dict  tid → list of
        {x_geo, z_surface, depth, z_rock, n_pairs, v2_median}.
    """
    rock: dict[int, list[dict]] = {}

    for tid, pair_list in pm_results.items():
        if tid not in elev_data:
            continue
        elev_x, elev_z = elev_data[tid]

        # ── Collect individual (x, depth, v2) from accepted pairs ──
        raw: list[tuple[float, float, float]] = []   # (x, depth, v2)
        n_total = len(pair_list)
        n_v2_ok = 0
        n_r2_ok = 0
        n_zone_ok = 0

        for pr in pair_list:
            if np.isnan(pr.v2):
                continue
            # Pair-level: V₂ range
            if not (config.PM_V2_ACCEPT_MIN <= pr.v2 <= config.PM_V2_ACCEPT_MAX):
                continue
            n_v2_ok += 1

            # Pair-level: T⁻ R²
            if pr.v2_r2 < config.PM_TMINUS_R2_MIN:
                continue
            n_r2_ok += 1

            # Pair-level: genuine refraction zone (reject fallbacks)
            if any('No refraction zone' in w for w in pr.warnings):
                continue
            n_zone_ok += 1

            # Point-level: depth range (spatially aware)
            for x, d in zip(pr.geo_x, pr.depths):
                if np.isnan(d) or d <= 0:
                    continue
                # Minimum-depth filter
                d_min = (config.PM_DEPTH_MIN_RIVER
                         if x > config.PM_RIVER_X
                         else config.PM_DEPTH_MIN)
                if d < d_min or d > config.PM_DEPTH_MAX:
                    continue
                raw.append((float(x), float(d), float(pr.v2)))

        if not raw:
            print(f"  Transect {tid}: {n_total} PM pairs → "
                  f"V₂ ok {n_v2_ok} → R² ok {n_r2_ok} → "
                  f"zone ok {n_zone_ok} → 0 depth points after filtering")
            continue

        # ── Aggregate by x-position (median) ──────────────────────
        xs  = np.array([p[0] for p in raw])
        ds  = np.array([p[1] for p in raw])
        v2s = np.array([p[2] for p in raw])

        bin_w = config.PM_AGG_BIN_WIDTH
        x_min = np.floor(xs.min() / bin_w) * bin_w
        x_max = np.ceil(xs.max() / bin_w) * bin_w
        bin_edges = np.arange(x_min, x_max + bin_w, bin_w)

        tid_points: list[dict] = []
        for i in range(len(bin_edges) - 1):
            mask = (xs >= bin_edges[i]) & (xs < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            xc      = float(0.5 * (bin_edges[i] + bin_edges[i + 1]))
            med_d   = float(np.median(ds[mask]))
            med_v2  = float(np.median(v2s[mask]))
            n_pts   = int(mask.sum())

            z_surf = interpolate_elevation(elev_x, elev_z, xc)
            if z_surf is None:
                continue

            tid_points.append(dict(
                x_geo=xc,
                z_surface=z_surf,
                depth=med_d,
                z_rock=z_surf - med_d,
                n_pairs=n_pts,
                v2_median=med_v2,
            ))

        if tid_points:
            tid_points.sort(key=lambda p: p['x_geo'])
            rock[tid] = tid_points

        print(f"  Transect {tid}: {n_total} PM pairs → "
              f"V₂ ok {n_v2_ok} → R² ok {n_r2_ok} → "
              f"zone ok {n_zone_ok} → {len(raw)} depth pts → "
              f"{len(tid_points)} aggregated bins")

    return rock


# ═══════════════════════════════════════════════════════════════════
# PM vs geotech comparison diagnostic
# ═══════════════════════════════════════════════════════════════════

def print_pm_geotech_comparison(
        pm_by_tid: dict[int, list[dict]],
        geotech_by_tid: dict,        # tid → list[GeotechPoint]
) -> None:
    """
    Print a comparison table of PM refractor depths vs geotech rock
    depths, highlighting the systematic three-layer offset.

    This diagnostic helps verify that the PM detects the SHALLOW
    refractor (compaction boundary) and NOT the deep rock/boulders
    found by DPL/CPTu.
    """
    print("\n" + "=" * 72)
    print("  PM REFRACTOR vs GEOTECH ROCK DEPTH — COMPARISON")
    print("=" * 72)
    print("  NOTE: PM detects the shallow refractor (V₂≈870 m/s),")
    print("  which is the compaction boundary, NOT rock/boulders.")
    print("  Expect PM depths ≈ 20–40% of geotech rock depths.")
    print("-" * 72)

    all_tids = sorted(set(list(pm_by_tid.keys()) +
                          list(geotech_by_tid.keys())))
    comparisons: list[tuple[float, float]] = []   # (pm_d, gt_d)

    for tid in all_tids:
        pm_pts = pm_by_tid.get(tid, [])
        gt_pts = geotech_by_tid.get(tid, [])
        if not pm_pts and not gt_pts:
            continue

        # Build geotech rock points: (x, rock_depth, test_type)
        gt_rocks = []
        for gp in gt_pts:
            if gp.depth_of_rock is not None:
                gt_rocks.append((gp.dist_x, gp.depth_of_rock, gp.test_type))

        if not pm_pts:
            continue      # nothing to compare

        print(f"\n  Transect {tid}:")
        for p in pm_pts:
            # Find nearest geotech rock point
            nearest = None
            min_dx = 999.0
            for x, rd, tt in gt_rocks:
                dx = abs(x - p['x_geo'])
                if dx < min_dx:
                    min_dx = dx
                    nearest = (x, rd, tt)
            gt_str = ""
            if nearest and min_dx < 4.0:
                ratio = p['depth'] / nearest[1] if nearest[1] > 0 else 0
                gt_str = (f"  ← {nearest[2]} @x={nearest[0]:.0f}: "
                          f"rock={nearest[1]:.1f} m  "
                          f"(PM/GT={ratio:.2f})")
                comparisons.append((p['depth'], nearest[1]))
            print(f"    x={p['x_geo']:5.1f}  "
                  f"PM_refr={p['depth']:5.1f} m  "
                  f"V₂={p.get('v2_median', 0):4.0f} m/s{gt_str}")

    # Summary statistics
    if comparisons:
        ratios = [pm / gt for pm, gt in comparisons if gt > 0]
        print(f"\n  {'─' * 60}")
        print(f"  Matched points (PM within 4 m of geotech): {len(ratios)}")
        if ratios:
            print(f"  PM/Geotech ratio: "
                  f"min={min(ratios):.2f}  "
                  f"median={np.median(ratios):.2f}  "
                  f"max={max(ratios):.2f}")
            print(f"  → PM refractor is ~{np.median(ratios)*100:.0f}% "
                  f"of geotech rock depth (three-layer effect).")
    else:
        print(f"\n  No co-located PM–geotech points for comparison.")
    print("=" * 72)