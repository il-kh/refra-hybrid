from typing import Optional

import numpy as np

import config

# ===========================================================================
# Elevation helpers
# ===========================================================================

def interpolate_elevation(elev_x: np.ndarray, elev_z: np.ndarray, query_x: float) -> Optional[float]:
    # Implementation of elevation interpolation
    """Linearly interpolate surface elevation at *query_x*."""
    if len(elev_x) == 0:
        return None
    return float(np.interp(query_x, elev_x, elev_z))

# ===========================================================================
# Geometry helpers
# ===========================================================================

def compute_offsets(geophone_locs: np.ndarray,
                    times_ms: np.ndarray,
                    shot_pos: Optional[float] = None
                    ) -> tuple[np.ndarray, float, int]:
    """
    Compute absolute source–receiver offsets.

    When *shot_pos* is provided (and non-zero), it is used directly.
    When it is missing or zero the position is inferred from the
    travel-time pattern:

    - The geophone with the smallest travel time is identified.
    - The shot is placed 0.5 m toward the neighbouring geophone
      that has the lower travel time (standard survey convention:
      shots sit midway between geophones).
    - If the minimum travel time is suspiciously large for a 0.5 m
      offset (likely a far shot whose position was lost), the nearest
      geophone is used as-is and a warning is printed.

    Returns (offsets, true_shot_loc, shot_idx).
    """
    n           = len(geophone_locs)
    shot_idx    = int(np.argmin(times_ms))
    nearest_geo = geophone_locs[shot_idx]

    if shot_pos is not None and shot_pos != 0.0:
        true_shot = float(shot_pos)
    else:
        # ── Infer shot position from travel-time pattern ──────────
        min_time_ms = float(times_ms[shot_idx])

        # A 0.5 m offset at V1_PRIOR ≈ 350 m/s gives ~1.4 ms.
        # If min_time >> that, this is likely a far shot whose
        # position was lost — we cannot reconstruct it.
        far_threshold_ms = 3.0 / config.V1_PRIOR * 1000.0  # ~8.6 ms

        is_edge = (shot_idx == 0 or shot_idx == n - 1)

        if is_edge and min_time_ms > far_threshold_ms:
            # Far shot with lost position — cannot infer offset
            true_shot = nearest_geo
            print(f"    ⚠  shot_pos=0 with min_time={min_time_ms:.1f} ms "
                  f"at edge geophone — likely a far shot with lost "
                  f"position. Using nearest geo {nearest_geo:.1f} m "
                  f"(UNRELIABLE).")
        elif shot_idx == 0:
            # Shot is to the left of the first geophone
            true_shot = nearest_geo - 0.5
        elif shot_idx == n - 1:
            # Shot is to the right of the last geophone
            true_shot = nearest_geo + 0.5
        else:
            # Interior: step 0.5 m toward the neighbour with lower time
            if times_ms[shot_idx - 1] <= times_ms[shot_idx + 1]:
                true_shot = nearest_geo - 0.5
            else:
                true_shot = nearest_geo + 0.5

        print(f"    ℹ  shot_pos missing (0.0) → inferred: "
              f"{true_shot:.1f} m  (nearest geo: {nearest_geo:.1f} m, "
              f"min t: {min_time_ms:.1f} ms)")

    return np.abs(geophone_locs - true_shot), true_shot, shot_idx