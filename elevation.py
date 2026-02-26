from typing import Optional

import numpy as np

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

    Returns (offsets, true_shot_loc, shot_idx).
    """
    shot_idx    = int(np.argmin(times_ms))
    nearest_geo = geophone_locs[shot_idx]
    true_shot   = (float(shot_pos)
                   if shot_pos is not None and shot_pos != 0.0
                   else nearest_geo)
    return np.abs(geophone_locs - true_shot), true_shot, shot_idx