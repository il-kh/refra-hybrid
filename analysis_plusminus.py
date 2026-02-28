"""
analysis_plusminus.py
-----------------------------------------
OBJECTIVE:
Implement the Plus-Minus (Reciprocal) Method to derive 2D bedrock topography from
seismic refraction arrivals, specifically to validate sheet pile refusal depths.

INPUTS:
- geophone_locs: 1D array of x-coordinates for geophones (e.g., 0 to 24m).
- shot_pos: 1D array of shot x-coordinates (includes far offsets like -30m).
- times_ms: 2D array [shot_index, geophone_index] of first arrival times.
- v1: Calculated overburden velocity (m/ms).

LOGIC STEPS FOR IMPLEMENTATION:

1. SHOT SELECTION:
   - identify 'idx_A' as the far-left shot (min(shot_pos)).
   - identify 'idx_B' as the far-right shot (max(shot_pos)).

2. VELOCITY (V2) & RECIPROCAL TIME (T_AB) EXTRAPOLATION:
   - Because geophones do not exist at far shot locations, V2 must be estimated
     from the slope of the arrivals at the distal ends of the array.
   - Calculate V2_fwd: slope of the last 5 geophones from Shot A.
   - Calculate V2_rev: slope of the first 5 geophones from Shot B.
   - Average V2_fwd and V2_rev to get V2_refractor.
   - Extrapolate T_AB (Total travel time between shot A and B):
     T_AB = time_to_last_geophone + (dist_from_last_geophone_to_shot_B / V2_refractor)

3. PLUS-TIME CALCULATION (T_plus):
   - For each geophone 'i', calculate:
     T_plus[i] = (times_ms[idx_A, i] + times_ms[idx_B, i] - T_AB) / 2
   - Note: T_plus represents the 'delay time' directly under the geophone.

4. DEPTH CONVERSION (Z_G):
   - Apply the conversion factor based on the critical angle (Snell's Law):
     depth_factor = (v1 * V2_refractor) / sqrt(V2_refractor**2 - v1**2)
   - Z_G[i] = T_plus[i] * depth_factor

5. MASKING & OUTPUT:
   - Mask/Discard results in 'Blind Zones' (where the wave is still V1 direct).
   - Return a 2D array of [geophone_x, calculated_depth_Z].

6. GEOTECHNICAL VALIDATION (CONSTRAINT):
   - If Z_G < 11.0m, flag as 'Critical Sheet Pile Refusal Zone' (Matches 20MPa CPT).
"""