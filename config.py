# Configuration constants
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'pick_files'
ELEV_FILE_NAME = 'elev.csv'
GEOTECH_FILE_NAME = 'geotech.csv'
V1_MIN = 150.0
V1_MAX = 800.0
V2_MIN = 800.0
V2_MAX = 3000.0
V2_ROCK_MIN = 800.0

BP_SLOPE_TOLERANCE = 1.20
MIN_SEGMENT_POINTS = 3

# ITM quality control — V1 prior and quality gates
V1_PRIOR         = 350.0   # CPTu-constrained V1 fallback (m/s)
MIN_V2_POINTS    = 4       # minimum V2 data points for reliable wing
XC_COVERAGE_WARN = 1.0     # warn if max_offset / xc_est < this ratio

SHEET_PILE_DEPTH = 18.0  # m below surface


# Elevation plot axis limits – 1:1 aspect is enforced by set_aspect('equal').
# x: 0 → 28 m  (28 m range)
# y: -18 → +7 m (25 m range)  — clipped at -18 for readability
ELEV_X_MIN = 0.0
ELEV_X_MAX = 28.0
ELEV_Y_MIN = -18.0
ELEV_Y_MAX = 7.0

# Geotech style
DPL_COLOR:        str = 'darkorange'
CPTU_COLOR:       str = 'royalblue'
ROCK_MARKER_SIZE: int = 7
SHEET_PILE_COLOR: str = 'black'

# ===========================================================================
# Plus-Minus (Hagedoorn) Method
# ===========================================================================
PM_MIN_SHOT_SEP     = 8.0    # minimum distance (m) between reciprocal shots
PM_MIN_OVERLAP      = 4      # minimum geophones in refraction overlap zone
PM_GEO_TOLERANCE    = 0.6    # tolerance (m) for matching geophone positions
PM_EDGE_TRIM        = 2      # drop this many geophones at each edge of the
                              # overlap zone to improve stability
PM_TMINUS_R2_MIN    = 0.90   # minimum R² on T⁻ regression for V2 to be used
PM_V1_FOR_DEPTH     = V1_PRIOR   # V1 used in depth conversion (CPTu prior)
PM_V2_FLOOR_FACTOR  = 1.4    # T⁻ zone must yield V₂ > V1 × this factor;
                              # otherwise it's the direct-wave zone, not
                              # the refracted zone

# Plus-Minus quality filtering (applied in collect_pm_rock_points)
# NOTE: The PM two-layer analysis detects the SHALLOWEST velocity
# contrast ("shallow refractor"), which in a three-layer setting
# is NOT the rock/boulder layer.  See analysis_plusminus docstring.
PM_SHOT_MARGIN          = 1.0          # (m) margin inside shot positions for
                                       # the "between-shots" geophone filter
PM_V2_ACCEPT_MIN        = V2_ROCK_MIN  # pair V₂ must be ≥ this
PM_V2_ACCEPT_MAX        = V2_MAX       # pair V₂ must be ≤ this
PM_DEPTH_MIN            = 1.5          # (m) min plausible depth (landward)
PM_DEPTH_MAX            = 16.0         # (m) max plausible depth
PM_RIVER_X              = 18.0         # geophones at x > this are "near river"
PM_DEPTH_MIN_RIVER      = 0.3          # (m) relaxed min depth near river
PM_AGG_BIN_WIDTH        = 1.0          # (m) x-bin width for median aggregation