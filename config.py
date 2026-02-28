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