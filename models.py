from dataclasses import dataclass
from typing import Optional

@dataclass
class GeotechPoint:
    """One row from geotech.csv that carries an actual test result."""
    test_type:     str
    line_no:       int
    dist_x:        float           # dist_first_gp_m → x-axis (m)
    tested_depth:  float           # m below surface
    depth_of_rock: Optional[float] # m below surface, or None


@dataclass
class SheetPileLine:
    """Position of the planned sheet-pile wall for one transect."""
    line_no: int
    dist_x:  float  # dist_first_gp_m where dist_sheet_pile_m == 0
