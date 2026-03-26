"""
Camera Manager
===============
Predefined camera presets for ``dash_vtk.View`` that cover common
reservoir-inspection viewpoints.

Supports both corner-point and rectilinear grid extents.
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .corner_point_grid import CornerPointGrid

CameraPreset = Tuple[List[float], List[float], bool]


class CameraManager:
    """Manages named camera presets for the reservoir 3D viewport."""

    def __init__(
        self,
        nx: int, ny: int, nz: int,
        dx: float = 20.0, dy: float = 10.0, dz: float = 5.0,
        well_locations: Optional[Dict] = None,
        corner_point_grid: Optional["CornerPointGrid"] = None,
    ):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.cpg = corner_point_grid

        if self.cpg is not None:
            c = self.cpg.center
            self.cx, self.cy, self.cz = c
            self.extent_diag = self.cpg.extent_diagonal
        else:
            self.cx = nx * dx / 2
            self.cy = ny * dy / 2
            self.cz = nz * dz / 2
            self.extent_diag = np.sqrt((nx * dx) ** 2 + (ny * dy) ** 2 + (nz * dz) ** 2)

        self.well_locations = well_locations or {}

    def _well_xyz(self, name: str) -> Optional[List[float]]:
        for wtype in ("producers", "injectors"):
            if name in self.well_locations.get(wtype, {}):
                i, j, k = self.well_locations[wtype][name]
                if self.cpg is not None:
                    cx, cy, cz = self.cpg.cell_center_ijk(i, j, k)
                    return [cx, cy, cz]
                return [i * self.dx + self.dx / 2,
                        j * self.dy + self.dy / 2,
                        k * self.dz + self.dz / 2]
        return None

    # ------------------------------------------------------------------
    # Built-in presets
    # ------------------------------------------------------------------
    def get_preset(self, name: str) -> CameraPreset:
        d = self.extent_diag

        presets: Dict[str, CameraPreset] = {
            "perspective_3d": (
                [self.cx + d * 0.8, self.cy - d * 0.6, self.cz + d * 0.7],
                [0, 0, 1],
                False,
            ),
            "top_down": (
                [self.cx, self.cy, self.cz + d * 1.2],
                [0, 1, 0],
                True,
            ),
            "cross_section_xz": (
                [self.cx, self.cy - d * 1.2, self.cz],
                [0, 0, 1],
                True,
            ),
            "cross_section_yz": (
                [self.cx + d * 1.2, self.cy, self.cz],
                [0, 0, 1],
                True,
            ),
            "front": (
                [self.cx, self.cy - d, self.cz],
                [0, 0, 1],
                False,
            ),
            "side": (
                [self.cx + d, self.cy, self.cz],
                [0, 0, 1],
                False,
            ),
        }

        if name.startswith("well_focus_"):
            well_name = name[len("well_focus_"):]
            pos = self._well_xyz(well_name)
            if pos is not None:
                cam_pos = [pos[0] + d * 0.3, pos[1] - d * 0.3, pos[2] + d * 0.4]
                return (cam_pos, [0, 0, 1], False)

        if name in presets:
            return presets[name]

        return presets["perspective_3d"]

    def list_presets(self) -> List[str]:
        base = ["perspective_3d", "top_down", "cross_section_xz",
                "cross_section_yz", "front", "side"]
        for wtype in ("producers", "injectors"):
            for wname in self.well_locations.get(wtype, {}):
                base.append(f"well_focus_{wname}")
        return base
