"""
Well Renderer
==============
Renders 3D well trajectories (vertical tubes from surface to perforation depth),
colour-coded by well type (producer / injector).

Supports both corner-point and rectilinear grid modes.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pyvista as pv
from dash_vtk.utils import to_mesh_state

if TYPE_CHECKING:
    from .corner_point_grid import CornerPointGrid

_WELL_TYPE_SCALAR = {"producers": 1.0, "injectors": 0.0}


class WellRenderer:
    """Creates tube-based 3D well trajectories."""

    def __init__(
        self,
        nx: int, ny: int, nz: int,
        dx: float = 20.0, dy: float = 10.0, dz: float = 5.0,
        tube_radius: float = 0.5,
        corner_point_grid: Optional["CornerPointGrid"] = None,
    ):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.tube_radius = tube_radius
        self.cpg = corner_point_grid

        if self.cpg is not None:
            bb = self.cpg.bounding_box
            self.tube_radius = (bb["x"][1] - bb["x"][0]) / self.nx * 0.02

    def _well_xyz(self, i: int, j: int, k: int):
        """Return (x, y, z_perf, z_top) for a well at grid index (i, j, k)."""
        if self.cpg is not None:
            cx, cy, cz = self.cpg.cell_center_ijk(i, j, k)
            bb = self.cpg.bounding_box
            z_top = bb["z"][1] + (bb["z"][1] - bb["z"][0]) * 0.08
            return cx, cy, cz, z_top

        x = i * self.dx + self.dx / 2
        y = j * self.dy + self.dy / 2
        z_perf = k * self.dz + self.dz / 2
        z_top = self.nz * self.dz + self.dz * 2
        return x, y, z_perf, z_top

    def render(
        self,
        well_locations: Dict,
        well_observations: Optional[np.ndarray] = None,
    ) -> Dict:
        """Render all wells as tubes and return a single mesh-state dict."""
        meshes: List[pv.PolyData] = []

        for wtype in ("producers", "injectors"):
            for wname, ijk in well_locations.get(wtype, {}).items():
                i, j, k = ijk
                x, y, z_perf, z_top = self._well_xyz(i, j, k)

                line = pv.Line([x, y, z_top], [x, y, z_perf], resolution=2)
                tube = line.tube(radius=self.tube_radius, n_sides=12)
                tube.point_data["well_type"] = np.full(tube.n_points, _WELL_TYPE_SCALAR[wtype])
                meshes.append(tube)

        if not meshes:
            return {}

        combined = meshes[0]
        for m in meshes[1:]:
            combined = combined.merge(m)

        return to_mesh_state(combined)

    def render_perforations(
        self,
        well_locations: Dict,
    ) -> Dict:
        """Small sphere glyphs at each perforation point."""
        points = []
        scalars = []
        for wtype in ("producers", "injectors"):
            for wname, ijk in well_locations.get(wtype, {}).items():
                i, j, k = ijk
                x, y, z_perf, _ = self._well_xyz(i, j, k)
                points.append([x, y, z_perf])
                scalars.append(_WELL_TYPE_SCALAR[wtype])

        if not points:
            return {}

        cloud = pv.PolyData(np.array(points))
        cloud.point_data["well_type"] = np.array(scalars)
        glyphed = cloud.glyph(geom=pv.Sphere(radius=self.tube_radius * 3), orient=False, scale=False)
        return to_mesh_state(glyphed)
