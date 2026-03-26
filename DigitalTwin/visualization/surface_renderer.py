"""
Surface Renderer
=================
Extracts the top layer of the 3D reservoir grid and renders it as a 2D
surface mesh with optional Z-exaggeration and well-head markers.

Supports both corner-point and rectilinear grid modes.
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pyvista as pv
from dash_vtk.utils import to_mesh_state

if TYPE_CHECKING:
    from .corner_point_grid import CornerPointGrid


class SurfaceRenderer:
    """Renders the reservoir top-surface and well-head markers."""

    def __init__(
        self,
        nx: int, ny: int, nz: int,
        dx: float = 20.0, dy: float = 10.0, dz: float = 5.0,
        z_exaggeration: float = 3.0,
        corner_point_grid: Optional["CornerPointGrid"] = None,
    ):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.z_exag = z_exaggeration
        self.cpg = corner_point_grid

    # ------------------------------------------------------------------
    # Top-surface mesh
    # ------------------------------------------------------------------
    def render_surface(
        self,
        field_data_3d: np.ndarray,
        field_name: str = "pressure",
    ) -> Dict:
        """Build a coloured top-surface mesh from the top layer (k = nz-1).

        When a corner-point grid is available, the surface is built from the
        actual top-face corner coordinates of the uppermost layer.  Otherwise
        a uniform ``StructuredGrid`` is used as before.
        """
        top_layer = field_data_3d[:, :, -1]  # (Nx, Ny)

        if self.cpg is not None:
            return self._render_surface_cpg(top_layer, field_name)

        xc = np.arange(self.nx) * self.dx + self.dx / 2
        yc = np.arange(self.ny) * self.dy + self.dy / 2
        xx, yy = np.meshgrid(xc, yc, indexing="ij")

        vmin, vmax = float(np.nanmin(top_layer)), float(np.nanmax(top_layer))
        span = vmax - vmin if vmax != vmin else 1.0
        z_base = self.nz * self.dz
        zz = z_base + ((top_layer - vmin) / span) * self.dz * self.z_exag

        surface = pv.StructuredGrid(xx, yy, zz)
        surface.point_data[field_name] = top_layer.flatten(order="F")
        surface.set_active_scalars(field_name)
        return to_mesh_state(surface)

    def _render_surface_cpg(self, top_layer: np.ndarray, field_name: str) -> Dict:
        """Build a top-surface mesh from corner-point top-face geometry."""
        top_nodes = self.cpg.top_surface_nodes  # (NI, NJ, 4, 3)

        centres_xy = top_nodes.mean(axis=2)  # (NI, NJ, 3)
        xx = centres_xy[:, :, 0]
        yy = centres_xy[:, :, 1]

        bb = self.cpg.bounding_box
        z_top = bb["z"][1]
        vmin, vmax = float(np.nanmin(top_layer)), float(np.nanmax(top_layer))
        span = vmax - vmin if vmax != vmin else 1.0
        z_range = (bb["z"][1] - bb["z"][0]) * 0.05
        zz = z_top + ((top_layer - vmin) / span) * z_range

        surface = pv.StructuredGrid(xx, yy, zz)
        surface.point_data[field_name] = top_layer.flatten(order="F")
        surface.set_active_scalars(field_name)
        return to_mesh_state(surface)

    # ------------------------------------------------------------------
    # Well-head markers
    # ------------------------------------------------------------------
    def render_wellheads(
        self,
        well_locations: Dict,
        well_observations: Optional[np.ndarray] = None,
    ) -> Dict:
        """Render well-head positions as a point cloud on the surface."""
        points: List[List[float]] = []
        scalars: List[float] = []
        marker_val = {"producers": 1.0, "injectors": 0.0}

        for wtype in ("producers", "injectors"):
            for name, (i, j, k) in well_locations.get(wtype, {}).items():
                if self.cpg is not None:
                    cx, cy, cz = self.cpg.cell_center_ijk(i, j, k)
                    bb = self.cpg.bounding_box
                    z = bb["z"][1] + (bb["z"][1] - bb["z"][0]) * 0.03
                    points.append([cx, cy, z])
                else:
                    x = i * self.dx + self.dx / 2
                    y = j * self.dy + self.dy / 2
                    z = self.nz * self.dz + self.dz * self.z_exag * 0.5
                    points.append([x, y, z])
                scalars.append(marker_val[wtype])

        if not points:
            return {}

        cloud = pv.PolyData(np.array(points))
        cloud.point_data["well_type"] = np.array(scalars)

        radius = self.dx * 0.4
        if self.cpg is not None:
            bb = self.cpg.bounding_box
            radius = (bb["x"][1] - bb["x"][0]) / self.nx * 0.4
        glyphed = cloud.glyph(geom=pv.Sphere(radius=radius), orient=False, scale=False)
        return to_mesh_state(glyphed)
