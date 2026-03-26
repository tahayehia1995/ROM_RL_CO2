"""
Subsurface Renderer
====================
Creates 3D PyVista meshes of the reservoir from ROM output tensors.
Returns raw PyVista surface meshes that callers convert to their
preferred format (Plotly, VTK, etc.).

Supports two grid modes:
- **Corner-point** (preferred): uses a ``CornerPointGrid`` hexahedral mesh
  built from the actual CMG simulation geometry.
- **Rectilinear** (fallback): uniform ``dx/dy/dz`` spacing when no
  corner-point data is available.
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from .corner_point_grid import CornerPointGrid


class SubsurfaceRenderer:
    """Renders the 3D reservoir grid and returns PyVista surface meshes."""

    def __init__(self, nx: int, ny: int, nz: int,
                 dx: float = 20.0, dy: float = 10.0, dz: float = 5.0,
                 corner_point_grid: Optional["CornerPointGrid"] = None):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.cpg = corner_point_grid
        self._build_base_grid()

    def _build_base_grid(self):
        if self.cpg is not None:
            self.grid = self.cpg.grid
            bb = self.cpg.bounding_box
            self._cx = (bb["x"][0] + bb["x"][1]) / 2
            self._cy = (bb["y"][0] + bb["y"][1]) / 2
            self._cz = (bb["z"][0] + bb["z"][1]) / 2
        else:
            x = np.arange(0, self.nx + 1) * self.dx
            y = np.arange(0, self.ny + 1) * self.dy
            z = np.arange(0, self.nz + 1) * self.dz
            self.grid = pv.RectilinearGrid(x, y, z)
            self._cx = self.nx * self.dx / 2
            self._cy = self.ny * self.dy / 2
            self._cz = self.nz * self.dz / 2

    @staticmethod
    def _to_surface(mesh) -> Optional[pv.PolyData]:
        """Extract a triangulated surface mesh suitable for Plotly Mesh3d."""
        if mesh.n_points == 0:
            return None
        if isinstance(mesh, pv.PolyData) and len(mesh.faces) > 0:
            surf = mesh
        else:
            surf = mesh.extract_surface(algorithm="dataset_surface")
        if surf.n_points == 0:
            return None
        return surf.triangulate()

    def _apply_mask(self, grid, active_mask: Optional[np.ndarray]):
        if active_mask is None:
            return grid
        grid.cell_data["_active"] = active_mask.flatten(order="F").astype(np.float32)
        return grid.threshold(0.5, scalars="_active")

    # ------------------------------------------------------------------
    # Full-grid rendering -> PyVista surface mesh with point scalars
    # ------------------------------------------------------------------
    def render(self, field_data: np.ndarray, field_name: str = "pressure",
               active_mask: Optional[np.ndarray] = None) -> Optional[pv.PolyData]:
        grid = self.grid.copy()
        grid.cell_data[field_name] = field_data.flatten(order="F")
        grid.set_active_scalars(field_name)
        grid = self._apply_mask(grid, active_mask)
        mesh = grid.cell_data_to_point_data()
        mesh.set_active_scalars(field_name)
        return self._to_surface(mesh)

    # ------------------------------------------------------------------
    # Slicing
    # ------------------------------------------------------------------
    def _do_slice_plane(self, field_data, field_name, active_mask, normal, origin):
        grid = self.grid.copy()
        grid.cell_data[field_name] = field_data.flatten(order="F")
        grid.set_active_scalars(field_name)
        masked = self._apply_mask(grid, active_mask)
        if field_name in masked.cell_data:
            masked.set_active_scalars(field_name)
        sliced = masked.slice(normal=normal, origin=origin)
        if sliced.n_points == 0:
            return None
        if field_name not in sliced.point_data and field_name in sliced.cell_data:
            sliced = sliced.cell_data_to_point_data()
        return self._to_surface(sliced)

    def _do_slice_by_index(self, field_data, field_name, active_mask,
                           axis: str, index: int):
        ni, nj, nk = self.nx, self.ny, self.nz
        ids = []
        if axis == "i":
            for k in range(nk):
                for j in range(nj):
                    ids.append(index + j * ni + k * ni * nj)
        elif axis == "j":
            for k in range(nk):
                for i in range(ni):
                    ids.append(i + index * ni + k * ni * nj)
        else:
            for j in range(nj):
                for i in range(ni):
                    ids.append(i + j * ni + index * ni * nj)

        grid = self.grid.copy()
        grid.cell_data[field_name] = field_data.flatten(order="F")
        grid.set_active_scalars(field_name)

        if active_mask is not None:
            flat_mask = active_mask.flatten(order="F").astype(bool)
            ids = [c for c in ids if flat_mask[c]]

        if not ids:
            return None

        sub = grid.extract_cells(np.array(ids, dtype=int))
        if sub.n_cells == 0:
            return None
        mesh = sub.cell_data_to_point_data()
        mesh.set_active_scalars(field_name)
        return self._to_surface(mesh)

    def slice_i(self, field_data, i, field_name="pressure", active_mask=None):
        if self.cpg is not None:
            return self._do_slice_by_index(field_data, field_name, active_mask, "i", i)
        origin = [(i + 0.5) * self.dx, self._cy, self._cz]
        return self._do_slice_plane(field_data, field_name, active_mask, "x", origin)

    def slice_j(self, field_data, j, field_name="pressure", active_mask=None):
        if self.cpg is not None:
            return self._do_slice_by_index(field_data, field_name, active_mask, "j", j)
        origin = [self._cx, (j + 0.5) * self.dy, self._cz]
        return self._do_slice_plane(field_data, field_name, active_mask, "y", origin)

    def slice_k(self, field_data, k, field_name="pressure", active_mask=None):
        if self.cpg is not None:
            return self._do_slice_by_index(field_data, field_name, active_mask, "k", k)
        origin = [self._cx, self._cy, (k + 0.5) * self.dz]
        return self._do_slice_plane(field_data, field_name, active_mask, "z", origin)

    # ------------------------------------------------------------------
    # Colour range
    # ------------------------------------------------------------------
    def color_range(self, field_data: np.ndarray, field_name: str,
                    active_mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
        vals = field_data[active_mask] if active_mask is not None else field_data
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        if vmin == vmax:
            vmax = vmin + 1e-6
        return vmin, vmax
