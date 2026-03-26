"""
Corner-Point Grid Builder
==========================
Converts a parsed ``(NI, NJ, NK, 8, 3)`` corner-coordinate array into a
PyVista ``UnstructuredGrid`` of hexahedral cells suitable for the Digital
Twin visualization pipeline.

Features:
- Vertex deduplication via spatial hashing (reduces ~109 K to ~15 K vertices)
- Configurable vertical (Z) exaggeration around the mean reservoir depth
- Fortran-order (I-fastest) cell numbering compatible with existing scalar
  assignment via ``field.flatten(order="F")``
- Exposes cell centres, top-surface nodes, and bounding box for downstream
  renderers (wells, camera, surface)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pyvista as pv

_VTK_HEXAHEDRON = 12


class CornerPointGrid:
    """Hexahedral ``UnstructuredGrid`` built from CMG corner-point data.

    Parameters
    ----------
    corners : np.ndarray, shape ``(NI, NJ, NK, 8, 3)``
        Corner coordinates as returned by
        :func:`~DigitalTwin.engine.corner_point_parser.parse_corners_from_file`.
    z_exaggeration : float
        Vertical exaggeration factor applied around the mean depth.
    negate_y : bool
        If ``True``, flip the sign of the Y coordinates so that the grid
        has positive Y extent (CMG data often uses negative Y).
    """

    def __init__(
        self,
        corners: np.ndarray,
        z_exaggeration: float = 30.0,
        negate_y: bool = True,
    ):
        if corners.ndim != 5 or corners.shape[3:] != (8, 3):
            raise ValueError(f"Expected (NI,NJ,NK,8,3) array, got {corners.shape}")

        self.ni, self.nj, self.nk = corners.shape[:3]
        self.n_cells = self.ni * self.nj * self.nk
        self.z_exaggeration = z_exaggeration

        pts = corners.copy()
        if negate_y:
            pts[..., 1] *= -1.0

        # CMG Z is depth (increasing downward); negate to convert to elevation
        pts[..., 2] *= -1.0

        self._z_mean = float(pts[..., 2].mean())
        if z_exaggeration != 1.0:
            pts[..., 2] = self._z_mean + (pts[..., 2] - self._z_mean) * z_exaggeration

        self._raw_corners = pts

        self.grid: pv.UnstructuredGrid = self._build_grid(pts)

        self._cell_centers: Optional[np.ndarray] = None
        self._top_surface_nodes: Optional[np.ndarray] = None

    def _build_grid(self, pts: np.ndarray) -> pv.UnstructuredGrid:
        """Build an ``UnstructuredGrid`` with deduplicated vertices.

        Cells are added in Fortran order (I-fastest, J-next, K-slowest)
        so that ``cell_data`` from ``field.flatten(order='F')`` maps
        correctly.
        """
        all_points_raw = pts.reshape(-1, 3)

        unique_pts, inverse = self._deduplicate(all_points_raw)

        cell_conn = np.empty(self.n_cells * 8, dtype=np.int64)
        vtk_order = [0, 1, 3, 2, 4, 5, 7, 6]

        idx = 0
        for k in range(self.nk):
            for j in range(self.nj):
                for i in range(self.ni):
                    base = ((i * self.nj + j) * self.nk + k) * 8
                    for local in vtk_order:
                        cell_conn[idx] = inverse[base + local]
                        idx += 1

        n_per_cell = np.full(self.n_cells, 8, dtype=np.int64)
        cell_types = np.full(self.n_cells, _VTK_HEXAHEDRON, dtype=np.uint8)

        grid = pv.UnstructuredGrid(
            {_VTK_HEXAHEDRON: cell_conn.reshape(self.n_cells, 8)},
            unique_pts,
        )
        return grid

    @staticmethod
    def _deduplicate(points: np.ndarray, tol: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Merge coincident vertices using rounded-coordinate hashing."""
        decimals = max(0, int(-np.log10(tol)))
        rounded = np.round(points, decimals)

        _, inv, counts = np.unique(
            rounded, axis=0, return_inverse=True, return_counts=True,
        )
        unique_pts = np.zeros((inv.max() + 1, 3), dtype=np.float64)
        np.add.at(unique_pts, inv, points)
        weight = np.zeros(inv.max() + 1, dtype=np.float64)
        np.add.at(weight, inv, 1.0)
        unique_pts /= weight[:, None]

        return unique_pts, inv

    @property
    def cell_centers(self) -> np.ndarray:
        """``(NI, NJ, NK, 3)`` array of cell-centre coordinates."""
        if self._cell_centers is None:
            centres = self._raw_corners.mean(axis=3)
            self._cell_centers = centres
        return self._cell_centers

    @property
    def top_surface_nodes(self) -> np.ndarray:
        """``(NI, NJ, 4, 3)`` top-face corners of the uppermost layer.

        After depth-to-elevation conversion, CMG corners 0-3 (the shallow /
        small-K face) are at high Z elevation, so they form the geological
        top surface.  We use ``k=0`` (shallowest CMG layer).
        """
        if self._top_surface_nodes is None:
            self._top_surface_nodes = self._raw_corners[:, :, 0, 0:4, :]
        return self._top_surface_nodes

    @property
    def bounding_box(self) -> Dict[str, Tuple[float, float]]:
        """Axis-aligned bounding box of the (exaggerated) grid."""
        pts = self._raw_corners.reshape(-1, 3)
        return {
            "x": (float(pts[:, 0].min()), float(pts[:, 0].max())),
            "y": (float(pts[:, 1].min()), float(pts[:, 1].max())),
            "z": (float(pts[:, 2].min()), float(pts[:, 2].max())),
        }

    @property
    def center(self) -> Tuple[float, float, float]:
        bb = self.bounding_box
        return (
            (bb["x"][0] + bb["x"][1]) / 2,
            (bb["y"][0] + bb["y"][1]) / 2,
            (bb["z"][0] + bb["z"][1]) / 2,
        )

    @property
    def extent_diagonal(self) -> float:
        bb = self.bounding_box
        return float(np.sqrt(
            (bb["x"][1] - bb["x"][0]) ** 2
            + (bb["y"][1] - bb["y"][0]) ** 2
            + (bb["z"][1] - bb["z"][0]) ** 2
        ))

    def cell_center_ijk(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """Return the (x, y, z) centre of cell ``(i, j, k)``."""
        c = self.cell_centers[i, j, k]
        return (float(c[0]), float(c[1]), float(c[2]))
