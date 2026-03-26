"""
Reservoir Scene Builder
========================
Composes all visualization renderers into Plotly 3D figures
for the dashboard.  Uses ``go.Mesh3d`` for reservoir grids and
``go.Scatter3d`` for well trajectories.
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pyvista as pv
import plotly.graph_objects as go
from dash import html

from .subsurface_renderer import SubsurfaceRenderer
from .well_renderer import WellRenderer
from .camera_manager import CameraManager

if TYPE_CHECKING:
    from .corner_point_grid import CornerPointGrid

_JET = "Jet"

_SCENE_LAYOUT = dict(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    zaxis=dict(visible=False),
    bgcolor="rgb(10,10,26)",
    aspectmode="data",
)


class ReservoirSceneBuilder:
    """Builds complete 3D Plotly figures from ROM state tensors."""

    def __init__(
        self,
        nx: int, ny: int, nz: int,
        dx: float = 20.0, dy: float = 10.0, dz: float = 5.0,
        well_locations: Optional[Dict] = None,
        z_exaggeration: float = 3.0,
        tube_radius: float = 0.5,
        active_mask: Optional[np.ndarray] = None,
        corner_point_grid: Optional["CornerPointGrid"] = None,
    ):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.well_locations = well_locations or {}
        self.active_mask = active_mask
        self.cpg = corner_point_grid

        self.subsurface = SubsurfaceRenderer(
            nx, ny, nz, dx, dy, dz, corner_point_grid=corner_point_grid,
        )
        self.wells = WellRenderer(
            nx, ny, nz, dx, dy, dz, tube_radius,
            corner_point_grid=corner_point_grid,
        )
        self.camera = CameraManager(
            nx, ny, nz, dx, dy, dz, well_locations,
            corner_point_grid=corner_point_grid,
        )

        self.channel_map: Dict[str, int] = {
            "gas_saturation": 0,
            "pressure": 1,
            "permeability": 2,
            "porosity": 3,
        }

        self.fixed_color_ranges: Dict[str, Tuple[float, float]] = {
            "gas_saturation": (0.0, 1.0),
            "pressure": (1094.8, 4486.4),
            "permeability": (0.0, 1850.0),
            "porosity": (0.02, 0.179),
        }

    # ------------------------------------------------------------------
    # PyVista mesh -> Plotly Mesh3d trace
    # ------------------------------------------------------------------
    @staticmethod
    def _pv_to_mesh3d(
        mesh: pv.PolyData,
        field_name: str,
        data_range: Tuple[float, float],
        colorscale: str = _JET,
        opacity: float = 1.0,
        show_colorbar: bool = False,
        name: str = "",
    ) -> go.Mesh3d:
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        scalars = mesh.point_data.get(field_name)
        if scalars is None:
            scalars = np.zeros(mesh.n_points)

        return go.Mesh3d(
            x=mesh.points[:, 0],
            y=mesh.points[:, 1],
            z=mesh.points[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=scalars,
            colorscale=colorscale,
            cmin=data_range[0],
            cmax=data_range[1],
            opacity=opacity,
            name=name,
            showscale=show_colorbar,
            colorbar=dict(
                title=dict(text=name, side="right"),
                thickness=15, len=0.6, x=1.02,
            ) if show_colorbar else None,
            flatshading=True,
            hoverinfo="skip",
        )

    # ------------------------------------------------------------------
    # Well traces (simple vertical lines + perforation markers)
    # ------------------------------------------------------------------
    def _well_traces(self) -> List[go.Scatter3d]:
        traces: List[go.Scatter3d] = []
        colors = {"producers": "#00dd44", "injectors": "#4499ff"}

        for wtype in ("producers", "injectors"):
            for wname, ijk in self.well_locations.get(wtype, {}).items():
                i, j, k = ijk
                x, y, z_perf, z_top = self.wells._well_xyz(i, j, k)
                traces.append(go.Scatter3d(
                    x=[x, x], y=[y, y], z=[z_top, z_perf],
                    mode="lines+text",
                    line=dict(color=colors[wtype], width=5),
                    text=[wname, ""],
                    textposition="top center",
                    textfont=dict(size=9, color=colors[wtype]),
                    name=wname,
                    showlegend=False,
                    hoverinfo="name",
                ))
                traces.append(go.Scatter3d(
                    x=[x], y=[y], z=[z_perf],
                    mode="markers",
                    marker=dict(size=4, color=colors[wtype], symbol="diamond"),
                    name=f"{wname} perf",
                    showlegend=False,
                    hoverinfo="skip",
                ))
        return traces

    # ------------------------------------------------------------------
    # Common figure layout
    # ------------------------------------------------------------------
    def _make_layout(self, height: int = 500, camera_preset: str = "perspective_3d"):
        cam_pos, cam_up, _ = self.camera.get_preset(camera_preset)

        if self.cpg is not None:
            c = self.cpg.center
            diag = self.cpg.extent_diagonal
        else:
            c = [self.nx * self.dx / 2, self.ny * self.dy / 2, self.nz * self.dz / 2]
            diag = np.sqrt((self.nx * self.dx)**2 + (self.ny * self.dy)**2 + (self.nz * self.dz)**2)

        scale = 1.0 / (diag if diag > 0 else 1.0)
        eye = dict(
            x=(cam_pos[0] - c[0]) * scale * 2,
            y=(cam_pos[1] - c[1]) * scale * 2,
            z=(cam_pos[2] - c[2]) * scale * 2,
        )
        up = dict(x=cam_up[0], y=cam_up[1], z=cam_up[2])

        return go.Layout(
            scene=dict(
                **_SCENE_LAYOUT,
                camera=dict(eye=eye, up=up, center=dict(x=0, y=0, z=0)),
            ),
            paper_bgcolor="rgb(10,10,26)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=height,
            showlegend=False,
        )

    # ------------------------------------------------------------------
    # Full 3D scene -> Plotly Figure
    # ------------------------------------------------------------------
    def build_main_figure(
        self,
        state_3d: np.ndarray,
        field_name: str = "pressure",
        well_observations: Optional[np.ndarray] = None,
        camera_preset: str = "perspective_3d",
    ) -> go.Figure:
        ch_idx = self.channel_map.get(field_name, 0)
        field_data = state_3d[ch_idx]
        data_range = list(self.subsurface.color_range(
            field_data, field_name, self.active_mask))

        fig = go.Figure(layout=self._make_layout(height=500, camera_preset=camera_preset))

        mesh = self.subsurface.render(field_data, field_name, self.active_mask)
        if mesh is not None:
            _UNITS = {"pressure": "psi", "gas_saturation": "frac",
                      "permeability": "mD", "porosity": "frac"}
            label = f"{field_name.replace('_', ' ').title()} ({_UNITS.get(field_name, '')})"
            fig.add_trace(self._pv_to_mesh3d(
                mesh, field_name, data_range,
                show_colorbar=True, name=label,
            ))

        for tr in self._well_traces():
            fig.add_trace(tr)

        return fig

    # ------------------------------------------------------------------
    # Slice views -> Plotly Figure
    # ------------------------------------------------------------------
    def build_slice_figure(
        self,
        state_3d: np.ndarray,
        axis: str,
        index: int,
        field_name: str = "pressure",
    ) -> go.Figure:
        ch_idx = self.channel_map.get(field_name, 0)
        field_data = state_3d[ch_idx]
        data_range = list(self.subsurface.color_range(
            field_data, field_name, self.active_mask))

        slicer = {"i": self.subsurface.slice_i,
                  "j": self.subsurface.slice_j,
                  "k": self.subsurface.slice_k}
        mesh = slicer[axis](field_data, index, field_name, active_mask=self.active_mask)

        camera_map = {"i": "cross_section_yz", "j": "cross_section_xz", "k": "top_down"}
        fig = go.Figure(layout=self._make_layout(
            height=220, camera_preset=camera_map.get(axis, "perspective_3d"),
        ))

        if mesh is not None:
            fig.add_trace(self._pv_to_mesh3d(mesh, field_name, data_range))

        return fig

    # ------------------------------------------------------------------
    # Well legend (HTML, not part of the Plotly figure)
    # ------------------------------------------------------------------
    def build_well_legend(self) -> html.Div:
        if not self.well_locations:
            return html.Div()

        items = []
        colors = {"producers": "#00dd44", "injectors": "#4499ff"}

        for wtype in ("producers", "injectors"):
            for wname in self.well_locations.get(wtype, {}):
                items.append(html.Span([
                    html.Span("\u25CF ", style={"color": colors[wtype]}),
                    html.Span(wname, style={"color": "#ccc"}),
                    html.Span("  "),
                ], style={"fontSize": "0.7rem", "marginRight": "8px"}))

        return html.Div(
            items,
            style={"textAlign": "center", "padding": "4px 0"},
        )

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def get_camera(self, preset_name: str) -> Tuple[List[float], List[float], bool]:
        return self.camera.get_preset(preset_name)

    def camera_presets(self) -> List[str]:
        return self.camera.list_presets()
