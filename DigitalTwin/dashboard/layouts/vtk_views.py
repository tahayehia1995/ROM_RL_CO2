"""
3D Views Layout
================
Main 3D viewport (Plotly Mesh3d) and three smaller slice-view panels.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


def _empty_figure(height: int = 500) -> go.Figure:
    """Dark placeholder figure with no geometry."""
    fig = go.Figure()
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgb(10,10,26)",
        ),
        paper_bgcolor="rgb(10,10,26)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
    )
    return fig


def build_vtk_views() -> html.Div:
    return html.Div([
        # --- Top toolbar ---
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id="dd-field",
                    options=[
                        {"label": "Pressure (psi)", "value": "pressure"},
                        {"label": "Gas Saturation", "value": "gas_saturation"},
                        {"label": "Permeability (mD)", "value": "permeability"},
                        {"label": "Porosity", "value": "porosity"},
                    ],
                    value="pressure",
                    clearable=False,
                    style={"background": "#16213e", "color": "#e0e0e0"},
                ),
            ], width=3),
            dbc.Col([
                dcc.Dropdown(
                    id="dd-camera",
                    options=[],
                    value="perspective_3d",
                    clearable=False,
                    style={"background": "#16213e", "color": "#e0e0e0"},
                ),
            ], width=3),
            dbc.Col([
                html.Div(id="step-display", children="Step: 0 / 30",
                         style={"color": "#53d8fb", "fontSize": "1.1rem",
                                "textAlign": "right", "paddingTop": "6px"}),
            ], width=6),
        ], className="mb-2"),

        # --- Main 3D viewport (Plotly Mesh3d) ---
        dbc.Card([
            dcc.Graph(
                id="graph-main-3d",
                figure=_empty_figure(700),
                config={"displayModeBar": True, "scrollZoom": True},
                style={"width": "100%", "height": "700px"},
            ),
        ], style={"background": "#0a0a1a", "border": "1px solid #0f3460",
                  "marginBottom": "8px"}),

        # --- Slice views ---
        dbc.Row([
            dbc.Col([
                html.Div("I-Slice", style={"color": "#8899aa", "fontSize": "0.75rem",
                                           "textAlign": "center"}),
                dcc.Slider(id="slider-slice-i", min=0, max=33, step=1, value=17,
                           marks=None, tooltip={"placement": "bottom"}),
                dcc.Graph(
                    id="graph-slice-i",
                    figure=_empty_figure(400),
                    config={"displayModeBar": False, "scrollZoom": True},
                    style={"width": "100%", "height": "400px"},
                ),
            ], width=4),
            dbc.Col([
                html.Div("J-Slice", style={"color": "#8899aa", "fontSize": "0.75rem",
                                           "textAlign": "center"}),
                dcc.Slider(id="slider-slice-j", min=0, max=15, step=1, value=8,
                           marks=None, tooltip={"placement": "bottom"}),
                dcc.Graph(
                    id="graph-slice-j",
                    figure=_empty_figure(400),
                    config={"displayModeBar": False, "scrollZoom": True},
                    style={"width": "100%", "height": "400px"},
                ),
            ], width=4),
            dbc.Col([
                html.Div("K-Slice", style={"color": "#8899aa", "fontSize": "0.75rem",
                                           "textAlign": "center"}),
                dcc.Slider(id="slider-slice-k", min=0, max=24, step=1, value=12,
                           marks=None, tooltip={"placement": "bottom"}),
                dcc.Graph(
                    id="graph-slice-k",
                    figure=_empty_figure(400),
                    config={"displayModeBar": False, "scrollZoom": True},
                    style={"width": "100%", "height": "400px"},
                ),
            ], width=4),
        ]),
    ])
