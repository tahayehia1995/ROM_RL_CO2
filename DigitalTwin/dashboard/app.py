"""
Dash Application
=================
Assembles the full Digital-Twin dashboard layout and wires callbacks.
"""

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

from .layouts.control_panel import build_control_panel
from .layouts.vtk_views import build_vtk_views
from .layouts.time_series import build_time_series_panel
from .layouts.chat_panel import build_chat_panel


def create_dash_app(engine, dt_cfg: dict) -> Dash:
    """Build and return a fully wired Dash application.

    Args:
        engine: ``DigitalTwinEngine`` instance (already loaded with ROM).
        dt_cfg: Digital-Twin config dict (``DigitalTwin/config.yaml`` contents).

    Returns:
        Configured ``Dash`` app ready for ``app.run()``.
    """
    theme_url = dbc.themes.FLATLY

    app = Dash(
        __name__,
        external_stylesheets=[theme_url],
        suppress_callback_exceptions=True,
        title="Reservoir Digital Twin",
    )

    # ---- Layout ---------------------------------------------------------
    app.layout = dbc.Container([
        # Hidden store for shared state between callbacks
        dcc.Store(id="store-state", data=None),
        dcc.Store(id="store-auto", data=False),

        # ---- Header ----
        dbc.Row([
            dbc.Col([
                html.H1("Reservoir Digital Twin", style={"color": "#c0392b", "margin": 0, "fontSize": "1.5rem", "fontWeight": "700"}),
                html.Span("ROM-driven interactive 3D simulation", className="subtitle",
                           style={"color": "#666666", "fontSize": "0.8rem"}),
            ], width=6),
            dbc.Col([
                html.Div(id="status-bar", children=[
                    html.Span(className="status-dot active"),
                    html.Span("Engine Ready", style={"color": "#2980b9", "fontSize": "0.85rem"}),
                ], style={"textAlign": "right", "paddingTop": "10px"}),
            ], width=6),
        ], className="dt-header", style={
            "background": "linear-gradient(135deg, #ecf0f1, #ffffff)",
            "padding": "12px 24px",
            "borderBottom": "2px solid #c0392b",
            "marginBottom": "8px",
        }),

        # ---- Main body: 3 columns ----
        dbc.Row([
            # Left: Controls
            dbc.Col(build_control_panel(), width=3, style={"paddingRight": "4px"}),

            # Centre: 3D viewports
            dbc.Col(build_vtk_views(), width=6, style={"padding": "0 4px"}),

            # Right: Time-series
            dbc.Col(build_time_series_panel(), width=3, style={"paddingLeft": "4px"}),
        ], style={"minHeight": "calc(100vh - 80px)"}),

        # ---- Bottom: AI Chat panel ----
        build_chat_panel(),

    ], fluid=True, style={"padding": "0", "background": "#f8f9fa"})

    # ---- Register callbacks ----
    from .callbacks import register_callbacks  # deferred to avoid circular import
    register_callbacks(app, engine)

    return app
