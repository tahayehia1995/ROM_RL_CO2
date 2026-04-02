"""
Time-Series Panel Layout
==========================
Per-well observation charts arranged in 3 rows of 3, plus a cumulative NPV
chart at the bottom.  Each row shares the same vertical (Y) scale so wells
can be compared at a glance.

Row layout::

    Injector BHP:    I1  |  I2  |  I4
    Gas Production:  P1  |  P2  |  P3
    Water Production: P1 |  P2  |  P3
    ---- Cumulative NPV (full width) ----
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


_LIGHT_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(248,249,250,0.8)",
    font=dict(color="#2c3e50", size=10),
    margin=dict(l=50, r=15, t=30, b=30),
    height=200,
)


def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=dict(text=title, font=dict(size=11)), **_LIGHT_LAYOUT)
    return fig


def _well_chart(chart_id: str, title: str) -> dbc.Col:
    return dbc.Col(
        dcc.Graph(
            id=chart_id,
            figure=_empty_fig(title),
            config={"displayModeBar": False},
            style={"height": "200px"},
        ),
        width=4,
        style={"padding": "4px"},
    )


def build_time_series_panel() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(html.H4(
            "Well Observations & KPIs",
            className="mb-0",
            style={"color": "#c0392b", "fontSize": "1rem"},
        )),
        dbc.CardBody([

            # --- Row 1: Injector BHP (3 wells) ---
            html.Div("Injector BHP (psi)", style={
                "color": "#2980b9", "fontSize": "0.8rem", "fontWeight": "600",
                "marginBottom": "4px", "marginTop": "4px",
            }),
            dbc.Row([
                _well_chart("chart-bhp-I1", "I1"),
                _well_chart("chart-bhp-I2", "I2"),
                _well_chart("chart-bhp-I4", "I4"),
            ], className="g-1"),

            # --- Row 2: Gas Production (3 wells) ---
            html.Div("Gas Production (ft\u00b3/day)", style={
                "color": "#2980b9", "fontSize": "0.8rem", "fontWeight": "600",
                "marginBottom": "4px", "marginTop": "16px",
            }),
            dbc.Row([
                _well_chart("chart-gas-P1", "P1"),
                _well_chart("chart-gas-P2", "P2"),
                _well_chart("chart-gas-P3", "P3"),
            ], className="g-1"),

            # --- Row 3: Water Production (3 wells) ---
            html.Div("Water Production (BBL/day)", style={
                "color": "#2980b9", "fontSize": "0.8rem", "fontWeight": "600",
                "marginBottom": "4px", "marginTop": "16px",
            }),
            dbc.Row([
                _well_chart("chart-wat-P1", "P1"),
                _well_chart("chart-wat-P2", "P2"),
                _well_chart("chart-wat-P3", "P3"),
            ], className="g-1"),

            # --- Row 4: Cumulative NPV (full width) ---
            html.Div("Economics", style={
                "color": "#2980b9", "fontSize": "0.8rem", "fontWeight": "600",
                "marginBottom": "4px", "marginTop": "16px",
            }),
            dcc.Graph(
                id="chart-npv",
                figure=_empty_fig("Cumulative NPV"),
                config={"displayModeBar": False},
                style={"height": "200px"},
            ),
        ], style={"padding": "8px"}),
    ], style={
        "background": "#ffffff", "border": "1px solid #dcdde1",
        "height": "100%", "overflowY": "auto",
    })
