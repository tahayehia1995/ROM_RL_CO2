"""
Control Panel Layout
=====================
Model selector, well-control sliders, action buttons, RL toggle,
prediction mode, and editable economics.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def _well_slider(well_id: str, label: str, min_val: float, max_val: float, default: float, unit: str):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(label, className="well-slider-label", style={"fontWeight": "600"}),
                html.Span(id=f"val-{well_id}", className="well-slider-value"),
            ], style={"display": "flex", "justifyContent": "space-between"}),
            dcc.Slider(
                id=f"slider-{well_id}",
                min=0, max=1, step=0.01, value=default,
                marks={0: "Min", 0.5: "Mid", 1: "Max"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(f"{min_val:.0f} -- {max_val:.0f} {unit}",
                     style={"fontSize": "0.65rem", "color": "#667788", "textAlign": "center"}),
        ], style={"padding": "8px"}),
    ], className="mb-2", style={"background": "#1a1a40", "border": "1px solid #0f3460"})


def _econ_input(input_id: str, label: str, default: float, unit: str):
    return dbc.Row([
        dbc.Col(html.Span(label, style={"fontSize": "0.75rem", "color": "#8899aa"}), width=7),
        dbc.Col(dbc.Input(
            id=input_id, type="number", value=default, size="sm",
            style={"background": "#1a1a40", "color": "#53d8fb", "border": "1px solid #0f3460",
                   "fontSize": "0.75rem", "padding": "2px 6px"},
        ), width=5),
    ], className="mb-1", align="center")


def build_control_panel() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(html.H4("Controls", className="mb-0", style={"color": "#e94560"})),
        dbc.CardBody([

            # ============ MODEL SELECTOR ============
            html.H6("ROM Model", style={"color": "#53d8fb", "marginBottom": "6px"}),
            dcc.Dropdown(
                id="dd-model",
                options=[],
                value=None,
                clearable=False,
                style={"background": "#1a1a40", "color": "#e0e0e0", "marginBottom": "6px"},
            ),
            dbc.Row([
                dbc.Col(
                    dbc.Button("Load Model", id="btn-load-model", color="success", size="sm",
                               n_clicks=0, className="w-100"),
                    width=6,
                ),
                dbc.Col(
                    dbc.RadioItems(
                        id="radio-pred-mode",
                        options=[
                            {"label": " State", "value": "state_based"},
                            {"label": " Latent", "value": "latent"},
                        ],
                        value="latent",
                        inline=True,
                        style={"fontSize": "0.75rem", "color": "#8899aa"},
                    ),
                    width=6,
                ),
            ], className="mb-1"),
            html.Div(id="model-status",
                     style={"fontSize": "0.7rem", "color": "#667788", "marginBottom": "6px"}),

            # ============ CASE SELECTOR ============
            html.H6("Simulation Case", style={"color": "#53d8fb", "marginBottom": "4px", "marginTop": "6px"}),
            dcc.Slider(
                id="slider-case", min=0, max=99, step=1, value=0,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(id="case-status",
                     style={"fontSize": "0.65rem", "color": "#667788", "marginBottom": "4px", "textAlign": "center"}),

            html.Hr(style={"borderColor": "#0f3460"}),

            # ============ WELL CONTROLS ============
            html.H6("Producers -- BHP (psi)", style={"color": "#53d8fb", "marginBottom": "8px"}),
            _well_slider("P1-bhp", "P1 BHP", 1087, 1305, 0.5, "psi"),
            _well_slider("P2-bhp", "P2 BHP", 1087, 1305, 0.5, "psi"),
            _well_slider("P3-bhp", "P3 BHP", 1087, 1305, 0.5, "psi"),

            html.Hr(style={"borderColor": "#0f3460"}),

            html.H6("Injectors -- Gas Rate (ft3/day)", style={"color": "#53d8fb", "marginBottom": "8px"}),
            _well_slider("I1-gas", "I1 Gas", 6180072, 100646896, 0.5, "ft3/d"),
            _well_slider("I2-gas", "I2 Gas", 6180072, 100646896, 0.5, "ft3/d"),
            _well_slider("I4-gas", "I4 Gas", 6180072, 100646896, 0.5, "ft3/d"),

            html.Hr(style={"borderColor": "#0f3460"}),

            # ============ ACTION BUTTONS ============
            dbc.ButtonGroup([
                dbc.Button("Step", id="btn-step", color="danger", className="dt-btn-primary", n_clicks=0),
                dbc.Button("Reset", id="btn-reset", color="secondary", className="dt-btn-secondary", n_clicks=0),
                dbc.Button("Auto >>", id="btn-auto", color="info", n_clicks=0),
            ], className="w-100 mb-3"),

            dbc.Checklist(
                id="toggle-rl",
                options=[{"label": "  RL Auto-pilot", "value": "rl_on"}],
                value=[], switch=True, style={"color": "#8899aa"},
            ),

            html.Div(id="rl-policy-section", children=[
                dcc.Dropdown(
                    id="dd-rl-policy",
                    options=[],
                    value=None,
                    placeholder="Select RL policy...",
                    clearable=False,
                    style={"background": "#1a1a40", "color": "#e0e0e0",
                           "marginTop": "6px", "marginBottom": "4px"},
                ),
                html.Div(id="rl-policy-status",
                         style={"fontSize": "0.7rem", "color": "#667788",
                                "marginBottom": "4px"}),
            ], style={"display": "none"}),

            dcc.Store(id="store-rl-policies", data=[]),

            html.Hr(style={"borderColor": "#0f3460"}),

            # ============ ECONOMICS ============
            html.Div([
                dbc.Button("Economics Parameters", id="btn-econ-collapse", color="link", n_clicks=0,
                           style={"color": "#e94560", "fontSize": "0.85rem", "padding": "0", "textDecoration": "none"}),
                dbc.Collapse(
                    id="econ-collapse", is_open=False,
                    children=dbc.Card([dbc.CardBody([
                        _econ_input("econ-gas-inj-rev", "Gas Inj Revenue ($/ton)", 50.0, "$/ton"),
                        _econ_input("econ-gas-inj-cost", "Gas Inj Cost ($/ton)", 10.0, "$/ton"),
                        _econ_input("econ-water-pen", "Water Prod Penalty ($/bbl)", 5.0, "$/bbl"),
                        _econ_input("econ-gas-prod-pen", "Gas Prod Penalty ($/ton)", 50.0, "$/ton"),
                        _econ_input("econ-scale", "Scale Factor", 1000000.0, ""),
                    ], style={"padding": "8px"})],
                    style={"background": "#1a1a40", "border": "1px solid #0f3460", "marginTop": "6px"}),
                ),
            ]),

            dcc.Interval(id="auto-interval", interval=1000, n_intervals=0, disabled=True),

            # Hidden store for model list (JSON-serialisable)
            dcc.Store(id="store-models", data=[]),
        ]),
    ], style={"background": "#16213e", "border": "1px solid #1a1a40", "height": "100%"})
