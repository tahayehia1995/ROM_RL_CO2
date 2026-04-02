"""
Chat Panel Layout
==================
AI assistant chat interface with LLM provider selector, API key input,
and scrollable conversation history.  Placed as a collapsible bottom
row in the main dashboard.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

_CHAT_STYLE = {
    "background": "#f5f6fa",
    "border": "1px solid #dcdde1",
    "borderRadius": "4px",
    "padding": "10px",
    "height": "380px",
    "overflowY": "auto",
}

_INPUT_STYLE = {
    "background": "#ffffff",
    "color": "#2c3e50",
    "border": "1px solid #bdc3c7",
    "fontSize": "0.85rem",
}


def build_chat_panel() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    "AI Reservoir Assistant",
                    id="btn-chat-collapse",
                    color="link",
                    n_clicks=0,
                    style={
                        "color": "#c0392b",
                        "fontSize": "1rem",
                        "fontWeight": "600",
                        "padding": "0",
                        "textDecoration": "none",
                    },
                ),
            ], width="auto"),
            dbc.Col([
                html.Span(
                    "Powered by LLM",
                    style={"color": "#7f8c8d", "fontSize": "0.75rem",
                           "paddingTop": "4px"},
                ),
            ]),
        ], className="mb-2", align="center"),

        dbc.Collapse(
            id="chat-collapse",
            is_open=False,
            children=dbc.Card([
                dbc.CardBody([
                    # --- Config row: Provider, Model, API Key ---
                    dbc.Row([
                        dbc.Col([
                            html.Label("Provider", style={
                                "fontSize": "0.7rem", "color": "#555555"}),
                            dcc.Dropdown(
                                id="dd-llm-provider",
                                options=[
                                    {"label": "Google Gemini (FREE)", "value": "gemini"},
                                    {"label": "Puter (FREE -- GPT/Claude/Gemini)", "value": "puter"},
                                    {"label": "OpenAI GPT (paid)", "value": "openai"},
                                    {"label": "Anthropic Claude (paid)", "value": "anthropic"},
                                ],
                                value="gemini",
                                clearable=False,
                                style={"fontSize": "0.8rem"},
                            ),
                        ], width=3),
                        dbc.Col([
                            html.Label("Model", style={
                                "fontSize": "0.7rem", "color": "#555555"}),
                            dbc.Input(
                                id="input-llm-model",
                                type="text",
                                value="gemini-2.0-flash",
                                size="sm",
                                style=_INPUT_STYLE,
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Label("API Key / Puter Token", style={
                                "fontSize": "0.7rem", "color": "#555555"}),
                            dbc.Input(
                                id="input-api-key",
                                type="password",
                                value="AIzaSyDqDh0Oev4Cv860o9y2QCPXq7ba0wvDdDs",
                                placeholder="API key from aistudio.google.com",
                                size="sm",
                                style=_INPUT_STYLE,
                            ),
                        ], width=3),
                        dbc.Col([
                            html.Label("Temp", style={
                                "fontSize": "0.7rem", "color": "#555555"}),
                            dbc.Input(
                                id="input-llm-temp",
                                type="number",
                                value=0.7,
                                min=0, max=2, step=0.1,
                                size="sm",
                                style=_INPUT_STYLE,
                            ),
                        ], width=1),
                        dbc.Col([
                            html.Div(
                                id="chat-llm-status",
                                style={"fontSize": "0.7rem", "color": "#7f8c8d",
                                       "paddingTop": "22px"},
                            ),
                        ], width=3),
                    ], className="mb-1"),

                    html.Div(
                        "FREE: Get a Gemini API key at aistudio.google.com (no credit card).  "
                        "Or use Puter (puter.com/dashboard) for GPT/Claude/Gemini.",
                        style={"fontSize": "0.65rem", "color": "#7f8c8d",
                               "marginBottom": "6px", "fontStyle": "italic"},
                    ),

                    # --- Chat history ---
                    html.Div(
                        id="chat-history",
                        children=[
                            html.Div(
                                "Ask me anything about the reservoir state, "
                                "well controls, or optimisation strategies.",
                                style={
                                    "color": "#7f8c8d",
                                    "fontStyle": "italic",
                                    "fontSize": "0.8rem",
                                    "textAlign": "center",
                                    "padding": "40px 0",
                                },
                            ),
                        ],
                        style=_CHAT_STYLE,
                    ),

                    # --- Input row ---
                    dbc.Row([
                        dbc.Col([
                            dbc.Textarea(
                                id="input-chat-message",
                                placeholder="Type your question here...",
                                style={
                                    **_INPUT_STYLE,
                                    "resize": "none",
                                    "height": "60px",
                                },
                            ),
                        ], width=9),
                        dbc.Col([
                            dbc.Button(
                                "Send",
                                id="btn-chat-send",
                                color="danger",
                                n_clicks=0,
                                className="w-100 mb-1",
                                style={"height": "28px", "fontSize": "0.8rem",
                                       "padding": "2px"},
                            ),
                            dbc.Button(
                                "Clear Chat",
                                id="btn-chat-clear",
                                color="secondary",
                                outline=True,
                                n_clicks=0,
                                className="w-100",
                                size="sm",
                                style={"fontSize": "0.75rem"},
                            ),
                        ], width=3, style={"display": "flex", "flexDirection": "column",
                                           "justifyContent": "center", "gap": "4px"}),
                    ], className="mt-2"),
                ], style={"padding": "12px"}),
            ], style={"background": "#ffffff", "border": "1px solid #dcdde1"}),
        ),

        dcc.Store(id="store-chat-history", data=[]),
    ], style={"padding": "8px 24px", "marginTop": "8px"})
