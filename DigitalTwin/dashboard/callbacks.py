"""
Dashboard Callbacks
====================
Wires model selection, well controls, RL toggle, prediction mode,
economics, 3D viewports (Plotly Mesh3d), time-series charts, and
AI chat assistant.
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate

from DigitalTwin.visualization import ReservoirSceneBuilder
from DigitalTwin.analysis import ChatManager

_CHART = dict(
    template="plotly_white",
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(248,249,250,0.8)",
    font=dict(color="#2c3e50", size=10),
    margin=dict(l=45, r=15, t=35, b=35),
    legend=dict(orientation="h", y=-0.25),
    height=220,
)

_WELL_NAMES_PROD = ["P1", "P2", "P3"]
_WELL_NAMES_INJ = ["I1", "I2", "I4"]


def register_callbacks(app, engine):

    chat_mgr = ChatManager(engine)

    dx, dy, dz = engine.grid_spacing
    nx, ny, nz = engine.grid_dims
    scene = ReservoirSceneBuilder(
        nx, ny, nz, dx, dy, dz,
        well_locations=engine.well_locations,
        z_exaggeration=engine.dt_cfg.get("visualization", {}).get("surface", {}).get("z_exaggeration", 3.0),
        tube_radius=engine.dt_cfg.get("visualization", {}).get("well_tube_radius", 0.5),
        active_mask=engine.active_mask,
        corner_point_grid=engine.corner_point_grid,
    )

    # ---- Camera dropdown ----
    @app.callback(Output("dd-camera", "options"), Input("dd-camera", "id"))
    def _cameras(_):
        return [{"label": p.replace("_", " ").title(), "value": p} for p in scene.camera_presets()]

    # ---- Populate model dropdown + case slider on page load ----
    @app.callback(
        Output("dd-model", "options"),
        Output("dd-model", "value"),
        Output("store-models", "data"),
        Output("model-status", "children"),
        Output("slider-case", "max"),
        Output("slider-case", "marks"),
        Output("case-status", "children"),
        Input("dd-model", "id"),
    )
    def _populate_models(_):
        models = engine.scan_models()
        n = max(engine.num_cases - 1, 0)
        marks = {0: "0", n: str(n)} if n > 0 else {0: "0"}
        case_txt = f"Case 0 / {engine.num_cases}"

        if not models:
            return ([{"label": "No models found", "value": "none"}], "none", [],
                    "No models in saved_models/", n, marks, case_txt)

        options = [{"label": m["label"], "value": str(i)} for i, m in enumerate(models)]
        status = f"Loaded: {engine._loaded_model_label}"
        return options, "0", models, status, n, marks, case_txt

    # ---- Load model button ----
    @app.callback(
        Output("model-status", "children", allow_duplicate=True),
        Output("store-state", "data", allow_duplicate=True),
        Input("btn-load-model", "n_clicks"),
        State("dd-model", "value"),
        State("store-models", "data"),
        State("radio-pred-mode", "value"),
        prevent_initial_call=True,
    )
    def _load_model(n, selected_idx, models_data, pred_mode):
        if not n or not models_data:
            raise PreventUpdate
        try:
            idx = int(selected_idx)
        except (TypeError, ValueError):
            raise PreventUpdate

        if idx < 0 or idx >= len(models_data):
            raise PreventUpdate

        engine.prediction_mode = pred_mode or "latent"
        msg = engine.load_model(models_data[idx])
        return msg, {"step": 0}

    # ---- Economics collapse ----
    @app.callback(
        Output("econ-collapse", "is_open"),
        Input("btn-econ-collapse", "n_clicks"),
        State("econ-collapse", "is_open"),
    )
    def _toggle_econ(n, is_open):
        return (not is_open) if n else is_open

    # ------------------------------------------------------------------
    # RL POLICY: populate dropdown and load agent
    # ------------------------------------------------------------------
    @app.callback(
        Output("dd-rl-policy", "options"),
        Output("dd-rl-policy", "value"),
        Output("store-rl-policies", "data"),
        Output("rl-policy-section", "style"),
        Output("rl-policy-status", "children"),
        Input("toggle-rl", "value"),
        prevent_initial_call=True,
    )
    def _populate_rl_policies(rl_toggle):
        rl_on = isinstance(rl_toggle, list) and "rl_on" in rl_toggle
        if not rl_on:
            return [], None, [], {"display": "none"}, ""

        policies = engine.scan_rl_policies()
        if not policies:
            return ([], None, [],
                    {"display": "block"},
                    "No RL checkpoints found in RL_Refactored/checkpoints/")

        options = [{"label": p["label"], "value": str(idx)}
                   for idx, p in enumerate(policies)]
        return (options, "0", policies,
                {"display": "block"},
                f"Found {len(policies)} policy checkpoint(s)")

    @app.callback(
        Output("rl-policy-status", "children", allow_duplicate=True),
        Input("dd-rl-policy", "value"),
        State("store-rl-policies", "data"),
        prevent_initial_call=True,
    )
    def _load_rl_policy(selected_idx, policies_data):
        if selected_idx is None or not policies_data:
            raise PreventUpdate
        try:
            idx = int(selected_idx)
        except (TypeError, ValueError):
            raise PreventUpdate
        if idx < 0 or idx >= len(policies_data):
            raise PreventUpdate

        policy = policies_data[idx]
        algo = policy["algorithm"]
        ckpt_path = policy["path"]

        ok = engine.load_rl_agent(algorithm_type=algo, checkpoint_path=ckpt_path)
        if ok:
            return f"✓ {algo} loaded: {policy['filename']}"
        else:
            return f"✗ Failed to load {algo} from {policy['filename']}"

    # ------------------------------------------------------------------
    # MAIN CALLBACK: STEP / RESET / AUTO-PLAY
    # ------------------------------------------------------------------
    @app.callback(
        Output("graph-main-3d", "figure"),
        Output("graph-slice-i", "figure"),
        Output("graph-slice-j", "figure"),
        Output("graph-slice-k", "figure"),
        Output("chart-bhp-I1", "figure"),
        Output("chart-bhp-I2", "figure"),
        Output("chart-bhp-I4", "figure"),
        Output("chart-gas-P1", "figure"),
        Output("chart-gas-P2", "figure"),
        Output("chart-gas-P3", "figure"),
        Output("chart-wat-P1", "figure"),
        Output("chart-wat-P2", "figure"),
        Output("chart-wat-P3", "figure"),
        Output("chart-npv", "figure"),
        Output("step-display", "children"),
        Output("store-state", "data"),
        Output("status-bar", "children"),
        # Slider value outputs (synced when RL drives)
        Output("slider-P1-bhp", "value"),
        Output("slider-P2-bhp", "value"),
        Output("slider-P3-bhp", "value"),
        Output("slider-I1-gas", "value"),
        Output("slider-I2-gas", "value"),
        Output("slider-I4-gas", "value"),
        # Inputs -- actions
        Input("btn-step", "n_clicks"),
        Input("btn-reset", "n_clicks"),
        Input("auto-interval", "n_intervals"),
        # Inputs -- viz controls
        Input("dd-field", "value"),
        Input("dd-camera", "value"),
        Input("slider-slice-i", "value"),
        Input("slider-slice-j", "value"),
        Input("slider-slice-k", "value"),
        Input("slider-case", "value"),
        # State -- well controls
        State("slider-P1-bhp", "value"),
        State("slider-P2-bhp", "value"),
        State("slider-P3-bhp", "value"),
        State("slider-I1-gas", "value"),
        State("slider-I2-gas", "value"),
        State("slider-I4-gas", "value"),
        State("store-state", "data"),
        State("toggle-rl", "value"),
        # State -- prediction mode
        State("radio-pred-mode", "value"),
        # State -- economics
        State("econ-gas-inj-rev", "value"),
        State("econ-gas-inj-cost", "value"),
        State("econ-water-pen", "value"),
        State("econ-gas-prod-pen", "value"),
        State("econ-scale", "value"),
        prevent_initial_call=False,
    )
    def _step_or_reset(
        n_step, n_reset, n_auto,
        field, camera, si, sj, sk, case_idx,
        p1, p2, p3, i1, i2, i4,
        store, rl_toggle,
        pred_mode,
        econ_rev, econ_cost, econ_wpen, econ_gpen, econ_scale,
    ):
        ctx = callback_context
        triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        viz_triggers = {"dd-field", "dd-camera", "slider-slice-i", "slider-slice-j", "slider-slice-k"}

        engine.prediction_mode = pred_mode or "latent"

        engine.state_mgr.update_economics(
            gas_inj_revenue=_safe_float(econ_rev, 50.0),
            gas_inj_cost=_safe_float(econ_cost, 10.0),
            water_penalty=_safe_float(econ_wpen, 5.0),
            gas_prod_penalty=_safe_float(econ_gpen, 50.0),
            scale_factor=_safe_float(econ_scale, 1e6),
        )

        rl_active = isinstance(rl_toggle, list) and "rl_on" in rl_toggle
        mode_label = "latent" if engine.prediction_mode == "latent" else "state"
        case_val = int(case_idx) if case_idx is not None else 0
        status_msg = _status_html(f"{engine._loaded_model_label} [{mode_label}] Case {case_val}", "#2980b9")
        slider_vals = (no_update,) * 6

        if triggered == "slider-case":
            engine.set_case(case_val)
            scene.active_mask = engine.active_mask
            state_3d = engine.reset()
            chat_mgr.reset()
            obs = np.zeros(engine.num_inj + 2 * engine.num_prod)
        elif triggered == "btn-reset" or store is None:
            state_3d = engine.reset()
            chat_mgr.reset()
            obs = np.zeros(engine.num_inj + 2 * engine.num_prod)
        elif triggered in viz_triggers:
            state_3d = engine.get_current_state()
            if state_3d is None:
                state_3d = engine.reset()
            obs = np.zeros(engine.num_inj + 2 * engine.num_prod)
        elif triggered in ("btn-step", "auto-interval"):
            if rl_active:
                if not engine._rl_agent_loaded:
                    status_msg = _status_html(
                        "Select an RL policy from the dropdown first", "#c0392b")
                    controls = np.array([p1, p2, p3, i1, i2, i4], dtype=np.float32)
                else:
                    rl_action = engine.get_rl_action()
                    if rl_action is not None:
                        controls = rl_action
                        algo_tag = engine._rl_algorithm_type or "RL"
                        status_msg = _status_html(
                            f"{algo_tag} driving [{mode_label}] Step {engine.step_index}",
                            "#2980b9")
                        slider_vals = tuple(float(controls[j]) for j in range(min(6, len(controls))))
                    else:
                        controls = np.array([p1, p2, p3, i1, i2, i4], dtype=np.float32)
            else:
                controls = np.array([p1, p2, p3, i1, i2, i4], dtype=np.float32)

            try:
                result = engine.step(controls, rl_source=rl_active)
                state_3d = result["state_3d"]
                obs = result["well_observations"]
            except Exception:
                state_3d = engine.reset()
                obs = np.zeros(engine.num_inj + 2 * engine.num_prod)
        else:
            state_3d = engine.get_current_state()
            if state_3d is None:
                state_3d = engine.reset()
            obs = np.zeros(engine.num_inj + 2 * engine.num_prod)

        # ---- Build 3D figures (Plotly Mesh3d) ----
        cur_field = field or "pressure"
        cam_preset = camera or "perspective_3d"

        fig_main = scene.build_main_figure(state_3d, cur_field, obs, cam_preset)
        fig_si = scene.build_slice_figure(state_3d, "i", si, cur_field)
        fig_sj = scene.build_slice_figure(state_3d, "j", sj, cur_field)
        fig_sk = scene.build_slice_figure(state_3d, "k", sk, cur_field)

        # ---- Time-series charts ----
        mgr = engine.state_mgr
        steps = list(range(mgr.n_steps))

        bhp_data = mgr.injector_bhp_history()
        gas_data = mgr.gas_production_history()
        wat_data = mgr.water_production_history()
        wat_data_bbl = wat_data / 5.615 if wat_data.ndim == 2 else wat_data

        bhp_figs = _per_well_charts(bhp_data, _WELL_NAMES_INJ, steps, "#2980b9")
        gas_figs = _per_well_charts(gas_data, _WELL_NAMES_PROD, steps, "#27ae60")
        wat_figs = _per_well_charts(wat_data_bbl, _WELL_NAMES_PROD, steps, "#e67e22")

        kpis = mgr.compute_kpis()
        cum = np.cumsum(kpis.get("step_rewards", [0])).tolist()
        fig_npv = go.Figure()
        fig_npv.add_trace(go.Scatter(x=steps[:len(cum)], y=cum, mode="lines+markers",
                                     name="NPV", line=dict(color="#c0392b", width=2)))
        fig_npv.update_layout(title=dict(text="Cumulative NPV", font=dict(size=10)), **_CHART)

        return (
            fig_main, fig_si, fig_sj, fig_sk,
            *bhp_figs, *gas_figs, *wat_figs, fig_npv,
            f"Step: {engine.step_index} / {engine.max_steps}",
            {"step": engine.step_index}, status_msg,
            *slider_vals,
        )

    # ---- Auto-play toggle ----
    @app.callback(
        Output("auto-interval", "disabled"), Output("btn-auto", "children"),
        Input("btn-auto", "n_clicks"), State("auto-interval", "disabled"),
    )
    def _toggle_auto(n, disabled):
        if not n:
            raise PreventUpdate
        new = not disabled
        return new, ("Auto >>" if new else "Stop")

    # ------------------------------------------------------------------
    # AI CHAT ASSISTANT
    # ------------------------------------------------------------------

    @app.callback(
        Output("chat-collapse", "is_open"),
        Input("btn-chat-collapse", "n_clicks"),
        State("chat-collapse", "is_open"),
    )
    def _toggle_chat(n, is_open):
        return (not is_open) if n else is_open

    @app.callback(
        Output("input-llm-model", "value"),
        Input("dd-llm-provider", "value"),
        prevent_initial_call=True,
    )
    def _update_default_model(provider):
        from DigitalTwin.analysis.llm_client import PROVIDER_DEFAULTS
        return PROVIDER_DEFAULTS.get(provider, {}).get("model", "")

    @app.callback(
        Output("chat-history", "children"),
        Output("store-chat-history", "data"),
        Output("input-chat-message", "value"),
        Output("chat-llm-status", "children"),
        Input("btn-chat-send", "n_clicks"),
        Input("btn-chat-clear", "n_clicks"),
        State("input-chat-message", "value"),
        State("dd-llm-provider", "value"),
        State("input-llm-model", "value"),
        State("input-api-key", "value"),
        State("input-llm-temp", "value"),
        State("store-chat-history", "data"),
        prevent_initial_call=True,
    )
    def _chat_send_or_clear(
        n_send, n_clear,
        message, provider, model, api_key, temperature,
        stored_history,
    ):
        ctx = callback_context
        triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        if triggered == "btn-chat-clear":
            chat_mgr.reset()
            placeholder = html.Div(
                "Chat cleared. Ask me anything about the reservoir.",
                style={"color": "#667788", "fontStyle": "italic",
                       "fontSize": "0.8rem", "textAlign": "center",
                       "padding": "40px 0"},
            )
            return [placeholder], [], "", "Chat cleared"

        if triggered == "btn-chat-send":
            if not message or not message.strip():
                raise PreventUpdate

            chat_mgr.configure_llm(
                provider=provider or "openai",
                api_key=api_key or "",
                model=model or "",
                temperature=_safe_float(temperature, 0.7),
            )

            reply = chat_mgr.send(message.strip())

            history = chat_mgr.history_for_display()
            ui_children = _render_chat_history(history)
            serialisable = [{"role": m["role"], "content": m["content"]}
                            for m in history]
            provider_label = (provider or "openai").capitalize()
            status = f"{provider_label} | {model or 'default'}"

            return ui_children, serialisable, "", status

        raise PreventUpdate


def _render_chat_history(history):
    """Convert chat history list into styled Dash HTML components."""
    children = []
    for msg in history:
        is_user = msg["role"] == "user"
        bubble_style = {
            "background": "#d5e8f0" if is_user else "#eaf2e6",
            "color": "#2c3e50",
            "padding": "8px 12px",
            "borderRadius": "12px",
            "marginBottom": "6px",
            "maxWidth": "85%",
            "fontSize": "0.82rem",
            "lineHeight": "1.4",
            "whiteSpace": "pre-wrap",
            "wordBreak": "break-word",
        }
        align = "flex-end" if is_user else "flex-start"
        label = html.Span(
            "You" if is_user else "AI",
            style={
                "fontSize": "0.65rem",
                "color": "#2980b9" if is_user else "#27ae60",
                "fontWeight": "700",
                "marginBottom": "2px",
            },
        )
        children.append(
            html.Div([
                label,
                html.Div(msg["content"], style=bubble_style),
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": align,
                "marginBottom": "4px",
            })
        )
    return children


def _safe_float(val, default: float) -> float:
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _per_well_charts(data, names, steps, color):
    """Build one figure per well with a shared Y-axis range across the row."""
    n_wells = len(names)
    figs = []

    if data.ndim == 2 and data.shape[0] > 0:
        y_min = float(np.nanmin(data[:, :n_wells]))
        y_max = float(np.nanmax(data[:, :n_wells]))
    else:
        y_min, y_max = 0.0, 1.0
    y_pad = (y_max - y_min) * 0.08 if y_max != y_min else 0.5
    y_range = [y_min - y_pad, y_max + y_pad]

    for idx, name in enumerate(names):
        fig = go.Figure()
        if data.ndim == 2 and data.shape[0] > 0 and idx < data.shape[1]:
            fig.add_trace(go.Scatter(
                x=steps, y=data[:, idx].tolist(),
                mode="lines+markers", name=name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
            ))
        fig.update_layout(
            title=dict(text=name, font=dict(size=10)),
            yaxis=dict(range=y_range),
            showlegend=False,
            **_CHART,
        )
        figs.append(fig)

    while len(figs) < n_wells:
        fig = go.Figure()
        fig.update_layout(title=dict(text="--", font=dict(size=10)), **_CHART)
        figs.append(fig)

    return figs


def _status_html(msg, color="#2980b9"):
    return html.Div([
        html.Span(className="status-dot active" if "fail" not in msg.lower() else "status-dot error"),
        html.Span(msg, style={"color": color, "fontSize": "0.85rem"}),
    ], style={"textAlign": "right", "paddingTop": "10px"})
