"""
Context Builder
================
Converts the current Digital-Twin engine state into a concise text
summary suitable for an LLM system prompt.  The summary includes:

* Grid dimensions and active-cell count
* Current simulation step / max steps
* Per-channel spatial statistics (min, mean, max)
* Well observations (injector BHP, gas production, water production)
* Applied controls (producer BHP, gas injection rate)
* Economic KPIs (NPV, cumulative production)
* Loaded model label and prediction mode

The output is kept under ~800 tokens so that it fits comfortably inside
the context window of any major LLM alongside the conversation history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from DigitalTwin.engine.digital_twin_engine import DigitalTwinEngine

_CHANNEL_LABELS = {
    0: ("Gas Saturation (SG)", "fraction"),
    1: ("Pressure (PRES)", "psi"),
    2: ("Permeability (PERMI)", "mD"),
    3: ("Porosity (POROS)", "fraction"),
}

_WELL_NAMES_PROD = ["P1", "P2", "P3"]
_WELL_NAMES_INJ = ["I1", "I2", "I4"]


class ContextBuilder:
    """Builds a text snapshot of the reservoir state for LLM consumption."""

    def __init__(self, engine: "DigitalTwinEngine"):
        self._engine = engine

    def build_system_prompt(self) -> str:
        """Static system prompt describing the assistant's role."""
        return (
            "You are a reservoir-engineering AI assistant embedded in a "
            "Digital-Twin dashboard.  The dashboard wraps a trained Reduced-Order "
            "Model (ROM) that predicts 3-D spatial fields (gas saturation, pressure, "
            "permeability, porosity) and well observations (injector BHP, gas "
            "production rate, water production rate) given well controls (producer "
            "BHP, gas injection rate).  At each simulation step the user supplies "
            "controls and the ROM advances the state.\n\n"
            "Your job:\n"
            "1. Explain what is happening in the reservoir based on the current "
            "   state summary attached to each message.\n"
            "2. Suggest optimal well-control strategies (BHP / injection rates) "
            "   to maximise NPV while honouring physical constraints.\n"
            "3. Warn about potential issues (e.g. water breakthrough, pressure "
            "   depletion, gas coning).\n"
            "4. Answer any reservoir-engineering or ROM-related questions.\n\n"
            "Keep answers concise and actionable.  Use bullet points where helpful.  "
            "When suggesting control changes, give approximate slider values "
            "(0-1 normalised range) the user can apply."
        )

    def build_state_context(self) -> str:
        """Snapshot of the current engine state as plain text."""
        eng = self._engine
        mgr = eng.state_mgr
        lines: list[str] = ["--- Current Reservoir State ---"]

        lines.append(
            f"Grid: {eng.nx} x {eng.ny} x {eng.nz}  |  "
            f"Step: {eng.step_index} / {eng.max_steps}  |  "
            f"Mode: {eng.prediction_mode}"
        )
        if eng._loaded_model_label:
            lines.append(f"Model: {eng._loaded_model_label}")
        lines.append(f"Simulation case index: {eng._current_case_idx}")

        active = eng.active_mask
        if active is not None:
            n_active = int(active.sum())
            n_total = int(active.size)
            lines.append(f"Active cells: {n_active} / {n_total} "
                         f"({100 * n_active / max(n_total, 1):.1f}%)")

        state = eng.get_current_state()
        if state is not None:
            lines.append("")
            lines.append("Spatial field statistics:")
            for ch_idx in range(min(state.shape[0], len(_CHANNEL_LABELS))):
                label, unit = _CHANNEL_LABELS.get(ch_idx, (f"Ch{ch_idx}", ""))
                field = state[ch_idx]
                if active is not None:
                    field = field[active.astype(bool)]
                lines.append(
                    f"  {label}: min={field.min():.4g}, "
                    f"mean={field.mean():.4g}, max={field.max():.4g} {unit}"
                )

        if mgr.n_steps > 0:
            lines.append("")
            obs = mgr.observations[-1]
            ctrl = mgr.controls[-1]

            lines.append("Latest well observations:")
            for i, name in enumerate(_WELL_NAMES_INJ):
                if i < len(obs):
                    lines.append(f"  {name} BHP: {obs[i]:.1f} psi")
            for i, name in enumerate(_WELL_NAMES_PROD):
                idx = eng.num_inj + i
                if idx < len(obs):
                    lines.append(f"  {name} Gas Rate: {obs[idx]:.1f} ft3/d")
            for i, name in enumerate(_WELL_NAMES_PROD):
                idx = eng.num_inj + eng.num_prod + i
                if idx < len(obs):
                    lines.append(f"  {name} Water Rate: {obs[idx]:.1f} ft3/d")

            lines.append("")
            lines.append("Applied controls (normalised 0-1):")
            ctrl_labels = [f"{n} BHP" for n in _WELL_NAMES_PROD] + \
                          [f"{n} GasInj" for n in _WELL_NAMES_INJ]
            for i, lbl in enumerate(ctrl_labels):
                if i < len(ctrl):
                    lines.append(f"  {lbl}: {ctrl[i]:.4f}")

        kpis = mgr.compute_kpis()
        if kpis:
            lines.append("")
            lines.append("Economic KPIs:")
            lines.append(f"  NPV: {kpis.get('npv', 0):.4f}")
            lines.append(f"  Cumulative Gas Prod: {kpis.get('cumulative_gas_prod', 0):.2f}")
            lines.append(f"  Cumulative Water Prod: {kpis.get('cumulative_water_prod', 0):.2f}")
            lines.append(f"  Cumulative Gas Inj: {kpis.get('cumulative_gas_inj', 0):.2f}")

        return "\n".join(lines)

    def build_user_message(self, user_text: str) -> str:
        """Combine the live state context with the user's question."""
        ctx = self.build_state_context()
        return f"{ctx}\n\n--- User Question ---\n{user_text}"
