"""
FastAPI REST API
=================
Lightweight HTTP service that exposes the Digital Twin engine for
programmatic control and integration with external systems.

Endpoints::

    POST /step      – advance one time step with given controls
    GET  /state     – retrieve the current 3D state
    POST /reset     – reset to initial conditions
    GET  /history   – retrieve the full trajectory history
"""

import numpy as np
from fastapi import FastAPI, HTTPException

from .schemas import (
    StepRequest, StepResponse, WellObservations, KPIs,
    StateResponse, ResetResponse, HistoryResponse,
)

_engine = None  # set by create_api_app


def create_api_app(engine) -> FastAPI:
    """Build a FastAPI application wired to *engine*."""
    global _engine
    _engine = engine

    app = FastAPI(
        title="Reservoir Digital Twin API",
        version="0.1.0",
        description="REST interface for the ROM-based reservoir digital twin.",
    )

    @app.post("/step", response_model=StepResponse)
    def step(req: StepRequest):
        controls = np.array(req.controls, dtype=np.float32)
        try:
            result = _engine.step(controls)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        obs = result["well_observations"]
        ni = _engine.num_inj
        np_ = _engine.num_prod

        return StepResponse(
            step=result["step"],
            done=result["done"],
            well_observations=WellObservations(
                injector_bhp=obs[:ni].tolist(),
                gas_production=obs[ni:ni + np_].tolist(),
                water_production=obs[ni + np_:ni + 2 * np_].tolist(),
            ),
            controls_physical=result["controls_physical"].tolist(),
            kpis=KPIs(**result["kpis"]),
            state_shape=list(result["state_3d"].shape),
        )

    @app.get("/state", response_model=StateResponse)
    def get_state():
        state = _engine.get_current_state()
        if state is None:
            raise HTTPException(status_code=404, detail="No state – call /reset first")
        return StateResponse(
            step=_engine.step_index,
            state=state.tolist(),
        )

    @app.post("/reset", response_model=ResetResponse)
    def reset():
        state = _engine.reset()
        return ResetResponse(state_shape=list(state.shape))

    @app.get("/history", response_model=HistoryResponse)
    def get_history():
        mgr = _engine.state_mgr
        if mgr.n_steps == 0:
            raise HTTPException(status_code=404, detail="No history – run steps first")
        return HistoryResponse(
            n_steps=mgr.n_steps,
            observations=[o.tolist() for o in mgr.observations],
            controls=[c.tolist() for c in mgr.controls],
            kpis=KPIs(**mgr.compute_kpis()),
        )

    return app
