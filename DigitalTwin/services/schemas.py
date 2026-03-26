"""
Pydantic Schemas
=================
Request and response models for the Digital Twin REST API.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class StepRequest(BaseModel):
    controls: List[float] = Field(
        ...,
        min_length=6, max_length=6,
        description="Well controls in [0,1]: [P1_BHP, P2_BHP, P3_BHP, I1_Gas, I2_Gas, I4_Gas]",
    )


class WellObservations(BaseModel):
    injector_bhp: List[float] = Field(description="Injector BHP values (psi)")
    gas_production: List[float] = Field(description="Gas production rates (ft³/day)")
    water_production: List[float] = Field(description="Water production rates (ft³/day)")


class KPIs(BaseModel):
    npv: float
    cumulative_gas_prod: float
    cumulative_water_prod: float
    cumulative_gas_inj: float
    step_rewards: List[float] = []


class StepResponse(BaseModel):
    step: int
    done: bool
    well_observations: WellObservations
    controls_physical: List[float]
    kpis: KPIs
    state_shape: List[int] = Field(description="Shape of the 3D state tensor")


class StateResponse(BaseModel):
    step: int
    state: List[List[List[List[float]]]] = Field(description="4D state [C, Nx, Ny, Nz] as nested lists")


class ResetResponse(BaseModel):
    message: str = "Reset successful"
    state_shape: List[int]


class HistoryResponse(BaseModel):
    n_steps: int
    observations: List[List[float]]
    controls: List[List[float]]
    kpis: KPIs
