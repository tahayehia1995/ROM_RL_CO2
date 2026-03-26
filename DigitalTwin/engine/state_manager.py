"""
State Manager
=============
Maintains the trajectory history (3D states, well observations, controls)
and computes key performance indicators (KPIs) such as cumulative production
and Net Present Value (NPV).

Economics logic mirrors RL_Refactored/environment/reward.py.
Supports live updates to economics parameters from the dashboard.
"""

from typing import Dict, List, Optional

import numpy as np


class StateManager:
    """Tracks simulation history and derives economic KPIs."""

    def __init__(
        self,
        n_channels: int,
        nx: int, ny: int, nz: int,
        num_prod: int,
        num_inj: int,
        economics_config: Dict,
        time_step_days: float = 365.0,
    ):
        self.n_channels = n_channels
        self.nx, self.ny, self.nz = nx, ny, nz
        self.num_prod = num_prod
        self.num_inj = num_inj
        self.econ = economics_config
        self.time_step_days = time_step_days

        self.states: List[np.ndarray] = []
        self.observations: List[np.ndarray] = []
        self.controls: List[np.ndarray] = []

    def reset(self):
        self.states.clear()
        self.observations.clear()
        self.controls.clear()

    def record(self, state: np.ndarray, obs: np.ndarray, ctrl: np.ndarray):
        self.states.append(state.copy())
        self.observations.append(obs.copy())
        self.controls.append(ctrl.copy())

    def update_economics(
        self,
        gas_inj_revenue: float = None,
        gas_inj_cost: float = None,
        water_penalty: float = None,
        gas_prod_penalty: float = None,
        scale_factor: float = None,
    ):
        """Live-update economics prices from dashboard inputs."""
        prices = self.econ.setdefault("prices", {})
        if gas_inj_revenue is not None:
            prices["gas_injection_revenue"] = gas_inj_revenue
        if gas_inj_cost is not None:
            prices["gas_injection_cost"] = gas_inj_cost
        if water_penalty is not None:
            prices["water_production_penalty"] = water_penalty
        if gas_prod_penalty is not None:
            prices["gas_production_penalty"] = gas_prod_penalty
        if scale_factor is not None:
            self.econ["scale_factor"] = scale_factor

    @property
    def n_steps(self) -> int:
        return len(self.states)

    # ------------------------------------------------------------------
    # Observation accessors
    # ------------------------------------------------------------------
    def _obs_array(self) -> np.ndarray:
        if not self.observations:
            return np.empty((0,))
        return np.stack(self.observations)

    def _ctrl_array(self) -> np.ndarray:
        if not self.controls:
            return np.empty((0,))
        return np.stack(self.controls)

    def injector_bhp_history(self) -> np.ndarray:
        arr = self._obs_array()
        return arr[:, :self.num_inj] if arr.ndim >= 2 else arr

    def gas_production_history(self) -> np.ndarray:
        arr = self._obs_array()
        return arr[:, self.num_inj:self.num_inj + self.num_prod] if arr.ndim >= 2 else arr

    def water_production_history(self) -> np.ndarray:
        arr = self._obs_array()
        return arr[:, self.num_inj + self.num_prod:self.num_inj + 2 * self.num_prod] if arr.ndim >= 2 else arr

    def producer_bhp_history(self) -> np.ndarray:
        arr = self._ctrl_array()
        return arr[:, :self.num_prod] if arr.ndim >= 2 else arr

    def gas_injection_history(self) -> np.ndarray:
        arr = self._ctrl_array()
        return arr[:, self.num_prod:self.num_prod + self.num_inj] if arr.ndim >= 2 else arr

    # ------------------------------------------------------------------
    # KPIs / economics
    # ------------------------------------------------------------------
    def compute_kpis(self) -> Dict:
        if len(self.observations) == 0:
            return {"npv": 0.0, "cumulative_gas_prod": 0.0, "cumulative_water_prod": 0.0,
                    "cumulative_gas_inj": 0.0, "step_rewards": []}

        obs = self._obs_array()
        ctrl = self._ctrl_array()

        conv = self.econ.get("conversion", {})
        gas_conv = conv.get("lf3_to_intermediate", 0.1167) * conv.get("intermediate_to_ton", 4.536e-4)
        water_conv = conv.get("ft3_to_barrel", 5.61458)

        prices = self.econ.get("prices", {})
        gas_inj_net = prices.get("gas_injection_revenue", 50.0) - prices.get("gas_injection_cost", 10.0)
        water_pen = prices.get("water_production_penalty", 5.0)
        gas_prod_pen = prices.get("gas_production_penalty", 50.0)
        scale = self.econ.get("scale_factor", 1e6)

        gas_prod_raw = obs[:, self.num_inj:self.num_inj + self.num_prod]
        water_prod_raw = obs[:, self.num_inj + self.num_prod:self.num_inj + 2 * self.num_prod]
        gas_inj_raw = ctrl[:, self.num_prod:self.num_prod + self.num_inj] if ctrl.ndim >= 2 else np.zeros_like(gas_prod_raw)

        gas_prod = np.clip(gas_prod_raw, 0.0, None).sum(axis=1)
        water_prod = np.clip(water_prod_raw, 0.0, None).sum(axis=1)
        gas_inj = np.clip(gas_inj_raw, 0.0, None).sum(axis=1)

        step_pv = (
            gas_inj_net * gas_conv * gas_inj
            - water_pen * (water_prod / water_conv)
            - gas_prod_pen * gas_conv * gas_prod
        ) / scale

        return {
            "npv": float(step_pv.sum()),
            "cumulative_gas_prod": float(gas_prod.sum() * self.time_step_days),
            "cumulative_water_prod": float(water_prod.sum() * self.time_step_days),
            "cumulative_gas_inj": float(gas_inj.sum() * self.time_step_days),
            "step_rewards": step_pv.tolist(),
        }
