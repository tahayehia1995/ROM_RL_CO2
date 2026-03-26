"""
Reservoir Colormaps
====================
Domain-specific colour look-up tables for reservoir fields:
pressure, gas saturation, porosity, and permeability.

Returns ``(N, 3)`` uint8 RGB arrays that can be used by
``dash_vtk.DataArray`` or converted to VTK lookup tables.
"""

import numpy as np


def _lerp_lut(colors: list, n: int = 256) -> np.ndarray:
    """Linearly interpolate between a list of ``(r, g, b)`` anchor colours."""
    colors = np.asarray(colors, dtype=np.float64)
    m = len(colors) - 1
    lut = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        t = i / (n - 1)
        idx = min(int(t * m), m - 1)
        frac = t * m - idx
        c = colors[idx] * (1 - frac) + colors[idx + 1] * frac
        lut[i] = np.clip(c * 255, 0, 255).astype(np.uint8)
    return lut


class ReservoirColormaps:
    """Pre-built LUTs for reservoir visualisation."""

    def __init__(self, n: int = 256):
        self._luts = {
            "pressure": _lerp_lut(
                [[0.1, 0.2, 0.7], [0.3, 0.6, 0.9], [0.95, 0.95, 0.3], [0.9, 0.3, 0.1], [0.6, 0.0, 0.0]],
                n,
            ),
            "gas_saturation": _lerp_lut(
                [[1.0, 1.0, 1.0], [1.0, 0.85, 0.4], [0.9, 0.5, 0.05], [0.6, 0.2, 0.0]],
                n,
            ),
            "porosity": _lerp_lut(
                [[0.95, 0.9, 0.75], [0.85, 0.7, 0.45], [0.7, 0.45, 0.2], [0.45, 0.25, 0.1]],
                n,
            ),
            "permeability": _lerp_lut(
                [[0.95, 0.95, 0.8], [0.6, 0.8, 0.3], [0.3, 0.55, 0.15], [0.4, 0.25, 0.1]],
                n,
            ),
        }

    def get_lut(self, field_name: str) -> np.ndarray:
        return self._luts.get(field_name, self._luts["pressure"])

    def available(self) -> list:
        return list(self._luts.keys())

    def get_rgb_for_field(self, field_data: np.ndarray, field_name: str) -> np.ndarray:
        """Map a flat scalar array to ``(N, 3)`` uint8 RGB."""
        flat = field_data.ravel()
        vmin, vmax = float(np.nanmin(flat)), float(np.nanmax(flat))
        span = vmax - vmin if vmax != vmin else 1e-12
        normed = np.clip((flat - vmin) / span, 0, 1)
        lut = self.get_lut(field_name)
        indices = (normed * (len(lut) - 1)).astype(int)
        return lut[indices]
