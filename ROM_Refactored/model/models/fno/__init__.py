"""
FNO-based E2C Architecture
============================
Fourier Neural Operator encoder/decoder for the Embed-to-Control framework.
Drop-in replacement for MSE2C / MultimodalMSE2C that uses spectral
convolution layers with approximately-invertible blocks.

No external dependencies beyond PyTorch (uses torch.fft).
"""

try:
    from .fno_e2c import FNOE2C
    FNO_AVAILABLE = True
except ImportError as e:
    FNO_AVAILABLE = False
    _IMPORT_ERROR = str(e)

    class FNOE2C:
        """Placeholder that raises a helpful error on import failure."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"FNO module import failed.\n"
                f"Original error: {_IMPORT_ERROR}"
            )

__all__ = ['FNOE2C', 'FNO_AVAILABLE']
