"""
GNN-based E2C Architecture
===========================
Graph Neural Network encoder/decoder for the Embed-to-Control framework.
Drop-in replacement for MSE2C / MultimodalMSE2C that operates on
graph-structured reservoir data instead of regular 3D grids.

Requires: torch_geometric (PyG)
"""

import sys
import os

# Ensure user site-packages is on sys.path for environments (e.g. Jupyter
# kernels) that may not include it automatically.
_user_site = os.path.join(
    os.path.expanduser("~"),
    "AppData", "Roaming", "Python",
    f"Python{sys.version_info.major}{sys.version_info.minor}",
    "site-packages",
)
if os.path.isdir(_user_site) and _user_site not in sys.path:
    sys.path.insert(0, _user_site)

try:
    from .gnn_e2c import GNNE2C
    GNN_AVAILABLE = True
except ImportError as e:
    GNN_AVAILABLE = False
    _IMPORT_ERROR = str(e)

    class GNNE2C:
        """Placeholder that raises a helpful error when PyG is not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"GNN module requires PyTorch Geometric (torch_geometric). "
                f"Install with: pip install torch_geometric\n"
                f"Original error: {_IMPORT_ERROR}"
            )

__all__ = ['GNNE2C', 'GNN_AVAILABLE']
