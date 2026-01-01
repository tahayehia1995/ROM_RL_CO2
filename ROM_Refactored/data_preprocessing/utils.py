"""
Utility functions for data preprocessing module
Includes formatting, matplotlib configuration, and widget availability checks
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import os

# Configure matplotlib for interactive widgets and animations
# Use inline backend by default for better stability (ipympl can have manager issues)
# Set environment variable MATPLOTLIB_BACKEND to override (e.g., 'ipympl' or 'inline')
backend_preference = os.environ.get('MATPLOTLIB_BACKEND', 'inline').lower()

if backend_preference == 'ipympl':
    # Try ipympl backend if explicitly requested
    try:
        import ipympl
        matplotlib.use('module://ipympl.backend_nbagg')
    except (ImportError, AttributeError, RuntimeError, ValueError):
        # Fallback to inline if ipympl fails
        try:
            matplotlib.use('module://ipykernel.pylab.backend_inline')
        except:
            matplotlib.use('Agg')
else:
    # Default to inline backend for better stability
    try:
        matplotlib.use('module://ipykernel.pylab.backend_inline')
    except:
        # Fallback to non-interactive backend if inline not available
        try:
            matplotlib.use('Agg')
        except:
            pass  # Use default backend

# Set matplotlib for high-resolution plots with modern styling
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Consistent font styling across all plots
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def format_3digits(x, pos):
    """Format numbers to 3 significant digits"""
    if x == 0:
        return '0'
    elif abs(x) >= 1000:
        return f'{x:.0f}'
    elif abs(x) >= 100:
        return f'{x:.0f}'
    elif abs(x) >= 10:
        return f'{x:.1f}'
    elif abs(x) >= 1:
        return f'{x:.2f}'
    else:
        return f'{x:.3g}'

# Set global formatter for all plots
plt.rcParams['axes.formatter.limits'] = [-3, 3]

# Import for interactive visualization (with fallback)
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None
    display = None
    clear_output = None

# Import config loader for well locations
try:
    from utilities.config_loader import load_config
    CONFIG_LOADER_AVAILABLE = True
except ImportError:
    try:
        from config_loader import load_config
        CONFIG_LOADER_AVAILABLE = True
    except ImportError:
        CONFIG_LOADER_AVAILABLE = False

