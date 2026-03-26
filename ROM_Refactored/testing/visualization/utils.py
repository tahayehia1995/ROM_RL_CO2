"""
Visualization utility functions
"""

from .dashboard import InteractiveVisualizationDashboard
from .metrics import ModelEvaluationMetrics

# Widget availability check
try:
    import ipywidgets as widgets
    from IPython.display import display
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None
    display = None


def create_visualization_dashboard(state_pred, state_seq_true_aligned, yobs_pred, yobs_seq_true, 
                                 test_case_indices, norm_params, Nx, Ny, Nz, num_tstep=24, channel_names=None,
                                 my_rom=None, test_controls=None, test_observations=None, device=None,
                                 train_state_pred=None, train_state_seq_true_aligned=None, 
                                 train_yobs_pred=None, train_yobs_seq_true=None, train_case_indices=None,
                                 loaded_data=None):
    """
    Convenience function to create and display the interactive visualization dashboard
    
    Args:
        state_pred: Predicted state data (num_case, num_tstep, 3, Nx, Ny, Nz)
        state_seq_true_aligned: True state data (num_case, 3, num_tstep, Nx, Ny, Nz)
        yobs_pred: Predicted observations (num_case, num_tstep, 9)
        yobs_seq_true: True observations (num_case, 9, num_tstep)
        test_case_indices: Array of test case indices
        norm_params: Dictionary of normalization parameters
        Nx, Ny, Nz: Grid dimensions
        num_tstep: Number of time steps
        channel_names: Names of state channels (optional)
        my_rom: ROM model for comparison predictions (optional)
        test_controls: Test control data for comparison predictions (optional)
        test_observations: Test observation data for comparison predictions (optional)
        device: Device for computation (optional)
        train_state_pred: Training predicted state data (optional)
        train_state_seq_true_aligned: Training true state data (optional)
        train_yobs_pred: Training predicted observations (optional)
        train_yobs_seq_true: Training true observations (optional)
        train_case_indices: Array of training case indices (optional)
        loaded_data: Dictionary containing loaded processed data for on-demand training prediction generation (optional)
        
    Returns:
        InteractiveVisualizationDashboard: Dashboard instance
    """
    if not WIDGETS_AVAILABLE:
        print("❌ Interactive widgets not available.")
        print("   Please install ipywidgets for interactive visualization: pip install ipywidgets")
        return None
    
    # Initialize evaluation metrics
    metrics_evaluator = ModelEvaluationMetrics(
        state_pred, state_seq_true_aligned, yobs_pred, yobs_seq_true, 
        channel_names=channel_names
    )
    
    # Create and display the dashboard
    dashboard = InteractiveVisualizationDashboard(
        state_pred, state_seq_true_aligned, yobs_pred, yobs_seq_true,
        test_case_indices, norm_params, Nx, Ny, Nz, num_tstep, channel_names,
        my_rom, test_controls, test_observations, device,
        train_state_pred, train_state_seq_true_aligned, train_yobs_pred, train_yobs_seq_true, train_case_indices,
        loaded_data
    )
    
    dashboard.display_dashboard()
    return dashboard


