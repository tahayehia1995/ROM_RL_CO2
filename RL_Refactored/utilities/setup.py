"""
Setup utilities for RL_Refactored
Handles path configuration and hardware setup
"""
import sys
import torch
from pathlib import Path


def setup_paths():
    """Setup Python paths for RL_Refactored and ROM_Refactored"""
    # Get project root (parent of RL_Refactored)
    project_root = Path(__file__).parent.parent.parent
    
    # Add parent directory to path so RL_Refactored can be imported
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Add ROM_Refactored to path
    rom_path = project_root / 'ROM_Refactored'
    if str(rom_path) not in sys.path:
        sys.path.insert(0, str(rom_path))


def print_hardware_info():
    """Print hardware configuration information"""
    print("=" * 70)
    print("Hardware Configuration")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.backends, 'mps'):
        print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    print("=" * 70)
    return device


def get_training_orchestrator(training_dashboard=None):
    """
    Get training orchestrator from training dashboard
    
    Args:
        training_dashboard: Training dashboard instance (optional)
    
    Returns:
        EnhancedTrainingOrchestrator or None
    """
    # Try to get from provided dashboard
    if training_dashboard:
        return getattr(training_dashboard, 'training_orchestrator', None)
    
    # Try to get from global scope (for Jupyter notebooks)
    try:
        import builtins
        if 'training_dashboard' in globals():
            training_dashboard = globals()['training_dashboard']
            return getattr(training_dashboard, 'training_orchestrator', None)
        elif hasattr(builtins, 'rl_training_orchestrator'):
            return builtins.rl_training_orchestrator
    except:
        pass
    
    return None

