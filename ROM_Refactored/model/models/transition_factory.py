"""
Factory for creating transition models based on configuration
"""

from .linear_transition import LinearTransitionModel, LinearMultiTransitionModel
from .fno_transition import FNOTransitionModel
from .hybrid_fno_transition import HybridFNOTransitionModel


def create_transition_model(config):
    """
    Create transition model based on configuration
    
    Args:
        config: Configuration object with transition type
        
    Returns:
        Transition model instance
    """
    transition_type = config['transition'].get('type', 'linear').lower()
    
    if transition_type == 'linear':
        return LinearTransitionModel(config)
    elif transition_type == 'fno':
        return FNOTransitionModel(config)
    elif transition_type == 'hybrid_fno' or transition_type == 'hybrid':
        return HybridFNOTransitionModel(config)
    else:
        raise ValueError(f"Unknown transition type: {transition_type}. "
                        f"Supported types: 'linear', 'fno', 'hybrid_fno'")

