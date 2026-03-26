"""
Factory for creating transition models based on configuration
"""

from .linear_transition import LinearTransitionModel, LinearMultiTransitionModel
from .clru_transition import CLRUTransitionModel
from .s4d_transition import S4DTransitionModel
from .s4d_dplr_transition import S4DPLRTransitionModel
from .s5_transition import S5TransitionModel
from .koopman_transition import KoopmanTransitionModel
from .ct_koopman_transition import CTKoopmanTransitionModel
from .nonlinear_transition import NonlinearTransitionModel
from .mamba_transition import MambaTransitionModel
from .mamba2_transition import Mamba2TransitionModel
from .stable_koopman_transition import StableKoopmanTransitionModel
from .deep_koopman_transition import DeepKoopmanTransitionModel
from .gru_transition import GRUTransitionModel
from .lstm_transition import LSTMTransitionModel
from .hamiltonian_transition import HamiltonianTransitionModel
from .skolr_transition import SKOLRTransitionModel
from .ren_transition import RENTransitionModel
from .koopman_aft_transition import KoopmanAFTTransitionModel
from .dissipative_koopman_transition import DissipativeKoopmanTransitionModel
from .bilinear_koopman_transition import BilinearKoopmanTransitionModel
from .isfno_transition import ISFNOTransitionModel
from .sindy_transition import SINDyTransitionModel
from .neural_cde_transition import NeuralCDETransitionModel
from .latent_sde_transition import LatentSDETransitionModel
from .transformer_transition import TransformerTransitionModel
from .deeponet_transition import DeepONetTransitionModel


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
    elif transition_type == 'clru':
        return CLRUTransitionModel(config)
    elif transition_type == 's4d':
        return S4DTransitionModel(config)
    elif transition_type == 's4d_dplr':
        return S4DPLRTransitionModel(config)
    elif transition_type == 's5':
        return S5TransitionModel(config)
    elif transition_type == 'koopman':
        return KoopmanTransitionModel(config)
    elif transition_type == 'ct_koopman':
        return CTKoopmanTransitionModel(config)
    elif transition_type == 'nonlinear':
        return NonlinearTransitionModel(config)
    elif transition_type == 'mamba':
        return MambaTransitionModel(config)
    elif transition_type == 'mamba2':
        return Mamba2TransitionModel(config)
    elif transition_type == 'stable_koopman':
        return StableKoopmanTransitionModel(config)
    elif transition_type == 'deep_koopman':
        return DeepKoopmanTransitionModel(config)
    elif transition_type == 'gru':
        return GRUTransitionModel(config)
    elif transition_type == 'lstm':
        return LSTMTransitionModel(config)
    elif transition_type == 'hamiltonian':
        return HamiltonianTransitionModel(config)
    elif transition_type == 'skolr':
        return SKOLRTransitionModel(config)
    elif transition_type == 'ren':
        return RENTransitionModel(config)
    elif transition_type == 'koopman_aft':
        return KoopmanAFTTransitionModel(config)
    elif transition_type == 'dissipative_koopman':
        return DissipativeKoopmanTransitionModel(config)
    elif transition_type == 'bilinear_koopman':
        return BilinearKoopmanTransitionModel(config)
    elif transition_type == 'isfno':
        return ISFNOTransitionModel(config)
    elif transition_type == 'sindy':
        return SINDyTransitionModel(config)
    elif transition_type == 'neural_cde':
        return NeuralCDETransitionModel(config)
    elif transition_type == 'latent_sde':
        return LatentSDETransitionModel(config)
    elif transition_type == 'transformer':
        return TransformerTransitionModel(config)
    elif transition_type == 'deeponet':
        return DeepONetTransitionModel(config)
    else:
        raise ValueError(f"Unknown transition type: {transition_type}. "
                        f"Supported types: 'linear', 'clru', 's4d', 's4d_dplr', "
                        f"'s5', 'koopman', 'ct_koopman', 'nonlinear', "
                        f"'mamba', 'mamba2', 'stable_koopman', 'deep_koopman', "
                        f"'gru', 'lstm', 'hamiltonian', 'skolr', 'ren', "
                        f"'koopman_aft', 'dissipative_koopman', 'bilinear_koopman', "
                        f"'isfno', 'sindy', 'neural_cde', 'latent_sde', "
                        f"'transformer', 'deeponet'")
