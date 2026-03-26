"""
Reward Function for RL
"""
import torch


def reward_fun(yobs, action, num_prod, num_inj, config):
    """
    ✅ CORRECTED: Reward function using optimal ROM structure
    
    Optimal observation order: [Injector_BHP(0-2), Gas_Production(3-5), Water_Production(6-8)]
    Optimal action order: [Producer_BHP(0-2), Gas_Injection(3-5)]
    
    Args:
        yobs: Observations using optimal order [Injector_BHP(3), Gas_Production(3), Water_Production(3)]
        action: Actions using optimal order [Producer_BHP(3), Gas_Injection(3)]
        num_prod: Number of production wells
        num_inj: Number of injection wells  
        config: Configuration object
        
    Returns:
        Economic value using optimal structure and corrected parameter mapping
    """
    econ_config = config.rl_model['economics']
    
    # Unit conversion factors from config
    gas_conversion_factor = econ_config['conversion']['lf3_to_intermediate'] * econ_config['conversion']['intermediate_to_ton']
    water_conversion_factor = econ_config['conversion']['ft3_to_barrel']
    
    # Economic parameters from config
    prices = econ_config['prices']
    gas_injection_net = prices['gas_injection_revenue'] - prices['gas_injection_cost']
    
    # ✅ CORRECTED: Extract observations using optimal order
    # [Injector_BHP(0-2), Gas_Production(3-5), Water_Production(6-8)]
    
    # Gas production (indices 3-5 in optimal order)
    gas_production_ft3_day = torch.sum(yobs[:, num_inj:num_inj+num_prod], dim=1)
    
    # Water production (indices 6-8 in optimal order)  
    water_production_ft3_day = torch.sum(yobs[:, num_inj+num_prod:num_inj+num_prod*2], dim=1)
    water_production_bbl_day = water_production_ft3_day / water_conversion_factor
    
    # ✅ CORRECTED: Extract actions using optimal order
    # [Producer_BHP(0-2), Gas_Injection(3-5)]
    
    # Gas injection (indices 3-5 in optimal action order)
    gas_injection_ft3_day = torch.sum(action[:, num_prod:num_prod+num_inj], dim=1)
    
    # Calculate economic value with optimal structure:
    # - Gas injection revenue: ($/ton) × (ft³/day → tons/day) [POSITIVE - enhances oil recovery]
    # - Water production penalty: ($/bbl) × (bbl/day) [NEGATIVE - unwanted byproduct]
    # - Gas production penalty: ($/ton) × (ft³/day → tons/day) [NEGATIVE - pressure loss]
    PV = (gas_injection_net * gas_conversion_factor * gas_injection_ft3_day - 
          prices['water_production_penalty'] * water_production_bbl_day - 
          prices['gas_production_penalty'] * gas_conversion_factor * gas_production_ft3_day) / econ_config['scale_factor']
    
    return PV

