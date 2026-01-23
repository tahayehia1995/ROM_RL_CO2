"""
Model Evaluation Metrics
Calculates evaluation metrics for E2C model predictions
"""

import numpy as np
import torch
try:
    from sklearn import metrics as skmetrics
except ImportError:
    # Fallback if sklearn not available
    skmetrics = None


class ModelEvaluationMetrics:
    """
    Class to calculate and provide evaluation metrics for E2C model predictions
    Supports both spatial field metrics and timeseries metrics
    
    Metrics provided:
    - R2 (Coefficient of determination)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - APE (Absolute Percentage Error)
    """
    
    def __init__(self, state_pred, state_true, yobs_pred, yobs_true, channel_names=None, obs_names=None):
        """
        Initialize with prediction and ground truth data
        
        Args:
            state_pred: Predicted state tensor (num_case, num_tstep, n_channels, Nx, Ny, Nz)
            state_true: True state tensor (num_case, n_channels, num_tstep, Nx, Ny, Nz)
            yobs_pred: Predicted observations (num_case, num_tstep, n_obs)
            yobs_true: True observations (num_case, n_obs, num_tstep)
            channel_names: Names of state channels (optional)
            obs_names: Names of observation variables (optional)
        """
        self.state_pred = state_pred
        self.state_true = state_true
        self.yobs_pred = yobs_pred
        self.yobs_true = yobs_true
        self.channel_names = channel_names if channel_names else ["Channel " + str(i) for i in range(state_pred.shape[2])]
        self.obs_names = obs_names if obs_names else ["Observation " + str(i) for i in range(yobs_pred.shape[2])]
        
        # Cache for computed metrics to avoid recalculation
        self.spatial_metrics_cache = {}
        self.timeseries_metrics_cache = {}
    
    def _get_field_unit(self, field_name):
        """
        Determine unit for spatial field based on field name
        
        Args:
            field_name: Name of the field/channel
            
        Returns:
            str: Unit string for the field
        """
        field_upper = field_name.upper()
        
        if field_upper in ['SW', 'SWAT', 'SG', 'SGAS', 'SO', 'SOIL']:
            return 'fraction'  # Saturations are fractions (0-1)
        elif field_upper in ['PRES', 'PRESSURE']:
            return 'psi'       # Pressure in psi
        elif 'PERM' in field_upper:
            return 'mD'        # Permeability in millidarcy
        elif field_upper in ['PORO', 'POROSITY']:
            return 'fraction'  # Porosity is fraction (0-1)
        else:
            return 'units'     # Generic fallback
    
    def _get_obs_unit(self, obs_idx):
        """
        Determine unit for observation based on observation index
        
        Args:
            obs_idx: Index of the observation
            
        Returns:
            str: Unit string for the observation
        """
        # Standard observation order: [Inj1 BHP, Inj2 BHP, Inj3 BHP, Prod1 Gas, Prod2 Gas, Prod3 Gas, Prod1 Water, Prod2 Water, Prod3 Water]
        if obs_idx < 3:
            return 'psi'       # BHP observations (indices 0-2)
        elif obs_idx < 6:
            return 'ft3/day'   # Gas production observations (indices 3-5)
        elif obs_idx < 9:
            return 'ft3/day'   # Water production observations (indices 6-8)
        else:
            return 'units'     # Generic fallback for additional observations
    
    def _compute_metrics(self, y_true, y_pred, epsilon=None, filter_negative_predictions=True, selected_metrics=None):
        """
        Compute metrics for given true and predicted values (optimized with selective computation)
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            epsilon: Small value to avoid division by zero (auto-calculated if None)
            filter_negative_predictions: Whether to filter out negative predictions (default True for spatial fields)
            selected_metrics: List of metric names to compute (None = all metrics: ['r2', 'rmse', 'mae', 'ape'])
            
        Returns:
            Dictionary with computed metrics
        """
        # Default to all metrics if none specified
        if selected_metrics is None:
            selected_metrics = ['r2', 'rmse', 'mae', 'ape']
        else:
            # Ensure 'ape' is included if not explicitly excluded (for compatibility)
            if 'ape' not in selected_metrics and 'ape' not in [m.lower() for m in selected_metrics]:
                selected_metrics = list(selected_metrics) + ['ape']
        
        # Flatten arrays if needed
        if len(y_true.shape) > 1:
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
        else:
            y_true_flat = y_true
            y_pred_flat = y_pred
        
        # Remove NaN values (from masked cells)
        valid_idx = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
        
        # Filter out negative predictions for physical realism (e.g., saturation fractions should be >= 0)
        if filter_negative_predictions:
            valid_idx = valid_idx & (y_pred_flat >= 0)
        
        y_true_valid = y_true_flat[valid_idx]
        y_pred_valid = y_pred_flat[valid_idx]
        
        # Skip computation if no valid data
        if len(y_true_valid) == 0:
            return {metric: 0.0 for metric in selected_metrics}
        
        # Initialize result dictionary
        result = {}
        
        # Compute only requested metrics using optimized numpy operations
        try:
            # Compute common values used by multiple metrics
            n = len(y_true_valid)
            if n == 0:
                return {metric: 0.0 for metric in selected_metrics}
            
            # Calculate R² (Coefficient of determination) - optimized numpy version
            if 'r2' in selected_metrics:
                ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
                ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)
                if ss_tot > 0:
                    r2 = 1 - (ss_res / ss_tot)
                else:
                    r2 = 0.0
                result['r2'] = r2
            
            # Calculate RMSE (Root Mean Squared Error) - optimized numpy version
            if 'rmse' in selected_metrics:
                mse = np.mean((y_true_valid - y_pred_valid) ** 2)
                rmse = np.sqrt(mse)
                result['rmse'] = rmse
                
                # If MSE is also requested, compute it from RMSE
                if 'mse' in selected_metrics:
                    result['mse'] = mse
            elif 'mse' in selected_metrics:
                # Only MSE requested
                mse = np.mean((y_true_valid - y_pred_valid) ** 2)
                result['mse'] = mse
            
            # Calculate MAE (Mean Absolute Error) - optimized numpy version
            if 'mae' in selected_metrics:
                mae = np.mean(np.abs(y_true_valid - y_pred_valid))
                result['mae'] = mae
            
            # Calculate APE (Absolute Percentage Error) with robust handling
            if 'ape' in selected_metrics:
                # Auto-calculate epsilon as percentage of data range
                if epsilon is None:
                    data_range = np.max(y_true_valid) - np.min(y_true_valid)
                    epsilon = max(0.01 * data_range, 1e-6)  # 1% of range or minimum threshold
                
                # Filter out near-zero values for APE calculation
                # Only compute APE for values that are significantly above noise level
                significant_mask = np.abs(y_true_valid) > epsilon
                
                if np.sum(significant_mask) > 0:
                    y_true_significant = y_true_valid[significant_mask]
                    y_pred_significant = y_pred_valid[significant_mask]
                    
                    # Compute APE only for significant values
                    ape_values = np.abs((y_true_significant - y_pred_significant) / y_true_significant) * 100
                    
                    # Cap extreme values (anything over 1000% is likely noise/outlier)
                    ape_values = np.clip(ape_values, 0.0, 1000.0)
                    ape = np.mean(ape_values)
                else:
                    # If no significant values, set APE to 0
                    ape = 0.0
                result['ape'] = ape
                
        except Exception as e:
            # Fallback if calculation fails - return zeros for requested metrics
            result = {metric: 0.0 for metric in selected_metrics}
            
        return result
    
    
    def get_spatial_metrics(self, case_idx, field_idx, layer_idx, timestep_idx, norm_params=None, dashboard=None):
        """
        Get metrics for spatial field prediction at specific case, field, layer and timestep
        
        Args:
            case_idx: Case index
            field_idx: Field/channel index
            layer_idx: Layer index
            timestep_idx: Time step index
            norm_params: Normalization parameters for denormalization (optional)
            
        Returns:
            Dictionary with metrics
        """
        # Check if metrics already computed and cached
        cache_key = f"{case_idx}_{field_idx}_{layer_idx}_{timestep_idx}"
        if cache_key in self.spatial_metrics_cache:
            return self.spatial_metrics_cache[cache_key]
        
        # Extract true and predicted data
        pred = self.state_pred[case_idx, timestep_idx, field_idx, :, :, layer_idx].cpu().detach().numpy()
        true = self.state_true[case_idx, field_idx, timestep_idx, :, :, layer_idx].cpu().numpy()
        
        # Denormalize if normalization parameters provided
        if norm_params and field_idx < len(self.channel_names):
            field_key = self.channel_names[field_idx]
            if field_key in norm_params:
                params = norm_params[field_key]
                if params.get('type') == 'none':
                    # Data was not normalized, use as-is
                    pass
                elif params.get('type') == 'log':
                    # Reverse log normalization
                    log_min = params['log_min']
                    log_max = params['log_max']
                    
                    # Step 1: Reverse min-max scaling of log data
                    pred_log = pred * (log_max - log_min) + log_min
                    true_log = true * (log_max - log_min) + log_min
                    
                    # Step 2: Reverse log transform
                    epsilon = params.get('epsilon', 1e-8)
                    data_shift = params.get('data_shift', 0)
                    
                    pred = np.exp(pred_log) - epsilon + data_shift
                    true = np.exp(true_log) - epsilon + data_shift
                else:
                    # Standard min-max denormalization
                    field_min = params['min']
                    field_max = params['max']
                    pred = pred * (field_max - field_min) + field_min
                    true = true * (field_max - field_min) + field_min
        
        # Apply inactive cell masking if dashboard is provided
        dashboard_ref = dashboard or getattr(self, 'dashboard_ref', None)
        if dashboard_ref is not None and hasattr(dashboard_ref, '_get_layer_mask'):
            # Get layer mask for this case and layer
            layer_mask = dashboard_ref._get_layer_mask(case_idx, layer_idx)
            # Apply mask (set inactive cells to NaN)
            pred = np.where(layer_mask, pred, np.nan)
            true = np.where(layer_mask, true, np.nan)
            
            # Count inactive cells for logging
            total_cells = pred.size
            active_cells = np.sum(layer_mask)
            inactive_cells = total_cells - active_cells
            if inactive_cells > 0:
                print(f"   🎭 Inactive cell masking: {active_cells}/{total_cells} active cells used for metrics")
        
        # Compute metrics with negative prediction filtering for spatial fields
        # Check if this is a fraction-based field (saturation, porosity) that should not have negative values
        field_name = self.channel_names[field_idx] if field_idx < len(self.channel_names) else "unknown"
        field_unit = self._get_field_unit(field_name)
        filter_negative = (field_unit == 'fraction')  # Filter negatives for saturation and porosity fields
        
        metrics = self._compute_metrics(true, pred, filter_negative_predictions=filter_negative)
        
        # Cache the result
        self.spatial_metrics_cache[cache_key] = metrics
        
        return metrics
    
    def get_timeseries_metrics(self, case_idx, obs_idx, norm_params=None):
        """
        Get metrics for timeseries prediction at specific case and observation variable
        
        Args:
            case_idx: Case index
            obs_idx: Observation variable index
            norm_params: Normalization parameters for denormalization (optional)
            
        Returns:
            Dictionary with metrics
        """
        # Check if metrics already computed and cached
        cache_key = f"{case_idx}_{obs_idx}"
        if cache_key in self.timeseries_metrics_cache:
            return self.timeseries_metrics_cache[cache_key]
        
        # Extract true and predicted data
        pred = self.yobs_pred[case_idx, :, obs_idx].cpu().detach().numpy()
        true = self.yobs_true[case_idx, obs_idx, :].cpu().numpy()
        
        # Denormalize if normalization parameters provided
        if norm_params:
            if obs_idx < 3:  # BHP data
                if 'BHP' in norm_params:
                    params = norm_params['BHP']
                    if params.get('type') == 'none':
                        pass  # No normalization was applied, use data as-is
                    elif params.get('type') == 'log':
                        # Reverse log normalization
                        log_min = params['log_min']
                        log_max = params['log_max']
                        pred_log = pred * (log_max - log_min) + log_min
                        true_log = true * (log_max - log_min) + log_min
                        epsilon = params.get('epsilon', 1e-8)
                        data_shift = params.get('data_shift', 0)
                        pred = np.exp(pred_log) - epsilon + data_shift
                        true = np.exp(true_log) - epsilon + data_shift
                    else:
                        # Standard min-max denormalization
                        obs_min = params['min']
                        obs_max = params['max']
                        pred = pred * (obs_max - obs_min) + obs_min
                        true = true * (obs_max - obs_min) + obs_min
            elif obs_idx < 6:  # Water production
                if 'WATRATSC' in norm_params:
                    params = norm_params['WATRATSC']
                    if params.get('type') == 'none':
                        pass  # No normalization was applied, use data as-is
                    elif params.get('type') == 'log':
                        # Reverse log normalization
                        log_min = params['log_min']
                        log_max = params['log_max']
                        pred_log = pred * (log_max - log_min) + log_min
                        true_log = true * (log_max - log_min) + log_min
                        epsilon = params.get('epsilon', 1e-8)
                        data_shift = params.get('data_shift', 0)
                        pred = np.exp(pred_log) - epsilon + data_shift
                        true = np.exp(true_log) - epsilon + data_shift
                    else:
                        # Standard min-max denormalization
                        obs_min = params['min'] 
                        obs_max = params['max']
                        pred = pred * (obs_max - obs_min) + obs_min
                        true = true * (obs_max - obs_min) + obs_min
            else:  # Gas production
                if 'GASRATSC' in norm_params:
                    params = norm_params['GASRATSC']
                    if params.get('type') == 'none':
                        pass  # No normalization was applied, use data as-is
                    elif params.get('type') == 'log':
                        # Reverse log normalization
                        log_min = params['log_min']
                        log_max = params['log_max']
                        pred_log = pred * (log_max - log_min) + log_min
                        true_log = true * (log_max - log_min) + log_min
                        epsilon = params.get('epsilon', 1e-8)
                        data_shift = params.get('data_shift', 0)
                        pred = np.exp(pred_log) - epsilon + data_shift
                        true = np.exp(true_log) - epsilon + data_shift
                    else:
                        # Standard min-max denormalization
                        obs_min = params['min']
                        obs_max = params['max']
                        pred = pred * (obs_max - obs_min) + obs_min
                        true = true * (obs_max - obs_min) + obs_min
        
        # 🚫 Ensure non-negative observations: Round any negative predictions to zero
        pred = np.maximum(pred, 0.0)
        
        # Compute metrics without negative filtering for timeseries (some observations can be negative)
        metrics = self._compute_metrics(true, pred, filter_negative_predictions=False)
        
        # Cache the result
        self.timeseries_metrics_cache[cache_key] = metrics
        
        return metrics
    
    def get_overall_spatial_metrics(self, field_idx=None):
        """
        Get overall metrics across all cases for a specific field or all fields
        
        Args:
            field_idx: Field/channel index (optional, if None computes for all fields)
            
        Returns:
            Dictionary with metrics
        """
        if field_idx is not None:
            # Compute metrics for specific field across all cases
            all_pred = []
            all_true = []
            
            for case_idx in range(self.state_pred.shape[0]):
                for timestep_idx in range(self.state_pred.shape[1]):
                    for layer_idx in range(self.state_pred.shape[5]):
                        pred = self.state_pred[case_idx, timestep_idx, field_idx, :, :, layer_idx].cpu().detach().numpy()
                        true = self.state_true[case_idx, field_idx, timestep_idx, :, :, layer_idx].cpu().numpy()
                        
                        all_pred.append(pred.flatten())
                        all_true.append(true.flatten())
            
            all_pred = np.concatenate(all_pred)
            all_true = np.concatenate(all_true)
            
            return self._compute_metrics(all_true, all_pred)
        else:
            # Compute metrics for all fields
            metrics_by_field = {}
            
            for field_idx in range(self.state_pred.shape[2]):
                metrics_by_field[self.channel_names[field_idx]] = self.get_overall_spatial_metrics(field_idx)
            
            return metrics_by_field
    
    def get_overall_timeseries_metrics(self, obs_idx=None):
        """
        Get overall metrics across all cases for a specific observation or all observations
        
        Args:
            obs_idx: Observation variable index (optional, if None computes for all observations)
            
        Returns:
            Dictionary with metrics
        """
        if obs_idx is not None:
            # Compute metrics for specific observation across all cases
            all_pred = []
            all_true = []
            
            for case_idx in range(self.yobs_pred.shape[0]):
                pred = self.yobs_pred[case_idx, :, obs_idx].cpu().detach().numpy()
                true = self.yobs_true[case_idx, obs_idx, :].cpu().numpy()
                
                all_pred.append(pred)
                all_true.append(true)
            
            all_pred = np.concatenate(all_pred)
            all_true = np.concatenate(all_true)
            
            return self._compute_metrics(all_true, all_pred)
        else:
            # Compute metrics for all observations
            metrics_by_obs = {}
            
            for obs_idx in range(self.yobs_pred.shape[2]):
                metrics_by_obs[self.obs_names[obs_idx]] = self.get_overall_timeseries_metrics(obs_idx)
            
            return metrics_by_obs
    
    def plot_spatial_metrics(self, case_idx, field_idx, layer_idx, timestep_idx, ax=None, norm_params=None, 
                            dashboard=None):
        """
        Plot metrics for spatial field prediction as actual vs predicted scatter plot
        
        Args:
            case_idx: Case index
            field_idx: Field/channel index
            layer_idx: Layer index
            timestep_idx: Time step index
            ax: Matplotlib axis (optional)
            norm_params: Normalization parameters for denormalization (optional)
            dashboard: InteractiveVisualizationDashboard instance for inactive cell masking (optional)
            
        Returns:
            Matplotlib axis
        """
        # Get data and metrics
        metrics = self.get_spatial_metrics(case_idx, field_idx, layer_idx, timestep_idx, norm_params, dashboard)
        
        # Extract true and predicted data
        pred = self.state_pred[case_idx, timestep_idx, field_idx, :, :, layer_idx].cpu().detach().numpy()
        true = self.state_true[case_idx, field_idx, timestep_idx, :, :, layer_idx].cpu().numpy()
        
        # Denormalize if normalization parameters provided
        if norm_params and field_idx < len(self.channel_names):
            field_key = self.channel_names[field_idx]
            if field_key in norm_params:
                params = norm_params[field_key]
                if params.get('type') == 'none':
                    # Data was not normalized, use as-is
                    pass
                elif params.get('type') == 'log':
                    # Reverse log normalization
                    log_min = params['log_min']
                    log_max = params['log_max']
                    
                    # Step 1: Reverse min-max scaling of log data
                    pred_log = pred * (log_max - log_min) + log_min
                    true_log = true * (log_max - log_min) + log_min
                    
                    # Step 2: Reverse log transform
                    epsilon = params.get('epsilon', 1e-8)
                    data_shift = params.get('data_shift', 0)
                    
                    pred = np.exp(pred_log) - epsilon + data_shift
                    true = np.exp(true_log) - epsilon + data_shift
                else:
                    # Standard min-max denormalization
                    field_min = params['min']
                    field_max = params['max']
                    pred = pred * (field_max - field_min) + field_min
                    true = true * (field_max - field_min) + field_min
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Apply inactive cell masking if dashboard is provided
        dashboard_ref = dashboard or getattr(self, 'dashboard_ref', None)  
        if dashboard_ref is not None and hasattr(dashboard_ref, '_get_layer_mask'):
            # Get layer mask for this case and layer
            layer_mask = dashboard_ref._get_layer_mask(case_idx, layer_idx)
            # Apply mask (set inactive cells to NaN)
            pred = np.where(layer_mask, pred, np.nan)
            true = np.where(layer_mask, true, np.nan)
        
        # Flatten arrays for scatter plot
        true_flat = true.flatten()
        pred_flat = pred.flatten()
        
        # Remove NaN values (which now includes inactive cells)
        valid_idx = ~np.isnan(true_flat) & ~np.isnan(pred_flat)
        
        # Filter out negative predictions for physical realism (e.g., saturation fractions should be >= 0)
        # Check if this is a fraction-based field (saturation, porosity) that should not have negative values
        field_name = self.channel_names[field_idx] if field_idx < len(self.channel_names) else f"Field {field_idx}"
        field_unit = self._get_field_unit(field_name)
        if field_unit == 'fraction':  # Filter negatives for saturation and porosity fields
            valid_idx = valid_idx & (pred_flat >= 0)
        
        true_valid = true_flat[valid_idx]
        pred_valid = pred_flat[valid_idx]
        
        # Log masking information  
        dashboard_ref = dashboard or getattr(self, 'dashboard_ref', None)
        if dashboard_ref is not None and hasattr(dashboard_ref, '_get_layer_mask'):
            total_cells = pred_flat.size
            
            # Count cells before any filtering
            mask_filtered = np.sum(~np.isnan(pred_flat))  # Cells after inactive masking
            inactive_cells = total_cells - mask_filtered
            
            # Count negative predictions that were filtered out (for fraction fields)
            if field_unit == 'fraction':
                negative_preds = np.sum((~np.isnan(pred_flat)) & (pred_flat < 0))
                if negative_preds > 0:
                    print(f"   🚫 Filtered {negative_preds} negative predictions (physically unrealistic for {field_name})")
            
            active_cells = np.sum(valid_idx)
            print(f"   🎭 Inactive cell masking: {mask_filtered}/{total_cells} active cells")
            print(f"   🎯 Final analysis: {active_cells}/{total_cells} valid cells used for metrics and visualization")
        
        # Create scatter plot
        ax.scatter(true_valid, pred_valid, alpha=0.5, s=5, color='blue')
        
        # Add reference lines
        if len(true_valid) > 0:
            # Get min and max for line plotting
            min_val = min(np.min(true_valid), np.min(pred_valid))
            max_val = max(np.max(true_valid), np.max(pred_valid))
            
            # Plot y=x line (perfect prediction) - bold and visible
            ax.plot([min_val, max_val], [min_val, max_val], 'r-', alpha=1.0, linewidth=3, label='Perfect Prediction')
            
            # Add ±10% reference lines (parallel to perfect prediction line) - bold and visible
            x_line = np.linspace(min_val, max_val, 100)
            data_range = max_val - min_val
            offset_10_percent = 0.10 * data_range  # 10% of the data range as constant offset
            y_plus_10 = x_line + offset_10_percent   # +10% line (parallel)
            y_minus_10 = x_line - offset_10_percent  # -10% line (parallel)
            ax.plot(x_line, y_plus_10, 'darkorange', alpha=1.0, linewidth=2.5, linestyle='--', label='+10% Variance')
            ax.plot(x_line, y_minus_10, 'darkorange', alpha=1.0, linewidth=2.5, linestyle='--', label='-10% Variance')
            
            # Create legend for reference lines
            legend = ax.legend(loc='upper left')
            for text in legend.get_texts():
                text.set_fontweight('bold')
        
        # Determine field unit based on field name
        field_name = self.channel_names[field_idx] if field_idx < len(self.channel_names) else f"Field {field_idx}"
        field_unit = self._get_field_unit(field_name)
        
        # Set axis labels with bold font and units
        ax.set_xlabel(f'True Values ({field_unit})', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Predicted Values ({field_unit})', fontsize=12, fontweight='bold')
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Add grid with bold appearance
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
        
        # Make tick labels bold
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Get field name (already obtained above)
        # field_name = self.channel_names[field_idx] if field_idx < len(self.channel_names) else f"Field {field_idx}"
        
        # Add title with metrics - bold formatting
        title = f"{field_name} - Case {case_idx}, Layer {layer_idx}, Time {timestep_idx}\n"
        title += f"R² = {metrics['r2']:.2f}, RMSE = {metrics['rmse']:.2f}, MAE = {metrics['mae']:.2f}, APE = {metrics['ape']:.2f}%"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        return ax
    
    def plot_timeseries_metrics(self, case_idx, obs_idx, ax=None, norm_params=None):
        """
        Plot metrics for timeseries prediction as actual vs predicted scatter plot
        
        Args:
            case_idx: Case index
            obs_idx: Observation variable index
            ax: Matplotlib axis (optional)
            norm_params: Normalization parameters for denormalization (optional)
            
        Returns:
            Matplotlib axis
        """
        # Get data and metrics
        metrics = self.get_timeseries_metrics(case_idx, obs_idx, norm_params)
        
        # Extract true and predicted data
        pred = self.yobs_pred[case_idx, :, obs_idx].cpu().detach().numpy()
        true = self.yobs_true[case_idx, obs_idx, :].cpu().numpy()
        
        # Denormalize if normalization parameters provided
        if norm_params:
            if obs_idx < 3:  # BHP data
                if 'BHP' in norm_params:
                    params = norm_params['BHP']
                    if params.get('type') != 'none':  # Only denormalize if not 'none'
                        obs_min = params['min']
                        obs_max = params['max']
                        pred = pred * (obs_max - obs_min) + obs_min
                        true = true * (obs_max - obs_min) + obs_min
            elif obs_idx < 6:  # Water production
                if 'WATRATSC' in norm_params:
                    params = norm_params['WATRATSC']
                    if params.get('type') != 'none':  # Only denormalize if not 'none'
                        obs_min = params['min'] 
                        obs_max = params['max']
                        pred = pred * (obs_max - obs_min) + obs_min
                        true = true * (obs_max - obs_min) + obs_min
            else:  # Gas production
                if 'GASRATSC' in norm_params:
                    params = norm_params['GASRATSC']
                    if params.get('type') != 'none':  # Only denormalize if not 'none'
                        obs_min = params['min']
                        obs_max = params['max']
                        pred = pred * (obs_max - obs_min) + obs_min
                        true = true * (obs_max - obs_min) + obs_min
        
        # 🚫 Ensure non-negative observations: Round any negative predictions to zero
        pred = np.maximum(pred, 0.0)
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Remove NaN values
        valid_idx = ~np.isnan(true) & ~np.isnan(pred)
        true_valid = true[valid_idx]
        pred_valid = pred[valid_idx]
        
        # Create scatter plot
        ax.scatter(true_valid, pred_valid, alpha=0.7, s=30, color='blue')
        
        # Add reference lines
        if len(true_valid) > 0:
            # Get min and max for line plotting
            min_val = min(np.min(true_valid), np.min(pred_valid))
            max_val = max(np.max(true_valid), np.max(pred_valid))
            
            # Plot y=x line (perfect prediction) - bold and visible
            ax.plot([min_val, max_val], [min_val, max_val], 'r-', alpha=1.0, linewidth=3, label='Perfect Prediction')
            
            # Add ±10% reference lines (parallel to perfect prediction line) - bold and visible
            x_line = np.linspace(min_val, max_val, 100)
            data_range = max_val - min_val
            offset_10_percent = 0.10 * data_range  # 10% of the data range as constant offset
            y_plus_10 = x_line + offset_10_percent   # +10% line (parallel)
            y_minus_10 = x_line - offset_10_percent  # -10% line (parallel)
            ax.plot(x_line, y_plus_10, 'darkorange', alpha=1.0, linewidth=2.5, linestyle='--', label='+10% Variance')
            ax.plot(x_line, y_minus_10, 'darkorange', alpha=1.0, linewidth=2.5, linestyle='--', label='-10% Variance')
            
            # Create legend for reference lines
            legend = ax.legend(loc='upper left')
            for text in legend.get_texts():
                text.set_fontweight('bold')
        
        # Determine observation unit based on observation index
        obs_unit = self._get_obs_unit(obs_idx)
        
        # Set axis labels with bold font and units
        ax.set_xlabel(f'True Values ({obs_unit})', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Predicted Values ({obs_unit})', fontsize=12, fontweight='bold')
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Add grid with bold appearance
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
        
        # Make tick labels bold
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Get observation name
        obs_name = self.obs_names[obs_idx] if obs_idx < len(self.obs_names) else f"Observation {obs_idx}"
        
        # Add title with metrics - bold formatting
        title = f"{obs_name} - Case {case_idx}\n"
        title += f"R² = {metrics['r2']:.2f}, RMSE = {metrics['rmse']:.2f}, MAE = {metrics['mae']:.2f}, APE = {metrics['ape']:.2f}%"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        return ax

def load_normalization_parameters(file_path):
    """
    Load normalization parameters from a saved file
    
    Args:
        file_path (str): Path to the normalization parameters file (.json or .pkl)
        
    Returns:
        dict: Loaded normalization configuration
    """
    import json
    import pickle
    import os
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Normalization parameters file not found: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.json':
            print(f"📖 Loading normalization parameters from JSON: {file_path}")
            with open(file_path, 'r') as f:
                norm_config = json.load(f)
        elif file_extension == '.pkl':
            print(f"📖 Loading normalization parameters from pickle: {file_path}")
            with open(file_path, 'rb') as f:
                norm_config = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Use .json or .pkl files.")
        
        # Validate structure
        required_keys = ['metadata', 'spatial_channels', 'control_variables', 'observation_variables']
        missing_keys = [key for key in required_keys if key not in norm_config]
        if missing_keys:
            raise ValueError(f"Invalid normalization file format. Missing keys: {missing_keys}")
        
        # Print summary of loaded configuration
        metadata = norm_config['metadata']
        print(f"✅ Loaded normalization configuration:")
        print(f"   📅 Created: {metadata.get('created_timestamp', 'Unknown')}")
        print(f"   📁 Data directory: {metadata.get('data_directory', 'Unknown')}")
        print(f"   🏔️ Spatial channels: {len(norm_config['spatial_channels'])}")
        print(f"   🎛️ Control variables: {len(norm_config['control_variables'])}")
        print(f"   📊 Observation variables: {len(norm_config['observation_variables'])}")
        
        return norm_config
        
    except Exception as e:
        raise RuntimeError(f"Error loading normalization parameters: {e}")

    def load_normalization_from_file(self, file_path):
        """
        Load previously saved normalization parameters into the current dashboard
        
        Args:
            file_path (str): Path to the normalization parameters file
        """
        try:
            norm_config = load_normalization_parameters(file_path)
            
            # Load the normalization parameters into current session
            self.norm_params = {}
            
            # Extract norm_params from all variable types
            for var_name, info in norm_config['spatial_channels'].items():
                self.norm_params[var_name] = info['parameters']
            
            for var_name, info in norm_config['control_variables'].items():
                self.norm_params[var_name] = info['parameters']
                
            for var_name, info in norm_config['observation_variables'].items():
                self.norm_params[var_name] = info['parameters']
            
            # Store the full configuration for reference
            self.loaded_norm_config = norm_config
            
            print(f"✅ Normalization parameters loaded successfully!")
            print(f"   📊 Available for denormalization: {list(self.norm_params.keys())}")
            
            return norm_config
            
        except Exception as e:
            print(f"❌ Error loading normalization parameters: {e}")
            return None