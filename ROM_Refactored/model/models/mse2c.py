# =====================================
# SECTION 0: IMPORT STATEMENTS
# =====================================
import torch
import torch.nn as nn
from model.models.encoder import Encoder
from model.models.decoder import Decoder
from model.models.transition_factory import create_transition_model
from model.losses.spatial_enhancements import Discriminator3D

# =====================================
# SECTION 1: MSE2C MODEL ARCHITECTURE
# =====================================
class MSE2C(nn.Module):
    def __init__(self, config):
        super(MSE2C, self).__init__()
        self.config = config
        self._build_model()
        self.n_steps = config['training']['nsteps']

    def _build_model(self):
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        
        # Choose transition model based on configuration
        transition_type = self.config['transition'].get('type', 'linear')
        
        if transition_type == 'fno':
            from model.models.fno_transition import FNOTransitionModel
            self.transition = FNOTransitionModel(self.config)
            self.transition_mode = 'spatial'  # FNO operates on spatial states
        elif transition_type == 'hybrid_fno':
            from model.models.hybrid_fno_transition import HybridFNOTransitionModel
            self.transition = HybridFNOTransitionModel(self.config)
            self.transition_mode = 'hybrid'  # Can operate on both spatial and latent
        else:
            from model.models.linear_transition import LinearTransitionModel
            self.transition = LinearTransitionModel(self.config)
            self.transition_mode = 'latent'  # Linear operates on latent states
        
        if getattr(self.config.runtime, 'verbose', False):
            print(f"🔧 E2C MODEL: Using {transition_type} transition model in {self.transition_mode} mode")
        
        # Add discriminator for adversarial training if enabled
        if getattr(self.config.loss, 'enable_adversarial_loss', False):
            self.discriminator = Discriminator3D(self.config)
        else:
            self.discriminator = None

    def forward(self, inputs):
        
        # X, U, Y, dt, perm = inputs
        X, U, Y, dt = inputs

        # Validate data structure matches n_steps
        expected_n_states = self.n_steps
        expected_n_controls = self.n_steps - 1
        expected_n_observations = self.n_steps - 1
        
        if len(X) != expected_n_states:
            raise ValueError(
                f"Data structure mismatch: Expected {expected_n_states} states in X (based on n_steps={self.n_steps}), "
                f"but got {len(X)}. Please ensure preprocessing n_steps matches training n_steps."
            )
        if len(U) != expected_n_controls:
            raise ValueError(
                f"Data structure mismatch: Expected {expected_n_controls} controls in U (based on n_steps={self.n_steps}), "
                f"but got {len(U)}. Please ensure preprocessing n_steps matches training n_steps."
            )
        if len(Y) != expected_n_observations:
            raise ValueError(
                f"Data structure mismatch: Expected {expected_n_observations} observations in Y (based on n_steps={self.n_steps}), "
                f"but got {len(Y)}. Please ensure preprocessing n_steps matches training n_steps."
            )

        X_next = X[1:]
        x0 = X[0]
        
        # Encode and decode initial state
        # VAE mode: encoder returns (z, mu, logvar); AE mode: returns (z, None, None)
        z0, mu0, logvar0 = self.encoder(x0)
        x0_rec = self.decoder(z0)
        
        # Collect VAE parameters for KL loss computation
        mu_list = [mu0] if mu0 is not None else None
        logvar_list = [logvar0] if logvar0 is not None else None
        
        X_next_pred = []
        Z_next = []
        Z_next_pred = []
        Y_next_pred = []
        
        # Choose prediction method based on transition model type
        if self.transition_mode == 'spatial':
            # FNO operates directly on spatial states
            X_next_pred_raw, Y_next_pred = self.transition.forward_nsteps(x0, dt, U)
            X_next_pred = X_next_pred_raw
            
            # Encode predicted spatial states to get latent representations
            for x_next_pred in X_next_pred:
                z_next_pred, mu_pred, logvar_pred = self.encoder(x_next_pred)
                Z_next_pred.append(z_next_pred)
                if mu_list is not None:
                    mu_list.append(mu_pred)
                    logvar_list.append(logvar_pred)
            
            # Encode true states for consistency
            for i_step in range(len(X_next)):
                z_next, mu_next, logvar_next = self.encoder(X[i_step + 1])
                Z_next.append(z_next)
                if mu_list is not None:
                    mu_list.append(mu_next)
                    logvar_list.append(logvar_next)
                
        elif self.transition_mode == 'hybrid':
            # Hybrid model can use both approaches
            hybrid_fno_config = getattr(self.config.transition, 'hybrid_fno', {})
            mode = getattr(hybrid_fno_config, 'forward_mode', 'fno_only')
            
            if mode == 'fno_only':
                # Use spatial approach
                X_next_pred_raw, Y_next_pred = self.transition.forward_nsteps(x0, dt, U)
                X_next_pred = X_next_pred_raw
                
                # Encode predicted spatial states
                for x_next_pred in X_next_pred:
                    z_next_pred, mu_pred, logvar_pred = self.encoder(x_next_pred)
                    Z_next_pred.append(z_next_pred)
                    if mu_list is not None:
                        mu_list.append(mu_pred)
                        logvar_list.append(logvar_pred)
            else:
                # Use traditional latent approach
                Z_next_pred, Y_next_pred = self.transition.forward_nsteps(z0, dt, U)
                
                # Decode predicted latent states
                for z_next_pred in Z_next_pred:
                    x_next_pred = self.decoder(z_next_pred)
                    X_next_pred.append(x_next_pred)
            
            # Encode true states
            for i_step in range(len(X_next)):
                z_next, mu_next, logvar_next = self.encoder(X[i_step + 1])
                Z_next.append(z_next)
                if mu_list is not None:
                    mu_list.append(mu_next)
                    logvar_list.append(logvar_next)
                
        else:
            # Traditional latent-based approach
            Z_next_pred, Y_next_pred = self.transition.forward_nsteps(z0, dt, U)
            
            for i_step in range(len(Z_next_pred)):
                z_next_pred = Z_next_pred[i_step]

                # Decode predicted latent state
                x_next_pred = self.decoder(z_next_pred)
                z_next, mu_next, logvar_next = self.encoder(X[i_step + 1])
                
                X_next_pred.append(x_next_pred)
                Z_next.append(z_next)
                if mu_list is not None:
                    mu_list.append(mu_next)
                    logvar_list.append(logvar_next)
            
        return X_next_pred, X_next, Z_next_pred, Z_next, Y_next_pred, Y, z0, x0, x0_rec, mu_list, logvar_list
    
    def predict(self, inputs):
        """
        Single-step prediction for inference.
        In inference mode, we use the mean (deterministic) for VAE.
        """
        # xt, ut, yt, dt, perm = inputs
        xt, ut, yt, dt = inputs

        if self.transition_mode == 'spatial':
            # FNO operates directly on spatial states
            xt_next_pred, yt_next = self.transition(xt, dt, ut)
            return xt_next_pred, yt_next
            
        elif self.transition_mode == 'hybrid':
            # Hybrid model can use different modes
            hybrid_fno_config = getattr(self.config.transition, 'hybrid_fno', {})
            mode = getattr(hybrid_fno_config, 'predict_mode', 'fno_only')
            
            if mode == 'fno_only':
                xt_next_pred, yt_next = self.transition.fno_model(xt, dt, ut)
                return xt_next_pred, yt_next
            else:
                # Use traditional latent approach
                # VAE: unpack tuple, use z (which is mean in inference if not training)
                zt, _, _ = self.encoder(xt)
                zt_next, yt_next = self.transition.linear_model(zt, dt, ut)
                xt_next_pred = self.decoder(zt_next)
                return xt_next_pred, yt_next
        else:
            # Traditional latent-based approach
            # VAE: unpack tuple, use z (sampled during training, mean during eval)
            zt, _, _ = self.encoder(xt)
            zt_next, yt_next = self.transition(zt, dt, ut)
            xt_next_pred = self.decoder(zt_next)
                
            return xt_next_pred, yt_next

    def predict_latent(self, zt, dt, ut):

        zt_next, yt_next = self.transition(zt, dt, ut)

        return zt_next, yt_next

    def load_weights_from_file(self, encoder_file, decoder_file, transition_file):
        # Determine the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        # Load encoder state dict first to check for VAE
        encoder_state = torch.load(encoder_file, map_location=device, weights_only=False)
        
        # Detect if weights contain VAE layers (fc_logvar)
        weights_have_vae = 'fc_logvar.weight' in encoder_state
        model_has_vae = hasattr(self.encoder, 'fc_logvar') and self.encoder.fc_logvar is not None
        
        if weights_have_vae and not model_has_vae:
            # Weights have VAE but model doesn't - add fc_logvar layer dynamically
            print("   🔧 Detected VAE weights - adding fc_logvar layer to encoder...")
            import torch.nn as nn
            from model.utils.initialization import weights_init
            
            # Get dimensions from the fc_logvar weights in the checkpoint
            latent_dim = encoder_state['fc_logvar.weight'].shape[0]
            flattened_size = encoder_state['fc_logvar.weight'].shape[1]
            
            # Add the fc_logvar layer
            self.encoder.fc_logvar = nn.Linear(flattened_size, latent_dim)
            self.encoder.fc_logvar.apply(weights_init)
            self.encoder.enable_vae = True
            self.encoder.to(device)
        
        # Load weights with appropriate device mapping
        self.encoder.load_state_dict(encoder_state)
        self.decoder.load_state_dict(torch.load(decoder_file, map_location=device, weights_only=False))
        self.transition.load_state_dict(torch.load(transition_file, map_location=device, weights_only=False))
        
    def save_weights_to_file(self, encoder_file, decoder_file, transition_file):
        torch.save(self.encoder.state_dict(), encoder_file)
        torch.save(self.decoder.state_dict(), decoder_file)
        torch.save(self.transition.state_dict(), transition_file)

    def find_and_load_weights(self, models_dir='./saved_models/', base_pattern='e2co', specific_pattern=None):
        """
        Automatically find and load the model weights from the models_dir.
        
        Args:
            models_dir: Directory containing model files
            base_pattern: Base prefix of model files
            specific_pattern: Optional pattern to further filter model files
            
        Returns:
            Dictionary with paths of loaded model files
        """
        import os
        import glob
        import re
        
        # Find all encoder files
        encoder_pattern = os.path.join(models_dir, f"{base_pattern}_encoder*.h5")
        encoder_files = glob.glob(encoder_pattern)
        
        # Find all decoder files
        decoder_pattern = os.path.join(models_dir, f"{base_pattern}_decoder*.h5")
        decoder_files = glob.glob(decoder_pattern)
        
        # Find all transition files
        transition_pattern = os.path.join(models_dir, f"{base_pattern}_transition*.h5")
        transition_files = glob.glob(transition_pattern)
        
        if not encoder_files or not decoder_files or not transition_files:
            raise FileNotFoundError(f"No matching model files found in {models_dir}")
        
        # Further filter by specific pattern if provided
        if specific_pattern:
            encoder_files = [f for f in encoder_files if specific_pattern in f]
            decoder_files = [f for f in decoder_files if specific_pattern in f]
            transition_files = [f for f in transition_files if specific_pattern in f]
        
        # Extract epochs from filenames using regex
        def extract_epoch(filename):
            match = re.search(r'ep(\d+)', filename)
            if match:
                return int(match.group(1))
            return 0
        
        # Sort files by epoch (highest first)
        encoder_files.sort(key=extract_epoch, reverse=True)
        decoder_files.sort(key=extract_epoch, reverse=True)
        transition_files.sort(key=extract_epoch, reverse=True)
        
        # Select the highest epoch files
        encoder_file = encoder_files[0]
        decoder_file = decoder_files[0]
        transition_file = transition_files[0]
        
        print(f"Loading model weights from:")
        print(f"  Encoder: {encoder_file}")
        print(f"  Decoder: {decoder_file}")
        print(f"  Transition: {transition_file}")
        
        # Load the weights
        self.load_weights_from_file(encoder_file, decoder_file, transition_file)
        
        return {
            'encoder': encoder_file,
            'decoder': decoder_file,
            'transition': transition_file
        }

    def smart_load_compatible_weights(self, models_dir='./saved_models/', base_pattern='e2co', verbose=True):
        """
        Intelligently find and load compatible model weights, even with different configurations.
        This method provides robust loading that handles:
        - Different normalization settings (spatial and timeseries)
        - Compatible channel configurations
        - Different well configurations
        - Different training epochs
        
        Priority order:
        1. Exact configuration match
        2. Compatible channel count match
        3. Same architecture with different normalization
        4. Fallback to any available compatible model
        
        Args:
            models_dir: Directory containing model files
            base_pattern: Base prefix of model files  
            verbose: Print detailed loading information
            
        Returns:
            Dictionary with paths of loaded model files and compatibility info
        """
        import os
        import glob
        import re
        from pathlib import Path
        
        if verbose:
            print("🧠 Smart Model Loading: Searching for compatible weights...")
            print(f"   📂 Search directory: {models_dir}")
            print(f"   🔍 Base pattern: {base_pattern}")
        
        # Find all model files
        encoder_pattern = os.path.join(models_dir, f"{base_pattern}_encoder*.h5")
        decoder_pattern = os.path.join(models_dir, f"{base_pattern}_decoder*.h5") 
        transition_pattern = os.path.join(models_dir, f"{base_pattern}_transition*.h5")
        
        encoder_files = glob.glob(encoder_pattern)
        decoder_files = glob.glob(decoder_pattern)
        transition_files = glob.glob(transition_pattern)
        
        if not encoder_files or not decoder_files or not transition_files:
            raise FileNotFoundError(f"No model files found in {models_dir} with pattern {base_pattern}")
        
        if verbose:
            print(f"   📊 Found {len(encoder_files)} encoder, {len(decoder_files)} decoder, {len(transition_files)} transition files")
        
        # Extract model configuration from filenames
        def parse_model_config(filename):
            """Extract configuration parameters from model filename"""
            config = {}
            
            # Extract various parameters using regex
            patterns = {
                'nt': r'nt(\d+)',           # Number of training samples
                'latent': r'l(\d+)',        # Latent dimension
                'lr': r'lr([\d.e-]+)',      # Learning rate
                'epoch': r'ep(\d+)',        # Epoch number
                'steps': r'steps(\d+)',     # Number of steps
                'channels': r'channels(\d+)', # Number of channels
                'wells': r'wells(\d+)'      # Number of wells
            }
            
            for param, pattern in patterns.items():
                match = re.search(pattern, filename)
                if match:
                    if param in ['nt', 'latent', 'epoch', 'steps', 'channels', 'wells']:
                        config[param] = int(match.group(1))
                    elif param == 'lr':
                        config[param] = float(match.group(1))
                        
            return config
        
        # Get current model configuration
        current_config = {
            'channels': self.config.model['n_channels'],
            'latent': self.config.model['latent_dim'],
            'steps': self.config.training['nsteps'],
            'wells': self.config.data['num_prod'] + self.config.data['num_inj']
        }
        
        if verbose:
            print(f"   🎯 Current model config: {current_config}")
        
        # Analyze all available models and rank by compatibility
        model_candidates = []
        
        for encoder_file in encoder_files:
            base_name = Path(encoder_file).stem
            decoder_file = None
            transition_file = None
            
            # Find corresponding decoder and transition files
            for df in decoder_files:
                if base_name.replace('encoder', 'decoder') == Path(df).stem:
                    decoder_file = df
                    break
            
            for tf in transition_files:
                if base_name.replace('encoder', 'transition') == Path(tf).stem:
                    transition_file = tf
                    break
            
            if decoder_file and transition_file:
                file_config = parse_model_config(encoder_file)
                
                # Calculate compatibility score
                compatibility_score = 0
                compatibility_notes = []
                
                # Critical compatibility checks
                if file_config.get('channels') == current_config['channels']:
                    compatibility_score += 100  # Exact channel match (most important)
                    compatibility_notes.append("✅ Exact channel count match")
                elif file_config.get('channels', 0) > 0:
                    compatibility_notes.append(f"⚠️ Channel mismatch: model={file_config.get('channels')} vs current={current_config['channels']}")
                
                if file_config.get('latent') == current_config['latent']:
                    compatibility_score += 50   # Latent dimension match
                    compatibility_notes.append("✅ Latent dimension match")
                elif file_config.get('latent', 0) > 0:
                    compatibility_notes.append(f"⚠️ Latent dim mismatch: model={file_config.get('latent')} vs current={current_config['latent']}")
                
                if file_config.get('steps') == current_config['steps']:
                    compatibility_score += 30   # Steps match
                    compatibility_notes.append("✅ N-steps match")
                elif file_config.get('steps', 0) > 0:
                    compatibility_notes.append(f"⚠️ N-steps mismatch: model={file_config.get('steps')} vs current={current_config['steps']}")
                
                if file_config.get('wells') == current_config['wells']:
                    compatibility_score += 20   # Wells match
                    compatibility_notes.append("✅ Well count match")
                elif file_config.get('wells', 0) > 0:
                    compatibility_notes.append(f"⚠️ Well count mismatch: model={file_config.get('wells')} vs current={current_config['wells']}")
                
                # Bonus for higher epoch (more trained)
                epoch = file_config.get('epoch', 0)
                compatibility_score += min(epoch, 10)  # Cap bonus at 10 points
                
                model_candidates.append({
                    'encoder': encoder_file,
                    'decoder': decoder_file,
                    'transition': transition_file,
                    'config': file_config,
                    'compatibility_score': compatibility_score,
                    'compatibility_notes': compatibility_notes,
                    'epoch': epoch
                })
        
        if not model_candidates:
            raise FileNotFoundError("No complete model file sets found (need encoder, decoder, transition)")
        
        # Sort by compatibility score (highest first), then by epoch
        model_candidates.sort(key=lambda x: (x['compatibility_score'], x['epoch']), reverse=True)
        
        if verbose:
            pass  # Model compatibility analysis
            for i, candidate in enumerate(model_candidates[:5]):  # Show top 5
                print(f"      {i+1}. Score: {candidate['compatibility_score']}, Epoch: {candidate['epoch']}")
                print(f"         Config: {candidate['config']}")
                for note in candidate['compatibility_notes']:
                    print(f"         {note}")
                print()
        
        # Select the most compatible model
        best_candidate = model_candidates[0]
        
        # Validate critical compatibility
        critical_mismatch = False
        critical_issues = []
        
        if best_candidate['config'].get('channels', 0) != current_config['channels']:
            critical_issues.append(f"Channel count mismatch: {best_candidate['config'].get('channels')} vs {current_config['channels']}")
            
        if best_candidate['config'].get('latent', 0) != current_config['latent']:
            critical_issues.append(f"Latent dimension mismatch: {best_candidate['config'].get('latent')} vs {current_config['latent']}")
        
        if critical_issues:
            print("⚠️ CRITICAL COMPATIBILITY WARNINGS:")
            for issue in critical_issues:
                print(f"   • {issue}")
            print("   This may cause loading errors or poor performance.")
            print("   Consider retraining with current configuration for best results.")
            
            # Ask user if they want to proceed
            try:
                response = input("   Continue loading anyway? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    raise RuntimeError("Model loading cancelled due to compatibility issues")
            except (EOFError, KeyboardInterrupt):
                # In non-interactive environments, proceed with warning
                print("   Non-interactive environment detected. Proceeding with warnings...")
        
        # Load the selected model
        encoder_file = best_candidate['encoder']
        decoder_file = best_candidate['decoder']
        transition_file = best_candidate['transition']
        
        if verbose:
            print(f"\n🚀 Loading best compatible model:")
            print(f"   📊 Compatibility Score: {best_candidate['compatibility_score']}")
            print(f"   📈 Epoch: {best_candidate['epoch']}")
            print(f"   📁 Files:")
            print(f"      • Encoder: {Path(encoder_file).name}")
            print(f"      • Decoder: {Path(decoder_file).name}")
            print(f"      • Transition: {Path(transition_file).name}")
        
        try:
            # Load the weights with error handling
            self.load_weights_from_file(encoder_file, decoder_file, transition_file)
            
            if verbose:
                print("✅ Model weights loaded successfully!")
                
                # Print compatibility summary
                pass  # Compatibility summary
                for note in best_candidate['compatibility_notes']:
                    print(f"   {note}")
                    
                if not critical_issues:
                    print("   🎉 No critical compatibility issues detected!")
                    
        except Exception as e:
            print(f"❌ Error loading model weights: {e}")
            print("   This might be due to architecture incompatibility.")
            print("   Consider retraining the model with current configuration.")
            raise
            
        return {
            'encoder': encoder_file,
            'decoder': decoder_file,
            'transition': transition_file,
            'config': best_candidate['config'],
            'compatibility_score': best_candidate['compatibility_score'],
            'compatibility_notes': best_candidate['compatibility_notes'],
            'critical_issues': critical_issues
        }


# =====================================
# SECTION 4: TRAINING WRAPPER
# ===================================== 
