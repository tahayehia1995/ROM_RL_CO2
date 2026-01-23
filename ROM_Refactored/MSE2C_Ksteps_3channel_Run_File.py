"""
Main Run File for E2C Model
==========================
This file orchestrates the three main parts of the ROM project:
1. Data Preprocessing Dashboard
2. Model Training Dashboard  
3. Testing & Visualization Dashboard

Each part is independent with its own configuration and functionality.
"""
#%%
import torch

# ============================================================================
# Hardware Configuration
# ============================================================================
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

# ============================================================================
# STEP 1: DATA PREPROCESSING DASHBOARD
# ============================================================================
# Import and create data preprocessing dashboard
from data_preprocessing import create_data_preprocessing_dashboard

# Create and display the interactive dashboard
preprocessing_dashboard = create_data_preprocessing_dashboard()



#%%
# ============================================================================
# STEP 2: MODEL TRAINING DASHBOARD
# ============================================================================
# Import and create training dashboard
from model.training import create_training_dashboard

# Create and display the training dashboard
training_dashboard = create_training_dashboard(config_path='config.yaml')


#%%
# ============================================================================
# STEP 3: TESTING & VISUALIZATION DASHBOARD
# ============================================================================
# Import and create testing dashboard
from testing import create_testing_dashboard

# Create and display the testing dashboard
testing_dashboard = create_testing_dashboard()


