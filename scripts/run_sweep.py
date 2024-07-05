import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now you can use os.getenv to get your API keys
wandb_api_key = os.getenv('WANDB_API_KEY')

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
import wandb
from data.data_preprocessing import load_and_preprocess_data
from training.train import train
from config.wandb_config import sweep_config

# Ensure wandb is logged in
wandb.login()

# Initialize wandb sweep
sweep_id = wandb.sweep(sweep_config, project='model-comparison')

# Check for available devices
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load and preprocess data
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = load_and_preprocess_data(device)

# Define the training function for wandb
def sweep_train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        train(config, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device)

# Run the sweep
wandb.agent(sweep_id, function=sweep_train)