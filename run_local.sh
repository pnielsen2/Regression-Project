#!/bin/bash

# run_local.sh

# Activate your virtual environment if you're using one
# source /path/to/your/venv/bin/activate

# Install project dependencies
pip install -r requirements.txt

# Set up wandb (you may need to run this manually the first time)
wandb login

# Run the sweep
python scripts/run_sweep.py