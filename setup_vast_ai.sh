#!/bin/bash

# Update and install system dependencies
apt-get update
apt-get install -y python3-pip

# Install project dependencies
pip3 install -r requirements.txt

# Set up wandb using the environment variable
wandb login $WANDB_API_KEY

# Run the sweep
python3 scripts/run_sweep.py