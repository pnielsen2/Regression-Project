#!/bin/bash

# Update and install system dependencies
apt-get update
apt-get install -y python3-pip

# Install project dependencies
pip3 install -r requirements.txt

# Set up wandb
wandb login

# Run the sweep
python3 scripts/run_sweep.py