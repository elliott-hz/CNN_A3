#!/bin/bash

# Script to run GoogLeNet with Auxiliary Classifiers experiment
echo "Starting GoogLeNet with Auxiliary Classifiers experiment..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python is not installed or not in PATH. Exiting."
    exit 1
fi

# Run the experiment
echo "Running experiment..."
python experiments/exp11_classification_googlenet_with_auxiliary.py

echo "Experiment completed!"