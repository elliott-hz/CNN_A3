#!/bin/bash

# Script to run GoogLeNet with auxiliary classifiers experiment
echo "Running GoogLeNet with auxiliary classifiers experiment..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the experiment
python experiments/exp11_classification_googlenet_with_auxiliary.py "$@"

if [ $? -eq 0 ]; then
    echo "Experiment completed successfully!"
else
    echo "Experiment failed!"
    exit 1
fi