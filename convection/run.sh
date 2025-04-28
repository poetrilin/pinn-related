#!/bin/bash

# Define an array of model names
model_names=("pinn" "kan" "powermlp")

# Loop through each model name
for model_name in "${model_names[@]}"; do
    echo "Running train.py with model_name=${model_name}"
    python3 train.py --model_name "${model_name}"
    
    echo "Running metric.py with model_name=${model_name}"
    python3 metric.py --model_name "${model_name}"
done

echo "All tasks completed."