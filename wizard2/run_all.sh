#!/bin/bash
# run_all.sh - Run experiments for all board sizes

echo "Starting Hex experiments..."

for size in 5 6 7 8; do
    echo ""
    echo "================================"
    echo "Running experiment for ${size}x${size} board"
    echo "================================"
    python run_experiment.py $size
done

echo ""
echo "All experiments completed!"