# Generate data first
    bash generate_data.sh

# Run with optimized settings for 5x5
    python run_experiment.py 5

# Or run with custom hyperparameters
    python hex_solution.py --board-size 5 --epochs 60 --number-of-clauses 4000 --T 3000 --s 3.0