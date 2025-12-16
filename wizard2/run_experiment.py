# run_experiment.py
"""
Run Hex experiments with optimized hyperparameters
"""

import sys
from hex_solution import default_args, train_and_evaluate
from hex_configs import get_config

def run_optimized_experiment(board_size):
    """Run experiment with optimized hyperparameters for the given board size"""
    
    config = get_config(board_size)
    
    # Override default args with optimized config
    args = default_args(**config, board_size=board_size)
    
    print(f"\n🎯 Running OPTIMIZED experiment for {board_size}x{board_size} board")
    print(f"Using configuration: {config}\n")
    
    results = train_and_evaluate(args)
    
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        board_size = int(sys.argv[1])
    else:
        board_size = 5  # Default to 5x5
    
    results = run_optimized_experiment(board_size)
    
    # Check if we achieved 100% on all scenarios
    all_perfect = all(r['test_acc'] >= 100.0 for r in results.values())
    
    if all_perfect:
        print(f"\n🏆 SUCCESS! Achieved 100% accuracy on ALL scenarios for {board_size}x{board_size} board!")
    else:
        print(f"\n📊 Results Summary:")
        for scenario, result in results.items():
            status = "✅" if result['test_acc'] >= 100.0 else "❌"
            print(f"  {status} {scenario}: {result['test_acc']:.2f}%")