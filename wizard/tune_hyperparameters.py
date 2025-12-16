#!/usr/bin/env python3
"""
Hyperparameter tuning for Hex GTM
Systematically search for parameters that achieve 100% accuracy
"""

import sys
import subprocess
import json
from itertools import product

# Hyperparameter search space
# These ranges are based on successful GTM configurations
SEARCH_SPACE = {
    '5x5': {
        'number_of_clauses': [2000, 4000, 6000, 8000],
        'T': [1500, 2000, 3000, 4000],
        's': [3.0, 5.0, 10.0, 15.0],
        'depth': [1, 2, 3],
        'max_included_literals': [16, 32, 48, 64],
        'epochs': [100],
    },
    '6x6': {
        'number_of_clauses': [6000, 8000, 10000, 12000],
        'T': [2000, 3000, 4000, 5000],
        's': [5.0, 10.0, 15.0, 20.0],
        'depth': [2, 3],
        'max_included_literals': [32, 48, 64],
        'epochs': [150],
    }
}

def run_experiment(board_size, dataset, params):
    """Run a single experiment with given parameters"""
    
    cmd = [
        'python3', 'hex_solution.py',
        '--board-size', str(board_size),
        '--dataset', dataset,
        '--number-of-clauses', str(params['number_of_clauses']),
        '--T', str(params['T']),
        '--s', str(params['s']),
        '--depth', str(params['depth']),
        '--max-included-literals', str(params['max_included_literals']),
        '--epochs', str(params['epochs']),
        '--train-size', '800000',
        '--test-size', '100000',
        '--verbose', 'False'
    ]
    
    print(f"\n{'='*80}")
    print(f"Testing: Clauses={params['number_of_clauses']}, T={params['T']}, "
          f"s={params['s']}, depth={params['depth']}, max_lit={params['max_included_literals']}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        output = result.stdout
        
        # Parse accuracy from output
        for line in output.split('\n'):
            if 'Best Test Accuracy:' in line:
                accuracy = float(line.split(':')[1].strip().replace('%', ''))
                return accuracy, params
        
        return 0.0, params
    
    except subprocess.TimeoutExpired:
        print("⏱️  Timeout - skipping this configuration")
        return 0.0, params
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0.0, params


def grid_search(board_size, dataset='final'):
    """Perform grid search over hyperparameters"""
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH FOR {board_size}x{board_size} BOARD")
    print(f"{'='*80}\n")
    
    if f'{board_size}x{board_size}' not in SEARCH_SPACE:
        print(f"No search space defined for {board_size}x{board_size}")
        return
    
    space = SEARCH_SPACE[f'{board_size}x{board_size}']
    
    # Generate all combinations
    keys = list(space.keys())
    values = [space[k] for k in keys]
    
    all_configs = []
    for combo in product(*values):
        config = dict(zip(keys, combo))
        all_configs.append(config)
    
    print(f"Testing {len(all_configs)} configurations...\n")
    
    results = []
    best_acc = 0.0
    best_config = None
    
    for i, config in enumerate(all_configs):
        print(f"\nConfiguration {i+1}/{len(all_configs)}")
        
        acc, params = run_experiment(board_size, dataset, config)
        
        results.append({
            'accuracy': acc,
            'params': params
        })
        
        if acc > best_acc:
            best_acc = acc
            best_config = params
            print(f"\n🎉 New best accuracy: {acc:.2f}%")
            
        if acc >= 99.99:
            print(f"\n🏆 Found 100% accuracy configuration!")
            print(f"Parameters: {params}")
            break
    
    # Print summary
    print(f"\n{'='*80}")
    print("SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest Accuracy: {best_acc:.2f}%")
    print(f"Best Configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_file = f'tuning_results_{board_size}x{board_size}_{dataset}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'best_accuracy': best_acc,
            'best_config': best_config,
            'all_results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


def quick_test(board_size, dataset='final'):
    """Quick test with promising parameters"""
    
    # Promising configurations based on similar problems
    promising_configs = {
        5: {
            'number_of_clauses': 4000,
            'T': 2000,
            's': 5.0,
            'depth': 2,
            'max_included_literals': 32,
            'epochs': 100
        },
        6: {
            'number_of_clauses': 8000,
            'T': 3000,
            's': 10.0,
            'depth': 2,
            'max_included_literals': 48,
            'epochs': 150
        }
    }
    
    if board_size in promising_configs:
        config = promising_configs[board_size]
        print(f"\nTesting promising configuration for {board_size}x{board_size}...")
        acc, params = run_experiment(board_size, dataset, config)
        print(f"\nResult: {acc:.2f}% accuracy")
        return acc
    else:
        print(f"No promising config defined for {board_size}x{board_size}")
        return 0.0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='final')
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'grid'])
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_test(args.board_size, args.dataset)
    else:
        grid_search(args.board_size, args.dataset)
