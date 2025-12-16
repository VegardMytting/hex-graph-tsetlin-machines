# analyze_clauses.py
"""
Analyze and interpret learned clauses
"""

import numpy as np
import pickle
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from hex_solution import create_hex_graph

def analyze_model(model_path, board_size):
    """Analyze a trained model to understand learned patterns"""
    
    print(f"Loading model from {model_path}...")
    
    with open(model_path, 'rb') as f:
        state_dict = pickle.load(f)
    
    # Load model
    tm = MultiClassGraphTsetlinMachine(
        state_dict['number_of_clauses'],
        state_dict['T'],
        state_dict['s'][0] if isinstance(state_dict['s'], tuple) else state_dict['s'],
        number_of_state_bits=state_dict['number_of_state_bits'],
        depth=state_dict['depth'],
        message_size=state_dict['message_size'],
        message_bits=state_dict['message_bits']
    )
    
    tm.load(fname=model_path)
    
    # Get weights
    weights = tm.get_weights()
    
    print(f"\n{'='*60}")
    print(f"Model Analysis for {board_size}x{board_size} Hex")
    print(f"{'='*60}\n")
    
    print(f"Number of clauses: {tm.number_of_clauses}")
    print(f"Depth: {tm.depth}")
    print(f"Number of literals: {tm.number_of_literals}")
    
    # Analyze clause weights
    print(f"\nClause Weight Statistics:")
    for class_id in range(weights.shape[0]):
        class_weights = weights[class_id]
        print(f"\nClass {class_id} ({'X wins' if class_id == 1 else 'O wins'}):")
        print(f"  Positive weights: {(class_weights > 0).sum()}")
        print(f"  Negative weights: {(class_weights < 0).sum()}")
        print(f"  Max weight: {class_weights.max()}")
        print(f"  Min weight: {class_weights.min()}")
        print(f"  Mean absolute weight: {np.abs(class_weights).mean():.2f}")
    
    # Find most important clauses
    print(f"\nTop 10 clauses by absolute weight:")
    for class_id in range(weights.shape[0]):
        print(f"\nClass {class_id}:")
        class_weights = weights[class_id]
        top_indices = np.argsort(np.abs(class_weights))[-10:][::-1]
        
        for rank, idx in enumerate(top_indices, 1):
            weight = class_weights[idx]
            print(f"  {rank}. Clause {idx}: weight = {weight:+d}")

if __name__ == "__main__":
    import sys
    import glob
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        board_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    else:
        # Find most recent model
        models = glob.glob("models/hex_*.pkl")
        if not models:
            print("No models found in models/ directory")
            sys.exit(1)
        model_path = max(models, key=lambda x: x.split('acc')[-1])
        board_size = 5
    
    analyze_model(model_path, board_size)