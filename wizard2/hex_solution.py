# hex_solution.py

import numpy as np
import pandas as pd
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse
from tqdm import tqdm
import os

def default_args(**kwargs):
    parser = argparse.ArgumentParser(description='Hex Winner Prediction with GTM')
    
    # Data parameters
    parser.add_argument("--board-size", default=5, type=int, help="Size of the Hex board (5, 6, 7, or 8)")
    parser.add_argument("--data-dir", default="data", type=str, help="Directory containing the data files")
    parser.add_argument("--train-size", default=800000, type=int, help="Number of training samples")
    parser.add_argument("--test-size", default=200000, type=int, help="Number of test samples")
    
    # GTM hyperparameters - optimized for Hex
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--number-of-clauses", default=8000, type=int, help="Number of clauses")
    parser.add_argument("--T", default=5000, type=int, help="Threshold T")
    parser.add_argument("--s", default=5.0, type=float, help="Specificity s")
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=2, type=int, help="Depth of graph reasoning")
    parser.add_argument("--hypervector-size", default=128, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument("--max-included-literals", default=32, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument('--one-hot-encoding', dest='one_hot_encoding', default=False, action='store_true')
    
    # Training options
    parser.add_argument("--save-model", default=True, type=bool, help="Save the trained model")
    parser.add_argument("--model-path", default="models", type=str, help="Path to save models")
    
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

def load_hex_data(filepath, board_size, num_samples=None):
    """Load Hex game data from CSV file"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    if num_samples:
        df = df.iloc[:num_samples]
    
    # Extract board positions
    board_cols = [col for col in df.columns if col.startswith('cell')]
    X = df[board_cols].values
    
    # Extract winner labels (1 for X/first player, -1 for O/second player)
    Y = df['winner'].values
    # Convert to binary: 0 for player O (-1), 1 for player X (1)
    Y = (Y == 1).astype(np.uint32)
    
    return X, Y

def create_hex_graph(board_size):
    """
    Create the graph structure for a Hex board.
    Returns a dictionary mapping each cell to its neighbors.
    """
    neighbors = {}
    
    for row in range(board_size):
        for col in range(board_size):
            cell_id = row * board_size + col
            neighbors[cell_id] = []
            
            # Six possible directions in hex grid
            directions = [
                (-1, 0),   # North
                (-1, 1),   # Northeast
                (0, 1),    # East
                (1, 0),    # South
                (1, -1),   # Southwest
                (0, -1)    # West
            ]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < board_size and 0 <= new_col < board_size:
                    neighbor_id = new_row * board_size + new_col
                    neighbors[cell_id].append(neighbor_id)
    
    return neighbors

def build_graphs(X, Y, board_size, args, init_with=None):
    """
    Build Graphs object for GTM from board positions.
    """
    num_samples = X.shape[0]
    num_cells = board_size * board_size
    
    # Symbols: X_piece, O_piece, Empty
    symbols = ['X', 'O']
    
    print(f"Creating graphs for {num_samples} games...")
    
    if init_with is None:
        graphs = Graphs(
            num_samples,
            symbols=symbols,
            hypervector_size=args.hypervector_size,
            hypervector_bits=args.hypervector_bits,
            double_hashing=args.double_hashing,
            one_hot_encoding=args.one_hot_encoding
        )
    else:
        graphs = Graphs(num_samples, init_with=init_with)
    
    # Get hex connectivity
    hex_neighbors = create_hex_graph(board_size)
    
    # Set number of nodes for each graph
    for graph_id in range(num_samples):
        graphs.set_number_of_graph_nodes(graph_id, num_cells)
    
    graphs.prepare_node_configuration()
    
    # Add nodes and count edges
    for graph_id in range(num_samples):
        for node_id in range(num_cells):
            num_edges = len(hex_neighbors[node_id])
            graphs.add_graph_node(graph_id, node_id, num_edges)
    
    graphs.prepare_edge_configuration()
    
    # Add edges for graph structure
    print("Adding edges...")
    for graph_id in tqdm(range(num_samples), desc="Building graph structure"):
        for node_id in range(num_cells):
            for neighbor_id in hex_neighbors[node_id]:
                graphs.add_graph_node_edge(graph_id, node_id, neighbor_id, "adjacent")
    
    # Add node features (piece positions)
    print("Adding node features...")
    for graph_id in tqdm(range(num_samples), desc="Adding features"):
        board = X[graph_id]
        for node_id in range(num_cells):
            cell_value = board[node_id]
            if cell_value == 1:  # X piece
                graphs.add_graph_node_property(graph_id, node_id, 'X')
            elif cell_value == -1:  # O piece
                graphs.add_graph_node_property(graph_id, node_id, 'O')
            # Empty cells get no properties
    
    graphs.encode()
    print("Graphs built successfully!")
    
    return graphs

def train_and_evaluate(args):
    """Main training and evaluation function"""
    
    board_size = args.board_size
    
    # Load data for all three scenarios
    scenarios = {
        'final': f'hex_{board_size}x{board_size}_*_final.csv',
        'minus2': f'hex_{board_size}x{board_size}_*_minus2.csv',
        'minus5': f'hex_{board_size}x{board_size}_*_minus5.csv'
    }
    
    results = {}
    
    for scenario_name, file_pattern in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Training for scenario: {scenario_name}")
        print(f"{'='*60}\n")
        
        # Find the data file
        import glob
        files = glob.glob(os.path.join(args.data_dir, file_pattern))
        if not files:
            print(f"Warning: No file found for pattern {file_pattern}")
            continue
        
        data_file = files[0]
        
        # Load data
        X_all, Y_all = load_hex_data(data_file, board_size)
        
        # Split into train/test
        train_size = min(args.train_size, len(X_all) - args.test_size)
        X_train = X_all[:train_size]
        Y_train = Y_all[:train_size]
        X_test = X_all[train_size:train_size + args.test_size]
        Y_test = Y_all[train_size:train_size + args.test_size]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Build graphs
        print("\nBuilding training graphs...")
        graphs_train = build_graphs(X_train, Y_train, board_size, args)
        
        print("\nBuilding test graphs...")
        graphs_test = build_graphs(X_test, Y_test, board_size, args, init_with=graphs_train)
        
        # Create and train GTM
        print("\nInitializing Graph Tsetlin Machine...")
        tm = MultiClassGraphTsetlinMachine(
            args.number_of_clauses,
            args.T,
            args.s,
            number_of_state_bits=args.number_of_state_bits,
            depth=args.depth,
            message_size=args.message_size,
            message_bits=args.message_bits,
            max_included_literals=args.max_included_literals,
            double_hashing=args.double_hashing,
            one_hot_encoding=args.one_hot_encoding
        )
        
        # Training
        print(f"\nTraining for {args.epochs} epochs...")
        best_test_acc = 0.0
        
        for epoch in range(args.epochs):
            start_train = time()
            tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
            train_time = time() - start_train
            
            start_test = time()
            predictions_test = tm.predict(graphs_test)
            test_acc = 100 * (predictions_test == Y_test).mean()
            test_time = time() - start_test
            
            predictions_train = tm.predict(graphs_train)
            train_acc = 100 * (predictions_train == Y_train).mean()
            
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}% | "
                  f"Train Time: {train_time:.2f}s | "
                  f"Test Time: {test_time:.2f}s")
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                
                # Save model if requested
                if args.save_model and test_acc >= 99.0:
                    os.makedirs(args.model_path, exist_ok=True)
                    model_file = os.path.join(
                        args.model_path, 
                        f'hex_{board_size}x{board_size}_{scenario_name}_acc{test_acc:.1f}.pkl'
                    )
                    tm.save(model_file)
                    print(f"Model saved to {model_file}")
            
            # Early stopping if we hit 100%
            if test_acc >= 100.0:
                print(f"\n🎉 Achieved 100% accuracy on {scenario_name}!")
                break
        
        results[scenario_name] = {
            'train_acc': train_acc,
            'test_acc': best_test_acc,
            'num_clauses': args.number_of_clauses
        }
        
        print(f"\nBest test accuracy for {scenario_name}: {best_test_acc:.2f}%")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS for {board_size}x{board_size} board")
    print(f"{'='*60}")
    for scenario, result in results.items():
        print(f"{scenario:10s}: Test Acc = {result['test_acc']:.2f}%")
    print(f"{'='*60}\n")
    
    return results

if __name__ == "__main__":
    args = default_args()
    
    print(f"\n{'='*60}")
    print(f"Hex Winner Prediction with Graph Tsetlin Machine")
    print(f"Board Size: {args.board_size}x{args.board_size}")
    print(f"{'='*60}\n")
    
    print("Hyperparameters:")
    print(f"  Clauses: {args.number_of_clauses}")
    print(f"  T: {args.T}")
    print(f"  s: {args.s}")
    print(f"  Depth: {args.depth}")
    print(f"  Max included literals: {args.max_included_literals}")
    print(f"  Epochs: {args.epochs}\n")
    
    results = train_and_evaluate(args)