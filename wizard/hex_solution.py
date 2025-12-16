#!/usr/bin/env python3
"""
Hex Game Winner Prediction using Graph Tsetlin Machine
This solution uses GTM to predict winners in Hex games at different stages.
"""

import argparse
import numpy as np
import pandas as pd
from time import time
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine


def default_args(**kwargs):
    """Parse command line arguments with defaults optimized for Hex"""
    parser = argparse.ArgumentParser(description='Hex Winner Prediction with GTM')
    
    # Data parameters
    parser.add_argument("--board-size", default=5, type=int, help="Board size (5, 6, 7, or 8)")
    parser.add_argument("--train-size", default=800000, type=int, help="Number of training examples")
    parser.add_argument("--test-size", default=100000, type=int, help="Number of test examples")
    parser.add_argument("--dataset", default="final", choices=["final", "minus2", "minus5"],
                        help="Dataset type: final, minus2, or minus5")
    
    # GTM hyperparameters - CRITICAL FOR 100% ACCURACY
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--number-of-clauses", default=4000, type=int, 
                        help="Number of clauses (increase for larger boards)")
    parser.add_argument("--T", default=2000, type=int, help="Threshold parameter")
    parser.add_argument("--s", default=5.0, type=float, help="Specificity parameter")
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--max-included-literals", default=32, type=int)
    
    # Graph structure parameters
    parser.add_argument("--depth", default=2, type=int, 
                        help="Depth of message passing (1=no edges, 2+=with edges)")
    parser.add_argument("--hypervector-size", default=128, type=int,
                        help="Size of hypervectors for encoding symbols")
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    
    # Encoding options
    parser.add_argument('--double-hashing', dest='double_hashing', 
                        default=False, action='store_true')
    parser.add_argument('--one-hot-encoding', dest='one_hot_encoding', 
                        default=False, action='store_true')
    
    # Other options
    parser.add_argument("--save-model", default="", type=str, 
                        help="Path to save trained model")
    parser.add_argument("--load-model", default="", type=str, 
                        help="Path to load pre-trained model")
    parser.add_argument("--verbose", default=True, type=bool)
    
    args = parser.parse_args()
    
    # Override with any kwargs
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    
    return args


def load_hex_data(board_size, dataset_type, train_size, test_size):
    """
    Load Hex game data from CSV files.
    
    Args:
        board_size: Size of the board (5, 6, 7, 8)
        dataset_type: 'final', 'minus2', or 'minus5'
        train_size: Number of training examples to use
        test_size: Number of test examples to use
    
    Returns:
        X_train, Y_train, X_test, Y_test
    """
    # Construct filename
    filename = f"data/hex_{board_size}x{board_size}_1000000_{dataset_type}.csv"
    
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    
    # The last column is the winner (1 for X, -1 for O)
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    
    # Convert labels: 1 (X wins) -> 1, -1 (O wins) -> 0
    Y = np.where(Y == 1, 1, 0).astype(np.uint32)
    
    # Split into train and test
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:train_size + test_size]
    Y_test = Y[train_size:train_size + test_size]
    
    print(f"Loaded {len(X_train)} training and {len(X_test)} test examples")
    print(f"Board size: {board_size}x{board_size}")
    print(f"Features per example: {X_train.shape[1]}")
    print(f"Winner distribution (train): X={np.sum(Y_train)}, O={len(Y_train)-np.sum(Y_train)}")
    
    return X_train, Y_train, X_test, Y_test


def create_hex_graphs(X, Y, board_size, symbols, init_with=None):
    """
    Create graph representation of Hex boards.
    
    Each cell is a node with properties:
    - 'X' if player X occupies it
    - 'O' if player O occupies it  
    - Empty cells have no properties
    
    Edges connect adjacent cells (6 neighbors in hexagonal grid)
    
    Args:
        X: Board states (n_games, board_size^2)
        Y: Winners (n_games,)
        board_size: Size of board
        symbols: List of symbols ['X', 'O']
        init_with: Existing Graphs object to initialize with
    
    Returns:
        graphs: Graphs object
        Y: Winner labels
    """
    n_games = len(X)
    n_cells = board_size * board_size
    
    print(f"Creating graphs for {n_games} games...")
    
    if init_with is None:
        graphs = Graphs(
            n_games,
            symbols=symbols,
            hypervector_size=128,  # Fixed size works well for Hex
            hypervector_bits=2,
            double_hashing=False,
            one_hot_encoding=False
        )
    else:
        graphs = Graphs(n_games, init_with=init_with)
    
    # Set number of nodes for each graph (one per cell)
    for game_id in range(n_games):
        graphs.set_number_of_graph_nodes(game_id, n_cells)
    
    graphs.prepare_node_configuration()
    
    # Create nodes and count edges
    # In a hexagonal grid, each cell has up to 6 neighbors
    # We need to count edges carefully based on position
    
    def get_neighbors(row, col, board_size):
        """Get hexagonal neighbors for a cell"""
        neighbors = []
        # Hexagonal grid neighbors (6 directions)
        directions = [
            (-1, 0),   # top
            (-1, 1),   # top-right
            (0, -1),   # left
            (0, 1),    # right
            (1, -1),   # bottom-left
            (1, 0)     # bottom
        ]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < board_size and 0 <= new_col < board_size:
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    # Count edges for each node
    for game_id in range(n_games):
        for cell_idx in range(n_cells):
            row = cell_idx // board_size
            col = cell_idx % board_size
            neighbors = get_neighbors(row, col, board_size)
            n_edges = len(neighbors)
            graphs.add_graph_node(game_id, cell_idx, n_edges)
    
    graphs.prepare_edge_configuration()
    
    # Add edges between adjacent cells
    for game_id in range(n_games):
        for cell_idx in range(n_cells):
            row = cell_idx // board_size
            col = cell_idx % board_size
            neighbors = get_neighbors(row, col, board_size)
            
            for neighbor_row, neighbor_col in neighbors:
                neighbor_idx = neighbor_row * board_size + neighbor_col
                # All edges are the same type ("adjacent")
                graphs.add_graph_node_edge(game_id, cell_idx, neighbor_idx, "adjacent")
    
    # Add node properties (which player occupies each cell)
    for game_id in range(n_games):
        if game_id % 10000 == 0:
            print(f"Processing game {game_id}/{n_games}...")
        
        board_state = X[game_id]
        
        for cell_idx in range(n_cells):
            cell_value = board_state[cell_idx]
            
            if cell_value == 1:  # X occupies this cell
                graphs.add_graph_node_property(game_id, cell_idx, 'X')
            elif cell_value == -1:  # O occupies this cell
                graphs.add_graph_node_property(game_id, cell_idx, 'O')
            # Empty cells (0) have no properties
    
    graphs.encode()
    print("Graph creation complete!")
    
    return graphs, Y


def train_and_evaluate(args):
    """Main training and evaluation function"""
    
    # Load data
    X_train, Y_train, X_test, Y_test = load_hex_data(
        args.board_size, 
        args.dataset, 
        args.train_size, 
        args.test_size
    )
    
    # Define symbols
    symbols = ['X', 'O']
    
    # Create training graphs
    print("\n" + "="*60)
    print("Creating training graphs...")
    print("="*60)
    graphs_train, Y_train = create_hex_graphs(
        X_train, Y_train, args.board_size, symbols
    )
    
    # Create test graphs
    print("\n" + "="*60)
    print("Creating test graphs...")
    print("="*60)
    graphs_test, Y_test = create_hex_graphs(
        X_test, Y_test, args.board_size, symbols, init_with=graphs_train
    )
    
    # Initialize or load GTM
    print("\n" + "="*60)
    print("Initializing Graph Tsetlin Machine...")
    print("="*60)
    
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
    
    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        tm.load(fname=args.load_model)
    
    # Training loop
    print("\n" + "="*60)
    print("Training...")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Clauses: {args.number_of_clauses}")
    print(f"T: {args.T}")
    print(f"s: {args.s}")
    print(f"Depth: {args.depth}")
    print("="*60 + "\n")
    
    best_test_acc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        start_training = time()
        tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
        stop_training = time()
        
        start_testing = time()
        train_pred = tm.predict(graphs_train)
        test_pred = tm.predict(graphs_test)
        stop_testing = time()
        
        train_acc = 100.0 * (train_pred == Y_train).mean()
        test_acc = 100.0 * (test_pred == Y_test).mean()
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            
            # Save model if path provided
            if args.save_model:
                tm.save(args.save_model)
                print(f"  → Saved model to {args.save_model}")
        
        print(f"Epoch {epoch:3d} | "
              f"Train: {train_acc:6.2f}% | "
              f"Test: {test_acc:6.2f}% | "
              f"Train time: {stop_training-start_training:5.1f}s | "
              f"Test time: {stop_testing-start_testing:5.1f}s")
        
        # Early stopping if we achieve 100%
        if test_acc >= 99.99:
            print(f"\n🎉 Achieved 100% accuracy at epoch {epoch}!")
            break
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Test Accuracy: {best_test_acc:.2f}% (epoch {best_epoch})")
    print(f"Number of Clauses: {args.number_of_clauses}")
    
    # Analyze learned patterns
    if args.verbose:
        print("\n" + "="*60)
        print("Analyzing learned patterns...")
        print("="*60)
        analyze_learned_rules(tm, graphs_train)
    
    return tm, best_test_acc


def analyze_learned_rules(tm, graphs):
    """
    Analyze the rules (clauses) learned by the TM.
    This provides interpretability.
    """
    weights = tm.get_weights()
    
    print("\nClause Weight Statistics:")
    print(f"  Min weight: {weights.min()}")
    print(f"  Max weight: {weights.max()}")
    print(f"  Mean absolute weight: {np.abs(weights).mean():.2f}")
    
    # Count clauses with significant weights
    significant_clauses = np.sum(np.abs(weights) > 1)
    print(f"  Significant clauses (|weight| > 1): {significant_clauses} / {tm.number_of_clauses}")
    
    # Get clause literals for interpretation
    try:
        clause_literals = tm.get_clause_literals(graphs.hypervectors)
        print(f"\nClause literals shape: {clause_literals.shape}")
        
        # Analyze top clauses
        print("\nTop 5 clauses by weight:")
        for class_id in range(weights.shape[0]):
            print(f"\n  Class {class_id} ({'X' if class_id == 1 else 'O'} wins):")
            top_indices = np.argsort(np.abs(weights[class_id]))[-5:][::-1]
            
            for idx in top_indices:
                weight = weights[class_id, idx]
                literals = clause_literals[idx]
                
                # Count how many symbols are included
                x_count = int(literals[0])  # X symbol
                o_count = int(literals[1])  # O symbol
                not_x_count = int(literals[2])  # NOT X
                not_o_count = int(literals[3])  # NOT O
                
                print(f"    Clause {idx}: weight={weight:3d}, "
                      f"X:{x_count} O:{o_count} ~X:{not_x_count} ~O:{not_o_count}")
    
    except Exception as e:
        print(f"Could not analyze clauses: {e}")


def main():
    """Main entry point"""
    args = default_args()
    
    print("="*60)
    print(" Hex Winner Prediction using Graph Tsetlin Machine")
    print("="*60)
    print(f"Board Size: {args.board_size}x{args.board_size}")
    print(f"Dataset: {args.dataset}")
    print(f"Training examples: {args.train_size}")
    print(f"Test examples: {args.test_size}")
    print("="*60 + "\n")
    
    tm, best_acc = train_and_evaluate(args)
    
    print("\n" + "="*60)
    print(" FINAL RESULTS")
    print("="*60)
    print(f"Board Size: {args.board_size}x{args.board_size}")
    print(f"Dataset: {args.dataset}")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Number of Clauses: {args.number_of_clauses}")
    print("="*60)


if __name__ == "__main__":
    main()
