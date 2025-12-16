from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
import numpy as np
import pandas as pd
from time import time
import argparse
import os

"""
Hex Winner Prediction - IMPROVED VERSION with diagnostics
"""

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    # Start with MORE AGGRESSIVE hyperparameters for learning
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--number-of-clauses", default=500, type=int)  # Start smaller
    parser.add_argument("--T", default=500, type=int)  # Much lower T
    parser.add_argument("--s", default=2.0, type=float)  # Lower s for more learning
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=1, type=int)  # Start without message passing
    parser.add_argument("--hypervector-size", default=64, type=int)  # Smaller
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=128, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument("--max-included-literals", default=16, type=int)  # More restrictive
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument('--one-hot-encoding', dest='one_hot_encoding', default=False, action='store_true')
    
    # Data parameters
    parser.add_argument("--board-size", default=5, type=int)
    parser.add_argument("--train-size", default=5000, type=int)  # Start smaller for faster iteration
    parser.add_argument("--test-size", default=1000, type=int)
    parser.add_argument("--moves-before-end", default=0, type=int)
    parser.add_argument("--data-path", default="hex_games.csv", type=str)
    
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

def parse_hex_board(row_data, board_size):
    """Parse CSV row into a 2D board array."""
    board_values = row_data[:-1].values
    board = board_values.reshape(board_size, board_size)
    return board

def get_hex_neighbors(row, col, board_size):
    """Get the 6 neighbors of a hexagonal cell."""
    neighbors = []
    directions = [
        (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)
    ]
    
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < board_size and 0 <= new_col < board_size:
            neighbors.append((new_row, new_col))
    
    return neighbors

def load_hex_data(data_path, board_size, num_samples, moves_before_end=0):
    """Load Hex game data from CSV."""
    if moves_before_end == 0:
        filename = f"data/hex_{board_size}x{board_size}_1000000_final.csv"
    elif moves_before_end == 2:
        filename = f"data/hex_{board_size}x{board_size}_1000000_minus2.csv"
    elif moves_before_end == 5:
        filename = f"data/hex_{board_size}x{board_size}_1000000_minus5.csv"
    else:
        raise ValueError("moves_before_end must be 0, 2, or 5")
    
    if not os.path.exists(data_path):
        data_path = filename
    
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Generating synthetic data.")
        return generate_synthetic_hex_data(board_size, num_samples)
    
    print(f"Loading from: {data_path}")
    df = pd.read_csv(data_path)
    
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    
    boards = []
    labels = []
    
    for idx, row in df.iterrows():
        board = parse_hex_board(row, board_size)
        winner = row['winner']
        label = 0 if winner == 1 else 1
        
        boards.append(board)
        labels.append(label)
    
    return np.array(boards), np.array(labels, dtype=np.uint32)

def generate_synthetic_hex_data(board_size, num_samples):
    """Generate synthetic data for testing."""
    print(f"Generating {num_samples} synthetic boards...")
    boards = []
    labels = []
    
    for _ in range(num_samples):
        board = np.random.choice([0, 1, -1], size=(board_size, board_size), p=[0.2, 0.4, 0.4])
        winner = np.random.choice([0, 1])
        boards.append(board)
        labels.append(winner)
    
    return np.array(boards), np.array(labels, dtype=np.uint32)

def create_hex_graphs_simple(boards, labels, args, init_with=None):
    """
    Create graphs with SIMPLE feature encoding - just pieces, no position initially.
    This helps debug if position encoding is causing issues.
    """
    num_samples = len(boards)
    board_size = args.board_size
    num_nodes = board_size * board_size
    
    # SIMPLE: Just X and O
    symbols = ['X', 'O']
    
    print(f"Creating SIMPLE graphs with {num_nodes} nodes, {len(symbols)} symbols...")
    
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
    
    for graph_id in range(num_samples):
        graphs.set_number_of_graph_nodes(graph_id, num_nodes)
    
    graphs.prepare_node_configuration()
    
    # Add nodes WITHOUT edges initially (like MNIST example)
    for graph_id in range(num_samples):
        for node_id in range(num_nodes):
            graphs.add_graph_node(graph_id, node_id, 0)  # 0 edges!
    
    graphs.prepare_edge_configuration()
    
    # Add node features - ONLY piece types, NO position
    for graph_id in range(num_samples):
        if graph_id % 1000 == 0:
            print(f"Processing graph {graph_id}/{num_samples}")
        
        board = boards[graph_id]
        
        for node_id in range(num_nodes):
            row = node_id // board_size
            col = node_id % board_size
            cell_value = board[row, col]
            
            if cell_value == 1:
                graphs.add_graph_node_property(graph_id, node_id, 'X')
            elif cell_value == -1:
                graphs.add_graph_node_property(graph_id, node_id, 'O')
    
    graphs.encode()
    return graphs

def create_hex_graphs_with_position(boards, labels, args, init_with=None):
    """Create graphs with position encoding but NO edges."""
    num_samples = len(boards)
    board_size = args.board_size
    num_nodes = board_size * board_size
    
    symbols = ['X', 'O']
    
    # Add positional encoding
    for i in range(board_size):
        symbols.append(f"Row:{i}")
        symbols.append(f"Col:{i}")
    
    print(f"Creating graphs with position encoding: {num_nodes} nodes, {len(symbols)} symbols...")
    
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
    
    for graph_id in range(num_samples):
        graphs.set_number_of_graph_nodes(graph_id, num_nodes)
    
    graphs.prepare_node_configuration()
    
    # Still NO edges
    for graph_id in range(num_samples):
        for node_id in range(num_nodes):
            graphs.add_graph_node(graph_id, node_id, 0)
    
    graphs.prepare_edge_configuration()
    
    # Add features: pieces + position
    for graph_id in range(num_samples):
        if graph_id % 1000 == 0:
            print(f"Processing graph {graph_id}/{num_samples}")
        
        board = boards[graph_id]
        
        for node_id in range(num_nodes):
            row = node_id // board_size
            col = node_id % board_size
            cell_value = board[row, col]
            
            if cell_value == 1:
                graphs.add_graph_node_property(graph_id, node_id, 'X')
            elif cell_value == -1:
                graphs.add_graph_node_property(graph_id, node_id, 'O')
            
            # Add position
            graphs.add_graph_node_property(graph_id, node_id, f"Row:{row}")
            graphs.add_graph_node_property(graph_id, node_id, f"Col:{col}")
    
    graphs.encode()
    return graphs

def train_and_evaluate(args, use_position=True):
    """Training pipeline with diagnostics."""
    print("="*60)
    print(f"Hex Prediction - {args.board_size}x{args.board_size}")
    print(f"Clauses: {args.number_of_clauses}, T: {args.T}, s: {args.s}")
    print(f"Depth: {args.depth}, Hypervector: {args.hypervector_size}")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    X_all, Y_all = load_hex_data(
        args.data_path, 
        args.board_size, 
        args.train_size + args.test_size, 
        args.moves_before_end
    )
    
    X_train = X_all[:args.train_size]
    Y_train = Y_all[:args.train_size]
    X_test = X_all[args.train_size:args.train_size + args.test_size]
    Y_test = Y_all[args.train_size:args.train_size + args.test_size]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train labels: X={np.sum(Y_train==0)}, O={np.sum(Y_train==1)}")
    print(f"Test labels:  X={np.sum(Y_test==0)}, O={np.sum(Y_test==1)}")
    
    # Baseline accuracy
    baseline_acc = max(np.mean(Y_train==0), np.mean(Y_train==1)) * 100
    print(f"\n⚠️  Baseline (always predict majority): {baseline_acc:.2f}%")
    
    # Create graphs
    print("\nCreating graphs...")
    if use_position:
        graphs_train = create_hex_graphs_with_position(X_train, Y_train, args)
        graphs_test = create_hex_graphs_with_position(X_test, Y_test, args, init_with=graphs_train)
    else:
        graphs_train = create_hex_graphs_simple(X_train, Y_train, args)
        graphs_test = create_hex_graphs_simple(X_test, Y_test, args, init_with=graphs_train)
    
    print("Graphs created!")
    
    # Initialize TM
    print("\nInitializing GTM...")
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
    
    print("\nTraining...")
    print("Epoch | Train Acc | Test Acc | Time")
    print("-" * 50)
    
    best_test_acc = 0.0
    no_improvement_count = 0
    
    for epoch in range(args.epochs):
        start = time()
        tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
        
        train_acc = 100 * (tm.predict(graphs_train) == Y_train).mean()
        test_acc = 100 * (tm.predict(graphs_test) == Y_test).mean()
        elapsed = time() - start
        
        print(f"{epoch:5d} | {train_acc:8.2f}% | {test_acc:7.2f}% | {elapsed:6.2f}s")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Early stopping
        if test_acc >= 99.9:
            print(f"\n🎉 100% at epoch {epoch}!")
            break
        
        if no_improvement_count >= 20:
            print(f"\n⚠️  No improvement for 20 epochs, stopping...")
            break
    
    print("\n" + "="*60)
    print(f"Best Test Acc: {best_test_acc:.2f}%")
    print(f"Improvement over baseline: {best_test_acc - baseline_acc:.2f}%")
    print("="*60)
    
    # Quick diagnostics
    if best_test_acc < baseline_acc + 5:
        print("\n❌ PROBLEM: Model barely learning!")
        print("Suggestions:")
        print("  1. Try lower T (current: {})".format(args.T))
        print("  2. Try lower s (current: {})".format(args.s))
        print("  3. Try more clauses (current: {})".format(args.number_of_clauses))
        print("  4. Check if data is loading correctly")
    
    return tm, best_test_acc

def main():
    """Run experiments with progressive complexity."""
    print("\n" + "="*60)
    print("HEX GTM - DIAGNOSTIC MODE")
    print("="*60)
    
    # Experiment 1: Simplest possible - no position
    print("\n\n" + "="*60)
    print("EXPERIMENT 1: Simple (pieces only, no position)")
    print("="*60)
    
    args = default_args(
        board_size=5,
        epochs=50,
        number_of_clauses=200,
        T=200,
        s=1.5,
        depth=1,
        hypervector_size=32,
        train_size=3000,
        test_size=1000
    )
    
    tm, acc1 = train_and_evaluate(args, use_position=False)
    
    # Experiment 2: Add position encoding
    print("\n\n" + "="*60)
    print("EXPERIMENT 2: With position encoding")
    print("="*60)
    
    args = default_args(
        board_size=5,
        epochs=50,
        number_of_clauses=300,
        T=300,
        s=2.0,
        depth=1,
        hypervector_size=64,
        train_size=3000,
        test_size=1000
    )
    
    tm, acc2 = train_and_evaluate(args, use_position=True)
    
    # Experiment 3: Increase capacity
    if acc2 > 65:  # If it's learning something
        print("\n\n" + "="*60)
        print("EXPERIMENT 3: Higher capacity")
        print("="*60)
        
        args = default_args(
            board_size=5,
            epochs=100,
            number_of_clauses=600,
            T=500,
            s=2.5,
            depth=1,
            hypervector_size=128,
            train_size=5000,
            test_size=1000
        )
        
        tm, acc3 = train_and_evaluate(args, use_position=True)
    
    print("\n\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"Exp 1 (simple): {acc1:.2f}%")
    print(f"Exp 2 (+ position): {acc2:.2f}%")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        args = default_args()
        tm, acc = train_and_evaluate(args, use_position=True)
    else:
        main()
