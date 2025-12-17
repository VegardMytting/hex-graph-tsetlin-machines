import numpy as np
import pandas as pd
import argparse
from time import time
from tqdm import tqdm
import os

from graphs import Graphs
from tm import MultiClassGraphTsetlinMachine

HYPERPARAMETERS = {
    3: {
        'number_of_clauses': 3000,
        'T': 2000,
        's': 8.0,
        'depth': 2,
        'hypervector_size': 256,
        'hypervector_bits': 2,
        'message_size': 256,
        'message_bits': 2,
        'max_included_literals': 40,
        'epochs': 60,
        'number_of_state_bits': 8,
        'train_samples': 20000,
        'boost_true_positive_feedback': 1,
        'q': 1.0
    },
    4: {
        'number_of_clauses': 4000,
        'T': 2500,
        's': 8.0,
        'depth': 2,
        'hypervector_size': 256,
        'hypervector_bits': 2,
        'message_size': 512,
        'message_bits': 2,
        'max_included_literals': 48,
        'epochs': 60,
        'number_of_state_bits': 8,
        'train_samples': 30000,
        'boost_true_positive_feedback': 1,
        'q': 1.0
    },
    5: {
        'number_of_clauses': 5000,
        'T': 3000,
        's': 8.5,
        'depth': 3,
        'hypervector_size': 512,
        'hypervector_bits': 2,
        'message_size': 1024,
        'message_bits': 2,
        'max_included_literals': 56,
        'epochs': 50,
        'number_of_state_bits': 8,
        'train_samples': 40000,
        'boost_true_positive_feedback': 1,
        'q': 1.0
    },
    6: {
        'number_of_clauses': 6000,
        'T': 3500,
        's': 9.0,
        'depth': 3,
        'hypervector_size': 512,
        'hypervector_bits': 2,
        'message_size': 1024,
        'message_bits': 2,
        'max_included_literals': 64,
        'epochs': 40,
        'number_of_state_bits': 8,
        'train_samples': 50000,
        'boost_true_positive_feedback': 1,
        'q': 1.0
    },
    7: {
        'number_of_clauses': 9000,
        'T': 3000,
        's': 6.5,
        'depth': 2,
        'hypervector_size': 1024,
        'hypervector_bits': 2,
        'message_size': 2048,
        'message_bits': 2,
        'max_included_literals': 48,
        'epochs': 30,
        'number_of_state_bits': 10,
        'train_samples': 20000,
        'boost_true_positive_feedback': 1,
        'q': 1.0
    },
    8: {
        'number_of_clauses': 12000,
        'T': 2000,
        's': 8.5,
        'depth': 2,
        'hypervector_size': 1024,
        'hypervector_bits': 2,
        'message_size': 2048,
        'message_bits': 2,
        'max_included_literals': 32,
        'epochs': 25,
        'number_of_state_bits': 12,
        'train_samples': 20000,
        'boost_true_positive_feedback': 1,
        'q': 1.0
    },
    9: {
        'number_of_clauses': 14000,
        'T': 2400,
        's': 6.5,
        'depth': 2,
        'hypervector_size': 1024,
        'hypervector_bits': 2,
        'message_size': 2048,
        'message_bits': 2,
        'max_included_literals': 28,
        'epochs': 22,
        'number_of_state_bits': 12,
        'train_samples': 18000,
        'boost_true_positive_feedback': 1,
        'q': 1.0
    },
    10: {
        'number_of_clauses': 16000,
        'T': 2800,
        's': 5.8,
        'depth': 2,
        'hypervector_size': 1024,
        'hypervector_bits': 2,
        'message_size': 2048,
        'message_bits': 2,
        'max_included_literals': 28,
        'epochs': 20,
        'number_of_state_bits': 12,
        'train_samples': 16000,
        'boost_true_positive_feedback': 1,
        'q': 1.0
    },
    11: {
        'number_of_clauses': 18000,
        'T': 3200,
        's': 5.5,
        'depth': 2,
        'hypervector_size': 1024,
        'hypervector_bits': 2,
        'message_size': 2048,
        'message_bits': 2,
        'max_included_literals': 24,
        'epochs': 18,
        'number_of_state_bits': 12,
        'train_samples': 14000,
        'boost_true_positive_feedback': 1,
        'q': 1.0
    },
}

def get_params(board_size: int) -> dict:
    if board_size in HYPERPARAMETERS:
        return dict(HYPERPARAMETERS[board_size])
      
    base = 5
    scale = max(1, board_size - base + 1)
    clauses = int(6000 + 800 * (scale ** 2))
    T = int(3500 + 400 * (scale ** 2))
    s = float(9.0 + 0.4 * scale)
    depth = 3 if board_size <= 9 else 2
    return {
        'number_of_clauses': clauses,
        'T': T,
        's': s,
        'depth': depth,
        'hypervector_size': 1024,
        'hypervector_bits': 2,
        'message_size': 2048,
        'message_bits': 2,
        'max_included_literals': 72,
        'epochs': 25,
        'number_of_state_bits': 8,
        'train_samples': 60000,
        'boost_true_positive_feedback': 1,
        'q': 1.0
    }

def hex_neighbors(i: int, j: int, n: int):
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < n and 0 <= nj < n:
            yield ni, nj


def random_subsample_indices(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if k >= n:
        return np.arange(n)
    return rng.choice(np.arange(n), size=k, replace=False)


def train_test_split_indices(n: int, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * train_ratio)
    return idx[:split], idx[split:]


def balance_train_only(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    k = min(len(idx0), len(idx1))
    if k == 0:
        return X, y
    take0 = rng.choice(idx0, size=k, replace=False)
    take1 = rng.choice(idx1, size=k, replace=False)
    idx = np.concatenate([take0, take1])
    rng.shuffle(idx)
    return X[idx], y[idx]

def load_hex_data(filepath: str):
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values.astype(np.int8)
    y = df.iloc[:, -1].values.astype(np.int8)

    y_encoded = np.where(y == 1, 0, 1).astype(np.uint32)
    return X, y_encoded

def build_symbols(board_size: int, include_neighbor_counts: bool) -> list[str]:
    symbols = ['X', 'O', 'Empty']

    for i in range(board_size):
        symbols.append(f'Row{i}')
        symbols.append(f'Col{i}')

    symbols.extend(['TopEdge', 'BottomEdge', 'LeftEdge', 'RightEdge'])

    if include_neighbor_counts:
        for i in range(7):
            symbols.append(f'XNeighbors{i}')
            symbols.append(f'ONeighbors{i}')

    symbols.extend(['XBorder', 'OBorder'])

    return symbols

def create_hex_graphs_dynamic(
    X: np.ndarray,
    board_size: int,
    symbols: list[str],
    params: dict,
    init_with: Graphs | None = None,
    use_border_nodes: bool = True,
    include_neighbor_counts: bool = False
) -> Graphs:
    num_examples = X.shape[0]
    n_cells = board_size * board_size
    n_border = 4 if use_border_nodes else 0
    n_nodes = n_cells + n_border

    if init_with is None:
        graphs = Graphs(
            num_examples,
            symbols=symbols,
            hypervector_size=params['hypervector_size'],
            hypervector_bits=params['hypervector_bits'],
            double_hashing=False,
            one_hot_encoding=False
        )
    else:
        graphs = Graphs(num_examples, init_with=init_with)

    X_LEFT = n_cells + 0
    X_RIGHT = n_cells + 1
    O_TOP = n_cells + 2
    O_BOTTOM = n_cells + 3

    for gid in range(num_examples):
        graphs.set_number_of_graph_nodes(gid, n_nodes)
    graphs.prepare_node_configuration()

    all_edges = [[] for _ in range(num_examples)]
    all_outdeg = [np.zeros(n_nodes, dtype=np.int32) for _ in range(num_examples)]

    for gid in range(num_examples):
        board = X[gid].reshape(board_size, board_size)

        for i in range(board_size):
            for j in range(board_size):
                v = board[i, j]
                if v == 0:
                    continue
                src = i * board_size + j

                if v == 1:
                    etype = "XConn"
                    for ni, nj in hex_neighbors(i, j, board_size):
                        if board[ni, nj] == 1:
                            dst = ni * board_size + nj
                            all_edges[gid].append((src, dst, etype))
                            all_outdeg[gid][src] += 1
                else:  # v == -1
                    etype = "OConn"
                    for ni, nj in hex_neighbors(i, j, board_size):
                        if board[ni, nj] == -1:
                            dst = ni * board_size + nj
                            all_edges[gid].append((src, dst, etype))
                            all_outdeg[gid][src] += 1

        if use_border_nodes:
            for i in range(board_size):
                if board[i, 0] == 1:
                    cell = i * board_size + 0
                    all_edges[gid].append((cell, X_LEFT, "XLeft"))
                    all_edges[gid].append((X_LEFT, cell, "XLeft"))
                    all_outdeg[gid][cell] += 1
                    all_outdeg[gid][X_LEFT] += 1
                if board[i, board_size - 1] == 1:
                    cell = i * board_size + (board_size - 1)
                    all_edges[gid].append((cell, X_RIGHT, "XRight"))
                    all_edges[gid].append((X_RIGHT, cell, "XRight"))
                    all_outdeg[gid][cell] += 1
                    all_outdeg[gid][X_RIGHT] += 1

            for j in range(board_size):
                if board[0, j] == -1:
                    cell = 0 * board_size + j
                    all_edges[gid].append((cell, O_TOP, "OTop"))
                    all_edges[gid].append((O_TOP, cell, "OTop"))
                    all_outdeg[gid][cell] += 1
                    all_outdeg[gid][O_TOP] += 1
                if board[board_size - 1, j] == -1:
                    cell = (board_size - 1) * board_size + j
                    all_edges[gid].append((cell, O_BOTTOM, "OBottom"))
                    all_edges[gid].append((O_BOTTOM, cell, "OBottom"))
                    all_outdeg[gid][cell] += 1
                    all_outdeg[gid][O_BOTTOM] += 1

    for gid in range(num_examples):
        for node_id in range(n_nodes):
            graphs.add_graph_node(gid, node_id, int(all_outdeg[gid][node_id]))

    graphs.prepare_edge_configuration()

    for gid in range(num_examples):
        for src, dst, et in all_edges[gid]:
            graphs.add_graph_node_edge(gid, src, dst, et)

    print("Adding node properties...")
    for gid in tqdm(range(num_examples), desc="Building graphs", disable=num_examples < 1000, leave=False):
        board = X[gid].reshape(board_size, board_size)

        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                cell_value = board[i, j]

                if cell_value == 1:
                    graphs.add_graph_node_property(gid, node_id, 'X')
                elif cell_value == -1:
                    graphs.add_graph_node_property(gid, node_id, 'O')
                else:
                    graphs.add_graph_node_property(gid, node_id, 'Empty')

                graphs.add_graph_node_property(gid, node_id, f'Row{i}')
                graphs.add_graph_node_property(gid, node_id, f'Col{j}')

                if i == 0:
                    graphs.add_graph_node_property(gid, node_id, 'TopEdge')
                if i == board_size - 1:
                    graphs.add_graph_node_property(gid, node_id, 'BottomEdge')
                if j == 0:
                    graphs.add_graph_node_property(gid, node_id, 'LeftEdge')
                if j == board_size - 1:
                    graphs.add_graph_node_property(gid, node_id, 'RightEdge')

                if include_neighbor_counts and cell_value != 0:
                    x_neighbors = 0
                    o_neighbors = 0
                    for ni, nj in hex_neighbors(i, j, board_size):
                        if board[ni, nj] == 1:
                            x_neighbors += 1
                        elif board[ni, nj] == -1:
                            o_neighbors += 1
                    if cell_value == 1:
                        graphs.add_graph_node_property(gid, node_id, f'XNeighbors{x_neighbors}')
                    else:
                        graphs.add_graph_node_property(gid, node_id, f'ONeighbors{o_neighbors}')

        if use_border_nodes:
            graphs.add_graph_node_property(gid, X_LEFT, "XBorder")
            graphs.add_graph_node_property(gid, X_RIGHT, "XBorder")
            graphs.add_graph_node_property(gid, O_TOP, "OBorder")
            graphs.add_graph_node_property(gid, O_BOTTOM, "OBorder")

    graphs.encode()
    return graphs

def diagnostics_binary(y_true: np.ndarray, y_pred: np.ndarray, pos_class: int = 0) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    y_pos = float((y_true == pos_class).mean())
    pred_pos = float((y_pred == pos_class).mean())
    maj_base = max(y_pos, 1.0 - y_pos) * 100.0

    return {"y_pos": y_pos, "pred_pos": pred_pos, "maj_base": maj_base}


def train_and_evaluate(board_size: int, args):
    print(f"\n{'='*70}")
    print(f"Training on {board_size}x{board_size} Hex board")
    print(f"{'='*70}\n")

    params = get_params(board_size)

    if args.train_samples is not None:
        params['train_samples'] = args.train_samples
    if args.epochs is not None:
        params['epochs'] = args.epochs

    symbols = build_symbols(board_size, include_neighbor_counts=args.neighbor_counts)

    scenarios = ['final', 'minus2', 'minus5']
    raw = {}

    for scenario in scenarios:
        path = os.path.join(args.data_dir, f"hex_{board_size}x{board_size}_{args.dataset_size}_{scenario}.csv")
        if not os.path.exists(path):
            print(f"Warning: missing {path}")
            continue
        X_all, y_all = load_hex_data(path)
        raw[scenario] = (X_all, y_all)

    if 'final' not in raw:
        print("Error: final scenario is required (missing file).")
        return None

    X_final, y_final = raw['final']
    total = len(X_final)

    k = min(params['train_samples'], total)
    sub_idx = random_subsample_indices(total, k, seed=args.seed)

    train_idx, test_idx = train_test_split_indices(len(sub_idx), train_ratio=0.8, seed=args.seed)
    sub_train_idx = sub_idx[train_idx]
    sub_test_idx = sub_idx[test_idx]

    data = {}
    for scenario, (X_all, y_all) in raw.items():
        X_train = X_all[sub_train_idx]
        y_train = y_all[sub_train_idx]
        X_test = X_all[sub_test_idx]
        y_test = y_all[sub_test_idx]
        data[scenario] = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

    train_scenario = args.train_on
    if train_scenario not in data:
        train_scenario = 'final'

    X_train = data[train_scenario]['X_train']
    y_train = data[train_scenario]['y_train']

    print(f"Train scenario: {train_scenario}")
    print(f"Train set size (before balance): {len(X_train)}")
    print(f"Train class counts (before balance) - Xwins(0): {np.sum(y_train==0)}, Owins(1): {np.sum(y_train==1)}")

    if args.balance_train:
        X_train, y_train = balance_train_only(X_train, y_train, seed=args.seed)
        
        print(f"Train set size (after balance): {len(X_train)}")
        print(f"Train class counts (after balance)  - Xwins(0): {np.sum(y_train==0)}, Owins(1): {np.sum(y_train==1)}")

    print("\nCreating training graphs (dynamic connectivity edges)...")
    graphs_train = create_hex_graphs_dynamic(
        X_train, board_size, symbols, params,
        init_with=None,
        use_border_nodes=(not args.no_border_nodes),
        include_neighbor_counts=args.neighbor_counts
    )

    print("\nInitializing Graph Tsetlin Machine...")
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=params['number_of_clauses'],
        T=params['T'],
        s=params['s'],
        q=params.get('q', 1.0),
        number_of_state_bits=params['number_of_state_bits'],
        depth=params['depth'],
        message_size=params['message_size'],
        message_bits=params['message_bits'],
        max_included_literals=params['max_included_literals'],
        boost_true_positive_feedback=params.get('boost_true_positive_feedback', 1),
        double_hashing=False,
        one_hot_encoding=False
    )

    print(f"\nTraining for {params['epochs']} epochs...")
    print(f"{'Epoch':<6} {'TrainAcc':<10} {'XAcc':<8} {'OAcc':<8} {'XPred':<7} {'OPred':<7} {'Time':<7}")
    print("-" * 70)

    history = []
    for epoch in range(params['epochs']):
        t0 = time()
        tm.fit(graphs_train, y_train, epochs=1, incremental=True)

        elapsed = time() - t0

        pred = tm.predict(graphs_train)
        train_acc = 100 * (pred == y_train).mean()

        x_mask = (y_train == 0)
        o_mask = (y_train == 1)
        x_acc = 100 * (pred[x_mask] == y_train[x_mask]).mean() if x_mask.any() else 0.0
        o_acc = 100 * (pred[o_mask] == y_train[o_mask]).mean() if o_mask.any() else 0.0

        x_preds = int(np.sum(pred == 0))
        o_preds = int(np.sum(pred == 1))

        history.append({
            "epoch": epoch,
            "train_acc": train_acc,
            "train_x_acc": x_acc,
            "train_o_acc": o_acc,
            "x_preds": x_preds,
            "o_preds": o_preds,
            "time": elapsed
        })

        if epoch % max(1, params['epochs'] // 20) == 0 or epoch == params['epochs'] - 1:
            print(f"{epoch:<6} {train_acc:>7.2f}%   {x_acc:>6.2f}% {o_acc:>6.2f}% {x_preds:<7} {o_preds:<7} {elapsed:>5.2f}s")

        if args.early_stop and board_size <= 3 and train_acc >= 100.0:
            break

    os.makedirs(args.results_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(args.results_dir, f"training_history_{board_size}x{board_size}_{args.train_on}.csv"), index=False)

    print("\n" + "=" * 70)
    print("EVALUATION (aligned splits; dynamic edges)")
    print("=" * 70)

    eval_rows = []
    results = {}
    for scenario in scenarios:
        if scenario not in data:
            continue

        X_test = data[scenario]['X_test']
        y_test = data[scenario]['y_test']

        graphs_test = create_hex_graphs_dynamic(
            X_test, board_size, symbols, params,
            init_with=graphs_train,
            use_border_nodes=(not args.no_border_nodes),
            include_neighbor_counts=args.neighbor_counts
        )

        t0 = time()
        pred = tm.predict(graphs_test)
        t_pred = time() - t0

        test_acc = 100 * (pred == y_test).mean()
        x_mask = (y_test == 0)
        o_mask = (y_test == 1)
        x_acc = 100 * (pred[x_mask] == y_test[x_mask]).mean() if x_mask.any() else 0.0
        o_acc = 100 * (pred[o_mask] == y_test[o_mask]).mean() if o_mask.any() else 0.0

        diag = diagnostics_binary(y_test, pred, pos_class=0)

        x_preds = int(np.sum(pred == 0))
        o_preds = int(np.sum(pred == 1))

        results[scenario] = {"accuracy": test_acc, "x_accuracy": x_acc, "o_accuracy": o_acc, "time": t_pred}

        eval_rows.append({
            "scenario": scenario,
            "test_acc": test_acc,
            "test_x_acc": x_acc,
            "test_o_acc": o_acc,
            "x_preds": x_preds,
            "o_preds": o_preds,
            "y_pos(Xwins)": diag["y_pos"],
            "pred_pos(Xwins)": diag["pred_pos"],
            "majority_baseline": diag["maj_base"],
            "test_time": t_pred
        })

        print(f"\nScenario: {scenario}")
        print(f"  Acc: {test_acc:.2f}%  (X: {x_acc:.2f}%, O: {o_acc:.2f}%)")
        print(f"  Pred counts: X={x_preds} O={o_preds}")
        print(f"  y_pos(Xwins)={diag['y_pos']:.3f} pred_pos(Xwins)={diag['pred_pos']:.3f} maj_base={diag['maj_base']:.2f}%")

    pd.DataFrame(eval_rows).to_csv(os.path.join(args.results_dir, f"evaluation_{board_size}x{board_size}.csv"), index=False)
    return results


def main():
    parser = argparse.ArgumentParser(description="Hex Winner Prediction with GTM (Improved Learning)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--dataset-size", type=str, default="1000000", help="Matches your file naming: hex_{n}x{n}_{dataset_size}_{scenario}.csv")

    parser.add_argument("--board-size", type=int, default=3, help="Board size (3..11)")
    parser.add_argument("--all-sizes", action="store_true", help="Train on all sizes from --min-size to --max-size")
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=11)

    parser.add_argument("--train-on", type=str, default="final", choices=["final", "minus2", "minus5"], help="Which scenario to train on (still evaluates all scenarios)")
    parser.add_argument("--train-samples", type=int, default=None, help="Override train_samples in hyperparams")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs in hyperparams")
    parser.add_argument("--seed", type=int, default=66)

    parser.add_argument("--balance-train", action="store_true", default=True, help="Balance TRAIN only (recommended).")
    parser.add_argument("--no-balance-train", dest="balance_train", action="store_false")

    parser.add_argument("--no-border-nodes", action="store_true", help="Disable 4 border nodes (usually hurts).")

    parser.add_argument("--neighbor-counts", action="store_true", default=False, help="Include XNeighbors/ONeighbors symbols (can be noisy).")

    parser.add_argument("--early-stop", action="store_true", default=True)
    parser.add_argument("--results-dir", type=str, default="results")

    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    if args.all_sizes:
        summary = []
        all_results = {}

        for size in range(args.min_size, args.max_size + 1):
            try:
                res = train_and_evaluate(size, args)
                if res is None:
                    continue
                all_results[size] = res

                row = {"board_size": f"{size}x{size}"}
                for scenario in ["final", "minus2", "minus5"]:
                    row[f"{scenario}_acc"] = res.get(scenario, {}).get("accuracy", np.nan)
                summary.append(row)

            except Exception as e:
                print(f"\nError at {size}x{size}: {e}")
                import traceback
                traceback.print_exc()

        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(args.results_dir, "all_sizes_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Saved summary to {summary_path}")

        print("\nFINAL RESULTS")
        print(f"{'Size':<8} {'Final':<10} {'Minus2':<10} {'Minus5':<10}")
        print("-" * 45)
        for size in sorted(all_results.keys()):
            r = all_results[size]
            print(f"{size}x{size:<5} {r.get('final',{}).get('accuracy',0):>6.2f}%    "
                  f"{r.get('minus2',{}).get('accuracy',0):>6.2f}%    "
                  f"{r.get('minus5',{}).get('accuracy',0):>6.2f}%")
    else:
        train_and_evaluate(args.board_size, args)


if __name__ == "__main__":
    main()


