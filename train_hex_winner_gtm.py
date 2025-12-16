import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import subprocess
import numpy as np

# --- Important: tm.py imports GraphTsetlinMachine.kernels, but you currently have kernels.py in the same folder.
# We create a lightweight package alias at runtime so you don't have to edit tm.py/kernels.py.
def _ensure_gtm_package_alias():
    import sys, types, importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent

    if "GraphTsetlinMachine" not in sys.modules:
        pkg = types.ModuleType("GraphTsetlinMachine")
        pkg.__path__ = [str(root)]
        sys.modules["GraphTsetlinMachine"] = pkg

    def _load_as(mod_name: str, file_name: str):
        if mod_name in sys.modules:
            return
        spec = importlib.util.spec_from_file_location(mod_name, str(root / file_name))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)

    _load_as("GraphTsetlinMachine.kernels", "kernels.py")


_ensure_gtm_package_alias()

from graphs import Graphs
from tm import MultiClassGraphTsetlinMachine


# ---------------------------
# Hex mechanics (winner check)
# ---------------------------

def hex_neighbors(size: int, r: int, c: int) -> List[Tuple[int, int]]:
    # Standard Hex 6-neighborhood on (row, col)
    nbrs = [
        (r - 1, c),     # up
        (r - 1, c + 1), # up-right
        (r, c - 1),     # left
        (r, c + 1),     # right
        (r + 1, c - 1), # down-left
        (r + 1, c),     # down
    ]
    return [(rr, cc) for rr, cc in nbrs if 0 <= rr < size and 0 <= cc < size]


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def winner_of_board(board: np.ndarray, size: int) -> int:
    """
    board: shape (size*size,), values: 0 empty, 1 P1, 2 P2
    Winner convention:
      - P1 connects TOP to BOTTOM
      - P2 connects LEFT to RIGHT
    Returns: 1 or 2 (assumes terminal positions where winner exists)
    """
    n_cells = size * size
    # Virtual nodes
    P1_TOP = n_cells
    P1_BOTTOM = n_cells + 1
    P2_LEFT = n_cells + 2
    P2_RIGHT = n_cells + 3

    uf = UnionFind(n_cells + 4)

    def idx(r: int, c: int) -> int:
        return r * size + c

    # Union within same player's stones, plus connect to borders
    for r in range(size):
        for c in range(size):
            v = board[idx(r, c)]
            if v == 0:
                continue

            here = idx(r, c)
            # same-player adjacency
            for rr, cc in hex_neighbors(size, r, c):
                there = idx(rr, cc)
                if board[there] == v:
                    uf.union(here, there)

            # border connections
            if v == 1:
                if r == 0:
                    uf.union(here, P1_TOP)
                if r == size - 1:
                    uf.union(here, P1_BOTTOM)
            else:  # v == 2
                if c == 0:
                    uf.union(here, P2_LEFT)
                if c == size - 1:
                    uf.union(here, P2_RIGHT)

    if uf.find(P1_TOP) == uf.find(P1_BOTTOM):
        return 1
    if uf.find(P2_LEFT) == uf.find(P2_RIGHT):
        return 2

    return 0  # non-terminal


# ---------------------------
# Dataset generation (random playout)
# ---------------------------

@dataclass
class GameSample:
    # moves are indices 0..size*size-1, in play order.
    moves: List[int]
    winner: int
    end_len: int  # number of moves until game ended (inclusive of winning move)

def generate_games_from_hex_c(exe_path: str, n_games: int) -> List[GameSample]:
    proc = subprocess.run(
        [exe_path, str(n_games)],
        check=True,
        capture_output=True,
        text=True
    )

def generate_balanced_games_from_hex_c(exe_path: str, n_total: int) -> List[GameSample]:
    """
    Generate approximately balanced set of games: ~n_total/2 P1 wins and ~n_total/2 P2 wins.
    """
    target = n_total // 2
    p1: List[GameSample] = []
    p2: List[GameSample] = []

    # Generate in batches to reduce subprocess overhead
    batch = max(200, n_total)

    while len(p1) < target or len(p2) < target:
        games = generate_games_from_hex_c(exe_path, batch)
        for g in games:
            if g.winner == 1 and len(p1) < target:
                p1.append(g)
            elif g.winner == 2 and len(p2) < target:
                p2.append(g)
            if len(p1) >= target and len(p2) >= target:
                break

    balanced = p1 + p2
    random.shuffle(balanced)

    # If n_total is odd, add one more random game (or just trim)
    return balanced[:n_total]


    games: List[GameSample] = []
    for line in proc.stdout.strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue

        winner_c = int(parts[0])         # 0 eller 1 fra C
        moves = [int(x) for x in parts[1:]]
        # map winner to your Python convention: 1=P1, 2=P2
        winner_py = 1 if winner_c == 0 else 2

        games.append(GameSample(moves=moves, winner=winner_py, end_len=len(moves)))

    return games


def generate_random_game(size: int, rng: random.Random) -> GameSample:
    cells = list(range(size * size))
    rng.shuffle(cells)

    board = np.zeros(size * size, dtype=np.uint8)
    player = 1

    moves_played: List[int] = []
    end_len = 0
    winner = 0

    for k, pos in enumerate(cells):
        board[pos] = player
        moves_played.append(pos)
        end_len = k + 1

        w = winner_of_board(board, size)
        if w != 0:
            winner = w
            break

        player = 2 if player == 1 else 1

    return GameSample(moves=moves_played, winner=winner, end_len=end_len)


def board_at_snap(game: GameSample, size: int, snap: int) -> np.ndarray:
    """
    snap=0: state at end of game (winning move included)
    snap=2: state two moves before end (i.e., end_len-2 moves played)
    snap=5: state five moves before end
    """
    t = max(0, game.end_len - snap)
    board = np.zeros(size * size, dtype=np.uint8)
    player = 1
    for i in range(t):
        board[game.moves[i]] = player
        player = 2 if player == 1 else 1
    return board


# ---------------------------
# Graph encoding
# ---------------------------

def _symbols_for_snap(size: int, snap: int) -> List[str]:
    symbols = []
    # Always include basic occupancy symbols for node properties (used in snap>=2/5)
    # For snap=0 we still include them so we can add extra node properties later if desired.
    symbols += ["P1", "P2", "EMPTY"]

    #Viktig for Hex: sidene må være observerbare
    symbols += ["TOP", "BOTTOM", "LEFT", "RIGHT"]


    # På 5x5 anbefaler jeg posisjon også for snap=0
    symbols += [f"ROW_{r}" for r in range(size)]
    symbols += [f"COL_{c}" for c in range(size)]

    if snap in (2, 5):
        for k in range(7):
            symbols += [f"N_P1_{k}", f"N_P2_{k}", f"N_E_{k}"]

    return symbols


def _edge_types_for_snap(snap: int) -> List[str]:
    # snap=0: required edge types only
    base = ["P1_CONN", "P2_CONN", "OPP_CONN", "EMPTY_CONN"]
    if snap in (2, 5):
        # "edges between all neighbors" => we add an unconditional adjacency edge too
        base += ["ADJ"]
    return base


def build_graphs_from_boards(boards: np.ndarray, size: int, snap: int, hypervector_size: int, hypervector_bits: int = 2):
    """
    boards: shape (N, size*size), values: 0 empty, 1 P1, 2 P2
    Returns a Graphs object ready for GTM.
    """
    n = boards.shape[0]
    node_count = size * size

    symbols = _symbols_for_snap(size, snap)

    graphs = Graphs(
        number_of_graphs=n,
        hypervector_size=hypervector_size,
        hypervector_bits=hypervector_bits,
        symbols=symbols,
    )

    for gid in range(n):
        graphs.set_number_of_graph_nodes(gid, node_count)

    graphs.prepare_node_configuration()

    # Precompute per-node neighbor list
    nbrs = []
    for r in range(size):
        for c in range(size):
            nbrs.append(hex_neighbors(size, r, c))

    add_adj = (snap in (2, 5))

    # Edge count per node depends on snap:
    # - We always add 1 directed edge per neighbor (6 max) with a state-dependent edge-type.
    # - For snap 2/5 we also add unconditional "ADJ" edges per neighbor (same multiplicity).
    per_node_edges = [len(nbrs[i]) * (1 + int(add_adj)) for i in range(node_count)]

    # Add nodes
    for gid in range(n):
        for r in range(size):
            for c in range(size):
                name = f"{r}:{c}"
                graphs.add_graph_node(gid, name, number_of_graph_node_edges=per_node_edges[r * size + c])

    graphs.prepare_edge_configuration()

    # Add edges + properties
    for gid in range(n):
        board = boards[gid]

        for r in range(size):
            for c in range(size):
                src_name = f"{r}:{c}"
                src_idx = r * size + c
                src_v = int(board[src_idx])

                # Node properties (ALL snaps)
                # ---------------------------

                # Occupancy (ALL snaps)
                if src_v == 0:
                    graphs.add_graph_node_property(gid, src_name, "EMPTY")
                elif src_v == 1:
                    graphs.add_graph_node_property(gid, src_name, "P1")
                else:
                    graphs.add_graph_node_property(gid, src_name, "P2")

                # Borders (ALL snaps) – critical for Hex winner logic
                if r == 0:
                    graphs.add_graph_node_property(gid, src_name, "TOP")
                if r == size - 1:
                    graphs.add_graph_node_property(gid, src_name, "BOTTOM")
                if c == 0:
                    graphs.add_graph_node_property(gid, src_name, "LEFT")
                if c == size - 1:
                    graphs.add_graph_node_property(gid, src_name, "RIGHT")

                # Position (ALL snaps) – strongly recommended for 5x5
                graphs.add_graph_node_property(gid, src_name, f"ROW_{r}")
                graphs.add_graph_node_property(gid, src_name, f"COL_{c}")

                # Local neighbor statistics (ONLY snap 2/5, as required)
                if snap in (2, 5):
                    n_p1 = n_p2 = n_e = 0
                    for rr, cc in nbrs[src_idx]:
                        v = int(board[rr * size + cc])
                        if v == 0:
                            n_e += 1
                        elif v == 1:
                            n_p1 += 1
                        else:
                            n_p2 += 1

                    graphs.add_graph_node_property(gid, src_name, f"N_P1_{n_p1}")
                    graphs.add_graph_node_property(gid, src_name, f"N_P2_{n_p2}")
                    graphs.add_graph_node_property(gid, src_name, f"N_E_{n_e}")


                # edges (required)
                # edges (required)
                for rr, cc in nbrs[src_idx]:
                    dst_name = f"{rr}:{cc}"
                    dst_v = int(board[rr * size + cc])

                    # State-aware connection edges
                    if dst_v == 0:
                        et = "EMPTY_CONN"
                    else:
                        if src_v == 0:
                            et = "P1_CONN" if dst_v == 1 else "P2_CONN"
                        else:
                            if src_v == dst_v:
                                et = "P1_CONN" if dst_v == 1 else "P2_CONN"
                            else:
                                et = "OPP_CONN"

                    # 1) alltid én “state edge” per nabo
                    graphs.add_graph_node_edge(gid, src_name, dst_name, et)

                    # 2) i tillegg: én ADJ per nabo når snap=2/5
                    if add_adj:
                        graphs.add_graph_node_edge(gid, src_name, dst_name, "ADJ")

                    

    graphs.encode()
    return graphs


# ---------------------------
# Training / Evaluation
# ---------------------------

def train_and_eval_for_snap(
    size: int,
    snap: int,
    n_train: int,
    n_test: int,
    clauses: int,
    s: float,
    epochs: int,
    hypervector_size: int,
    seed: int,
):
    rng = random.Random(seed + snap)

    # Generate games
    train_games = generate_balanced_games_from_hex_c("./hex5", n_train)
    test_games = generate_balanced_games_from_hex_c("./hex5", n_test)



    # Build boards at snap
    X_train = np.stack([board_at_snap(g, size, snap) for g in train_games], axis=0)
    y_train = np.array([g.winner - 1 for g in train_games], dtype=np.uint32)  # 0=P1, 1=P2

    X_test = np.stack([board_at_snap(g, size, snap) for g in test_games], axis=0)
    y_test = np.array([g.winner - 1 for g in test_games], dtype=np.uint32)

    graphs_train = build_graphs_from_boards(X_train, size, snap, hypervector_size=hypervector_size)
    graphs_test = build_graphs_from_boards(X_test, size, snap, hypervector_size=hypervector_size)

    depth = size + 2
    T = int(0.5 * clauses)

    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=clauses,
        T=T,
        s=s,
        depth=depth,
        # message_size/message_bits kan tunes, men default i tm.py fungerer som regel OK.
    )

    tm.fit(graphs_train, y_train, epochs=epochs, incremental=False)
    preds = tm.predict(graphs_test)

    # Confusion matrix for binary classes: 0=P1, 1=P2
    cm = np.zeros((2, 2), dtype=np.int64)
    for yt, yp in zip(y_test, preds):
        cm[int(yt), int(yp)] += 1

    # Distributions
    true_counts = np.bincount(y_test.astype(np.int64), minlength=2)
    pred_counts = np.bincount(preds.astype(np.int64), minlength=2)

    acc = float(np.mean(preds == y_test))

    print(f"[snap={snap}] accuracy={acc:.4f}")
    print(f"[snap={snap}] true counts:  P1={true_counts[0]}  P2={true_counts[1]}")
    print(f"[snap={snap}] pred counts:  P1={pred_counts[0]}  P2={pred_counts[1]}")
    print(f"[snap={snap}] confusion matrix (rows=true, cols=pred):")
    print("           pred P1   pred P2")
    print(f"true P1    {cm[0,0]:7d}  {cm[0,1]:7d}")
    print(f"true P2    {cm[1,0]:7d}  {cm[1,1]:7d}")

    return acc, tm



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board_size", type=int, default=11)
    ap.add_argument("--snaps", type=str, default="0,2,5")
    ap.add_argument("--train_games", type=int, default=2000)
    ap.add_argument("--test_games", type=int, default=500)
    ap.add_argument("--clauses", type=int, default=2000)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hypervector_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    snaps = [int(x.strip()) for x in args.snaps.split(",") if x.strip()]

    print(f"BOARD_SIZE={args.board_size}, depth={args.board_size + 2}")
    print(f"snaps={snaps}")
    print(f"train_games={args.train_games}, test_games={args.test_games}")
    print(f"clauses={args.clauses}, T={int(0.5*args.clauses)}, s={args.s}, epochs={args.epochs}")
    print(f"hypervector_size={args.hypervector_size}")

    for snap in snaps:
        acc, _ = train_and_eval_for_snap(
            size=args.board_size,
            snap=snap,
            n_train=args.train_games,
            n_test=args.test_games,
            clauses=args.clauses,
            s=args.s,
            epochs=args.epochs,
            hypervector_size=args.hypervector_size,
            seed=args.seed,
        )
        print(f"[snap={snap}] accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
