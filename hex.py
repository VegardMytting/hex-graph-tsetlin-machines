import random
import csv
from collections import deque

# ========= CONFIGURATION =========
BOARD_DIM = 11    # board size (can be 7, 9, 11, etc.)
NUM_GAMES = 5000  # number of random games to generate
RANDOM_SEED = 42  # for reproducibility
# =================================

random.seed(RANDOM_SEED)

# 6 neighbors for hex grid (axial coordinates on rhombus board)
NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

# ---------- HEX GAME LOGIC ----------
class HexGame:
    def __init__(self, dim=BOARD_DIM):
        self.dim = dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        self.moves = []
        self.open_positions = [(r, c) for r in range(dim) for c in range(dim)]

    def place_random(self, player):
        """Randomly place a piece for the player (0=X, 1=O)."""
        pos = random.choice(self.open_positions)
        self.open_positions.remove(pos)
        r, c = pos
        self.board[r][c] = player
        self.moves.append((r, c, player))
        return pos

    def is_full(self):
        return len(self.open_positions) == 0

    def neighbors(self, r, c):
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.dim and 0 <= nc < self.dim:
                yield nr, nc

    def check_winner(self):
        """Return 0 if X wins (top→bottom), 1 if O wins (left→right), else None."""
        # Player 0 (X): top to bottom
        visited = [[False]*self.dim for _ in range(self.dim)]
        q = deque([(0, c) for c in range(self.dim) if self.board[0][c] == 0])
        while q:
            r, c = q.popleft()
            if visited[r][c]:
                continue
            visited[r][c] = True
            if r == self.dim - 1:
                return 0
            for nr, nc in self.neighbors(r, c):
                if self.board[nr][nc] == 0 and not visited[nr][nc]:
                    q.append((nr, nc))

        # Player 1 (O): left to right
        visited = [[False]*self.dim for _ in range(self.dim)]
        q = deque([(r, 0) for r in range(self.dim) if self.board[r][0] == 1])
        while q:
            r, c = q.popleft()
            if visited[r][c]:
                continue
            visited[r][c] = True
            if c == self.dim - 1:
                return 1
            for nr, nc in self.neighbors(r, c):
                if self.board[nr][nc] == 1 and not visited[nr][nc]:
                    q.append((nr, nc))
        return None


# ---------- DATA GENERATION ----------
def flatten_board(board):
    """Convert 2D board → [X(0), O(0), X(1), O(1), ..., X(n-1), O(n-1)]"""
    features = []
    for row in board:
        for cell in row:
            features.append(1 if cell == 0 else 0)  # X
            features.append(1 if cell == 1 else 0)  # O
    return features


def simulate_random_game(dim):
    game = HexGame(dim)
    player = 0
    while not game.is_full():
        game.place_random(player)
        winner = game.check_winner()
        if winner is not None:
            return game, winner
        player = 1 - player
    return game, None


def save_snapshot(writer, game, winner, moves_to_keep):
    """Write one snapshot row (features + label) to CSV."""
    temp = HexGame(game.dim)
    for move in game.moves[:moves_to_keep]:
        r, c, p = move
        temp.board[r][c] = p
    row = flatten_board(temp.board) + [winner]
    writer.writerow(row)


def generate_dataset():
    random.seed(RANDOM_SEED)
    files = {
        "end": open("hex_end.csv", "w", newline=""),
        "m2": open("hex_m2.csv", "w", newline=""),
        "m5": open("hex_m5.csv", "w", newline=""),
    }
    writers = {k: csv.writer(f) for k, f in files.items()}

    for game_id in range(NUM_GAMES):
        game, winner = simulate_random_game(BOARD_DIM)
        if winner is None:
            continue

        total_moves = len(game.moves)
        if total_moves < 5:
            continue  # skip too-short games

        save_snapshot(writers["end"], game, winner, total_moves)
        save_snapshot(writers["m2"], game, winner, total_moves - 2)
        save_snapshot(writers["m5"], game, winner, total_moves - 5)

        if (game_id + 1) % 100 == 0:
            print(f"Simulated {game_id+1}/{NUM_GAMES} games")

    for f in files.values():
        f.close()
    print("✅ Datasets saved: hex_end.csv, hex_m2.csv, hex_m5.csv")


if __name__ == "__main__":
    generate_dataset()
