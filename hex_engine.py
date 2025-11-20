import subprocess
import shutil
import random
from hex_board import HexBoard

class HexEngine:
  def __init__(self, board_size, mohex_path="mohex"):
    self.N = board_size
    self.mohex_path = mohex_path
    self.use_mohex = shutil.which(mohex_path) is not None

  def mohex_move(self, board: HexBoard):
    """
    Sends the board to MoHex (GNUGo-style GTP) and gets a move back.
    """
    if not self.use_mohex:
      raise RuntimeError("MoHex not installed.")

    proc = subprocess.Popen(
      [self.mohex_path],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      text=True
    )

    proc.stdin.write(f"boardsize {self.N}\n")
    proc.stdin.write("clear_board\n")

    for move_id, (r,c) in enumerate(board.move_list):
      color = "black" if move_id % 2 == 0 else "white"
      proc.stdin.write(f"play {color} {c+1},{r+1}\n")

    proc.stdin.write("genmove black\n")
    proc.stdin.flush()

    for line in proc.stdout:
      if line.startswith("="):
        move = line[2:].strip()
        if move.lower() == "resign":
          return None
        x,y = move.split(",")
        return int(y)-1, int(x)-1

    return None

  def mcts_policy(self, board: HexBoard):
    """
    A very simple Monte-Carlo policy:
      - Try each legal move
      - Simulate random rollouts
      - Pick highest winrate
    Good enough for dataset generation on small boards.
    """
    moves = list(board.empty_cells())
    if not moves: return None
    if board.size <= 4:
      rollout_count = 200
    elif board.size <= 6:
      rollout_count = 40
    else:
      rollout_count = 10

    best_move = None
    best_score = -999

    for (r,c) in moves:
      wins = 0
      for _ in range(rollout_count):
        b = board.clone()
        b.play(r,c)
        wins += self.random_rollout(b)

      if wins > best_score:
        best_score = wins
        best_move = (r,c)

    return best_move

  def random_rollout(self, board: HexBoard):
    b = board.clone()
    cells = list(b.empty_cells())
    random.shuffle(cells)
    for r,c in cells:
      b.play(r,c)
      if b.is_terminal():
        break
    winner = b.winner()
    return 1 if winner == board.player_to_move else 0

  def choose_move(self, board: HexBoard):
    if self.use_mohex:
      return self.mohex_move(board)
    else:
      return self.mcts_policy(board)
