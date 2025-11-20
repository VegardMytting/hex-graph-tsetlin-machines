from hex_board import HexBoard
from hex_engine import HexEngine
from tqdm import tqdm, trange
import random

def generate_game(board_size, engine: HexEngine):
  b = HexBoard(board_size)
  while not b.is_terminal():
    move = engine.choose_move(b)
    if move is None:
      break
    b.play(*move)
  return b

def generate_dataset(board_size, n_games, method="mcts"):
  engine = HexEngine(board_size)

  games = []
  for _ in tqdm(range(n_games), desc=f"Generating {board_size}x{board_size} dataset", disable=False):
    g = generate_game(board_size, engine)
    games.append(g)
  return games
