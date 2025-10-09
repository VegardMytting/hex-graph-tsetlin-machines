import random
from collections import deque

NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

class HexGame:
  def __init__(self, dim=7):
    self.dim = dim
    self.board = [[None for _ in range(dim)] for _ in range(dim)]
    self.moves = []
    self.open_positions = [(r, c) for r in range(dim) for c in range(dim)]
  
  def neighbors(self, r, c):
    for dr, dc in NEIGHBORS:
      nr, nc = r + dr, c + dc
      if 0 <= nr < self.dim and 0 <= nc < self.dim:
        yield nr, nc
  
  def place_random(self, player):
    pos = random.choice(self.open_positions)
    self.open_positions.remove(pos)
    r, c = pos
    self.board[r][c] = player
    self.moves.append((r, c, player))
    return pos
  
  def is_full(self):
    return len(self.open_positions) == 0
  
  def winner(self):
    """Return 0 if X wins, 1 if O wins, None otherwise"""
    dim = self.dim
    
    visited = [[False]*dim for _ in range(dim)]
    q = deque([(0, c) for c in range(dim) if self.board[0][c] == 0])
    while q:
      r, c = q.popleft()
      if visited[r][c]:
        continue
      visited[r][c] = True
      if r == dim - 1:
        return 0
      for nr, nc in self.neighbors(r, c):
        if self.board[nr][nc] == 0 and not visited[nr][nc]:
          q.append((nr, nc))
    
    visited = [[False]*dim for _ in range(dim)]
    q = deque([(r, 0) for r in range(dim) if self.board[r][0] == 1])
    while q:
      r, c = q.popleft()
      if visited[r][c]:
        continue
      visited[r][c] = True
      if c == dim - 1:
        return 1
      for nr, nc in self.neighbors(r, c):
        if self.board[nr][nc] == 1 and not visited[nr][nc]:
          q.append((nr, nc))
    
    return None
  
  def print_board(self):
    """Prints the board in a hex-grid-like layout"""
    print()
    for i in range(self.dim):
      print(" " * i, end="")
      for j in range(self.dim):
        cell = self.board[i][j]
        if cell == 0:
          print(" X", end="")
        elif cell == 1:
          print(" O", end="")
        else:
          print(" ·", end="")
      print()
    print()

def simulate_and_display(dim=7, delay=0.3):
  import time
  game = HexGame(dim)
  player = 0
  
  while not game.is_full():
    game.place_random(player)
    game.print_board()
    time.sleep(delay)
    winner = game.winner()
    
    if winner is not None:
      print(f"🎉 Player {winner} ({'X' if winner==0 else 'O'}) wins!\n")
      break
    
    player = 1 - player

if __name__ == "__main__":
  simulate_and_display(dim=7, delay=0.2)
