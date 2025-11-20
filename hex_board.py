import random
from collections import deque

EMPTY = -1
RED = 0
BLUE = 1

DIRS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

class HexBoard:
  def __init__(self, size):
    self.size = size
    self.board = [[EMPTY]*size for _ in range(size)]
    self.move_list = []
    self.player_to_move = RED

  def clone(self):
    b = HexBoard(self.size)
    b.board = [row[:] for row in self.board]
    b.move_list = list(self.move_list)
    b.player_to_move = self.player_to_move
    return b

  def empty_cells(self):
    for r in range(self.size):
      for c in range(self.size):
        if self.board[r][c] == EMPTY:
          yield (r, c)

  def play(self, r, c):
    self.board[r][c] = self.player_to_move
    self.move_list.append((r, c))
    self.player_to_move ^= 1

  def is_terminal(self):
    return self.winner() != None

  def inside(self, r, c):
    return 0 <= r < self.size and 0 <= c < self.size

  def winner(self):
    """Return RED or BLUE or None."""
    return self._connected(RED) or self._connected(BLUE)

  def _connected(self, player):
    n = self.size
    visited = [[False]*n for _ in range(n)]
    q = deque()

    if player == RED:
      for c in range(n):
        if self.board[0][c] == RED:
          q.append((0,c))
    else:
      for r in range(n):
        if self.board[r][0] == BLUE:
          q.append((r,0))

    while q:
        r,c = q.popleft()
        if visited[r][c]: continue
        visited[r][c] = True

        if player == RED and r == n-1: 
          return RED
        if player == BLUE and c == n-1: 
          return BLUE

        for dr,dc in DIRS:
          nr,nc = r+dr, c+dc
          if self.inside(nr,nc) and not visited[nr][nc]:
            if self.board[nr][nc] == player:
              q.append((nr,nc))
    return None
