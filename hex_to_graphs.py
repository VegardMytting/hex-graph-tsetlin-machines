from graphs import Graphs
from hex_board import HexBoard, EMPTY, RED, BLUE

def hex_to_graphs(game_list, hypervector_size=128):
  symbols = ["empty", "red", "blue"]
  G = Graphs(
    number_of_graphs=len(game_list),
    symbols=symbols,
    hypervector_size=hypervector_size,
    hypervector_bits=2,
    double_hashing=False
  )

  for gid, game in enumerate(game_list):
    N = game.size
    G.set_number_of_graph_nodes(gid, N*N)

  G.prepare_node_configuration()

  for gid, game in enumerate(game_list):
    N = game.size
    for r in range(N):
      for c in range(N):
        idx = r*N + c
        deg = 0
        for dr,dc in [(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0)]:
          nr,nc = r+dr, c+dc
          if 0 <= nr < N and 0 <= nc < N:
            deg += 1
        G.add_graph_node(gid, f"{idx}", deg, "hexcell")

  G.prepare_edge_configuration()

  for gid, game in enumerate(game_list):
    N = game.size
    for r in range(N):
      for c in range(N):
        src = r*N + c
        for dr,dc in [(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0)]:
          nr,nc = r+dr, c+dc
          if 0 <= nr < N and 0 <= nc < N:
            dst = nr*N + nc
            G.add_graph_node_edge(gid, f"{src}", f"{dst}", "adj")
            
        stone = game.board[r][c]
        if stone == RED:
          G.add_graph_node_property(gid, f"{src}", "red")
        elif stone == BLUE:
          G.add_graph_node_property(gid, f"{src}", "blue")
        else:
          G.add_graph_node_property(gid, f"{src}", "empty")

  G.encode()
  return G
