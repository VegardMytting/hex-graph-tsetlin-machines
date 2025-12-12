from graphs import Graphs
from tqdm import tqdm

DIRS = [(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0)]

def csv_to_graphs(df, board_size, hypervector_size=128):
  symbols = ["empty", "red", "blue"]

  G = Graphs(
    number_of_graphs=len(df),
    symbols=symbols,
    hypervector_size=hypervector_size,
    hypervector_bits=2,
    double_hashing=False
  )

  N = board_size
  num_cells = N * N

  for gid in range(len(df)):
    G.set_number_of_graph_nodes(gid, num_cells)

  G.prepare_node_configuration()

  print("Building graph nodes...")
  for gid, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), leave=False):
    board_values = row.to_numpy(dtype=int).reshape(N, N)

    for r in range(N):
      for c in range(N):
        idx = r * N + c
        deg = 0
        
        for dr,dc in DIRS:
          rr, cc = r+dr, c+dc
          if 0 <= rr < N and 0 <= cc < N:
            deg += 1
            
        G.add_graph_node(gid, f"{idx}", deg, "hexcell")

  G.prepare_edge_configuration()

  print("Building graph edges and properties...")
  for gid, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), leave=False):
    board_values = row.to_numpy(dtype=int).reshape(N, N)

    for r in range(N):
      for c in range(N):
        src = r * N + c
        
        for dr,dc in DIRS:
          rr, cc = r+dr, c+dc
          if 0 <= rr < N and 0 <= cc < N:
            dst = rr * N + cc
            G.add_graph_node_edge(gid, f"{src}", f"{dst}", "adj")
            
        val = board_values[r][c]
        if val == 1:
          G.add_graph_node_property(gid, f"{src}", "red")
        elif val == -1:
          G.add_graph_node_property(gid, f"{src}", "blue")
        else:
          G.add_graph_node_property(gid, f"{src}", "empty")

  print("Encoding GTM graphs...")
  G.encode()

  print("Done.")
  return G
