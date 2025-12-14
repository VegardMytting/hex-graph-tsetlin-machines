from graphs import Graphs
from tqdm import tqdm

DIRS = {
  (-1, 0): "NW",
  (-1, 1): "NE",
  (0, -1): "W",
  (0, 1): "E",
  (1, -1): "SW",
  (1, 0): "SE",
}

def csv_to_graphs(df, board_size, hypervector_size=128, init_with=None):
  symbols = ["empty", "red", "blue"]

  G = Graphs(
    number_of_graphs=len(df),
    symbols=symbols,
    hypervector_size=hypervector_size,
    hypervector_bits=4,
    double_hashing=False,
    init_with=init_with
  )

  N = board_size
  num_cells = N * N

# Fortell Graph hvor mange noder hver graf har
  for gid in range(len(df)):
    G.set_number_of_graph_nodes(gid, num_cells)

# Fordeler interne arrays basert på nodestrukturen
  G.prepare_node_configuration()

  print("Building graph nodes...")
  for gid, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), leave=False):
    # Row = 49 celler flatt -> reshape til 7x7
    board_values = row.to_numpy(dtype=int).reshape(N, N)

    for r in range(N):
      for c in range(N):
        idx = r * N + c

        # Celleverdi
        val = int(board_values[r][c])

        # Node-type basert på celleverdi
        if val == 0:
            node_type = "empty_cell"
        elif val == 1:
            node_type = "red_cell"
        elif val == -1:
            node_type = "blue_cell"
        else:
            raise ValueError(f"Unexpected cell value: {val}")

        # Beregn antall naboer
        deg = 0
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < N and 0 <= cc < N:
                deg += 1

        G.add_graph_node(gid, f"{idx}", deg, node_type)
            


# Fordeler edge-arrays basert på deg for alle noder
  G.prepare_edge_configuration()

  print("Building graph edges and properties...")
  for gid, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), leave=False):
    board_values = row.to_numpy(dtype=int).reshape(N, N)

    for r in range(N):
      for c in range(N):
        src = r * N + c
        
        # Legg inn alle nabokanter
        for (dr, dc), etype in DIRS.items():
          rr, cc = r+dr, c+dc
          if 0 <= rr < N and 0 <= cc < N:
            dst = rr * N + cc
            G.add_graph_node_edge(gid, f"{src}", f"{dst}", etype)
            
        # Legg inn node-feature (SYMBOL) basert på cellenes verdi
        val = int(board_values[r][c])

        # Her må det matche datasettets encoding.
        # Vanligst er 0 = empty, 1 = player 1, 2 = player 2
        if val == 0:
          G.add_graph_node_property(gid, f"{src}", "empty")
        elif val == 1:
          G.add_graph_node_property(gid, f"{src}", "red")
        elif val == -1:
          G.add_graph_node_property(gid, f"{src}", "blue")
        else:
          raise ValueError(f"Unexpected cell value: {val}")

  print("Encoding GTM graphs...")
  G.encode()

  print("Done.")
  return G
