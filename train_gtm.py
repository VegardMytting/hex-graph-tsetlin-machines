from tm import GraphTsetlinMachine
from hex_dataset import generate_dataset
from hex_to_graphs import hex_to_graphs
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

BOARD_SIZE = 6
N_GAMES = 5000

count = cuda.Device.count()
print("CUDA device count:", count)

if count == 0:
  print("No CUDA GPU detected")
  
for i in range(count):
  dev = cuda.Device(i)
  print(f"\nGPU {i}: {dev.name()}")
  print("Compute Capability:", dev.compute_capability())
  print("Memory: %.2f GB" % (dev.total_memory() / 1024**3))

print(f"Generating {N_GAMES} {BOARD_SIZE}x{BOARD_SIZE} board size games...")
games = generate_dataset(BOARD_SIZE, N_GAMES)

print("Converting to GTM graphs...")
G = hex_to_graphs(games)

print("Preparing labels...")
Y = np.array([1 if g.winner() == 0 else 0 for g in games], dtype=np.uint32)

print("Training GTM...")
tm = GraphTsetlinMachine(
  number_of_clauses=200,
  T=50,
  s=3.0,
  depth=1,
  message_size=256
)

tm.fit(G, Y)

print("Training done.")
print("Evaluating...")
pred = tm.predict(G)
acc = (pred == Y).mean()
print("Accuracy:", acc)
