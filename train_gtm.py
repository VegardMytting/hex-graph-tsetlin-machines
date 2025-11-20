import os
import numpy as np
from sklearn.model_selection import train_test_split
from hex_dataset import generate_dataset
from hex_to_graphs import hex_to_graphs
from tm import GraphTsetlinMachine

# -------------------------------------
# CONFIG
# -------------------------------------
BOARD_SIZE = 3
N_GAMES = 100

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DATA_PATH = f"{DATA_DIR}/hex_{BOARD_SIZE}x{BOARD_SIZE}_{N_GAMES}.npz"

# -------------------------------------
# LOAD OR GENERATE DATASET
# -------------------------------------
if os.path.exists(DATA_PATH):
    print(f"Loading cached dataset: {DATA_PATH}")
    data = np.load(DATA_PATH, allow_pickle=True)

    games = data["games"]
    Y = data["Y"]

else:
    print(f"No dataset found. Generating new {BOARD_SIZE}x{BOARD_SIZE} dataset ({N_GAMES} games)...")
    games = generate_dataset(BOARD_SIZE, N_GAMES)
    Y = np.array([1 if g.winner() == 0 else 0 for g in games], dtype=np.uint32)

    # Save dataset
    print(f"Saving dataset to {DATA_PATH}")
    np.savez_compressed(DATA_PATH, games=games, Y=Y)

print("Dataset ready.")

# -------------------------------------
# TRAIN / VAL / TEST SPLIT
# -------------------------------------
games_train, games_temp, y_train, y_temp = train_test_split(
    games, Y, test_size=0.3, shuffle=True, random_state=66
)

games_val, games_test, y_val, y_test = train_test_split(
    games_temp, y_temp, test_size=0.5, shuffle=True, random_state=66
)

# -------------------------------------
# CONVERT TO GRAPHS
# -------------------------------------
print("Converting training graphs...")
G_train = hex_to_graphs(games_train)

print("Converting validation graphs...")
G_val = hex_to_graphs(games_val)

print("Converting test graphs...")
G_test = hex_to_graphs(games_test)

# -------------------------------------
# TRAIN GTM
# -------------------------------------
print("Training GTM...")
tm = GraphTsetlinMachine(
    number_of_clauses=300,
    T=50,
    s=25.0,
    depth=1
)
tm.fit(G_train, y_train)

# -------------------------------------
# VALIDATION ACCURACY
# -------------------------------------
val_pred = tm.predict(G_val)
val_acc = (val_pred == y_val).mean()
print("Validation Accuracy:", val_acc)

# -------------------------------------
# TEST ACCURACY
# -------------------------------------
test_pred = tm.predict(G_test)
test_acc = (test_pred == y_test).mean()
print("Test Accuracy:", test_acc)
