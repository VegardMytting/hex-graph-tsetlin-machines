import pandas as pd
from sklearn.model_selection import train_test_split
from converter import csv_to_graphs
from tm import MultiClassGraphTsetlinMachine
import numpy as np
from InquirerPy import inquirer

BOARD_SIZE = 7

number_of_clauses_choice = int(inquirer.number(
  message="Choose Number of Clauses:",
  min_allowed=1,
  default=5000
).execute())

T_choice = int(inquirer.number(
  message="Choose T:",
  min_allowed=1,
  default=5000
).execute())

s_choice = float(inquirer.number(
  message="Choose s:",
  min_allowed=1,
  max_allowed=25,
  default = 5,
).execute())

depth_choice = int(inquirer.number(
  message="Choose Depth:",
  min_allowed=1,
  default=1
).execute())

#Les bare første N rader for rask test
FAST_N = 1000000
# Les data fra data-mappen 
df = pd.read_csv("data/hex_games_1_000_000_size_7.csv", nrows=FAST_N)

# Sanity Check: Bekreft encoding i CSV
vals = df.drop(columns=["winner"]).to_numpy().ravel()

print("Cell min/max: ", vals.min(), vals.max())
print("Unique cell values (sample):", np.unique(vals)[:10])
print("Winner distribution:\n", df["winner"].value_counts())

epochs_choice = int(inquirer.number(
  message="Choose Epochs:",
  min_allowed=1,
  default=10
).execute())

# Label: lag klasse-id 0/1 (MultiClass forventer class IDs)
w = df["winner"].to_numpy()

uniq = set(np.unique(w))
if uniq.issubset({-1, 1}):
    # -1 -> 0, 1 -> 1
    Y = (w == 1).astype(np.uint32)
elif uniq.issubset({1, 2}):
    # 1 -> 0, 2 -> 1 (evt. bytt hvis du vil motsatt)
    Y = (w == 2).astype(np.uint32)
else:
    raise ValueError(f"Unexpected winner encoding: {sorted(list(uniq))[:10]}")

board_df = df.drop(columns=["winner"])

X_train, X_temp, y_train, y_temp = train_test_split(
  board_df, Y, test_size=0.3, shuffle=True, random_state=66
)

X_val, X_test, y_val, y_test = train_test_split(
  X_temp, y_temp, test_size=0.5, shuffle=True, random_state=66
)

TRAIN_CAP = 200000
VAL_CAP = 20000
TEST_CAP = 20000

X_train = X_train.iloc[:TRAIN_CAP]
y_train = y_train[:TRAIN_CAP]
X_val = X_val.iloc[:VAL_CAP]
y_val = y_val[:VAL_CAP]
X_test = X_test.iloc[:TEST_CAP]
y_test = y_test[:TEST_CAP]


y_train_np = np.asarray(y_train, dtype=np.uint32)
y_val_np   = np.asarray(y_val, dtype=np.uint32)
y_test_np  = np.asarray(y_test, dtype=np.uint32)

baseline_val  = max((y_val_np == 0).mean(),  (y_val_np == 1).mean())
baseline_test = max((y_test_np == 0).mean(), (y_test_np == 1).mean())
print("Baseline VAL:", baseline_val)
print("Baseline TEST:", baseline_test)

# Bygg grafene
# - Train lager ny random hypervector mapping
# - Val/test gjenbruker train sin mapping (kritisk)
G_train = csv_to_graphs(X_train, BOARD_SIZE, hypervector_size=512)
G_val   = csv_to_graphs(X_val, BOARD_SIZE, hypervector_size=512, init_with=G_train)
G_test  = csv_to_graphs(X_test, BOARD_SIZE, hypervector_size=512, init_with=G_train)

tm = MultiClassGraphTsetlinMachine(
  number_of_clauses=number_of_clauses_choice,
  T=T_choice,
  s=s_choice,
  depth=depth_choice
)

# Tren
tm.fit(G_train, y_train_np, epochs=epochs_choice)

# Evaluer robust (numpy på begge sider)
pred_val = tm.predict(G_val).astype(np.uint32)
pred_test = tm.predict(G_test).astype(np.uint32)

print("VAL:", (pred_val == y_val_np).mean())
print("TEST:", (pred_test == y_test_np).mean())


# Sanity-Check: ser om modellen predikerer konstant
print("VAL pred dist  :", np.bincount(pred_val, minlength=2) / pred_val.size)
print("VAL label dist :", np.bincount(y_val_np, minlength=2) / y_val_np.size)
print("TEST pred dist :", np.bincount(pred_test, minlength=2) / pred_test.size)
print("TEST label dist:", np.bincount(y_test_np, minlength=2) / y_test_np.size)


