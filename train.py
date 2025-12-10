import pandas as pd
from sklearn.model_selection import train_test_split
from converter import csv_to_graphs
from tm import GraphTsetlinMachine
import numpy as np
from InquirerPy import inquirer

BOARD_SIZE = 7

number_of_clauses_choice = int(inquirer.number(
  message="Choose Number of Clauses:",
  min_allowed=1,
  default=100
).execute())

T_choice = int(inquirer.number(
  message="Choose T:",
  min_allowed=1,
).execute())

s_choice = float(inquirer.number(
  message="Choose s:",
  min_allowed=1,
  max_allowed=25,
).execute())

depth_choice = int(inquirer.number(
  message="Choose Depth:",
  min_allowed=1,
  default=3
).execute())

df = pd.read_csv("data/hex_games_1_000_000_size_7.csv")

Y = (df["winner"] == 1).astype(np.uint32)
board_df = df.drop(columns=["winner"])

X_train, X_temp, y_train, y_temp = train_test_split(
  board_df, Y, test_size=0.3, shuffle=True, random_state=66
)

X_val, X_test, y_val, y_test = train_test_split(
  X_temp, y_temp, test_size=0.5, shuffle=True, random_state=66
)

G_train = csv_to_graphs(X_train, BOARD_SIZE)
G_val = csv_to_graphs(X_val, BOARD_SIZE)
G_test = csv_to_graphs(X_test, BOARD_SIZE)

tm = GraphTsetlinMachine(
  number_of_clauses=number_of_clauses_choice,
  T=T_choice,
  s=s_choice,
  depth=depth_choice
)
tm.fit(G_train, y_train)

print("VAL:", (tm.predict(G_val) == y_val).mean())
print("TEST:", (tm.predict(G_test) == y_test).mean())
