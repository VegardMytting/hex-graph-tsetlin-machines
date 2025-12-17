# IKT466 - Final Project
#### Hex Winner Prediction using Graph Tsetlin Machines

This project trains and evaluates a Multi-Class Graph Tsetlin Machine (GTM) to predict the winner of a Hex game from board states. The system supports **multiple board sizes**, **different game-end scenarios**, **and large-scale datasets**.

## Project Structure

    ├── main.py
    ├── hex.c
    ├── requirements.txt
    ├── data/
    ├── results/
    ├── Makefile
    └── README.md



## Requirements

To install required Python packages:

    pip install -r requirements.txt

## Dataset Generation

Compile the dataset generator:

    make hex

This produces the executable:

    ./hex

This will generate CSV files in the `data/` directory with the name scheme:

    hex_{N}x{N}_{DATASET_SIZE}_{SCENARIO}.csv

Example:

hex_7x7_1000000_final.csv
hex_7x7_1000000_minus2.csv
hex_7x7_1000000_minus5.csv

### Scenarios
- `final` - Final board state
- `minus2` - Board with last 2 moves removed
- `minus5` - Board with last 5 moves removed

> [!CAUTION]
All board sizes and scenarios you want to train on **must exist** in `data/`

## Running the Program

Train on a single board size:

    python main.py --board-size 7 --data-dir data --dataset-size 1000000

Train on multiple board sizes:

    python main.py --all-sizes --min-size 3 --max-size 11 --data-dir data --dataset-size 1000000


This will:
- Train a separate model for each board size
- Evaluate on all scenarios
- Save results to `results/`

## Command-Line Arguments

### Board Sizes
| Argument | Description |
|--------|------------|
| `--board-size` | Train on a single board size |
| `--all-sizes` | Train on all sizes from `--min-size` to `--max-size` |
| `--min-size` | Minimum board size |
| `--max-size` | Maximum board size |

---

### Dataset
| Argument | Description |
|--------|------------|
| `--data-dir` | Directory containing datasets |
| `--dataset-size` | Must match dataset file names |

---

### Training Control
| Argument | Description |
|--------|------------|
| `--train-on` | Scenario to train on (`final`, `minus2`, `minus5`) |
| `--train-samples` | Override number of training samples |
| `--epochs` | Override training epochs |
| `--seed` | Random seed (default: 66) |

---

### Model Options
| Argument | Description |
|--------|------------|
| `--balance-train` | Balance training set (default: ON) |
| `--no-balance-train` | Disable class balancing |
| `--no-border-nodes` | Disable Hex border nodes |
| `--neighbor-counts` | Add neighbor count symbols |
| `--early-stop` | Early stopping on small boards |

---

### Output
| Argument | Description |
|--------|------------|
| `--results-dir` | Directory for results (default: `results/`) |

---

> [!NOTE] 
Large datasets (1M samples) require significant RAM and time.