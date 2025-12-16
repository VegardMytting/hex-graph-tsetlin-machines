# Hex Winner Prediction using Graph Tsetlin Machine

This project uses Graph Tsetlin Machines (GTM) to predict winners in the board game Hex with near-perfect or perfect accuracy.

## 📋 Project Structure

```
hex_project/
├── hex_solution.py              # Main solution file
├── tune_hyperparameters.py      # Hyperparameter tuning script  
├── generate_data.sh             # Script to generate Hex datasets
├── hex.c                        # C program for data generation
├── data/                        # Directory for CSV datasets
└── README.md                    # This file
```

## 🎯 Goal

Predict the winner of Hex games at three different stages:
1. **Final board** - Complete game
2. **Minus 2 moves** - Two moves before the end
3. **Minus 5 moves** - Five moves before the end

**Target**: Achieve 100% accuracy on the largest possible board size.

## 🚀 Quick Start

### 1. Generate Data

First, generate the Hex game datasets:

```bash
cd hex_project
bash generate_data.sh
```

This will create CSV files in the `data/` directory for different board sizes (5x5, 6x6, 7x7, 8x8).

### 2. Run Training

Train on 5x5 board (easiest to get 100%):

```bash
python hex_solution.py --board-size 5 --dataset final --epochs 100
```

Train on 6x6 board:

```bash
python hex_solution.py --board-size 6 --dataset final --epochs 150 --number-of-clauses 8000
```

### 3. Test Different Game Stages

```bash
# Final board
python hex_solution.py --board-size 5 --dataset final

# Two moves before end
python hex_solution.py --board-size 5 --dataset minus2

# Five moves before end  
python hex_solution.py --board-size 5 --dataset minus5
```

## 🔧 Key Hyperparameters

The most important hyperparameters for achieving 100% accuracy:

| Parameter | Description | 5x5 | 6x6 | 7x7 |
|-----------|-------------|-----|-----|-----|
| `--number-of-clauses` | Number of logical rules | 4000 | 8000 | 12000 |
| `--T` | Threshold (voting strength) | 2000 | 3000 | 4000 |
| `--s` | Specificity (how specific rules are) | 5.0 | 10.0 | 15.0 |
| `--depth` | Message passing depth | 2 | 2 | 2-3 |
| `--max-included-literals` | Max literals per clause | 32 | 48 | 64 |
| `--epochs` | Training epochs | 100 | 150 | 200 |

## 🎓 Understanding the Solution

### Graph Representation

Each Hex board is represented as a graph where:

- **Nodes**: Each cell on the board is a node
- **Node Properties**: 
  - `'X'` if player X occupies the cell
  - `'O'` if player O occupies the cell  
  - Empty cells have no properties
- **Edges**: Connect adjacent cells in the hexagonal grid (6 neighbors per cell)

### Why GTM Works Well for Hex

1. **Logical Patterns**: Hex games have clear logical winning patterns (chains, connections)
2. **Local + Global**: GTM can learn both local patterns (cell configurations) and global patterns (through message passing)
3. **Interpretability**: We can inspect the learned clauses to understand what patterns the model uses

### Message Passing (Depth Parameter)

- `depth=1`: Only look at individual cells (no edge information)
- `depth=2`: Cells can send messages to neighbors (local neighborhoods)  
- `depth=3`: Information propagates further (better for larger boards)

## 🔬 Hyperparameter Tuning

To find optimal parameters automatically:

```bash
# Quick test with promising parameters
python tune_hyperparameters.py --board-size 5 --mode quick

# Full grid search (slower but thorough)
python tune_hyperparameters.py --board-size 5 --mode grid
```

## 📊 Expected Results

### 5x5 Board
- **Final**: Should achieve 99-100% accuracy with default parameters
- **Minus2**: Should achieve 95-100% accuracy
- **Minus5**: Should achieve 85-95% accuracy (harder to predict)

### 6x6 Board
- **Final**: Should achieve 95-100% accuracy with increased clauses
- **Minus2**: Should achieve 90-98% accuracy
- **Minus5**: Should achieve 80-90% accuracy

## 💡 Tips for Achieving 100%

1. **Start Small**: Get 100% on 5x5 before trying larger boards

2. **Increase Clauses**: If accuracy plateaus, increase `--number-of-clauses`

3. **Balance T and s**:
   - Higher `T` → More aggressive learning
   - Higher `s` → More specific patterns

4. **Try Different Depths**:
   - `depth=1`: Faster, good for simple patterns
   - `depth=2`: Better accuracy, considers neighbors
   - `depth=3`: Best for larger boards, slower

5. **More Epochs**: If accuracy is improving slowly, train longer

6. **Adjust max-included-literals**:
   - Too low → Can't capture complex patterns
   - Too high → May overfit or be slow

## 🐛 Troubleshooting

### Low Accuracy (<90%)

```bash
# Try more clauses
python hex_solution.py --board-size 5 --number-of-clauses 8000

# Try higher specificity
python hex_solution.py --board-size 5 --s 10.0

# Try message passing
python hex_solution.py --board-size 5 --depth 2
```

### Training Too Slow

```bash
# Reduce clauses
python hex_solution.py --board-size 5 --number-of-clauses 2000

# Use depth=1 (no message passing)
python hex_solution.py --board-size 5 --depth 1

# Use fewer training examples
python hex_solution.py --board-size 5 --train-size 400000
```

### Out of Memory

```bash
# Reduce clauses
python hex_solution.py --board-size 5 --number-of-clauses 2000

# Use smaller training set
python hex_solution.py --board-size 5 --train-size 400000 --test-size 50000
```

## 📈 Advanced Usage

### Save and Load Models

```bash
# Train and save
python hex_solution.py --board-size 5 --save-model models/hex_5x5.pkl

# Load and test
python hex_solution.py --board-size 5 --load-model models/hex_5x5.pkl --epochs 0
```

### Analyze Learned Rules

The verbose output shows the top clauses learned:

```bash
python hex_solution.py --board-size 5 --verbose True
```

This will show:
- Clause weights (importance)
- Which symbols each clause includes
- Pattern statistics

## 📝 Report Requirements

Your report should include:

1. **Description of Solution**
   - Graph representation details
   - Hyperparameter choices and why
   - Any special techniques used

2. **Performance Statistics**
   - Accuracy on all three datasets (final, minus2, minus5)
   - Board sizes tested
   - Number of clauses used
   - Training time

3. **Interpretability Analysis**
   - What patterns did the model learn?
   - Show example clauses and their meanings
   - Visualize important board configurations

## 🏆 Leaderboard Criteria

**Winner**: Highest board size with 100% accuracy
**Tiebreaker**: Fewest number of clauses

Example competitive results:
```
Team A: 5x5 board, 100% accuracy, 4000 clauses
Team B: 6x6 board, 100% accuracy, 8000 clauses  ← Winner
Team C: 7x7 board, 98% accuracy, 12000 clauses
```

## 📚 References

- Graph Tsetlin Machine: https://github.com/cair/GraphTsetlinMachine
- Hex Game Rules: https://www.krammer.nl/hex/
- Original Paper: "Logic-based AI for Interpretable Board Game Winner Prediction with Tsetlin Machine" (IJCNN 2022)

## 🤝 Support

If you encounter issues:

1. Check this README thoroughly
2. Verify data files are generated correctly
3. Try the quick test first: `python tune_hyperparameters.py --mode quick`
4. Start with small board sizes (5x5) before scaling up

Good luck achieving 100% accuracy! 🎯
