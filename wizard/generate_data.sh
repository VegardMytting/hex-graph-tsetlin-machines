#!/bin/bash

# Script to generate Hex game datasets for different board sizes
# Usage: bash generate_data.sh

echo "====================================="
echo "Hex Dataset Generation Script"
echo "====================================="

# Create data directory if it doesn't exist
mkdir -p data

# Generate 5x5 boards (1 million games)
echo ""
echo "Generating 5x5 board data (1M games)..."
gcc -DBOARD_DIM=5 -DNUM_GAMES=1000000 hex.c -o hex_5x5
if [ $? -eq 0 ]; then
    ./hex_5x5
    rm hex_5x5
    echo "✓ 5x5 data generated successfully!"
else
    echo "✗ Failed to compile for 5x5"
fi

# Generate 6x6 boards (1 million games)
echo ""
echo "Generating 6x6 board data (1M games)..."
gcc -DBOARD_DIM=6 -DNUM_GAMES=1000000 hex.c -o hex_6x6
if [ $? -eq 0 ]; then
    ./hex_6x6
    rm hex_6x6
    echo "✓ 6x6 data generated successfully!"
else
    echo "✗ Failed to compile for 6x6"
fi

# Generate 7x7 boards (500k games - takes longer)
echo ""
echo "Generating 7x7 board data (500K games)..."
gcc -DBOARD_DIM=7 -DNUM_GAMES=500000 hex.c -o hex_7x7
if [ $? -eq 0 ]; then
    ./hex_7x7
    rm hex_7x7
    echo "✓ 7x7 data generated successfully!"
else
    echo "✗ Failed to compile for 7x7"
fi

# Generate 8x8 boards (250k games - takes even longer)
echo ""
echo "Generating 8x8 board data (250K games)..."
gcc -DBOARD_DIM=8 -DNUM_GAMES=250000 hex.c -o hex_8x8
if [ $? -eq 0 ]; then
    ./hex_8x8
    rm hex_8x8
    echo "✓ 8x8 data generated successfully!"
else
    echo "✗ Failed to compile for 8x8"
fi

echo ""
echo "====================================="
echo "Dataset Generation Complete!"
echo "====================================="
echo ""
echo "Generated files in data/ directory:"
ls -lh data/

echo ""
echo "You can now run the training script:"
echo "  python hex_solution.py --board-size 5"
echo "  python hex_solution.py --board-size 6"
echo "  etc."
