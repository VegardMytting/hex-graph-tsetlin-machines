#!/bin/bash

# Quick Start Script for Hex Winner Prediction
# This script guides you through the entire process

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  Hex Winner Prediction - Quick Start                  ║"
echo "║  Using Graph Tsetlin Machines                         ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Check if data exists
if [ ! -d "data" ] || [ -z "$(ls -A data)" ]; then
    echo "📊 Step 1: Generate Data"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "No data found. Generating datasets..."
    echo "This will take 10-30 minutes depending on your system."
    echo ""
    read -p "Generate data now? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        chmod +x generate_data.sh
        bash generate_data.sh
    else
        echo "Skipping data generation. You'll need to generate data manually."
    fi
    echo ""
else
    echo "✓ Data directory exists"
    echo ""
fi

# Main menu
while true; do
    echo ""
    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║  What would you like to do?                           ║"
    echo "╚═══════════════════════════════════════════════════════╝"
    echo ""
    echo "  1) Quick test on 5x5 board (fast, ~95% accuracy)"
    echo "  2) Optimal run on 5x5 board (balanced, ~99-100% accuracy)"
    echo "  3) Aggressive run on 5x5 board (best chance at 100%)"
    echo "  4) Optimal run on 6x6 board (challenging, ~95-100%)"
    echo "  5) Aggressive run on 6x6 board (maximum effort)"
    echo "  6) Custom configuration (expert mode)"
    echo "  7) Hyperparameter tuning (find best parameters)"
    echo "  8) View optimal configurations"
    echo "  9) Test all three datasets (final, minus2, minus5)"
    echo "  0) Exit"
    echo ""
    read -p "Enter your choice (0-9): " choice
    
    case $choice in
        1)
            echo ""
            echo "🚀 Running quick test on 5x5 board..."
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            python3 optimal_configs.py --board-size 5 --dataset final --mode fast
            ;;
        2)
            echo ""
            echo "🎯 Running optimal configuration on 5x5 board..."
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            python3 optimal_configs.py --board-size 5 --dataset final --mode optimal
            ;;
        3)
            echo ""
            echo "💪 Running aggressive configuration on 5x5 board..."
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            python3 optimal_configs.py --board-size 5 --dataset final --mode aggressive
            ;;
        4)
            echo ""
            echo "🎯 Running optimal configuration on 6x6 board..."
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            python3 optimal_configs.py --board-size 6 --dataset final --mode optimal
            ;;
        5)
            echo ""
            echo "💪 Running aggressive configuration on 6x6 board..."
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            python3 optimal_configs.py --board-size 6 --dataset final --mode aggressive
            ;;
        6)
            echo ""
            echo "⚙️  Custom Configuration"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            read -p "Board size (5/6/7/8): " bsize
            read -p "Number of clauses: " clauses
            read -p "T value: " tval
            read -p "s value: " sval
            read -p "Depth (1/2/3): " depth
            read -p "Max included literals: " maxlit
            read -p "Epochs: " epochs
            
            python3 hex_solution.py \
                --board-size $bsize \
                --number-of-clauses $clauses \
                --T $tval \
                --s $sval \
                --depth $depth \
                --max-included-literals $maxlit \
                --epochs $epochs
            ;;
        7)
            echo ""
            echo "🔍 Hyperparameter Tuning"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            read -p "Board size (5/6): " bsize
            echo ""
            echo "Choose mode:"
            echo "  1) Quick - Test one promising configuration"
            echo "  2) Grid - Test multiple configurations (slow)"
            read -p "Mode (1/2): " mode_choice
            
            if [ "$mode_choice" == "1" ]; then
                python3 tune_hyperparameters.py --board-size $bsize --mode quick
            else
                python3 tune_hyperparameters.py --board-size $bsize --mode grid
            fi
            ;;
        8)
            echo ""
            echo "📋 Viewing Optimal Configurations"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            read -p "Board size (5/6/7/8): " bsize
            python3 optimal_configs.py --board-size $bsize --print-only
            ;;
        9)
            echo ""
            echo "🧪 Testing All Three Datasets"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            read -p "Board size (5/6): " bsize
            
            echo ""
            echo "Testing FINAL dataset..."
            python3 optimal_configs.py --board-size $bsize --dataset final --mode optimal
            
            echo ""
            echo "Testing MINUS2 dataset..."
            python3 optimal_configs.py --board-size $bsize --dataset minus2 --mode optimal
            
            echo ""
            echo "Testing MINUS5 dataset..."
            python3 optimal_configs.py --board-size $bsize --dataset minus5 --mode optimal
            ;;
        0)
            echo ""
            echo "👋 Thanks for using the Hex GTM solver!"
            echo ""
            exit 0
            ;;
        *)
            echo ""
            echo "❌ Invalid choice. Please enter a number between 0 and 9."
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done
