# hex_configs.py
"""
Optimized hyperparameters for different board sizes.
These configurations are tuned to achieve high accuracy.
"""

CONFIGS = {
    5: {  # 5x5 board - easiest
        'number_of_clauses': 4000,
        'T': 3000,
        's': 3.0,
        'depth': 2,
        'max_included_literals': 32,
        'epochs': 60,
        'hypervector_size': 128,
        'message_size': 256,
    },
    6: {  # 6x6 board
        'number_of_clauses': 8000,
        'T': 5000,
        's': 5.0,
        'depth': 2,
        'max_included_literals': 32,
        'epochs': 100,
        'hypervector_size': 128,
        'message_size': 256,
    },
    7: {  # 7x7 board
        'number_of_clauses': 12000,
        'T': 8000,
        's': 7.0,
        'depth': 3,
        'max_included_literals': 48,
        'epochs': 150,
        'hypervector_size': 256,
        'message_size': 512,
    },
    8: {  # 8x8 board
        'number_of_clauses': 16000,
        'T': 10000,
        's': 10.0,
        'depth': 3,
        'max_included_literals': 64,
        'epochs': 200,
        'hypervector_size': 256,
        'message_size': 512,
    }
}

def get_config(board_size):
    """Get optimized configuration for a specific board size"""
    if board_size not in CONFIGS:
        raise ValueError(f"No configuration for board size {board_size}")
    return CONFIGS[board_size]