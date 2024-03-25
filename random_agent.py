import numpy as np
import matplotlib.pyplot as plt
from game import (
    create_board,
    drop_piece,
    is_valid_location,
    get_next_open_row,
    winning_move,
    check_draw,
)
import random

ROW_COUNT = 6
COLUMN_COUNT = 7


def simulate_game():
    board = create_board()
    game_over = False
    turn = 0

    while not game_over:
        col = random.randint(0, COLUMN_COUNT - 1)
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, 1 + turn % 2)

            if winning_move(board, 1 + turn % 2):
                return (1 + turn % 2, turn)  # Return winner and number of turns
            elif check_draw(board):
                return (0, turn)  # Return draw and number of turns

            turn += 1

    return None


# Simulate games
n_games = 1000
results = [simulate_game() for _ in range(n_games)]

# Count wins and draws
wins = [result[0] for result in results]
draws = wins.count(0)
wins = [win for win in wins if win != 0]
win1 = wins.count(1)
win2 = wins.count(2)

# Count number of turns
turns = [result[1] for result in results]
avg_turns = sum(turns) / n_games

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.pie([win1, win2, draws], labels=["Player 1", "Player 2", "Draw"], autopct="%1.1f%%")
plt.title("Game Outcomes")
plt.subplot(1, 2, 2)
plt.hist(turns, bins=range(min(turns), max(turns) + 2), align="left", rwidth=0.8)
plt.title("Number of Turns")
plt.xlabel("Turns")
plt.ylabel("Games")
plt.show()
