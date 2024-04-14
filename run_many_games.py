import matplotlib.pyplot as plt
from players.random_player import RandomPlayer  # Adjust import as necessary
from game.board import Board
import numpy as np


class Stats:
    def __init__(self, player1, player2, n_games):
        self.player1 = player1
        self.player2 = player2
        self.n_games = n_games
        self.results = {"Player 1 Wins": 0, "Player 2 Wins": 0, "Draws": 0}
        self.turns = []

    def simulate_games(self):
        for _ in range(self.n_games):
            board = Board()
            game_over = False
            turn = 0
            current_turn = 0

            while not game_over:
                current_player = self.player1 if turn == 0 else self.player2
                col = current_player.move(board)

                if board.is_valid_location(col):
                    row = board.get_next_open_row(col)
                    board.drop_piece(row, col, current_player.piece)
                    current_turn += 1

                    if board.winning_move(current_player.piece):
                        winner = "Player 1 Wins" if turn == 0 else "Player 2 Wins"
                        self.results[winner] += 1
                        game_over = True

                    elif board.check_draw():
                        self.results["Draws"] += 1
                        game_over = True

                turn = 1 - turn

            self.turns.append(current_turn)

    def display_stats(self):
        # Plotting results
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Pie chart for game outcomes
        labels = self.results.keys()
        sizes = self.results.values()
        colors = ["gold", "lightcoral", "lightskyblue"]
        axs[0].pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140
        )
        axs[0].axis("equal")
        axs[0].set_title("Game Outcomes")

        # Histogram for number of turns
        axs[1].hist(
            self.turns,
            bins=np.arange(min(self.turns) - 0.5, max(self.turns) + 1.5, 1),
            rwidth=0.8,
        )
        axs[1].set_title("Number of Turns per Game")
        axs[1].set_xlabel("Turns")
        axs[1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    player1 = RandomPlayer(1)
    player2 = RandomPlayer(2)
    n_games = 10000  # Example: simulate 100 games
    stats = Stats(player1, player2, n_games)
    stats.simulate_games()
    stats.display_stats()
