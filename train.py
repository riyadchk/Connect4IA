import matplotlib.pyplot as plt
from game.board import Board
from players.dqn_agent import DQNPlayer  # Adjust import as necessary
import numpy as np
from tqdm import tqdm


class Training:
    def __init__(self, player1, player2, n_games, batch_size):
        self.player1 = player1
        self.player2 = player2
        self.n_games = n_games
        self.batch_size = batch_size  # For DQN replay
        self.results = {"Player 1 Wins": 0, "Player 2 Wins": 0, "Draws": 0}
        self.turns = []

    def simulate_games(self):

        for game in tqdm(range(self.n_games)):
            board = Board()
            game_over = False
            turn = 0
            state_history = []
            current_turn = 0

            while not game_over:
                current_player = self.player1 if turn % 2 == 0 else self.player2
                state = current_player._get_state_from_board(board)

                action = current_player.act(state)  # DQN chooses action

                if board.is_valid_location(action):
                    row = board.get_next_open_row(action)
                    board.drop_piece(row, action, current_player.piece)
                    if turn % 2 == 0:
                        current_turn += 1
                    next_state = current_player._get_state_from_board(board)
                    reward = 0  # Define your reward here

                    if board.winning_move(current_player.piece):
                        reward = 1  # Example reward
                        game_over = True
                        self.results[
                            "Player 1 Wins" if turn == 0 else "Player 2 Wins"
                        ] += 1
                    elif board.check_draw():
                        reward = 0.5  # Example reward for a draw
                        game_over = True
                        self.results["Draws"] += 1

                    done = 1 if game_over else 0  # Flag to indicate game over
                    current_player.remember(
                        state, action, reward, next_state, done
                    )  # Store experience
                    state_history.append((state, action, reward, next_state, done))

                turn = 1 - turn

            self.turns.append(current_turn)

            print(f"Game {game + 1}/{self.n_games}, Turns: {current_turn}")

            # Learning phase
            self.player1.replay(self.batch_size)
            self.player2.replay(self.batch_size)

            # update target model
            self.player1.update_target_model()
            self.player2.update_target_model()

            # Switch players to ensure learning from both perspectives
            self.player1, self.player2 = self.player2, self.player1

        # Save models after training
        self.player1.save_model(f"players/trained_models/player1_{self.n_games}.pth")
        self.player2.save_model(f"players/trained_models/player2_{self.n_games}.pth")

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
    input_dim = 42  # Assuming a flattened board state
    player1 = DQNPlayer(piece=1, input_dim=input_dim)
    player2 = DQNPlayer(
        piece=2, input_dim=input_dim
    )  # Could share model weights with player1 for true self-play
    player1 = player1
    player2 = player2
    n_games = 300
    batch_size = 256  # Example batch size for training
    stats = Training(player1, player2, n_games, batch_size)
    stats.simulate_games()
    stats.display_stats()
