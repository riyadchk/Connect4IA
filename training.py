import numpy as np
from tqdm import tqdm
from game.board import Board
from players.dqn_agent import DQNPlayer
from players.random_player import RandomPlayer


class Trainer:
    def __init__(self, dqn_player, random_player, n_games, batch_size):
        self.dqn_player = dqn_player
        self.random_player = random_player
        self.n_games = n_games
        self.batch_size = batch_size
        self.results = {"DQN Wins": 0, "Random Wins": 0, "Draws": 0}

    def play_game(self):
        board = Board()
        game_over = False
        while not game_over:
            for player in [self.dqn_player, self.random_player]:
                state = self._get_state_from_board(board, player.piece)
                action = player.move(board)

                if board.is_valid_location(action):
                    row = board.get_next_open_row(action)
                    board.drop_piece(row, action, player.piece)

                    if board.winning_move(player.piece):
                        game_over = True
                        if isinstance(player, DQNPlayer):
                            self.results["DQN Wins"] += 1
                            reward = 1
                        else:
                            self.results["Random Wins"] += 1
                            reward = -1
                        self.dqn_player.remember(state, action, reward, None, game_over)
                        break

                    if board.check_draw():
                        self.results["Draws"] += 1
                        game_over = True
                        reward = 0.5
                        self.dqn_player.remember(state, action, reward, None, game_over)
                        break

                next_state = self._get_state_from_board(board)
                if not game_over and isinstance(player, DQNPlayer):
                    reward = 0
                    self.dqn_player.remember(
                        state, action, reward, next_state, game_over
                    )

    def train(self):
        for _ in tqdm(range(self.n_games)):
            self.play_game()
            if len(self.dqn_player.memory) > self.batch_size:
                self.dqn_player.replay(self.batch_size)
            self.dqn_player.update_target_model()

        # After training, you may want to save the model.
        self.dqn_player.save_model("dqn_player_model.pth")

    def _get_state_from_board(self, board):
        return board.get_state_representation()


if __name__ == "__main__":
    dqn_player = DQNPlayer(piece=1, input_dim=6 * 7 * 2, output_dim=7)
    random_player = RandomPlayer(piece=2)
    trainer = Trainer(dqn_player, random_player, n_games=10000, batch_size=64)
    trainer.train()
