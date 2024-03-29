from game.board import Board
from players.random_player import RandomPlayer
from players.human_player import HumanPlayer
from players.dqn_agent import DQNPlayer
from utils.gui import GUI  # Make sure this path matches your project structure
import sys
import argparse


def simulate_game(use_gui=False):
    board = Board()
    game_over = False
    turn = 0

    gui = GUI() if use_gui else None

    player1 = HumanPlayer(1, gui)
    player2 = HumanPlayer(2, gui)
    players = [player1, player2]

    if use_gui:
        gui.draw_board(board.board)

    while not game_over:
        current_player = players[turn % 2]
        col = current_player.move(board)

        if board.is_valid_location(col):
            row = board.get_next_open_row(col)
            board.drop_piece(row, col, current_player.piece)

            if use_gui:
                gui.draw_board(board.board)

            if board.winning_move(current_player.piece):
                if use_gui:
                    gui.display_message(f"Player {current_player.piece} wins!")
                else:
                    print(f"Player {current_player.piece} wins!")
                game_over = True

            elif board.check_draw():
                if use_gui:
                    gui.display_message("Game is a draw!")
                else:
                    print("Game is a draw!")
                game_over = True

        turn += 1

    if use_gui:
        # Wait for a few seconds to show the end game state
        gui.wait(3000)
        gui.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a game of Connect 4.")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable GUI mode to visualize the game.",
        default=True,
    )
    args = parser.parse_args()

    simulate_game(args.gui)
    sys.exit()
