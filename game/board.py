import numpy as np
from .constants import ROW_COUNT, COLUMN_COUNT  


class Board:
    """
    Represents the game board for Connect 4, encapsulating the state and behaviors related to the board itself.
    """

    def __init__(self):
        """
        Initializes a new Board instance with an empty board.
        """
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT))

    def drop_piece(self, row, col, piece):
        """
        Drops a piece into the board.
        """
        self.board[row][col] = piece

    def is_valid_location(self, col):
        """
        Checks if a column can accept a new piece.
        """
        return self.board[ROW_COUNT - 1][col] == 0

    def get_next_open_row(self, col):
        """
        Finds the next open row in a column.
        """
        for r in range(ROW_COUNT):
            if self.board[r][col] == 0:
                return r

    def winning_move(self, piece):
        """
        Checks for a winning move on the board.
        """
        # Check horizontal locations for win
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if (
                    self.board[r][c] == piece
                    and self.board[r][c + 1] == piece
                    and self.board[r][c + 2] == piece
                    and self.board[r][c + 3] == piece
                ):
                    return True

        # Check vertical locations for win
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if (
                    self.board[r][c] == piece
                    and self.board[r + 1][c] == piece
                    and self.board[r + 2][c] == piece
                    and self.board[r + 3][c] == piece
                ):
                    return True

        # Check positively sloped diagonals
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if (
                    self.board[r][c] == piece
                    and self.board[r + 1][c + 1] == piece
                    and self.board[r + 2][c + 2] == piece
                    and self.board[r + 3][c + 3] == piece
                ):
                    return True

        # Check negatively sloped diagonals
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if (
                    self.board[r][c] == piece
                    and self.board[r - 1][c + 1] == piece
                    and self.board[r - 2][c + 2] == piece
                    and self.board[r - 3][c + 3] == piece
                ):
                    return True

        return False

    def print_board(self):
        """
        Prints the board state, flipping the board so that the bottom row is shown as the bottom.
        """
        print(np.flip(self.board, 0))

    def check_draw(self):
        """
        Checks if the game is a draw (i.e., no more valid moves).
        """
        return np.all(self.board != 0)

    def get_state_representation(self):
        """
        Returns the current state of the board as a 3D tensor suitable for input to a neural network.
        """
        state = np.zeros((ROW_COUNT, COLUMN_COUNT, 2))
        for r in range(ROW_COUNT):
            for c in range(COLUMN_COUNT):
                if self.board[r][c] == 1:
                    state[r][c][0] = 1
                elif self.board[r][c] == 2:
                    state[r][c][1] = 1
        return state
