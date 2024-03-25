import numpy as np

# Game Configuration Constants
ROW_COUNT = 6
COLUMN_COUNT = 7


# Functions to manage the game state
def create_board():
    """Creates and returns a new game board."""
    return np.zeros((ROW_COUNT, COLUMN_COUNT))


def drop_piece(board, row, col, piece):
    """Drops a piece into the board."""
    board[row][col] = piece


def is_valid_location(board, col):
    """Checks if a column can receive another piece."""
    return board[ROW_COUNT - 1][col] == 0


def get_next_open_row(board, col):
    """Finds the next open row within a column."""
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r
    return None


def print_board(board):
    """Prints the board."""
    print(np.flip(board, 0))


def winning_move(board, piece):
    """Checks if the last move was a winning move."""
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if (
                board[r][c] == piece
                and board[r][c + 1] == piece
                and board[r][c + 2] == piece
                and board[r][c + 3] == piece
            ):
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if (
                board[r][c] == piece
                and board[r + 1][c] == piece
                and board[r + 2][c] == piece
                and board[r + 3][c] == piece
            ):
                return True

    # Check positively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if (
                board[r][c] == piece
                and board[r + 1][c + 1] == piece
                and board[r + 2][c + 2] == piece
                and board[r + 3][c + 3] == piece
            ):
                return True

    # Check negatively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if (
                board[r][c] == piece
                and board[r - 1][c + 1] == piece
                and board[r - 2][c + 2] == piece
                and board[r - 3][c + 3] == piece
            ):
                return True
    return False


def check_draw(board):
    """Checks if the game is a draw."""
    return np.all(board != 0)
