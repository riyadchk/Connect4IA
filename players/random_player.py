import random
from .player import Player  
from game.constants import COLUMN_COUNT  
from game.board import Board  


class RandomPlayer(Player):
    """
    A player that makes moves by randomly selecting a valid column.
    """

    def __init__(self, piece):
        """
        Initializes a new RandomPlayer instance with the specified piece type.

        Parameters:
            piece (int): The piece type that this player will use in the game.
        """
        super().__init__(piece)

    def move(self, board: Board):
        """
        Determines the column where the player decides to drop their piece by selecting randomly from available columns.

        Parameters:
            board (Board): The current state of the game board.

        Returns:
            int: The column number where the player decides to drop their piece.
        """
        valid_locations = []
        for col in range(COLUMN_COUNT):
            if board.is_valid_location(col):
                valid_locations.append(col)

        if valid_locations:
            return random.choice(valid_locations)
        else:
            raise Exception("No valid locations available.")
