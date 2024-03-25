from abc import ABC, abstractmethod


class Player(ABC):
    """
    An abstract base class for players in the Connect 4 game.

    Attributes:
        piece (int): The piece type that this player uses, 1 or 2 typically representing different colors.
    """

    def __init__(self, piece):
        """
        Initializes a new Player instance.

        Parameters:
            piece (int): The piece type that this player will use in the game.
        """
        self.piece = piece

    @abstractmethod
    def move(self, board):
        """
        Abstract method to determine the column in which to drop the piece.
        This method must be implemented by subclasses representing specific player types.

        Parameters:
            board (np.array): The current state of the game board as a numpy array.

        Returns:
            int: The column number where the player decides to drop their piece.
        """
        pass
