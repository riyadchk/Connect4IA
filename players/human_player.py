from .player import Player  # Adjust import as necessary


class HumanPlayer(Player):
    def __init__(self, piece, gui=None):
        super().__init__(piece)
        self.selected_column = None
        self.gui = gui

    def move(self, board):
        """
        Determines the column where the human player decides to drop their piece.
        This implementation assumes the use of a GUI for the human to make their move.

        Parameters:
            board (Board): The current state of the game board. Not used in this method but kept for consistency.
            gui (GUI): An instance of the GUI class to register human clicks. If None, waits for terminal input.

        Returns:
            int: The column number where the player decides to drop their piece.
        """
        if self.gui:
            return self.gui.wait_for_human_move()
        else:
            # Fallback for no GUI: Prompt for terminal input (not ideal for real-time gameplay)
            col = int(input("Select column (0-6): "))
            while not board.is_valid_location(col):
                col = int(
                    input("Column full or invalid. Select another column (0-6): ")
                )
            return col
