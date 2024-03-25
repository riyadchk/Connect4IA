import pygame
import sys
import math
from game.board import Board
from game.constants import (
    ROW_COUNT,
    COLUMN_COUNT,
    SQUARESIZE,
    SIZE,
    BLUE,
    BLACK,
    RED,
    YELLOW,
    RADIUS,
    HEIGHT,
)


class GUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SIZE)
        self.myfont = pygame.font.SysFont("monospace", 75)

    def draw_board(self, board):
        """
        Draws the game board onto the screen using Pygame.

        Parameters:
            board (np.ndarray): The game board as a 2D NumPy array.
        """
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT + 1):
                pygame.draw.rect(
                    self.screen,
                    BLUE,
                    (
                        c * SQUARESIZE,
                        r * SQUARESIZE + SQUARESIZE,
                        SQUARESIZE,
                        SQUARESIZE,
                    ),
                )
                pygame.draw.circle(
                    self.screen,
                    BLACK,
                    (
                        int(c * SQUARESIZE + SQUARESIZE / 2),
                        int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2),
                    ),
                    RADIUS,
                )

        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                if board[r][c] == 1:
                    pygame.draw.circle(
                        self.screen,
                        RED,
                        (
                            int(c * SQUARESIZE + SQUARESIZE / 2),
                            HEIGHT - int(r * SQUARESIZE + SQUARESIZE / 2),
                        ),
                        RADIUS,
                    )
                elif board[r][c] == 2:
                    pygame.draw.circle(
                        self.screen,
                        YELLOW,
                        (
                            int(c * SQUARESIZE + SQUARESIZE / 2),
                            HEIGHT - int(r * SQUARESIZE + SQUARESIZE / 2),
                        ),
                        RADIUS,
                    )

        pygame.display.update()

    def display_message(self, message):
        """
        Displays a message on the screen.

        Parameters:
            message (str): The message to display.
        """
        label = self.myfont.render(message, 1, RED)
        self.screen.blit(label, (40, 10))
        pygame.display.update()

    def clear_screen(self):
        """
        Clears the screen to prepare for the next frame.
        """
        self.screen.fill(BLACK)

    def wait(self, milliseconds):
        pygame.time.delay(milliseconds)

    # Inside the GUI class in utils/gui.py

    def wait_for_human_move(self):
        """
        Waits for the human player to make a move by clicking on a column.

        Returns:
            int: The index of the column where the human player wants to drop their piece.
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    xpos = event.pos[0]
                    col = int(math.floor(xpos / SQUARESIZE))
                    return col

    def close(self):
        """
        Closes the Pygame window.
        """
        pygame.quit()
