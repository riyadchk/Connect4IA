# Game board dimensions
ROW_COUNT = 6
COLUMN_COUNT = 7

# GUI dimensions and colors
SQUARESIZE = 100  # Size of squares in the game grid
RADIUS = int(SQUARESIZE / 2 - 5)  # Radius of the circular game pieces
WIDTH = COLUMN_COUNT * SQUARESIZE  # Width of the game window
HEIGHT = (
    ROW_COUNT + 1
) * SQUARESIZE  # Height of the game window (includes an additional row for piece dropping)
SIZE = (WIDTH, HEIGHT)  # Tuple representing the size of the game window

# Colors defined in RGB format
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
