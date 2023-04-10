import numpy as np
import pygame
import sys
import math
import random
from random import choice
import matplotlib.pyplot as plt
import numpy as np

BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7
EPISODES = 100

# Add a constant for the depth of the search
MAX_DEPTH = 6

# Define screen and myfont before training
pygame.init()
SQUARESIZE = 100
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE
size = (width, height)
RADIUS = int(SQUARESIZE/2 - 5)
screen = pygame.display.set_mode(size)
myfont = pygame.font.SysFont("monospace", 75)

def create_board():
    board = np.zeros((ROW_COUNT,COLUMN_COUNT))
    return board

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r
    # print(f"No open row found for column {col}")

def print_board(board):
    print(np.flip(board, 0))

def rule_based_player(board):
    valid_moves = [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]
    if len(valid_moves) == 0:
        return None

    for col in valid_moves:
        temp_board = board.copy()
        row = get_next_open_row(temp_board, col)
        drop_piece(temp_board, row, col, 2)
        if winning_move(temp_board, 2):
            return col

    for col in valid_moves:
        temp_board = board.copy()
        row = get_next_open_row(temp_board, col)
        drop_piece(temp_board, row, col, 1)
        if winning_move(temp_board, 1):
            return col

    return random.choice(valid_moves)

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def best_move(board, piece):
    global eval_count
    eval_count = 0
    valid_locations = get_valid_locations(board)
    if len(valid_locations) == 0:
        return None
    best_score = -math.inf
    best_col = choice(valid_locations)
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = minimax(temp_board, MAX_DEPTH, -math.inf, math.inf, True, piece)[1]
        if score > best_score:
            best_score = score
            best_col = col
    # Print the number of game states evaluated
    # print(f"Evaluated {eval_count} game states.")
    return best_col

def is_terminal_node(board):
    return winning_move(board, 1) or winning_move(board, 2) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximizing_player, piece):
    global eval_count
    eval_count += 1
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, 1):  # AI player wins
                return (None, 10000)
            elif winning_move(board, 2):  # Opponent player wins
                return (None, -10000)
            else:  # Game is over, no more valid moves (tie)
                return (None, 0)
        else:  # Depth is zero
            return (None, score_position(board, piece))
    if maximizing_player:
        value = -math.inf
        column = choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, piece)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, piece)[1]
            if new_score >= value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha <= beta:
                break
        return column, value

    else:  # Minimizing player
        value = math.inf
        column = choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, 3 - piece)  # Drop opponent's piece
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, piece)[1]
            if new_score <= value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha <= beta:
                break
        return column, value


def evaluate_window(window, piece):
    opponent_piece = 2
    score = 0

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(0) == 1:
        score += 30
    elif window.count(piece) == 2 and window.count(0) == 2:
        score += 20

    if window.count(opponent_piece) == 3 and window.count(0) == 1:
        score -= 40

    return score


def score_position(board, piece):
    score = 0

    # Score center column
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Score horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + 4]
            score += evaluate_window(window, piece)

    # Score vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + 4]
            score += evaluate_window(window, piece)

   # Score positive diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(4)]
            score += evaluate_window(window, piece)

    # Score negative diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score

def get_board_state(board):
    return tuple(map(tuple, board))

def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True
    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
def draw_board(board):
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT):
			pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
			pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
	
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT):		
			if board[r][c] == 1:
				pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
			elif board[r][c] == 2: 
				pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
	pygame.display.update()

def play_game(screen, player2_type='rule_based'):
    board = create_board()
    game_over = False
    turn = 0

    while not game_over:
        if turn == 0:
            col = best_move(board, 1)
            if col is not None:
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, 1)
                if winning_move(board, 1):
                    game_over = True
                    label = myfont.render("Player 1 wins!!", 1, RED)
                    screen.blit(label, (40, 10))
            else:
                print("No valid moves for player 1")
        else:
            if player2_type == 'rule_based':
                col = rule_based_player(board)
            else:
                print("Coming to minimax")
                col, _ = minimax(board, MAX_DEPTH, -math.inf, math.inf, True, 2)
            if col is None:  # Check if the board is full
                game_over = True
            else:
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, 2)
                if winning_move(board, 2):
                    game_over = True
                    label = myfont.render("Player 2 wins!!", 1, YELLOW)
                    screen.blit(label, (40, 10))

        draw_board(board)
        pygame.display.update()
        turn = (turn + 1) % 2

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

    pygame.time.wait(2)  # Wait for 3 seconds before starting the next episode

    # Return the winner (1 for player 1, 2 for player 2, 0 for a draw)
    if winning_move(board, 1):
        return 1
    elif winning_move(board, 2):
        return 2
    else:
        return 0

def calculate_wins_losses(episodes, screen, player2_type='rule_based'):
    wins_losses = {'player1': 0, 'player2': 0, 'draw': 0}
    for _ in range(episodes):
        result = play_game(screen, player2_type='rule_based')
        if result == 1:
            wins_losses['player1'] += 1
        elif result == 2:
            wins_losses['player2'] += 1
        else:
            wins_losses['draw'] += 1
    return wins_losses

def plot_progress_lines(wins_losses_history, iterations):
    labels = [f"{i+1}" for i in range(iterations)]

    player1_wins = [data['player1'] for data in wins_losses_history]
    player2_wins = [data['player2'] for data in wins_losses_history]
    draws = [data['draw'] for data in wins_losses_history]

    plt.plot(labels, player1_wins, label='Player 1 Wins', marker='o')
    plt.plot(labels, player2_wins, label='Player 2 Wins', marker='o')
    plt.plot(labels, draws, label='Draws', marker='o')

    plt.ylabel('Number of Wins and Draws')
    plt.title('Tic-Tac-Toe AI Progress')
    plt.legend()

    plt.show()

ITERATIONS = 10
wins_losses_history = []
for iteration in range(ITERATIONS):
    wins_losses = calculate_wins_losses(EPISODES, screen, player2_type='rule_based')
    wins_losses_history.append(wins_losses)
    print(f"{iteration + 1}:")
    print(f"Player 1 wins: {wins_losses['player1']}")
    print(f"Player 2 wins: {wins_losses['player2']}")
    print(f"Draws: {wins_losses['draw']}")
plot_progress_lines(wins_losses_history, ITERATIONS)