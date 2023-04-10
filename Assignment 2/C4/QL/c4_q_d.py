import numpy as np
import pygame
import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np

BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7
EPISODES = 100
ALPHA = 0.5
GAMMA = 0.6
EPSILON = 0.1

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
        drop_piece(temp_board, row, col, 1)
        if winning_move(temp_board, 1):
            return col

    # for col in valid_moves:
    #     temp_board = board.copy()
    #     row = get_next_open_row(temp_board, col)
    #     drop_piece(temp_board, row, col, 2)
    #     if winning_move(temp_board, 2):
    #         return col

    return random.choice(valid_moves)

def q_learning_player(board, Q, epsilon):
    if np.random.random() < epsilon:
        valid_moves = [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]
        if len(valid_moves) == 0:
            return None
        return random.choice(valid_moves)

    state = get_board_state(board)
    if state not in Q:
        Q[state] = np.zeros(COLUMN_COUNT)

    valid_moves = [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]
    if len(valid_moves) == 0:
        return None

    q_values = np.array([Q[state][c] if c in valid_moves else -np.inf for c in range(COLUMN_COUNT)])
    return np.argmax(q_values)

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

def play_game(Q, epsilon, screen, player2_type='rule_based'):
    board = create_board()
    game_over = False
    turn = 0
    states = []  # Keep track of the states at each step
    actions = []  # Keep track of the actions at each step

    while not game_over:
        if turn == 0:
            col = q_learning_player(board, Q, epsilon)
            if col is not None:
                state = get_board_state(board)
                states.append(state)
                actions.append(col)
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, 1)
                if winning_move(board, 1):
                    game_over = True
                    label = myfont.render("Player 1 wins!!", 1, RED)
                    screen.blit(label, (40, 10))
        else:
            col = rule_based_player(board)
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

    # Update the Q-values based on the final reward
    if winning_move(board, 1):
        reward = 1
    elif winning_move(board, 2):
        reward = -1
    else:
        reward = 0
    update_q_table(Q, states, actions, reward)

    # Clear the lists of states and actions for the next episode
    states.clear()
    actions.clear()

    # Return the winner (1 for player 1, 2 for player 2, 0 for a draw)
    if winning_move(board, 1):
        return 1
    elif winning_move(board, 2):
        return 2
    else:
        return 0


def update_q_table(Q, states, actions, reward):
    for state, action in zip(reversed(states), reversed(actions)):
        if state not in Q:
            Q[state] = np.zeros(COLUMN_COUNT)
        next_state = states[-1] if len(states) > 1 else state
        Q[state][action] = Q[state][action] + ALPHA * (reward + GAMMA * np.max(Q[next_state]) - Q[state][action])
        reward = 0

def train_q_learning(episodes, epsilon):
    Q = {}
    for episode in range(episodes):
        play_game(Q, epsilon, screen)
    return Q

Q = train_q_learning(EPISODES, EPSILON)

def calculate_wins_losses(Q, episodes, epsilon, screen, player2_type='rule_based'):
    wins_losses = {'player1': 0, 'player2': 0, 'draw': 0}
    for _ in range(episodes):
        result = play_game(Q, epsilon, screen, player2_type)
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
    plt.title('C4 AI Progress')
    plt.legend()

    plt.show()

ITERATIONS = 20
wins_losses_history = []
for iteration in range(ITERATIONS):
    wins_losses = calculate_wins_losses(Q, EPISODES, EPSILON, screen, player2_type='rule_based')
    wins_losses_history.append(wins_losses)
    print(f"Iteration {iteration + 1}:")
    print(f"Player 1 wins: {wins_losses['player1']}")
    print(f"Player 2 wins: {wins_losses['player2']}")
    print(f"Draws: {wins_losses['draw']}")
plot_progress_lines(wins_losses_history, ITERATIONS)

board = create_board()
game_over = False

