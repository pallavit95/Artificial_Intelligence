import pygame
import math
import random
import matplotlib.pyplot as plt
import numpy as np

pygame.init()

player1_moves = np.zeros((3, 3), dtype=int)
player2_moves = np.zeros((3, 3), dtype=int)

# Screen
WIDTH = 300
ROWS = 3
win = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("TicTacToe")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Images
X_IMAGE = pygame.transform.scale(pygame.image.load("images/x.svg"), (80, 80))
O_IMAGE = pygame.transform.scale(pygame.image.load("images/o.svg"), (80, 80))

# Fonts
END_FONT = pygame.font.SysFont('arial', 40)


def draw_grid():
    gap = WIDTH // ROWS

    # Starting points
    x = 0
    y = 0

    for i in range(ROWS):
        x = i * gap

        pygame.draw.line(win, GRAY, (x, 0), (x, WIDTH), 3)
        pygame.draw.line(win, GRAY, (0, x), (WIDTH, x), 3)


def initialize_grid():
    dis_to_cen = WIDTH // ROWS // 2

    # Initializing the array
    game_array = [[None, None, None], [None, None, None], [None, None, None]]

    for i in range(len(game_array)):
        for j in range(len(game_array[i])):
            x = dis_to_cen * (2 * j + 1)
            y = dis_to_cen * (2 * i + 1)

            # Adding centre coordinates
            game_array[i][j] = (x, y, "", True)

    return game_array


# Checking if someone has won
def has_won(game_array, display=True):

    # Checking columns
    for col in range(len(game_array)):
        if (game_array[0][col][2] == game_array[1][col][2] == game_array[2][col][2]) and game_array[0][col][2] != "":
            if display:
                display_message(game_array[0][col][2].upper() + " has won!")
            return True
        
    # Checking rows
    for row in range(len(game_array)):
        if (game_array[row][0][2] == game_array[row][1][2] == game_array[row][2][2]) and game_array[row][0][2] != "":
            if display:
                display_message(game_array[row][0][2].upper() + " has won!")
            return True

    # Checking main diagonal
    if (game_array[0][0][2] == game_array[1][1][2] == game_array[2][2][2]) and game_array[0][0][2] != "":
        if display:
            display_message(game_array[0][0][2].upper() + " has won!")
        return True

    # Checking reverse diagonal
    if (game_array[0][2][2] == game_array[1][1][2] == game_array[2][0][2]) and game_array[0][2][2] != "":
        if display:
            display_message(game_array[0][2][2].upper() + " has won!")
        return True

    return False


def has_drawn(game_array, display=True):
    for i in range(len(game_array)):
        for j in range(len(game_array[i])):
            if game_array[i][j][2] == "":
                return False

    if display:
        display_message("It's a draw!")
    return True


def display_message(content):
    pygame.time.delay(500)
    win.fill(WHITE)
    end_text = END_FONT.render(content, 1, BLACK)
    win.blit(end_text, ((WIDTH - end_text.get_width()) // 2, (WIDTH - end_text.get_height()) // 2))
    pygame.display.update()
    pygame.time.delay(3000)


def render():
    win.fill(WHITE)
    draw_grid()

    # Drawing X's and O's
    for image in images:
        x, y, IMAGE = image
        win.blit(IMAGE, (x - IMAGE.get_width() // 2, y - IMAGE.get_height() // 2))

    pygame.display.update()

def minimax(game_array, depth, alpha, beta, is_maximizing):
    global eval_count

    if has_won(game_array,display=False):
        if is_maximizing:
            return -10 + depth, depth
        else:
            return 10 + depth, depth
    elif has_drawn(game_array,display=False):
        return 0, depth
    
    if is_maximizing:
        best_score = -math.inf
        # alpha = -math.inf  # Initialize alpha to negative infinity
        for i in range(len(game_array)):
            for j in range(len(game_array[i])):
                if game_array[i][j][2] == "":
                    game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], 'o', False)
                    eval_count += 1
                    score, d = minimax(game_array, depth+1, alpha, beta, False)
                    game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], '', True)
                    best_score = max(best_score, score)
                    if score >= best_score:
                        best_depth = d
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break
        return best_score, best_depth
    else:
        best_score = math.inf
        # beta = math.inf  # Initialize beta to positive infinity
        for i in range(len(game_array)):
            for j in range(len(game_array[i])):
                if game_array[i][j][2] == "":
                    game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], 'x', False)
                    eval_count += 1
                    score, d = minimax(game_array, depth+1, alpha, beta, True)
                    game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], '', True)
                    best_score = min(best_score, score)
                    if score <= best_score:
                        best_depth = d
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break
        return best_score, best_depth

def get_available_moves(game_array):
    moves = []

    for i in range(len(game_array)):
        for j in range(len(game_array[i])):
            if game_array[i][j][2] == "":
                moves.append((i, j))

    return moves

def rule_based_move(game_array, player):
    global player1_moves, player2_moves
    opponent = 'x' if player == 'o' else 'o'
    available_moves = get_available_moves(game_array)

    if not available_moves:
        return game_array
    
    # Rule 1: If there is a winning move, play it.
    for move in available_moves:
        i, j = move
        game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], player, False)

        if has_won(game_array, display=False):
            game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], '', True)
            images.append((game_array[i][j][0], game_array[i][j][1], O_IMAGE))
            player2_moves[i][j] += 1
            return game_array
        
        game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], '', True)

    # Rule 2: If the opponent has a winning move, block it.
    for move in available_moves:
        i, j = move
        game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], opponent, False)

        if has_won(game_array, display=False):
            game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], player, False)
            images.append((game_array[i][j][0], game_array[i][j][1], O_IMAGE))
            player2_moves[i][j] += 1
            return game_array
        
        game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], '', True)

    # Rule 3: Play in the center, if available.
    # if game_array[1][1][2] == "":
    #     game_array[1][1] = (game_array[1][1][0], game_array[1][1][1], player, False)
    #     images.append((game_array[1][1][0], game_array[1][1][1], O_IMAGE))
    #     player2_moves[1][1] += 1
    #     return game_array

    # Rule 4: Choose a random available cell.
    if available_moves:
        random_move = random.choice(available_moves)
        i, j = random_move
        images.append((game_array[i][j][0], game_array[i][j][1], O_IMAGE))
        game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], player, False)

    player2_moves[i][j] += 1   
    return game_array

def minimax_move(game_array, player):
    global x_turn, o_turn, images, eval_count, player1_moves, player2_moves, best_depth

    eval_count = 0
    # Choose the best move for the AI player
    best_score = -math.inf if player == 'o' else math.inf
    best_move = None

    for i in range(len(game_array)):
        for j in range(len(game_array[i])):
            if game_array[i][j][2] == "":
                game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], player, False)
                score, d = minimax(game_array, 0, -math.inf, math.inf, player == 'x')
                game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], '', True)
                if (player == 'o' and score > best_score) or (player == 'x' and score < best_score):
                    best_score = score
                    best_move = (i, j)
                    best_depth = d

    # Update the game state with the AI player's move
    if best_move is not None:
        i, j = best_move
        images.append((game_array[i][j][0], game_array[i][j][1], X_IMAGE if player == 'x' else O_IMAGE))
        game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], player, False)

    if player == 'x':
        player1_moves[i][j] += 1
    else:
        player2_moves[i][j] += 1

    # print("Best depth achieved: ", best_depth)

    return game_array


def play_game(screen, player2_type='rule_based'):
    global x_turn, o_turn, images, draw, best_depth

    images = []
    draw = False
    best_depth = -math.inf

    run = True

    x_turn = True
    o_turn = False

    game_array = initialize_grid()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        if x_turn:
            game_array = minimax_move(game_array, 'x')
            x_turn = False
            o_turn = True

        if o_turn:
            if player2_type == 'minimax':
                print("playing minimax vs minimax!")
                game_array = minimax_move(game_array, 'o')
            else:
                game_array = rule_based_move(game_array, 'o')
            x_turn = True
            o_turn = False

        render()

        if has_won(game_array, display=False):
            if x_turn:
                return 2  # 'o' player wins
            else:
                return 1  # 'x' player wins
        elif has_drawn(game_array, display=False):
            run = False
            return 0  # Draw

    pygame.time.delay(3000)

def plot_progress_lines(wins_losses_history, iterations):
    labels = [f"Iteration {i+1}" for i in range(iterations)]

    player1_wins = [data['player1'] for data in wins_losses_history]
    player2_wins = [data['player2'] for data in wins_losses_history]
    draws = [data['draw'] for data in wins_losses_history]

    plt.plot(labels, player1_wins, label='Player O Wins', marker='o')
    plt.plot(labels, player2_wins, label='Player X Wins', marker='o')
    plt.plot(labels, draws, label='Draws', marker='o')

    plt.ylabel('Number of Wins and Draws')
    plt.title('Tic-Tac-Toe AI Progress')
    plt.legend()

    plt.show()

def plot_progress(wins_losses_history, iterations):
    labels = [f"Iteration {i+1}" for i in range(iterations)]

    player1_wins = [data['player1'] for data in wins_losses_history]
    player2_wins = [data['player2'] for data in wins_losses_history]
    draws = [data['draw'] for data in wins_losses_history]

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, player1_wins, width, label='Player 1 Wins')
    rects2 = ax.bar(x, player2_wins, width, label='Player 2 Wins')
    rects3 = ax.bar(x + width, draws, width, label='Draws')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of Wins and Draws')
    ax.set_title('Tic-Tac-Toe AI Progress')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    plt.show()

def plot_heatmaps(player1_moves, player2_moves):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    heatmap1 = ax[0].imshow(player1_moves, cmap='viridis', interpolation='nearest')
    ax[0].set_title("Player 1 (X) Moves")
    plt.colorbar(heatmap1, ax=ax[0], fraction=0.046, pad=0.04)

    heatmap2 = ax[1].imshow(player2_moves, cmap='viridis', interpolation='nearest')
    ax[1].set_title("Player 2 (O) Moves")
    plt.colorbar(heatmap2, ax=ax[1], fraction=0.046, pad=0.04)

    plt.show()

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

EPISODES = 10
ITERATIONS = 5
wins_losses_history = []

for iteration in range(ITERATIONS):
    wins_losses = calculate_wins_losses(EPISODES, win, player2_type='rule_based')
    wins_losses_history.append(wins_losses)
    print(f"Iteration {iteration + 1}:")
    print(f"Player O wins: {wins_losses['player1']}")
    print(f"Player X wins: {wins_losses['player2']}")
    print(f"Draws: {wins_losses['draw']}")

plot_progress_lines(wins_losses_history, ITERATIONS)
plot_heatmaps(player1_moves, player2_moves)