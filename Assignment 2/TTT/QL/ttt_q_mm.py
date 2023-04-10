import pygame
import math
import numpy as np
import random
import matplotlib.pyplot as plt

player1_moves = np.zeros((3, 3), dtype=int)
player2_moves = np.zeros((3, 3), dtype=int)
pygame.init()

# Q-learning parameters
alpha = 0.5
gamma = 0.5
epsilon = 0.5
one = 0
two = 0
three = 0
max_depth = 1
# Q-table: a dictionary with keys as the state and values as a list of Q-values for each action
Q_table = {}

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

def render():
    win.fill(WHITE)
    draw_grid()

    # Drawing X's and O's
    for image in images:
        x, y, IMAGE = image
        win.blit(IMAGE, (x - IMAGE.get_width() // 2, y - IMAGE.get_height() // 2))

    pygame.display.update()

def display_message(content):
    pygame.time.delay(1)
    win.fill(WHITE)
    end_text = END_FONT.render(content, 1, BLACK)
    win.blit(end_text, ((WIDTH - end_text.get_width()) // 2, (WIDTH - end_text.get_height()) // 2))
    pygame.display.update()
    pygame.time.delay(1)

# Checking if someone has won
def has_won(game_array, player):
    # Checking rows
    for row in range(len(game_array)):
        if (game_array[row][0][2] == game_array[row][1][2] == game_array[row][2][2]) and game_array[row][0][2] == player:
            return True

    # Checking columns
    for col in range(len(game_array)):
        if (game_array[0][col][2] == game_array[1][col][2] == game_array[2][col][2]) and game_array[0][col][2] == player:
            return True

    # Checking main diagonal
    if (game_array[0][0][2] == game_array[1][1][2] == game_array[2][2][2]) and game_array[0][0][2] == player:
        return True

    # Checking reverse diagonal
    if (game_array[0][2][2] == game_array[1][1][2] == game_array[2][0][2]) and game_array[0][2][2] == player:
        return True

    return False

def has_drawn(game_array):
    for i in range(len(game_array)):
        for j in range(len(game_array[i])):
            if game_array[i][j][2] == "":
                return False
    return True

def minimax_alpha_beta(game_array, depth, alpha, beta, maximizing_player):
    if depth == max_depth or has_won(game_array, 'x') or has_won(game_array, 'o') or has_drawn(game_array):
        return evaluate_game_state(game_array)

    if maximizing_player:
        max_eval = -float("inf")
        for move in get_available_moves(game_array):
            i, j = move
            game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], 'o', False)
            eval = minimax_alpha_beta(game_array, depth + 1, alpha, beta, False)
            game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], "", True)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for move in get_available_moves(game_array):
            i, j = move
            game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], 'x', False)
            eval = minimax_alpha_beta(game_array, depth + 1, alpha, beta, True)
            game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], "", True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def evaluate_game_state(game_array):
    if has_won(game_array, 'o'):
        return 1
    elif has_won(game_array, 'x'):
        return -1
    else:
        return 0

def minimax_opponent(game_array):
    best_move = None
    best_value = -float("inf")
    alpha = -float("inf")
    beta = float("inf")

    for move in get_available_moves(game_array):
        i, j = move
        game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], 'o', False)
        move_value = minimax_alpha_beta(game_array, 0, alpha, beta, False)
        game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], "", True)

        if move_value > best_value:
            best_value = move_value
            best_move = move

    return best_move


def rule_based_opponent(game_array):
    global x_r_m
    available_moves = get_available_moves(game_array)

    # Step 1: Check for opponent's winning move and block it
    for move in available_moves:
        if winning_move(game_array, move, 'o'):
            return move

    # Step 2: Check for winning move
    for move in available_moves:
        if winning_move(game_array, move, 'x'):
            return move

    # Step 3: Make a random move
    if available_moves:
        x_r_m+=1
        return random.choice(available_moves)
    else:
        return (None, None)
    
def get_available_moves(game_array):
    available_moves = []
    for i in range(3):
        for j in range(3):
            if game_array[i][j][2] == "":
                available_moves.append((i, j))
    return available_moves

def winning_move(game_array, move, player):
    i, j = move
    game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], player, False)
    if has_won(game_array, player):
        # Revert the game_array back to its original state
        game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], "", True)
        return move

    # Revert the game_array back to its original state
    game_array[i][j] = (game_array[i][j][0], game_array[i][j][1], "", True)
    return None

def click(game_array):
    global x_turn, o_turn, images, x_wins, o_wins, draw

    if x_turn:  # If it's X's turn
        prev_state = state_to_str(game_array)
        action = choose_action(game_array, epsilon)
        i, j = action
        x, y, char, can_play = game_array[i][j]
        images.append((x, y, X_IMAGE))
        x_turn = False
        o_turn = True
        game_array[i][j] = (x, y, 'x', False)
        player1_moves[i][j] += 1
        if has_won(game_array, 'x'):
            reward = 1
            update_q_table(prev_state, action, reward, state_to_str(game_array))
            display_message("X has won!")
            return
        elif has_drawn(game_array):
            reward = 1
            update_q_table(prev_state, action, reward, state_to_str(game_array))
            display_message("It's a draw!")
            draw = True
            return
        else:
            reward = -1
            next_state = state_to_str(game_array)
            update_q_table(prev_state, action, reward, next_state)  
    
    elif o_turn:  # If it's O's turn
        prev_state = state_to_str(game_array)
        # move = rule_based_opponent(game_array)
        move = minimax_opponent(game_array)
        # if move != (None, None):
        i, j = move
        x, y, char, can_play = game_array[i][j]
        images.append((x, y, O_IMAGE))
        x_turn = True
        o_turn = False
        game_array[i][j] = (x, y, 'o', False)
        player2_moves[i][j] += 1
        if has_won(game_array, 'o'):
            reward = -1
            update_q_table(prev_state, move, reward, state_to_str(game_array))
            display_message("O has won!")
            return
        elif has_drawn(game_array):
            display_message("It's a draw!")
            reward = 0
            update_q_table(prev_state, move, reward, state_to_str(game_array))
            display_message("It's a draw!")
            draw = True
            return


def state_to_str(game_array):
    state = ''
    for i in range(ROWS):
        for j in range(ROWS):
            state += game_array[i][j][2] if game_array[i][j][2] != '' else ' '
    return state


def str_to_state(state_str):
    dis_to_cen = WIDTH // ROWS // 2
    state = []
    for i in range(ROWS):
        row = []
        for j in range(ROWS):
            x = dis_to_cen * (2 * j + 1)
            y = dis_to_cen * (2 * i + 1)
            row.append((x, y, state_str[i * ROWS + j], False))
        state.append(row)
    return state


def get_valid_actions(game_array):
    valid_actions = []
    for i in range(ROWS):
        for j in range(ROWS):
            if game_array[i][j][2] == '':
                valid_actions.append((i, j))
    return valid_actions

def choose_action(game_array, epsilon):
    global one, two, three
    state = state_to_str(game_array)
    if state not in Q_table:
        Q_table[state] = [-1 for _ in range(ROWS * ROWS)]

    valid_actions = get_valid_actions(game_array)
    if not valid_actions:
        # No valid actions available, return a random action
        one+=1
        return random.choice([(i, j) for i in range(ROWS) for j in range(ROWS)])

    if np.random.uniform(0, 1) < epsilon:
        # Choose a random action with probability epsilon
        two+=1
        return valid_actions[np.random.randint(len(valid_actions))]
    else:
        # Choose the action with the highest Q-value
        three+=1
        q_values = Q_table[state]
        max_q = -float("inf")
        best_action = None
        for action in valid_actions:
            i, j = action
            action_q_value = q_values[i * ROWS + j]
            if action_q_value > max_q:
                max_q = action_q_value
                best_action = action
        return best_action
    
def update_q_table(prev_state, prev_action, reward, next_state):
    if prev_state not in Q_table:
        Q_table[prev_state] = [-1 for _ in range(ROWS * ROWS)]

    if next_state not in Q_table:
        Q_table[next_state] = [-1 for _ in range(ROWS * ROWS)]

    prev_q = Q_table[prev_state][prev_action[0] * ROWS + prev_action[1]]
    max_next_q = max(Q_table[next_state])

    Q_table[prev_state][prev_action[0] * ROWS + prev_action[1]] = prev_q + alpha * (reward + gamma * max_next_q - prev_q)

def train_q_learning(num_iterations, games_per_iteration, q_convergence_threshold=1e-6):
    global x_wins, o_wins, game_array, xo_draw, x_r_m, one, two, three, player1_moves, player2_moves, max_depth, gamma, epsilon, alpha
    total_x_wins = 0
    total_o_wins = 0
    total_draws = 0

    x_wins_list = []
    o_wins_list = []
    draws_list = []

    print(f"\nPlaying {games_per_iteration} games, for {num_iterations} iterations.")

    for iteration in range(num_iterations):
        o_wins = 0
        x_wins = 0
        xo_draw = 0
        x_r_m=0
        prev_q_table = Q_table.copy()
        for _ in range(games_per_iteration):
            if(iteration>=10 and iteration<=30):
                max_depth=2
                epsilon=0.1
                gamma=0.9
                alpha=0.8
            if(iteration>=40):
                max_depth=3
                epsilon=0.001
                gamma=0.9
                alpha=0.9
            result = main()
            if result is None:
                xo_draw +=1
            elif result:
                o_wins += 1
            else:
                x_wins += 1
            pygame.time.delay(1)
            game_array = initialize_grid()
        total_o_wins += o_wins
        total_x_wins += x_wins
        total_draws += xo_draw

        # Save the number of wins and draws for plotting
        x_wins_list.append(x_wins)
        o_wins_list.append(o_wins)
        draws_list.append(xo_draw)

        # Check for convergence
        if iteration > 0:
            q_values = np.array(list(Q_table.values()))
            prev_q_values = np.array(list(prev_q_table.values()))
            # pad prev_q_values with zeros to match shape of q_values
            if prev_q_values.shape[0] < q_values.shape[0]:
                pad_width = ((0, q_values.shape[0]-prev_q_values.shape[0]), (0, 0))
                prev_q_values = np.pad(prev_q_values, pad_width, mode='constant', constant_values=-1)
            else:
                prev_q_values = np.array(list(prev_q_table.values()))
            if q_values.size == 0 or prev_q_values.size == 0:
                diff = float("inf")
            else:
                diff = np.abs(q_values - prev_q_values).max()
                if diff < q_convergence_threshold:
                    print(f"Converged after iteration {iteration + 1}")
                
        if iteration > 0:
            print(f"Iteration {iteration + 1}: Player O won {o_wins} games, Player X won {x_wins} games, diff={diff}")
            print(f"one = {one}, two = {two}, three = {three}")
        else:
            print(f"Iteration {iteration + 1}: Player O won {o_wins} games, Player X won {x_wins} games")
            print(f"one = {one}, two = {two}, three = {three}")
        print(f"total random moves by x {x_r_m}")

    print(f"\nTotal: Player O won {total_o_wins} games, Player X won {total_x_wins} games.")

    # Plot the progress of the training
    plot_progress(x_wins_list, o_wins_list, draws_list, num_iterations, games_per_iteration)
    plot_heatmaps(player1_moves, player2_moves)

def plot_progress(x_wins, o_wins, draws, num_iterations, games_per_iteration):
    x = list(range(1, num_iterations + 1))
    plt.plot(x, x_wins, label='X Wins')
    plt.plot(x, o_wins, label='O Wins')
    plt.plot(x, draws, label='Draws')

    plt.xlabel('Iteration')
    plt.ylabel('Number of Games')
    plt.title(f'Progress of Training over {num_iterations} Iterations ({games_per_iteration} Games per Iteration)')
    plt.legend()
    plt.show()

def main():
    global x_turn, o_turn, images, draw, x_wins, o_wins, xo_draw, x_r_m, one, two, three

    images = []
    draw = False

    run = True

    x_turn = True
    o_turn = False

    game_array = initialize_grid()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        click(game_array)

        if has_won(game_array, 'o'):
            run = False
            return True
        elif has_drawn(game_array):
            run = False
            return None
        elif has_won(game_array, 'x'):
            run = False
            return False

        render()


def plot_heatmaps(player1_moves, player2_moves):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    heatmap1 = ax[0].imshow(player1_moves, cmap='viridis', interpolation='nearest')
    ax[0].set_title("Player 1 (X) Moves")
    plt.colorbar(heatmap1, ax=ax[0], fraction=0.046, pad=0.04)

    heatmap2 = ax[1].imshow(player2_moves, cmap='viridis', interpolation='nearest')
    ax[1].set_title("Player 2 (O) Moves")
    plt.colorbar(heatmap2, ax=ax[1], fraction=0.046, pad=0.04)

    plt.show()

if __name__ == '__main__':
    num_iterations = 50
    games_per_iteration = 200
    train_q_learning(num_iterations, games_per_iteration)