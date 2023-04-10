# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    import time
    import resource
    # Initialize the visited set, stack and path list
    visited = set()
    stack = util.Stack()
    path = []

    # Push the start state onto the stack
    start_state = problem.getStartState()
    stack.push((start_state, path))

    # Initialize the metrics
    start_time = time.time()
    num_nodes_expanded = 0
    max_memory_used = 0
    num_dead_ends = 0

    # Start the search
    while not stack.isEmpty():
        # Pop the next state and its path from the stack
        state, path = stack.pop()

        # Check if the state has already been visited
        if state in visited:
            continue

        # Mark the state as visited
        visited.add(state)

        # Check if the state is the goal state
        if problem.isGoalState(state):
            # path.append(path)
            end_time = time.time()
            path_length = len(path)
            branching_factor = num_nodes_expanded / (num_nodes_expanded - 1)
            print(f"Path found: {path}")
            print(f"Time taken: {end_time - start_time} seconds")
            print(f"Number of nodes expanded: {num_nodes_expanded}")
            print("Maximum memory used:", max_memory_used, "bytes")
            print(f"Path length: {path_length}")
            print(f"Branching factor: {branching_factor}")
            print(f"Number of dead-ends encountered: {num_dead_ends}")
            return path

        # Expand the state and add its successors to the stack
        successors = problem.getSuccessors(state)
        num_nodes_expanded += 1
        # Check for dead-ends
        if len(successors) == 1 and state != start_state:
            num_dead_ends += 1

        for successor in successors:
            next_state, action, cost = successor
            next_path = path + [action]
            stack.push((next_state, next_path))

        # Update the maximum memory used
        memory_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        max_memory_used = max(max_memory_used, memory_used)

    print("No path found!")
    return []
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    "*** YOUR CODE HERE ***"
    import time
    import resource
    start_time = time.time()
    visited = set()
    start_node = (problem.getStartState(), [], 0)
    frontier = util.Queue()
    frontier.push(start_node)
    max_memory_used = 0
    num_dead_ends = 0  # track number of dead-ends encountered
    num_nodes_generated = 1  # include start node
    while not frontier.isEmpty():
        current_node = frontier.pop()
        current_state, current_path, current_cost = current_node
        if problem.isGoalState(current_state):
            end_time = time.time()
            print("Path found:", current_path)
            print("Time taken:", end_time - start_time, "seconds")
            print("Number of nodes expanded:", len(visited))
            print("Maximum memory used:", max_memory_used, "bytes")
            print("Path length:", len(current_path))
            print("Branching factor:", num_nodes_generated / len(visited))
            print("Number of dead-ends encountered:", num_dead_ends)
            return current_path
        if current_state not in visited:
            visited.add(current_state)
            for child_state, child_action, child_cost in problem.getSuccessors(current_state):
                child_node = (child_state, current_path + [child_action], current_cost + child_cost)
                frontier.push(child_node)
                num_nodes_generated += 1
            if len(problem.getSuccessors(current_state)) == 0:  # dead-end encountered
                num_dead_ends += 1
        memory_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        max_memory_used = max(max_memory_used, memory_used)
    return []
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    import time
    import resource
    start_time = time.time()
    frontier = util.PriorityQueue()
    frontier.push((problem.getStartState(), [], 0), 0)
    explored = set()
    max_memory_used = 0
    num_nodes_expanded = 0
    num_dead_ends = 0
    while not frontier.isEmpty():
        current, actions, cost_so_far = frontier.pop()
        if problem.isGoalState(current):
            elapsed_time = time.time() - start_time
            print("Path found!")
            print("Time taken: {:.4f} seconds".format(elapsed_time))
            print("Nodes expanded: {}".format(num_nodes_expanded))
            print("Max memory used: {:.2f} MB".format(max_memory_used / 1024 / 1024))
            print("Path length: {}".format(len(actions)))
            print("Branching factor: {:.2f}".format(num_nodes_expanded / len(actions)))
            print("Dead ends encountered: {}".format(num_dead_ends))
            return actions
        if current not in explored:
            explored.add(current)
            for next_state, action, step_cost in problem.getSuccessors(current):
                new_cost = cost_so_far + step_cost
                num_nodes_expanded += 1
                if next_state not in explored:
                    frontier.push((next_state, actions + [action], new_cost), new_cost + heuristic(next_state, problem))
                else:
                    num_dead_ends += 1
        memory_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        max_memory_used = max(max_memory_used, memory_used)
    
    print("No path found!")
    print("Time taken: {:.4f} seconds".format(time.time() - start_time))
    print("Nodes expanded: {}".format(num_nodes_expanded))
    print("Max memory used: {:.2f} MB".format(max_memory_used / 1024 / 1024))
    print("Path length: 0")
    print("Branching factor: 0")
    print("Dead ends encountered: {}".format(num_dead_ends))
    return []
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
