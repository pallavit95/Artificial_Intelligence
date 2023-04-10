import numpy as np
import random
import util
from learningAgents import ValueEstimationAgent

class PolicyIterationAgent(ValueEstimationAgent):
    def __init__(self, mdp, gamma=0.9, iterations=10):
        self.mdp = mdp
        self.gamma = gamma
        self.policy = util.Counter()  # Initialize the policy with an empty dict
        self.iterations = iterations
        self.values = util.Counter() # Initialize the values with 0 for all states
        self.computePolicy()

    def getQValue(self, state, action):
        qvalue = 0
        # Check if the action is one of the possible actions for the given state
        if action in self.mdp.getPossibleActions(state):
            # Iterate over all possible next states and their corresponding transition probabilities
            for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                # Calculate the Q-value using the Bellman equation
                qvalue += prob * (self.mdp.getReward(state, action, nextState) + self.gamma * self.values[nextState])
        return qvalue

    def getValue(self, state):
        return self.values[state]
    

    def computePolicy(self):
        # Initialize the policy with an empty dictionary
        self.policy = util.Counter()
        # Initialize the values with 0 for all states
        self.values = util.Counter() #{state: 0 for state in self.mdp.getStates()}
        num_iterations = 0
        for i in range(self.iterations):
            # Policy Evaluation
            # Iterate until the values converge
            for j in range(self.iterations):
                # Keep track of the largest change in value for any state during this iteration
                delta = 0
                # For each state in the MDP, update its value using the current policy
                for state in self.mdp.getStates():
                    # Calculate the new value using the Bellman equation
                    newValue = self.getQValue(state, self.policy[state])
                    # Update the largest change in value seen so far
                    delta = max(delta, abs(newValue - self.values[state]))
                    # Update the value for the state
                    self.values[state] = newValue

                # If the largest change in value is smaller than a threshold, we have converged
                if delta < 1e-8:
                    num_iterations = i + 1
                    print("Convergence reached after", num_iterations, "iterations!")
                    break
                
                num_iterations+=1
            # Policy Improvement
            # Initialize a flag to keep track of whether the policy has changed
            policyChanged = False
            # Iterate over all states in the MDP
            for state in self.mdp.getStates():
                # Get the possible actions for the current state
                possibleActions = self.mdp.getPossibleActions(state)
                # If there are no possible actions, skip this state
                if not possibleActions:
                    continue
                # Calculate the Q-values for each action in the current state
                qvalues = [self.getQValue(state, action) for action in possibleActions]
                # Choose the action that maximizes the Q-value
                bestAction = possibleActions[np.argmax(qvalues)]
                # If the action that maximizes the Q-value is different from the current policy, update the policy
                if bestAction != self.policy.get(state, None):
                    self.policy[state] = bestAction
                    policyChanged = True

            # If the policy hasn't changed, we have converged
            if not policyChanged:
                break
        # Return the best policy found
        return self.policy

    def getAction(self, state):
        # Check if the state is in the policy
        if state in self.policy:
            # Return the action associated with the state in the policy
            return self.policy[state]
        else:
            # If the state is not in the policy, return a random action
            return np.random.choice(self.mdp.getPossibleActions(state))
        
    def getPolicy(self, state):
        if state == 'TERMINAL_STATE':
            return None  # or an empty list
        if state not in self.policy:
            return None  # or an empty list
        return self.policy[state]

