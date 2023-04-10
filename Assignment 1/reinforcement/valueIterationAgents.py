# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
      * Please read learningAgents.py before reading this.*
      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
            mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()

        # Perform value iteration to compute optimal values and policy
        self.valueIteration()

    def getValue(self, state):
        """
        Return the value of the given state under the
        current value function.
        """
        return self.values[state]

    def getQValue(self, state, action):
        """
        Return the Q-value of the given state-action pair
        under the current value function.
        """
        qValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discount * self.values[nextState])
        return qValue

    def getPolicy(self, state):
        """
        Return the optimal action for the given state
        under the current value function.
        """
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None

        # Find the action with the highest Q-value
        maxQValue = float('-inf')
        bestAction = None
        for action in actions:
            qValue = self.getQValue(state, action)
            if qValue > maxQValue:
                maxQValue = qValue
                bestAction = action

        return bestAction

    def getAction(self, state):
        """
        Return the optimal action for the given state
        under the current value function (which is the
        same as the optimal policy).
        """
        return self.getPolicy(state)

    def valueIteration(self):
        """
        Run value iteration for the specified number of iterations
        or until convergence.
        """
        num_iterations = 0
        for i in range(self.iterations):
            # Initialize the dictionary of new values
            newValues = util.Counter()

            # Compute the new value of each state
            for state in self.mdp.getStates():
                # Find the action with the highest Q-value
                actions = self.mdp.getPossibleActions(state)
                if not actions:
                    continue
                maxQValue = float('-inf')
                for action in actions:
                    qValue = self.getQValue(state, action)
                    if qValue > maxQValue:
                        maxQValue = qValue

                # Update the value of the state
                newValues[state] = maxQValue

            if self.has_converged(self.values, newValues):
                num_iterations = i + 1
                print("Convergence reached after", num_iterations, "iterations!")
                break
            # Update the value function with the new values
            self.values = newValues
            num_iterations+=1

    def has_converged(self, values, new_values):
        """
        Check if the state values have converged.
        """
        max_change = max(abs(new_values[s] - values[s]) for s in self.mdp.getStates())
        if (max_change < 1e-5):
            print("Convergence reached!")
            return max_change < 1e-5
        return
