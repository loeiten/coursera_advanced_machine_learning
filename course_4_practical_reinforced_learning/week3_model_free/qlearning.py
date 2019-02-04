from collections import defaultdict
import random, math
import numpy as np

class QLearningAgent:
    """
    Q-Learning Agent

    Notes
    -----
    Please avoid using self._qValues directly. 
    There's a special self.get_qvalue/set_qvalue for that.
        
    Attributes
    ----------
    epsilon : float
        The exploration probability
    alpha : float
        The learning rate
    discount : float
        The discount rate
    
    Methods
    -------
    get_legal_actions(state)
        Returns a list of the possible actions of a given state
    get_qvalue(state, action)
        Returns Q(state, action)
    set_qvalue(state, action, value)
        Sets Q(state,action) := value
    get_value(state)
        Returns the agent's estimate of V(s) using the current q-values
    update(state, action, reward, next_state)
        Q value update on the form 
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
    get_best_action(state)
        Returns the best action in a state using current q-values
    get_action(state)
        Returns the chosen action in the current state (includes exploration)
    
    References
    ----------
    Based on
    http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
    """
    
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent constructor
        
        Parameters
        ----------
        alpha : float
            The learning rate
        epsilon : float
            The exploration probability
        discount : float
            The discount rate
        get_legal_actions : function
            Function which takes the state as an input parameter and
            returns a list of the possible actions
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """
        Returns Q(state,action)
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        #If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        <YOUR CODE HERE>

        return value

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        #agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        <YOUR CODE HERE>
        
        self.set_qvalue(state, action, <YOUR_QVALUE>)

    
    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values). 
        """
        possible_actions = self.get_legal_actions(state)

        #If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        <YOUR CODE HERE>

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.  
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.getPolicy).
        
        Note: To pick randomly from a list, use random.choice(list). 
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)
        action = None

        #If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        #agent parameters:
        epsilon = self.epsilon

        <YOUR CODE HERE>
        
        return chosen_action