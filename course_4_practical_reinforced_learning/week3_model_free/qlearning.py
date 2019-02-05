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
        Returns the agent's estimate of V(s) using the current Q-values
    update(state, action, reward, next_state)
        Q value update on the form 
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
    get_best_action(state)
        Returns the best action in a state using current Q-values
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
        Returns Q(state, action)
        
        Parameters
        ----------
        state : object
            The state to get the Q-value from
            Must be a valid key in Q-value
        action : object
            The action to get the Q-value from
            Must be a valid key in the state
        
        Returns
        -------
        float
            The Q-value
        """
        
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """
        Sets the Qvalue for [state, action] to the given value
        
        Parameters
        ----------
        state : object
            The state to set the Q-value
            Must be a valid key in Q-value
        action : object
            The action to set the Q-value from
            Must be a valid key in the state
        value : float
            The Q-value
        """
        
        self._qvalues[state][action] = value

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current Q-values
        using the formula
        
        V(s) = max_over_action Q(state,action) over possible actions.
        
        Notes
        -----
        Q-values can be negative.
        
        Parameters
        ----------
        state : object
            The state to get the value function value from
            Must be a valid key in Q-value
            
        Returns
        -------
        value : float
            The estimated V(state)
        """
        
        possible_actions = self.get_legal_actions(state)

        #If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        values = np.empty(len(possible_actions))
        for i, action in enumerate(possible_actions):
            values[i] = self.get_qvalue(state, action)

        value = values.max()
        
        return value

    def update(self, state, action, reward, next_state):
        """
        Updates the Q-value using the formula
        
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        
        Parameters
        ----------
        state : object
            The state to update the Q-value
            Must be a valid key in Q-value
        action : object
            The action to update the Q-value
            Must be a valid key in the state
        reward : float
            The reward gained taking the action
        next_state : object
            The resulting state after taking the action
        """
        
        gamma = self.discount
        alpha = self.alpha

        new_qvalue = (1 - alpha) * self.get_qvalue(state, action) +\
                     alpha*(reward + gamma*self.get_value(next_state))
        
        self.set_qvalue(state, action, new_qvalue)

    
    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current Q-values).
        
        Parameters
        ----------
        state : object
            The state to find the best action from
            
        Returns
        -------
        best_action : None or object
            The action with the highest Q-value
            None is returned if there is no possible actions
        """
        
        possible_actions = self.get_legal_actions(state)

        #If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # NOTE: The best action is the one with the highest Q-value
        values = np.empty(len(possible_actions))
        for i, action in enumerate(possible_actions):
            values[i] = self.get_qvalue(state, action)
            
        best_action = possible_actions[np.argmax(values)]

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.  
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).
        
        Parameters
        ----------
        state : object
            The state to take an action in
            
        Returns
        -------
        chosen_action : None or object
            The action chosen
            None is returned if there is no possible actions
        
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

        epsilon = self.epsilon
        pick_best_action = np.random.choice([True, False], p=[(1-epsilon), epsilon])
        best_action = self.get_best_action(state)
        
        if pick_best_action:
            chosen_action = best_action
        else:
            # NOTE: We do not remove the best_action from possible actions
            #       possible_actions.remove(best_action)
            chosen_action = np.random.choice(possible_actions)
        
        return chosen_action