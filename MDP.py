import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import unittest

class MDP:
    def __init__(self, nrow, ncol, terminal, default_reward, discount):
        '''Create/store the following attributes:
        states -- list of all the states (x,y) tuples
        terminal_states -- is a dictionary with terminal state keys, and rewards as values
        default_reward -- is the reward for being in any non-terminal state
        df -- discount factor
        ... and anything else you decide will be useful!
        '''
        
        
        states = []
        for i in range(1, ncol+1):
            for j in range(1, nrow+1):
                states.append((i,j))
        #print(states)
        self.states = states
        self.terminal_states = terminal
        self.default_reward = default_reward
        self.df = discount
        
        

    def actions(self, state):
        '''Return a list of available actions from the given state.
        [None] are the actions available from a terminal state.
        '''
        
        # your code goes here...
        if state in self.terminal_states: #checking if state is a terminial state
            return {}
        
        actions = {}
        i = state[0]
        j = state[1]
        #move North by adding to j:
        if (i,j+1) in self.states:#check to make sure the move is within the possible moves
            actions['N'] = (i,j+1)
        #move South by subtracting from j:
        if (i,j-1) in self.states:
            actions['S'] = (i,j-1)
        #move West by subtracting from i:
        if (i-1,j) in self.states:
            actions['W'] = (i-1,j)
        #move East by adding to i:
        if (i+1,j) in self.states:
            actions['E'] = (i+1,j)
        return actions
        
    def reward(self, state):
        '''Return the reward for being in the given state'''
        
        # your code goes here...
        if state not in self.terminal_states:
            return self.default_reward
        else:
            return self.terminal_states[state]
        
        
    def result(self, state, action):
        '''Return the resulting state (as a tuple) from doing the given
        action in the given state, without uncertainty. Uncertainty
        is incorporated into the transition method.
        state -- a tuple representing the current state
        action -- one of N, S, E or W, as a string
        '''
        i = state[0]
        j = state[1]
        # your code goes here...
        if action == 'N':
            return (i,j+1)
        elif action == 'S':
            return (i,j-1)
        elif action == 'E':
            return (i+1,j)
        elif action == 'W':
            return (i-1,j)
        else:
            return None
        
                
    def transition(self, state, action):
        '''Return the probabilities and subsequent states associated
        with taking the given action from the given state. Can be done
        however you want, so that it works with your value/policy iteration.
        '''
        
        # your code goes here...
        
        # Use the following transition model for this decision process, if you are trying to move from state  𝑠  to state  𝑠′ :
        # you successfully move from  𝑠  to  𝑠′  with probability 0.6
        # the remaining 0.4 probability is spread equally likely across state  𝑠  and all adjacent (N/S/E/W) states except for  𝑠′
        # Note that this does not necessarily mean that all adjacent states have 0.1, because some states do not have 4 adjacent states.
        action_tup = self.result(state, action)#will turn 'N' into (i,j) cooridinate
        possible_actions = self.actions(state) 
        #print("possible_actions:", possible_actions)
        #possible_actions.append(state) #adding staying still at state as an option
        numActs = len(possible_actions) - 1 #numActs = len(possible_actions) - action
        pMessUp = 0.4/numActs
        results = {}
        for thing in possible_actions.values():
            #print("thing", thing)
            if thing != action_tup:
                results[thing] = pMessUp
            else:
                results[thing] = 0.6
        return results
    
def value_iteration(mdp, tol=1e-3):
    
    
    #initialize utility for all states
    #to zero? how do i do that? do I actually change mdp in some way? do I make a new dictionary?
    #for now making a utility dictionary where the current utility is set to zer0 for each state
    utility = {}
    cont = True
    for key in mdp.states:
        utility[key] = 0
    #print(mdp.states)
    uPrime = utility.copy()
    while cont == True:
        utility = uPrime #make a copy of current utility to be modified
        #uPrime = utility.copy()
        delta = 0 #initialize maximum change to 0
        for s in mdp.states: #for each state s
            if s not in mdp.terminal_states:
                #for each available action, what next states are possible, and their probabilities? 
                #Uprime = R(s) + discount * max (SUM(p(s'|s,a)*U[s'])
                actions = mdp.actions(s)
                #print("actions:", actions)
                options = []
                for a in actions:
                    transitionModel = mdp.transition(s,a)#holds coordinates and their probabilities with respect to s and a
                    #rint(transitionModel)
                    summation = 0
                    for key in transitionModel.keys():
                        #print(key)
                        #print(transitionModel[key])
                        #print(utility[key])
                        x = transitionModel[key]*utility[key] #p(s'|s,a)*U[s']
                        summation += x
                    #now summation = Sum(p(s'|s,a)*U[s']) for each possible s' given s and a  
                    options.append(summation)
                #now we have an option for each possible a and we need the max one
                maxOption = max(options)
                uPrime[s] = mdp.reward(s) + (mdp.df*maxOption)
                if abs(uPrime[s]-utility[s])>delta:
                    delta = abs(uPrime[s]-utility[s])
            else:
                #means in terminal state, what do?
                utility[s] = mdp.terminal_states[s]
        if delta <= tol:
            cont = False
    return utility
        
    
    
    
    

def find_policy(mdp, utility):
    
    # your code goes here...
    #initializing utility of all states... to zer0?
    
    #initialize a policy for each state, being a random action
    #like is policy a singular value or is it a dictionary {state: policy for that state}???
    #looking at pseudocode in book seems like theres a policy for every state, initially random
    utility = {}
    policies = {}
    cont = True
    initPol = np.random.choice(['N','W','E','S'])
    for key in mdp.states:
        utility[key] = 0
        policies[key] = initPol
    def policy_eval():
        newUtility = {}
        for s in mdp.states:
            if s not in mdp.terminal_states:
                polAct = policies[s]
                results = mdp.transition(s,polAct)
                #newUtility[s] = mdp.default_reward + mdp.df*()
                total = 0
                for thing in results.keys():
                    total += results[thing]*utility[thing]
                newUtility[s] = mdp.default_reward + mdp.df*total
            else:
                newUtility[s] = mdp.terminal_states[s]
        return newUtility
                
    while cont == True:
        utility = policy_eval()
        newPolicies = policies.copy()
        #Ok, with our new utilities based on the current policy, we now want to see if we should update policy
        for s in mdp.states:
            if s not in mdp.terminal_states:
                possible_actions = mdp.actions(s)
                possibilities = {}
                for a in possible_actions.keys():
                    total = 0
                    results = mdp.transition(s,a)
                    for thing in results.keys():
                        total += results[thing]*utility[thing]
                    possibilities[total] = a
                maxy = max(possibilities.keys())
                bestPol = possibilities[maxy]
                newPolicies[s] = bestPol
        if policies == newPolicies:#no change
            break
        else:
            policies = newPolicies
    return policies
    
    
    