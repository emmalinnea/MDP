# MDP
Created an MDP (Markov Decision Process) class in python. Made a value iteration function that will return dictionary with the state as the key as an (x,y) tuple and the utility of that state as the value. Then made a policy iteration function with a policy evaluation function defined within it, which returns a dictionary with each state as keys and the best policy (North, West, South, or East) from that state.
Very useful for searches that involve varying combinations of risk vs reward. You can set up the class to incentivise reaching some states while avoiding others.
