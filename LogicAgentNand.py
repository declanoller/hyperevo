import matplotlib.pyplot as plt
import numpy as np


class LogicAgentNand:


    def __init__(self, **kwargs):

        self.getStateVec()


        self.N_state_terms = 2
        self.N_actions = 2

        self.state_labels = ['x', 'y']
        self.action_labels = ['T', 'F']

        self.truth_table_dict = {
        np.array([0,0]) : 1,
        np.array([0,1]) : 1,
        np.array([1,0]) : 1,
        np.array([1,1]) : 0,
        }

        self.action_space_type = 'discrete'
        self.render_type = 'matplotlib'



    def getStateVec(self):
        return(self.state_vec)


    def getNewState(self):
        self.state_vec = np.random.random_integers(0,1,3)


    def initEpisode(self):
        self.getNewState()



    def iterate(self, action):
        # Action 0 is go L, action 1 is go R.

        return(self.reward(action), self.getStateVec(), False)



    def reward(self, action):

        correct = self.truth_table_dict[self.state_vec]
        return(-abs(correct - action))



    def drawState(self, ax):

        ax.clear()
        pass







#
