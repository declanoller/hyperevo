import matplotlib.pyplot as plt
import numpy as np


class agent_name:


    def __init__(self, **kwargs):

        self.position = 0
        self.target_position = None

        self.N_state_terms = 2
        self.N_actions = 2

        self.state_labels = ['pos_x', 'pos_target']
        self.action_labels = ['L', 'R']

        self.action_space_type = 'discrete'
        self.render_type = 'matplotlib'


    def getStateVec(self):
        return(np.array([self.position, self.target_position]))


    def initEpisode(self):
        pass



    def iterate(self, action):
        # Action 0 is go L, action 1 is go R.

        return(self.reward(), self.getStateVec(), False)



    def reward(self):
        pass



    def drawState(self, ax):

        ax.clear()
        pass







#
