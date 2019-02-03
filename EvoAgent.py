import numpy as np
from copy import deepcopy
from HyperEPANN import HyperEPANN
import FileSystemTools as fst
import matplotlib.pyplot as plt

'''

Last time, I had the EPANN class create an Agent object, but that doesn't really
make much sense -- the agent is really using the EPANN, not the other way around.
So here, EvoAgent creates a HyperEPANN object, which it uses.


So last time I had an GymAgent class and it made an env. for each one. I think that
was causing some real problems, so this time I think the smart thing to do is either
pass an env object that every EvoAgent will share, or, if nothing is passed, it will
create its own.


agent class (GymAgent or something like Walker_1D) needs members and methods:

-getStateVec()
-initEpisode()
-iterate(action) (returns (reward, state, done))
-drawState()

-state labels (list of strings corresponding to each state)
-action labels (same but with actions)
-action space type ('discrete' or 'continuous')
-N_state_terms
-N_actions



'''


class EvoAgent:

    def __init__(self, **kwargs):

        # Agent stuff
        self.agent_class = kwargs.get('agent_class', None)
        assert self.agent_class is not None, 'Need to provide an agent class! exiting'

        self.agent = self.agent_class(**kwargs)

        self.verbose = kwargs.get('verbose', False)
        self.action_space_type = self.agent.action_space_type
        self.render_type = self.agent.render_type

        self.N_inputs = self.agent.N_state_terms
        self.N_outputs = self.agent.N_actions


        # HyperEPANN stuff
        self.NN = HyperEPANN(N_inputs=self.N_inputs, N_outputs=self.N_outputs, **kwargs)


        self.weight_change_chance = kwargs.get('weight_change_chance', 0.98)
        self.weight_add_chance = kwargs.get('weight_add_chance', 0.09)
        self.weight_remove_chance = kwargs.get('weight_remove_chance', 0.05)
        self.node_add_chance = kwargs.get('node_add_chance', 0.0005)







############################## For interfacing with NN


    def forwardPass(self, state_vec):
        output_vec = self.NN.forwardPass(state_vec)
        a = self.greedyOutput(output_vec)
        return(a)


    def greedyOutput(self, vec):
        return(np.argmax(vec))


    def mutate(self, std=0.1):

        self.NN.mutate(std=std)


    def getNAtoms(self):
        return(len(self.NN.atom_list))


    def getNConnections(self):
        return(len(self.NN.weights_list))


    def plotNetwork(self, **kwargs):
        self.NN.plotNetwork(**kwargs)


    def saveNetworkAsAtom(self, **kwargs):
        self.NN.saveNetworkAsAtom(**kwargs)


    def saveNetworkToFile(self, **kwargs):
        self.NN.saveNetworkToFile(**kwargs)


    def loadNetworkFromFile(self, **kwargs):
        self.NN.loadNetworkFromFile(**kwargs)


    def clone(self):
        clone = deepcopy(self)
        return(clone)


########################### For interacting with the agent class


    def iterate(self, action):

        r, s, done = self.agent.iterate(action)
        return(r, s, done)


    def initEpisode(self):
        self.agent.initEpisode()


################################ For interfacing with gym env and playing


    def setMaxEpisodeSteps(self, N_steps):
        self.agent.setMaxEpisodeSteps(N_steps)



    def runEpisode(self, N_steps, **kwargs):


        R_tot = 0
        Rs = []

        show_episode = kwargs.get('show_episode', False)
        record_episode = kwargs.get('record_episode', False)

        if show_episode:
            self.createFig()

        if record_episode:
            self.agent.setMonitorOn(show_run=show_episode)

        self.initEpisode()

        for i in range(N_steps):
            self.NN.clearAllAtoms()

            if i%int(N_steps/10)==0:
                self.print('R_tot = {:.3f}'.format(R_tot))


            s = self.agent.getStateVec()
            a = self.forwardPass(s)
            self.print('s = {}, a = {}'.format(s, a))

            (r, s, done) = self.iterate(a)

            R_tot += r
            Rs.append(R_tot)

            if done:
                #return(R_tot)
                break

            if show_episode or record_episode:
                self.drawState()


        if record_episode:
            print('R_tot = {:.3f}'.format(R_tot))

        self.print('R_tot/N_steps = {:.3f}'.format(R_tot/N_steps))
        #self.agent.closeEnv()
        return(R_tot)




    def drawState(self):

        if self.render_type == 'gym':
            self.agent.drawState()

        if self.render_type == 'matplotlib':
            self.agent.drawState(self.ax)
            self.fig.canvas.draw()






############################# Misc/debugging stuff


    def print(self, str):

        if self.verbose:
            print(str)



    def createFig(self):

        if self.render_type == 'matplotlib':
            self.fig, self.ax = plt.subplots(1, 1, figsize=(4,4))
            plt.show(block=False)



#
