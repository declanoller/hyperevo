from HyperEPANN import HyperEPANN
from GymAgent import GymAgent
from EvoAgent import EvoAgent
from Population import Population
from Walker_1D import Walker_1D
from Walker_2D import Walker_2D
from Walker_ND import Walker_ND
from Atom import Atom
from time import time
import numpy as np





p1 = Population(agent_class=Walker_ND, N_pop=32, atom_add_chance=0.005, complex_atom_add_chance=0.95)

p1.evolve(N_gen=512, N_episode_steps=200, N_trials_per_agent=2, N_runs_with_best=2, record_final_runs=False, show_final_runs=False)

best_individ = p1.population[0]

#best_individ.NN.saveNetworkAsAtom()

best_individ.plotNetwork()

exit()




ea = EvoAgent(agent_class=Walker_2D)
path = '/home/declan/Documents/code/hyperevo/misc_runs/evolve_Walker_2D_02-02-2019_17-20-44_great'
NN = 'bestNN_Walker_2D_02-02-2019_17-20-44' + '.json'

'''path = '/home/declan/Documents/code/hyperevo/misc_runs/evolve_Walker_2D_02-02-2019_16-34-22'
NN = 'bestNN_Walker_2D_02-02-2019_16-34-22.json'
'''
ea.loadNetworkFromFile(fname=(path + '/' + NN))

#ea.plotNetwork()

#print(ea.runEpisode(200, show_episode=True))

exit()














#ea = EvoAgent(agent_class=GymAgent, env_name='Pendulum', verbose=False)
N_inputs=2
ea = HyperEPANN(N_inputs=N_inputs, N_outputs=2)

#ea.plotNetwork()

ea.addConnectingWeight((0,0,3,0), std=1.0)
ea.addConnectingWeight((1,0,3,0), std=1.0)
ea.addConnectingWeight((2,0,3,0), std=1.0)
ea.addConnectingWeight((0,0,4,0), std=1.0)
ea.addConnectingWeight((1,0,4,0), std=1.0)
ea.addAtomInBetween((1,0,4,0))

ea.addAtom('test_atom') #6

ea.addConnectingWeight((0,0,6,0), std=1.0)
ea.addConnectingWeight((1,0,6,3), std=1.0)
ea.addConnectingWeight((6,0,3,0), std=1.0)
#ea.addConnectingWeight((6,1,3,0), std=1.0)
ea.addConnectingWeight((6,1,4,0), std=1.0)
ea.plotNetwork()
exit()

ea.addConnectingWeight((2,0,6,1), std=1.0)

ea.addConnectingWeight((6,0,4,0), std=1.0)
ea.addConnectingWeight((6,1,3,0), std=1.0)

'''ea.addConnectingWeight((2,0,5,0), std=1.0)
ea.addAtom()
ea.addConnectingWeight((2,0,6,0), std=1.0)
ea.addConnectingWeight((3,0,5,0), std=1.0)
ea.addConnectingWeight((4,0,5,0), std=1.0)
ea.addConnectingWeight((4,0,6,0), std=1.0)
ea.addConnectingWeight((3,0,8,0), std=1.0)
ea.addConnectingWeight((8,0,5,0), std=1.0)'''
#ea.addAtomInBetween((2,0,5,0))


'''print(ea.analytic_NN_fn)

print(ea.NN_fn(*[8,9,5,3]))'''

ea.plotNetwork()

exit()




p1 = Population(agent_class=GymAgent, env_name='CartPole', N_pop=4)

p1.evolve(N_gen=128, N_episode_steps=200, N_trials_per_agent=2, N_runs_with_best=2, record_final_runs=False, show_final_runs=False)

best_individ = p1.population[0]

best_individ.NN.saveNetworkAsAtom()

best_individ.plotNetwork()

exit()










#ea = EvoAgent(agent_class=GymAgent, env_name='Pendulum', verbose=False)
N_inputs=4
ea = HyperEPANN(N_inputs=N_inputs, N_outputs=2)

#ea.plotNetwork()

ea.addConnectingWeight((0,0,5,0), std=1.0)
ea.addConnectingWeight((1,0,5,0), std=1.0)
ea.addAtomInBetween((1,0,5,0))

ea.addConnectingWeight((2,0,5,0), std=1.0)
ea.addAtom()
ea.addConnectingWeight((2,0,6,0), std=1.0)
ea.addConnectingWeight((3,0,5,0), std=1.0)
ea.addConnectingWeight((4,0,5,0), std=1.0)
ea.addConnectingWeight((4,0,6,0), std=1.0)
ea.addConnectingWeight((3,0,8,0), std=1.0)
ea.addConnectingWeight((8,0,5,0), std=1.0)
#ea.addAtomInBetween((2,0,5,0))


print(ea.analytic_NN_fn)

print(ea.NN_fn(*[8,9,5,3]))

ea.plotNetwork()

exit()







N_tests = 100000

inputs = np.random.random((N_tests, N_inputs))

st = time()
for i in range(N_tests):

    #ea.clearAllAtoms()
    ea.NN_fn(*inputs[i])


print('time elapsed:', time() - st)








a = Atom(0, 'Node')

import sympy

i_0, c = sympy.symbols('i_0 c')

a.addToInputsReceived(0, i_0)
a.addToInputsReceived(0, c)

'''a.addToInputsReceived(0, 5)
a.addToInputsReceived(0, 7)
'''
a.forwardPass()

exit()

ea = EvoAgent(agent_class=GymAgent, env_name='Pendulum', verbose=False)

ea.NN.addConnectingWeight((0,0,4,0))
ea.NN.addConnectingWeight((1,0,4,0))
ea.NN.addAtomInBetween((1,0,4,0))

#ea.plotNetwork()

ea.NN.getAtomFunction()

exit()


ea = EvoAgent(agent_class=GymAgent, env_name='LunarLander', verbose=False)

'''ea.initEpisode()
for i in range(200):
    ea.agent.drawState()
    ea.iterate(0)
'''
ea.runEpisode(200, show_episode=True)

exit()





e = HyperEPANN()

e.addConnectingWeight((0,0,1,0))

e.printNetwork()

#e.plotNetwork()

exit()

a = Atom(0, 'Node')



exit(0)

e.runEpisode(400, plot_run=False, record_episode=True)

# add thing to make it plan N runs with the best individ, to show it actually getting better as opposed to just lucky
###### Add node legend for plotnetwork!
############## oh shit, is it gonna shit the bed when it tries to add another weight but can't?? fix that for sure
p1 = Population(agent_class=LunarLanderAgent, N_pop=64, mut_type='change_topo', std=1.0, render_type='gym')

p1.evolve(N_gen=256, N_episode_steps=500, N_trials_per_agent=2, N_runs_with_best=9, record_final_runs=True, show_final_runs=False)

exit(0)






e = EPANN(agent_class=LunarLanderAgent, render_type='gym', N_init_hidden_nodes=0, init_IO_weights=False, verbose=True)


e.mutateAddWeight()
e.plotNetwork()
e.mutateAddWeight()
e.plotNetwork()
e.mutateAddWeight()
e.plotNetwork()
e.mutateAddWeight()
e.plotNetwork()
e.mutateAddWeight()
e.plotNetwork()








start = time()
e.clearAllNodes()
s = e.agent.getStateVec()
e.forwardPass(s)
end = time()

print('elapsed time:', end-start)

exit()




# This will be my "standard base case", that I'll center the others around, since I know it works.
'''
pt.varyParam(agent_class=LunarLanderAgent, N_pop=25, mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=16, N_gen=100, N_episode_steps=400, N_trials_per_agent=5, N_runs=3)
'''

'''pt.varyParam(agent_class=LunarLanderAgent, N_pop=25, mut_type='gauss_noise', std=[0.001, 0.01, 0.1, 1.0, 5.0], render_type='gym',
N_init_hidden_nodes=16, N_gen=150, N_episode_steps=400, N_trials_per_agent=5, N_runs=3)'''


'''pt.varyParam(agent_class=LunarLanderAgent, N_pop=25, mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=[4, 8, 16, 32, 64], N_gen=150, N_episode_steps=400, N_trials_per_agent=5, N_runs=3)'''


'''pt.varyParam(agent_class=LunarLanderAgent, N_pop=16, mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=8, N_gen=250, N_episode_steps=400, N_trials_per_agent=[1, 5, 10], N_runs=3)


pt.varyParam(agent_class=LunarLanderAgent, N_pop=16, mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=8, N_gen=250, N_episode_steps=400, N_trials_per_agent=5, N_runs=3, best_N_frac=[1/10.0, 1/5.0, 1/3.0])'''


'''pt.varyParam(agent_class=LunarLanderAgent, N_pop=[8, 16, 32], mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=8, N_gen=250, N_episode_steps=400, N_trials_per_agent=5, N_runs=3)'''

# Messing around with the best way to spend computation time:

'''pt.varyParam(agent_class=LunarLanderAgent, N_pop=[8, 16, 32, 64], mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=8, N_gen=[512, 256, 128, 64], N_episode_steps=400, N_trials_per_agent=5, N_runs=3)'''


'''pt.varyParam(agent_class=LunarLanderAgent, N_pop=[16, 32, 64], mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=8, N_gen=250, N_episode_steps=400, N_trials_per_agent=[8, 4, 2], N_runs=3)'''

exit(0)

pt.varyParam(agent_class=LunarLanderAgent, N_pop=16, mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=8, N_gen=[64, 128, 256, 512], N_episode_steps=400, N_trials_per_agent=[16, 8, 4, 2], N_runs=3)





exit(0)

p1 = Population(agent_class=LunarLanderAgent, N_pop=25, mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=24)

p1.evolve(N_gen=50, N_steps=400, N_runs=5)
p1.population[0].plotNetwork()
p1.population[0].printNetwork()

for i in range(10):
    r = p1.population[0].runEpisode(400, plot_run=False)
    print('r of best individ: ', r)

exit()
exit()




e = EPANN(agent_class=CartpoleAgent, render_type='gym')
e.runEpisode(400, plot_run=True)
exit(0)


'''p1.population[0].printNetwork()
p1.population[0].mutate()
p1.population[0].printNetwork()
exit()'''
#ep.runEpisode(200)







'''
e = EPANN(agent_class=Walker_1D)
e.printNetwork()
e.addNode(0, 2)
e.printNetwork()
e.addNode(0, 3)
e.printNetwork()
e.addNode(5, 3)
e.printNetwork()
e.addConnection(4, 5)
e.printNetwork()
print(e.nodeInLineage(4, 6))'''







#
