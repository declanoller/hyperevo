import numpy as np
from queue import Queue
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.collections import PatchCollection
import random
import networkx as nx
import pygraphviz as pgv
from Atom import Atom
import FileSystemTools as fst
import json
import sympy
import os
import traceback as tb


'''
So this will be fairly similar to EPANN(), but instead of just nodes,
there will be "atoms". There will still be simple nodes too, but atoms
can also be a fully formed EPANN. The idea is to be able to still propagate
and keep track of a fully feed-forward NN, but be able to insert complex, higher
level things into the NN.

Now I'm making the bias atom the last created, so the inputs can just be i_0, i_1,
etc.

'''


class HyperEPANN:

    def __init__(self, **kwargs):

        self.verbose = kwargs.get('verbose', False)
        #self.action_space_type = self.agent.action_space_type

        self.N_inputs = kwargs.get('N_inputs', 2)
        self.N_outputs = kwargs.get('N_outputs', 2)
        self.N_atoms = 0

        self.input_atom_indices = []
        self.output_atom_indices = []
        self.non_node_atom_indices = []
        self.other_node_atom_indices = []


        self.atom_list = []
        # This will be a set of tuples of the form (par index, par_out_ind, child index, child_in_ind)
        self.weights_list = set()
        self.propagate_order = []
        self.NN_fn = None

        self.weight_change_chance = kwargs.get('weight_change_chance', 0.98)
        self.weight_add_chance = kwargs.get('weight_add_chance', 0.09)
        self.weight_remove_chance = kwargs.get('weight_remove_chance', 0.05)
        self.atom_add_chance = kwargs.get('atom_add_chance', 0.0005)
        # This is the chance of adding a complex atom when mutating to
        # add some atom has already been selected.
        self.complex_atom_add_chance = kwargs.get('complex_atom_add_chance', 0.7)


        # Add input and output atoms
        [self.addAtom(is_input_atom=True) for i in range(self.N_inputs)]

        # Add bias atom (type 'Node' by default)
        self.addAtom(is_bias_atom=True)

        #self.printNetwork()
        [self.addAtom(is_output_atom=True) for i in range(self.N_outputs)]

        atom_modules_dir = 'atom_modules/'
        standard_modules = ['Node', 'InputNode', 'OutputNode', 'BiasNode']
        # Get a list of the modules in atom_modules_dir that aren't the standard ones.
        self.complex_atom_types = [x.replace('.json', '') for x in os.listdir(atom_modules_dir) if (x.replace('.json', '') not in standard_modules)]




########################### Stuff for altering the network

    def addAtom(self, atom_type='Node', is_input_atom=False, is_output_atom=False, is_bias_atom=False):

        #print('adding atom ', self.N_atoms, 'of type ', atom_type)
        if is_input_atom:
            new_atom = Atom(self.N_atoms, 'InputNode')
            self.input_atom_indices.append(new_atom.atom_index)

        elif is_output_atom:
            new_atom = Atom(self.N_atoms, 'OutputNode')
            self.output_atom_indices.append(new_atom.atom_index)

        elif is_bias_atom:
            new_atom = Atom(self.N_atoms, 'BiasNode')
            self.bias_atom_index = new_atom.atom_index

        elif atom_type=='Node':
            # Ordinary atom type
            new_atom = Atom(self.N_atoms, atom_type)
            self.other_node_atom_indices.append(new_atom.atom_index)

        else:
            # Non node atom types
            new_atom = Atom(self.N_atoms, atom_type)
            self.non_node_atom_indices.append(new_atom.atom_index)


        self.N_atoms += 1
        self.atom_list.append(new_atom)
        self.sortPropagateOrder()
        self.getNetworkFunction()
        return(new_atom.atom_index)


    def addConnectingWeight(self, weight_parchild_tuple, val=None, std=1.0):
        #print('adding weight: ', weight_parchild_tuple)
        # weight_parchild_tuple should now be of the form:
        # (par_atom_index, par_atom_output_index, child_atom_index, child_atom_input_index)

        # You shouldn't be calling this unless you already know it doesn't have that connection.
        assert weight_parchild_tuple not in self.weights_list, 'Problem in addConnectingWeight! Weight already in weights_list'

        (par_atom_index, par_atom_output_index, child_atom_index, child_atom_input_index) = weight_parchild_tuple
        assert par_atom_index not in self.output_atom_indices, 'Cant make an output atom a parent!'
        assert (child_atom_index not in self.input_atom_indices) and (child_atom_index != self.bias_atom_index) , 'Cant make an input or bias atom a child! weight_tuple {}'.format(weight_parchild_tuple)

        try:
            self.atom_list[child_atom_index].addToInputIndices(par_atom_index, par_atom_output_index, child_atom_input_index)
            self.atom_list[par_atom_index].addToOutputWeights(par_atom_output_index, child_atom_index, child_atom_input_index, val=val, std=std)
            self.weights_list.add(weight_parchild_tuple)
        except:
            print('problem adding weight: ', weight_parchild_tuple)
            print(tb.format_exc())
            self.plotNetwork(save_plot=True, show_plot=True)
            exit()
        #print('adding weight between atoms {} and {}'.format(par_atom_index, child_atom_index))
        self.sortPropagateOrder()
        self.getNetworkFunction()


    def removeConnectingWeight(self, weight_parchild_tuple):

        # You shouldn't be calling this unless you already know it has that connection.
        assert weight_parchild_tuple in self.weights_list, 'Problem in removeConnectingWeight!'

        (par_atom_index, par_atom_output_index, child_atom_index, child_atom_input_index) = weight_parchild_tuple
        assert par_atom_index not in self.output_atom_indices, 'Cant make an output atom a parent!'
        assert (child_atom_index not in self.input_atom_indices) and (child_atom_index != self.bias_atom_index) , 'Cant make an input or bias atom a child!'

        try:
            self.atom_list[child_atom_index].removeFromInputIndices(par_atom_index, par_atom_output_index, child_atom_input_index)
            self.atom_list[par_atom_index].removeFromOutputWeights(par_atom_output_index, child_atom_index, child_atom_input_index)
            self.weights_list.remove(weight_parchild_tuple)
        except:
            print('problem removing weight: ', weight_parchild_tuple)
            print(tb.format_exc())
            self.plotNetwork(save_plot=True, show_plot=True)
            exit()

        self.sortPropagateOrder()
        self.getNetworkFunction()


    def addAtomInBetween(self, weight_parchild_tuple, atom_type='Node'):
        # This adds a atom in between existing atoms par_index and child_index, where the output of par_index went to child_index.
        # Pass it the index of each.
        (par_atom_index, par_atom_output_index, child_atom_index, child_atom_input_index) = weight_parchild_tuple
        assert par_atom_index not in self.output_atom_indices, 'Cant make an output atom a parent!'
        assert (child_atom_index not in self.input_atom_indices) and (child_atom_index != self.bias_atom_index) , 'Cant make an input or bias atom a child!'

        self.print('adding atom between atoms {} and {}'.format(par_atom_index, child_atom_index))
        #print('adding atom between atoms {} and {}'.format(par_atom_index, child_atom_index))
        # Add atom:
        # New atom:
        old_weight = self.atom_list[par_atom_index].getOutputWeight(par_atom_output_index, child_atom_index, child_atom_input_index)
        self.removeConnectingWeight(weight_parchild_tuple)
        new_atom_index = self.addAtom(atom_type=atom_type)

        # Get random output, input indices for the new atom
        new_atom_input_index = random.choice(self.atom_list[new_atom_index].getAllInputIndices())
        new_atom_output_index = random.choice(self.atom_list[new_atom_index].getAllOutputIndices())

        new_parchild_tup_1 = (par_atom_index, par_atom_output_index, new_atom_index, new_atom_input_index)
        new_parchild_tup_2 = (new_atom_index, new_atom_output_index, child_atom_index, child_atom_input_index)

        self.addConnectingWeight(new_parchild_tup_1, val=old_weight)
        self.addConnectingWeight(new_parchild_tup_2, val=1)

        self.sortPropagateOrder()
        self.getNetworkFunction()


    def sortPropagateOrder(self):


        # So something to keep in mind here is that if a atom is isolated because its
        # atom was removed or something, it will just never enter the sort_prop list.
        # That's probably fine, because it can't add anything anyway.
        # Update: I now use propagate_order to get new pairs for adding weights,
        # so you have to manually add these "island" atoms in (what unadded_list is for).
        # I don't think it matters where they go. No wait, you need them to go first,
        # or it screws it up when adding weights later (at the beginning both inputs and
        # outputs aren't connected, but outputs get added to prop_order by default, so you
        # should manually make it so the inputs are first in prop_order.)
        #
        # I think even with this new "atom" way, all you need to know are
        # pars and children, the input and output indices of an atom shouldn't
        # change anything.

        sort_queue = Queue()
        prop_order_set = set()
        queue_set = set()
        self.propagate_order = []
        unadded_list = list(range(self.N_atoms))

        # First add the output indices, which we'll work backwards from.
        for ind in self.output_atom_indices:
            sort_queue.put(ind)
            queue_set.add(ind)

        while not sort_queue.empty():

            ind = sort_queue.get()
            queue_set.remove(ind)
            # Make sure all the children of this atom are already in the list/set. If
            # one isn't, add this child to the queue if it's not already there,
            # (this is in case there's a "dead end" atom that would never get seen by
            # tracing back from the outputs), put the atom back in the queue, and continue
            # to the next loop iteration.
            #
            # You could also have it not break immediately, and add all its unseen children,
            # which might speed it up, but might also not.
            all_children_in_prop_order = True

            #
            for child_ind in self.atom_list[ind].getAllChildAtoms():
                if child_ind not in prop_order_set:
                    all_children_in_prop_order = False
                    # Only want to add the child to the queue if it isn't already in it
                    if child_ind not in queue_set:
                        sort_queue.put(child_ind)
                        queue_set.add(child_ind)

                    break

            if all_children_in_prop_order:
                # This means that the atom can now be added to the prop_order and set,
                # and also add its pars to the queue if they're not already.
                self.propagate_order.append(ind)
                prop_order_set.add(ind)
                unadded_list.remove(ind)

                for par_ind in self.atom_list[ind].getAllParentAtoms():
                    if par_ind not in queue_set:
                        sort_queue.put(par_ind)
                        queue_set.add(par_ind)

            else:
                # If the children aren't all there already, put it back in the queue.
                sort_queue.put(ind)
                queue_set.add(ind)

        # Now it should be in order, where you can evaluate each atom, starting with the input ones,
        # and all the inputs should arrive in the right order.
        self.propagate_order += unadded_list
        self.propagate_order.reverse()


    def getNetworkFunction(self):

        # This will return an analytic expression for what the network does,
        # in terms of a list of inputs [i_0, i_1, ...] that's N_inputs long.
        # They're sympy symbols.

        # I think you have to do this, to clear all the inputs.
        # (I think if you switch to lambdifying the atoms as well, you won't
        # need to deal with inputs_received)
        self.clearAllAtoms()

        # Make the sympy symbols for the input atoms
        self.input_atom_symbols = [sympy.symbols('i_{}'.format(i)) for i in self.input_atom_indices]

        # Put the input vec into the input atoms
        for i, index in enumerate(self.input_atom_indices):
            self.atom_list[index].setInputAtomValue(self.input_atom_symbols[i])

        # For each atom in the sorted propagate list, propagate to its children
        for ind in self.propagate_order:
            self.propagateAtomOutput(ind)

        # Right now this is tricky because getValue() returns a vector, but we probably
        # only want nodes as outputs. the [0] will be a temp. hack
        output_vec = [self.atom_list[ind].getValue()[0] for ind in self.output_atom_indices]
        self.analytic_NN_fn = output_vec
        #print(output_vec)
        self.NN_fn = sympy.lambdify(self.input_atom_symbols, output_vec, 'numpy')


    def getsInputFrom(self, n1_index, n2_index):

        # This is to check if n1 gets input from n2, indirectly.

        n1 = self.atom_list[n1_index]
        n2 = self.atom_list[n2_index]
        lineage_q = Queue()
        [lineage_q.put(n) for n in n1.getAllParentAtoms()]

        while lineage_q.qsize() > 0:
            next = lineage_q.get()
            if n2_index in self.atom_list[next].getAllParentAtoms():
                return(True)
            else:
                [lineage_q.put(n) for n in self.atom_list[next].getAllParentAtoms()]

        return(False)


    def loadNetworkFromFile(self, **kwargs):

        # This loads a NN from a .json file that was saved with
        # saveNetworkToFile(). Note that it will overwrite any existing NN
        # for this object.

        fname = kwargs.get('fname', None)
        assert fname is not None, 'Need to supply an fname'

        with open(fname) as json_file:
            NN_dict = json.load(json_file)

        self.atom_list = []
        self.N_atoms = 0
        self.input_atom_indices = []
        self.output_atom_indices = []
        self.non_node_atom_indices = []
        self.other_node_atom_indices = []

        self.N_inputs = NN_dict['N_input_atoms']
        self.N_outputs = NN_dict['N_output_atoms']

        # Uhhhh yeah so right now this won't load other, special atoms.
        # I should add that soon but it might be complicated.

        # Add input nodes
        [self.addAtom(is_input_atom=True) for i in range(self.N_inputs)]
        N_other_atoms = NN_dict['N_atoms'] - (1 + self.N_inputs + self.N_outputs)
        # Add bias node
        self.addAtom(is_bias_atom=True)
        # output and other nodes
        [self.addAtom(is_output_atom=True) for i in range(self.N_outputs)]
        [self.addAtom() for i in range(N_other_atoms)]

        for weight_dict in NN_dict['weights']:
            self.addConnectingWeight((weight_dict['parent'], weight_dict['parent_output_index'], weight_dict['child'], weight_dict['child_input_index']), weight_dict['weight'])


########################################## Mutation stuff

    def mutateAddAtom(self):
        if len(self.weights_list)>0:
            w = random.choice(list(self.weights_list))
            if random.random() <= self.complex_atom_add_chance:
                type = random.choice(self.complex_atom_types)
                self.addAtomInBetween(w, atom_type=type)
            else:
                self.addAtomInBetween(w, atom_type='Node')


    def mutateAddWeight(self, std=0.1):
        N_attempts = 8
        i = 0
        while True:
            if i>N_attempts:
                return(0)
            else:
                i += 1


            (a1, a2) = random.sample(self.atom_list, 2)

            # check that they're not both input atoms or output atoms
            if (a1.is_input_atom or a1.is_bias_atom) and (a2.is_input_atom or a2.is_bias_atom):
                continue
            if a1.is_output_atom and a2.is_output_atom:
                continue

            # Figure out which one is earlier vs later in the prop_order list.
            a1_prop_order_index = self.propagate_order.index(a1.atom_index)
            a2_prop_order_index = self.propagate_order.index(a2.atom_index)

            par_atom_index = self.propagate_order[min(a1_prop_order_index, a2_prop_order_index)]
            child_atom_index = self.propagate_order[max(a1_prop_order_index, a2_prop_order_index)]

            # I thought the prop_order should make it so I don't need this, but
            # I just got a situation where a bias node was made a child in this function,
            # so I'm doing this just to be safe.
            if par_atom_index in self.output_atom_indices:
                continue
            if (child_atom_index in self.input_atom_indices) or (child_atom_index == self.bias_atom_index):
                continue

            par_atom = self.atom_list[par_atom_index]
            child_atom = self.atom_list[child_atom_index]

            # Get random output, input indices for the atoms
            par_atom_output_index = random.choice(par_atom.getAllOutputIndices())
            child_atom_input_index = random.choice(child_atom.getAllInputIndices())

            weight_parchild_tuple = (par_atom_index, par_atom_output_index, child_atom_index, child_atom_input_index)

            # Check if this tuple is in the weights list yet. If not, you're good!
            if weight_parchild_tuple in self.weights_list:
                continue

            self.addConnectingWeight(weight_parchild_tuple, val=None, std=std)
            break


    def mutateChangeWeight(self, std=0.1):
        if len(self.weights_list)>0:
            (par_atom_index, par_atom_output_index, child_atom_index, child_atom_input_index) = random.choice(list(self.weights_list))
            self.print('changing weight between {} and {}'.format(par_atom_index, child_atom_index))
            self.atom_list[par_atom_index].mutateOutputWeight(par_atom_output_index, child_atom_index, child_atom_input_index, std=0.1)
            self.getNetworkFunction()


    def mutateRemoveWeight(self):
        if len(self.weights_list)>0:
            (par_atom_index, par_atom_output_index, child_atom_index, child_atom_input_index) = random.choice(list(self.weights_list))
            self.print('removing weight between {} and {}'.format(par_atom_index, child_atom_index))
            self.removeConnectingWeight((par_atom_index, par_atom_output_index, child_atom_index, child_atom_input_index))


    def mutate(self, std=0.1):

        self.print('\n\nbefore mutate:')
        if self.verbose:
            self.printNetwork()

        if random.random() < self.atom_add_chance:
            # Add a atom by splitting an existing weight
            self.mutateAddAtom()


        if random.random() < self.weight_add_chance:
            # Add weight between two atoms
            self.mutateAddWeight(std=std)


        if random.random() < self.weight_change_chance:
            # Change weight
            self.mutateChangeWeight(std=std)


        if random.random() < self.weight_remove_chance:
            # Remove weight
            self.mutateRemoveWeight()


        self.print('\nafter mutate:')
        if self.verbose:
            self.printNetwork()






###################################### Stuff for propagating and actually using the network

    def propagateAtomOutput(self, atom_index):

        # This is now only used for finding the NN function, not for evaluating
        # specific results.

        # This assumes that the propagate_order list is already sorted!
        # If it isn't, you'll get some bad results.
        atom = self.atom_list[atom_index]

        for par_output_ind in atom.getAllOutputIndices():

            val = atom.getValueOfOutputIndex(par_output_ind)
            for child_atom_index in atom.getChildAtomsOfOutput(par_output_ind):
                #print('proping val {} of atom ind {} to child ind {}'.format(val, atom_index, child_atom_index))
                for child_atom_input_ind in atom.getChildAtomInputIndices(par_output_ind, child_atom_index):

                    w = atom.getOutputWeight(par_output_ind, child_atom_index, child_atom_input_ind)
                    self.atom_list[child_atom_index].addToInputsReceived(child_atom_input_ind, w*val)


    def forwardPass(self, input_vec):

        assert len(input_vec)==self.N_inputs, 'Input vec needs to be same size as N_inputs! ({} vs {})'.format(len(input_vec), self.N_inputs)
        return(self.NN_fn(*input_vec))



#################### Bigger scale stuff

    def clearAllAtoms(self):
        [a.clearAtom() for a in self.atom_list]





################################## Other stuff

    def clone(self):
        clone = deepcopy(self)
        return(clone)


    def createFig(self):
        if self.render_type == 'matplotlib':
            self.fig, self.axes = plt.subplots(1,2, figsize=(16,8))
            plt.show(block=False)


    def print(self, str):

        if self.verbose:
            print(str)


    def printNetwork(self):
        print('\n')
        for i, n in enumerate(self.atom_list):
            if n.is_bias_atom:
                print('\natom ', i, '(bias atom)')
            elif n.is_input_atom:
                print('\natom ', i, '(input atom)')
            elif n.is_output_atom:
                print('\natom ', i, '(output atom)')
            else:
                print('\natom ', i)
            print('input indices:', n.input_indices)
            print('output indices: ', n.getAllChildAtoms())
            print('output weights: ', n.getOutputWeightStr())

        print()


    def plotNetwork(self, **kwargs):

        show_plot = kwargs.get('show_plot', True)
        save_plot = kwargs.get('save_plot', False)
        fname = kwargs.get('fname', None)
        atom_legend = kwargs.get('atom_legend', False)




        fig, ax = plt.subplots(1, figsize=(12,8))
        DG = nx.DiGraph()

        other_atom_indices = [i for i,n in enumerate(self.atom_list) if ((i not in self.input_atom_indices) and (i not in self.output_atom_indices) and (i != self.bias_atom_index))]
        node_indices = (self.input_atom_indices + [self.bias_atom_index] + self.output_atom_indices + self.other_node_atom_indices)


        for i in self.input_atom_indices:
            DG.add_node(i)

        DG.add_node(self.bias_atom_index)

        for i in self.output_atom_indices:
            DG.add_node(i)

        # I think you have to add this, because if you have a node that doesn't have any connections
        # and it's not I/O/B, then it will never get entered into DG without this.
        for i in self.other_node_atom_indices:
            DG.add_node(i)

        for i in self.non_node_atom_indices:
            DG.add_node(i)

        # You need to make these before doing the layout, because the layout uses these
        # edges!
        for n in self.atom_list:
            for o in n.getAllChildAtoms():
                DG.add_edges_from([(n.atom_index, o)])

        #nx.draw(DG, with_labels=True, font_weight='bold', arrowsize=20)
        # Get layout
            atom_positions_dict = nx.drawing.nx_agraph.graphviz_layout(DG, prog='dot')
        atom_positions = list(atom_positions_dict.values())

        # Draw nodes
        node_draw_args = {'pos' : atom_positions_dict, 'node_size' : 600, 'linewidths' : 1.0, 'edgecolors' : 'black'}
        nx.draw_networkx_nodes(DG, nodelist=self.input_atom_indices, node_color='lightgreen', **node_draw_args)
        nx.draw_networkx_nodes(DG, nodelist=self.output_atom_indices, node_color='orange', **node_draw_args)
        nx.draw_networkx_nodes(DG, nodelist=[self.bias_atom_index], node_color='mediumseagreen', **node_draw_args)
        nx.draw_networkx_nodes(DG, nodelist=self.other_node_atom_indices, node_color='plum', **node_draw_args)


        # Geometry stuff for atoms
        atom_height = 40.0
        node_rad = atom_height/4
        io_circle_rad = node_rad/2
        side_margin = atom_height/6
        span = (atom_height - 2*side_margin)
        io_node_offset = 0.0*io_circle_rad

        # Draw more complex atoms
        draw_atom_arg_dict = {
        'linewidth' : 1,
        'edgecolor' : 'k',
        'facecolor' : 'khaki',
        'boxstyle' : 'round, rounding_size={}'.format(atom_height/6)
        }
        for index in self.non_node_atom_indices:
            atom_patch = mp.FancyBboxPatch(atom_positions_dict[index], atom_height, atom_height, **draw_atom_arg_dict)
            ax.add_patch(atom_patch)

        # Stuff for figuring out where atom i/o's go, and creating nodes for them.
        atom_io_nodes_positions = {}
        atom_input_to_node_ind = {}
        atom_output_to_node_ind = {}
         # Add nodes starting at index self.N_atoms, so we don't overlap with the real atoms.
        io_node_index = self.N_atoms

        # Only do this for more complex atoms with more than 1 i/o.
        for atom_ind in self.non_node_atom_indices:
            N_in = self.atom_list[atom_ind].N_inputs
            N_out = self.atom_list[atom_ind].N_outputs
            x_incr_in = span/(N_in - 1)
            x_incr_out = span/(N_out - 1)

            # This is the lower left corner of the atom box.
            atom_x = atom_positions_dict[atom_ind][0]
            atom_y = atom_positions_dict[atom_ind][1]

            for io_ind in range(N_in):
                DG.add_node(io_node_index)
                atom_input_to_node_ind[(atom_ind, io_ind)] = io_node_index
                pos_x = atom_x + side_margin + io_ind*x_incr_in
                pos_y = atom_y + atom_height + io_node_offset
                atom_io_nodes_positions[io_node_index] = (pos_x, pos_y)
                io_node_index += 1

            for io_ind in range(N_out):
                DG.add_node(io_node_index)
                atom_output_to_node_ind[(atom_ind, io_ind)] = io_node_index
                pos_x = atom_x + side_margin + io_ind*x_incr_out
                pos_y = atom_y - io_node_offset
                atom_io_nodes_positions[io_node_index] = (pos_x, pos_y)
                io_node_index += 1

        # Draw the i/o nodes for the atoms.
        nx.draw_networkx_nodes(DG, nodelist=atom_io_nodes_positions.keys(), pos=atom_io_nodes_positions, node_color='khaki', node_size=100, edgecolors='k', linewidths=1)

        # Combine the dicts so now atom_positions_dict has all of the nodes, including the new ones.
        atom_positions_dict = {**atom_positions_dict, **atom_io_nodes_positions}
        # Draw all network edges. If either the par or child has more than one O/I (respectively),
        # then you need to check the atom_output_to_node_ind dict to get the NODE index (note: this
        # is the index for the node list, only corresponding to the graph). If if does have a single
        # i/o though, then you can just use the par/child's index to get the node index.
        edge_labels = {}
        for w in self.weights_list:
            (par_atom_index, par_atom_output_index, child_atom_index, child_atom_input_index) = w
            weight = self.atom_list[w[0]].getOutputWeight(w[1], w[2], w[3])

            if self.atom_list[par_atom_index].N_outputs==1:
                parent_node_index = par_atom_index
            else:
                parent_node_index = atom_output_to_node_ind[(par_atom_index, par_atom_output_index)]

            if self.atom_list[child_atom_index].N_inputs==1:
                child_node_index = child_atom_index
            else:
                child_node_index = atom_input_to_node_ind[(child_atom_index, child_atom_input_index)]

            # The pair, in terms of the node indices (not atom indices!)
            atom_ind_pair = (parent_node_index, child_node_index)
            edge_labels[atom_ind_pair] = '{:.2f}'.format(weight)

            if weight < 0:
                nx.draw_networkx_edges(DG, pos=atom_positions_dict, edgelist=[atom_ind_pair], width=4.0, alpha=min(abs(weight), 1), edge_color='tomato')

            if weight >= 0:
                nx.draw_networkx_edges(DG, pos=atom_positions_dict, edgelist=[atom_ind_pair], width=4.0, alpha=min(abs(weight), 1), edge_color='dodgerblue')


        # Node labels
        node_labels = {i:str(i) for i in node_indices}
        nx.draw_networkx_labels(DG, pos=atom_positions, labels=node_labels, font_size=14)

        # Atom text labels
        for ind in self.non_node_atom_indices:
            x = atom_positions_dict[ind][0] + side_margin
            y = atom_positions_dict[ind][1] + atom_height/2
            ax.text(x, y, self.atom_list[ind].type, size=14)
            ax.text(x + 1*side_margin, y + atom_height/2 - io_circle_rad, 'inputs', size=10)
            ax.text(x + 1*side_margin, y - atom_height/2 + 0.5*io_circle_rad, 'outputs', size=10)

        # Draw edge weight labels
        nx.draw_networkx_edge_labels(DG, pos=atom_positions_dict, edge_labels=edge_labels, font_size=10, bbox={'alpha':0.2, 'pad':0.0}, label_pos=0.85)

        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(left=.2, bottom=0, right=1, top=1, wspace=1, hspace=0)
        ax.axis('off')

        if atom_legend:
            if (self.agent.state_labels is not None) and (self.agent.action_labels is not None):
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

                percent_offset = 0.02

                bias_str = 'Bias: atom {}\n\n'.format(self.bias_atom_index)
                input_str = bias_str + 'Inputs:\n\n' + '\n'.join(['atom {} = {}'.format(ind, self.agent.state_labels[i]) for i, ind in enumerate(self.input_atom_indices)])
                ax.text(-percent_offset, (1-3*percent_offset), input_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)

                output_str = 'Outputs:\n\n' + '\n'.join(['atom {} = {}'.format(ind, self.agent.action_labels[i]) for i, ind in enumerate(self.output_atom_indices)])
                ax.text(-percent_offset, 3*percent_offset, output_str, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=props)
                textstr = input_str + '\n\n' + output_str


                # place a text box in upper left in axes coords

        if save_plot:
            if fname is not None:
                plt.savefig(fname)
            else:
                plt.savefig(fst.getDateString() + '_NNplot.png')

        if show_plot:
            plt.show()

        plt.close()



    def saveNetworkAsAtom(self, **kwargs):

        try:

            dt = fst.getDateString()
            default_fname = 'misc_runs/NN_{}.json'.format(dt)

            fname = kwargs.get('fname', default_fname)
            atom_name = kwargs.get('atom_name', 'Atom_{}'.format(dt))

            # For saving the NN to file in a way that it can be read in again.
            NN_dict = {}
            NN_dict['Name'] = atom_name
            NN_dict['N_inputs'] = self.N_inputs
            NN_dict['N_outputs'] = self.N_outputs

            # This is to replace all the i_'s in the analytic NN fn with a_'s,
            # because that's how the atom will read them in.
            NN_fn_atom_format = copy(self.analytic_NN_fn)
            #atom_input_symbols_format = [sympy.symbols('a_{}'.format(ind)) for ind in range(self.N_inputs)]
            #self.input_atom_symbols
            for output_index in range(self.N_outputs):
                # Replace the atom input with the input to that input index of the atom.
                for input_index in range(self.N_inputs):
                    NN_fn_atom_format[output_index] = NN_fn_atom_format[output_index].subs(f'i_{input_index}', f'a_{input_index}')

            NN_dict['atom_function_vec'] = [str(fn) for fn in NN_fn_atom_format]

            with open(fname, 'w') as outfile:
                json.dump(NN_dict, outfile, indent=4)

        except:
            print('problem in saving saveNetworkAsAtom!')




    def saveNetworkToFile(self, **kwargs):

        default_fname = 'misc_runs/NN_{}.json'.format(fst.getDateString())

        fname = kwargs.get('fname', default_fname)

        # For saving the NN to file in a way that it can be read in again.
        NN_dict = {}
        NN_dict['N_atoms'] = self.N_atoms
        NN_dict['N_input_atoms'] = self.N_inputs
        NN_dict['N_output_atoms'] = self.N_outputs
        NN_dict['input_atoms'] = self.input_atom_indices
        NN_dict['output_atoms'] = self.output_atom_indices
        NN_dict['bias_atom'] = self.bias_atom_index

        NN_dict['weights'] = []
        for (par_atom_index, par_atom_output_index, child_atom_index, child_atom_input_index) in list(self.weights_list):
            NN_dict['weights'].append({
            'parent' : par_atom_index,
            'parent_output_index' : par_atom_output_index,
            'child' : child_atom_index,
            'child_input_index' : child_atom_input_index,
            'weight' : self.atom_list[par_atom_index].getOutputWeight(par_atom_output_index, child_atom_index, child_atom_input_index)
            })


        with open(fname, 'w') as outfile:
            json.dump(NN_dict, outfile, indent=4)









''' SCRAP




        print(self.bias_atom_index)
        print(self.input_atom_indices)
        print(self.output_atom_indices)





self.clearAllAtoms()

# Put the input vec into the input atoms
for i, index in enumerate(self.input_atom_indices):
    self.atom_list[index].setInputAtomValue(input_vec[i])

# For each atom in the sorted propagate list, propagate to its children
for ind in self.propagate_order:
    self.propagateAtomOutput(ind)

output_vec = np.array([self.atom_list[ind].getValue() for ind in self.output_atom_indices])
return(output_vec)



    def epsGreedyOutput(self, vec):
        if random.random() < self.epsilon:
            return(random.randint(0, len(vec)-1))
        else:
            return(self.greedyOutput(vec))




    def softmaxOutput(self, vec):
        a = np.array(vec)
        a = np.exp(a)
        a = a/sum(a)
        return(np.random.choice(list(range(len(a))), p=a))





'''



#
