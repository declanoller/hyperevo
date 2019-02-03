import json

atom_info_dict = {

'Node' : {
'Name' : 'Node',
'N_nodes' : 1,
'N_inputs' : 1,
'N_outputs' : 1,
'atom_function_vec' : ['tanh(a_0)']
},

'InputNode' : {
'Name' : 'InputNode',
'N_nodes' : 1,
'N_inputs' : 1,
'N_outputs' : 1,
'atom_function_vec' : ['tanh(a_0)']
},


'OutputNode' : {
'Name' : 'OutputNode',
'N_nodes' : 1,
'N_inputs' : 1,
'N_outputs' : 1,
'atom_function_vec' : ['a_0']
},


'BiasNode' : {
'Name' : 'BiasNode',
'N_nodes' : 1,
'N_inputs' : 1,
'N_outputs' : 1,
'atom_function_vec' : ['1']
},



}

for k, v in atom_info_dict.items():

    fname = 'atom_modules/{}.json'.format(k)

    with open(fname, 'w') as outfile:
        json.dump(v, outfile, indent=4)






''' SCRAP



'output_type' : 'nonlinear',
'weights_list' : [],
'propagate_order' : [],




'''


#
