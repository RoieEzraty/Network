import numpy as np
from numpy import zeros as zeros
from numpy import arange as arange
from numpy import array as array

def ConstraintMatrix(NodeData, Nodes, EdgeData, Edges, GroundNodes, NN, EI, EJ):
    ####
    # Builds constraint matrix, 
    # for constraints on edge voltage drops: 1 at input node index, -1 at output and voltage drop at NN+1 index, for every row
    # For constraints on node voltages: 1 at constrained node index, voltage at NN+1 index, for every row
    # For ground nodes: 1 at ground node index, 0 else.
    # Inputs
    #
    # NodeData    = 1D np.array at length as "Nodes" corresponding to pressures at each node from "Nodes"
    # Nodes       = 1D np.array of nodes that have a constraint
    # EdgeData    = 1D np.array at length "Edges" corresponding to pressure drops at each edge from "Edges"
    # Edges       = 1D np.array of edges that have a constraint
    # GroundNodes = 1D np.array of nodes that have a constraint of ground (outlet)
    # NN          = int, number of nodes in network
    # EI          = 1D np.array of nodes at each edge beginning
    # EJ          = 1D np.array of nodes at each edge ending corresponding to EI
    #
    # outputs
    #
    # Cstr_full = 2D array sized [Constraints, NN + 1] representing constraints on nodes and edges. last column is value of constraint
    #             (p value of row contains just +1. pressure drop if row contains +1 and -1)
    # Cstr      = 2D array without last column (which is f from Rocks and Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
    # f         = constraint vector (from Rocks and Katifori 2018)
    
    # ground nodes
    csg = len(GroundNodes)
    idg = arange(csg)
    CStr = zeros([csg, NN+1])
    CStr[idg, GroundNodes] = +1.
    CStr[:, NN] = 0.
    
    # constrained node pressures
    if len(Nodes):
        csn = len(Nodes)
        idn = arange(csn)
        SN = zeros([csn, NN+1])
        SN[idn, Nodes] = +1.
        SN[:, NN] = NodeData
        CStr = np.r_[CStr, SN]
    
    # constrained edge pressure drops
    if len(Edges):
        cse = len(Edges)
        ide = arange(cse)
        SE = zeros([cse, NN+1])
        SE[ide, EI[Edges]] = +1.
        SE[ide, EJ[Edges]] = -1.
        SE[:, NN] = EdgeData
        CStr = np.r_[CStr, SE]
        
    # last column of CStr is vector f 
    f = zeros([NN + len(CStr), 1])
    f[NN:,0] = CStr[:,-1]

    return CStr, CStr[:,:-1], f

def build_input_output_and_fixed(task_type, row, NGrid):
    """
    build_input_output_and_fixed builds the input and output pairs and fixed node values

    inputs:
    task_type - str, type of learning task the network should solve
                'Allostery_one_pair' = 1 pair of input and outputs
                'Allostery' = 2 pairs of input and outputs
                'XOR' = 2 inputs and 2 outputs. difference between output nodes encodes the XOR result of the 2 inputs
                'Channeling_diag' = 1st from input to diagonal output, then from output to 2 perpindicular nodes. 
                                    test from input to output
                'Channeling_straight' = 1st from input to output on same column, then from output to 2 perpindicular nodes. 
                                        test from input to output (same as 1st)
                'Counter' = column of cells bottomost and topmost nodes are input/output (switching), 
                            rightmost nodes (1 each row) ground. more about the task in "_counter.ipynb".
    row       - int, # of row (and column) of cell in network from which input and output are considered
    NGrid     - int, row dimension of cells in network

    outputs:
    input_output_pairs - np.array of input and output node pairs. State.flow_iterate() will know how to handle them.
    fixed_nodes        - np.array of nodes with fixed values, for 'XOR' task. default=0
    """
    if task_type == 'Allostery' or task_type == 'XOR' or task_type=='Flow_clockwise':
        input_output_pairs = np.array([[(row*NGrid+(row+1))*5-1, (NGrid*(NGrid-(row+1))+(row+1))*5-1], 
                                      [(NGrid*(NGrid-row)-row)*5-1, ((row+1)*NGrid-row)*5-1]])
        if task_type == 'XOR':
            fixed_nodes = np.array([(NGrid*int(np.floor(NGrid/2))+2)*5-1,
                                         (NGrid*int(np.floor(NGrid/2))+(NGrid-row))*5-1])
        else:
            fixed_nodes = np.array([])
    elif task_type == 'Channeling_diag' or task_type=='Channeling_straight':
        n_every_row = 3  # divide into 2 rows of n_every_row inputs/outputs
        from_node = 0  # input column
        to_node = 2  # output column
        all_nodes_bot = np.arange((row*NGrid+(row+1))*5-1,((row+1)*NGrid-row)*5, 5)  # bottom nodes
        all_nodes_top = np.arange((NGrid*(NGrid-(row+1))+(row+1))*5-1, (NGrid*(NGrid-row)-row)*5, 5)  # top nodes
        idx = np.round(np.linspace(0, len(all_nodes_bot) - 1, n_every_row)).astype(int)
        input_output_pairs = np.array([all_nodes_bot[idx], all_nodes_top[idx]])
        if task_type == 'Channeling_diag':  # output diagonal to input
            input_output_pairs = array([[input_output_pairs[0, from_node], input_output_pairs[1, to_node]], 
                                       [input_output_pairs[0, to_node], input_output_pairs[1, from_node]]])
        elif task_type == 'Channeling_straight':  # output in same row as input
            input_output_pairs = array([[input_output_pairs[0, from_node], input_output_pairs[1, from_node]], 
                                       [input_output_pairs[0, to_node], input_output_pairs[1, to_node]]])
                
        fixed_nodes = np.array([])
    elif task_type == 'Counter':
        input_output_pairs = np.array([[4, NGrid*5-1], [NGrid*5-1, 4]])
        fixed_nodes = np.array([np.linspace(7, (NGrid-1)*5+2, NGrid-1, dtype=int)])
    elif task_type == 'Allostery_one_pair': 
        input_output_pairs = np.array([[(row*NGrid+int(np.ceil(NGrid/2)))*5-1, 
                                        (NGrid*(NGrid-(row+1))+int(np.ceil(NGrid/2)))*5-1], 
                                      [(row*NGrid+int(np.ceil(NGrid/2)))*5-1, 
                                        (NGrid*(NGrid-(row+1))+int(np.ceil(NGrid/2)))*5-1]])
        fixed_nodes = np.array([])

    return input_output_pairs, fixed_nodes

