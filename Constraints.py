import numpy as np
import numpy.random as rand
from numpy import zeros as zeros
from numpy import arange as arange
from numpy import array as array
import copy

def ConstraintMatrix(NodeData, Nodes, EdgeData, Edges, GroundNodes, NN, EI, EJ):
    ####
    # Builds constraint matrix, 
    # for constraints on edge voltage drops: 1 at input node index, -1 at output and voltage drop at NN+1 index, for every row
    # For constraints on node voltages: 1 at constrained node index, voltage at NN+1 index, for every row
    # For ground nodes: 1 at ground node index, 0 else.
    # Inputs
    #
    # NodeData    = 1D array at length as "Nodes" corresponding to pressures at each node from "Nodes"
    # Nodes       = 1D array of nodes that have a constraint
    # EdgeData    = 1D array at length "Edges" corresponding to pressure drops at each edge from "Edges"
    # Edges       = 1D array of edges that have a constraint
    # GroundNodes = 1D array of nodes that have a constraint of ground (outlet)
    # NN          = int, number of nodes in network
    # EI          = 1D array of nodes at each edge beginning
    # EJ          = 1D array of nodes at each edge ending corresponding to EI
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

def build_input_output_and_ground(task_type, sub_task_type, row, NGrid, Nin=2, Nout=2):
    """
    build_input_output_and_ground builds the input and output pairs and ground node values

    inputs:
    task_type     - str, type of learning task the network should solve
                    'Allostery_one_pair' = 1 pair of input and outputs
                    'Allostery' = 2 pairs of input and outputs
                    'XOR' = 2 inputs and 2 outputs. difference between output nodes encodes the XOR result of the 2 inputs
                    'Channeling_diag' = 1st from input to diagonal output, then from output to 2 perpindicular nodes. 
                                        test from input to output
                    'Channeling_straight' = 1st from input to output on same column, then from output to 2 perpindicular nodes. 
                                            test from input to output (same as 1st)
                    'Counter' = column of cells bottomost and topmost nodes are input/output (switching), 
                                rightmost nodes (1 each row) ground. more about the task in "_counter.ipynb".
    sub_task_type - str, another specification of task, for regression whether there are 2 outputs or not etc.
    row           - int, # of row (and column) of cell in network from which input and output are considered
    NGrid         - int, row dimension of cells in network
    Nin           - int, # input nodes
    Nout          - int, # output nodes

    outputs:
    input_nodes_lst  - array of all input nodes in task, even when they switch roles. State.flow_iterate() will handle them.
    ground_nodes_lst - array of all output nodes in task, even when they switch roles. State.flow_iterate() will handle them.
    output_nodes        - array of nodes with fixed values, for 'XOR' task. default=0
    """
    if task_type == 'dual_no_cell':
        input_nodes_lst = array([i for i in range(Nin)])  # input nodes are first ones named
        output_nodes = array([Nin + i for i in range(Nout)])  # output nodes are named later
        ground_nodes_lst = array([Nin + Nout])  # 1 last node is ground
    if task_type == 'Allostery' or task_type == 'XOR' or task_type=='Flow_clockwise':
        input_nodes_lst = array([(row*NGrid+(row+1))*5-1, (NGrid*(NGrid-row)-row)*5-1])
        ground_nodes_lst = array([(NGrid*(NGrid-(row+1))+(row+1))*5-1, ((row+1)*NGrid-row)*5-1])
        # input_output_pairs = array([[(row*NGrid+(row+1))*5-1, (NGrid*(NGrid-(row+1))+(row+1))*5-1], 
        #                               [(NGrid*(NGrid-row)-row)*5-1, ((row+1)*NGrid-row)*5-1]])
        if task_type == 'XOR':
            output_nodes = array([(NGrid*int(np.floor(NGrid/2))+2)*5-1, (NGrid*int(np.floor(NGrid/2))+(NGrid-row))*5-1])
        else:
            output_nodes = array([])
    elif task_type == 'Allostery_contrastive':
        input_nodes_lst = array([(row*NGrid+(row+1))*5-1])
        ground_nodes_lst = array([(NGrid*(NGrid-row)-row)*5-1])
        output_nodes = array([((row+1)*NGrid-row)*5-1, (NGrid*(NGrid-(row+1))+(row+1))*5-1])
    elif task_type == 'Regression_contrastive':
        input_nodes_lst = array([(row*NGrid+(row+1))*5-1, (NGrid*(NGrid-row)-row)*5-1])
        if sub_task_type == '2in2out':
            ground_nodes_lst = array([int((np.ceil(NGrid*NGrid/2))*5-1)])  # These will be ground
            output_nodes = array([(NGrid*(NGrid-(row+1))+(row+1))*5-1, ((row+1)*NGrid-row)*5-1])  # These will be measured fixed nodes
        else:
            ground_nodes_lst = array([((row+1)*NGrid-row)*5-1])  # These will be ground
            output_nodes = array([(NGrid*(NGrid-(row+1))+(row+1))*5-1])  # These will be measured fixed nodes
    elif task_type == 'Channeling_diag' or task_type=='Channeling_straight' or task_type=='Iris':
        n_every_row = 4  # divide into 2 rows of n_every_row inputs/outputs
        from_node = 0  # input column
        to_node = 3  # output column
        all_nodes_bot = np.arange((row*NGrid+(row+1))*5-1,((row+1)*NGrid-row)*5, 5)  # bottom nodes
        all_nodes_top = np.arange((NGrid*(NGrid-(row+1))+(row+1))*5-1, (NGrid*(NGrid-row)-row)*5, 5)  # top nodes
        idx = np.round(np.linspace(0, len(all_nodes_bot) - 1, n_every_row)).astype(int)
        nodes_bot = all_nodes_bot[idx]
        nodes_top = all_nodes_top[idx]
        if task_type == 'Channeling_diag':  # output diagonal to input
            input_nodes_lst = array([nodes_bot[from_node], nodes_bot[to_node]])
            ground_nodes_lst = array([nodes_top[to_node], nodes_top[from_node]])
        elif task_type == 'Channeling_straight':  # output in same row as input
            input_nodes_lst = array([nodes_bot[from_node], nodes_bot[to_node]])
            ground_nodes_lst = array([nodes_top[from_node], nodes_top[to_node]])
        elif task_type == 'Iris':
            input_nodes_lst = array([nodes_top[0], nodes_bot[1], nodes_top[2], nodes_bot[3]])
            ground_nodes_lst = array([nodes_bot[0], nodes_top[1], nodes_bot[2]])
        output_nodes = array([])
    elif task_type == 'Counter':
        input_nodes_lst = array([4, NGrid*5-1])
        ground_nodes_lst = array([NGrid*5-1, 4])
        output_nodes = np.linspace(5*1+2, 5*(NGrid-2)+2, NGrid-2, dtype=int)
    elif task_type == 'Memristor':
        input_nodes_lst = array([4])
        ground_nodes_lst = array([NGrid*5-1])
        output_nodes = array([])
    elif task_type == 'Allostery_one_pair': 
        input_nodes_lst = array([(row*NGrid+int(np.ceil(NGrid/2)))*5-1])
        ground_nodes_lst = array([(NGrid*(NGrid-(row+1))+int(np.ceil(NGrid/2)))*5-1])
        output_nodes = array([])
    return input_nodes_lst, ground_nodes_lst, output_nodes

def Constraints_afo_task(Variabs, Strctr, State, sim_type, i, train_sample):
    """
    Constraints_afo_task sets up the constraints on nodes and edges for specific learning task, and for specific step.
    This comes after Setup_constraints which sets them for the whole task

    inputs:
    Variabs      - class instance that includes all variables
    Strctr       - class instance that includes network structure and related sizes
    State        - class instance that includes the state of the network for all times and related sizes
    sim_type     - simulation type: 'no marbles'     - flows from 1 input at a time and all outputs, all edges have high State.K
                                    'allostery test' - flows from 1 input at a time and all outputs, K's as in State.K
                                    'w marbles'      - (default) flows from 1 input and one output K's as in State.K, 
                                                       update conductivities as due to flow u and calculate flow again so
                                                       further update of conductivities will not change a thing
    i            - int, index from which to extract B.C.s, corresponding to some mod of the training / test step
    train_sample - 1D array, pressure values for input nodes given chosen sample from training set
    p_nudge      - 1D array of pressure values to force on output nodes, for the clamped stage in "Contrastive" task type
    """
    task_type = copy.copy(Variabs.task_type)
    flow_scheme = copy.copy(Variabs.flow_scheme)
    OutputNodeData_full = copy.copy(Strctr.OutputNodeData_full)
    OutputNodes_full = copy.copy(Strctr.OutputNodes_full)
    GroundNodes_full = copy.copy(Strctr.GroundNodes_full)
    InNodeData_full = copy.copy(Strctr.InNodeData_full)
    InNodes_full = copy.copy(Strctr.InNodes_full)
    EdgeData_full = copy.copy(Strctr.EdgeData_full)
    Edges_full = copy.copy(Strctr.Edges_full)


    if task_type == 'Allostery_contrastive' or task_type == 'Regression_contrastive':
        m = i % 2  # iterate between 1st and 2nd inputs
        if np.shape(State.p_nudge)==np.shape(OutputNodeData_full):  # account for size of user input
            State.p_nudge = np.ones(np.shape(OutputNodeData_full))*State.p_nudge
        if i % 4 > 1 and flow_scheme == 'taktak':  # flip inputs, ground and output at 2nd cycle
            if np.size(GroundNodes_full)==1:  # switch between input and output
                InNodes_full_temp = copy.copy(GroundNodes_full)  # dummy, to remember original value of InNodeData
                GroundNodes_full = copy.copy(InNodes_full)  
                InNodes_full = copy.copy(InNodes_full_temp)
            else:
                InNodes_full = copy.copy(np.flip(InNodes_full))
            if np.size(State.p_nudge)>1:  # switch between values of input
                State.p_nudge = copy.copy(np.flip(State.p_nudge))
        # if m==0 and i!=0:  # large input p, for if we want to first push, then measure
        if m==1:  # large input p, for if we want to first measure, then push           
            # nodes will be input magnified by factor concatenated to output nodes with assigned nudged pressure
            if task_type == 'Allostery_contrastive':  # no sample of new training point for allostery task
                if flow_scheme == 'dual':
                    InNodeData_full = State.p_nudge
                    OutputNodeData_full_nudged = State.outputs_dual
                else:
                    OutputNodeData_full_nudged = State.p_nudge
                NodeData, Nodes = np.append(InNodeData_full, OutputNodeData_full_nudged)*Variabs.mag_factor, np.append(InNodes_full, OutputNodes_full)
            elif task_type == 'Regression_contrastive':  # no sample of new training point for allostery task
                if flow_scheme == 'dual':
                    InNodeData_full = State.p_nudge
                    OutputNodeData_full_nudged = State.outputs_dual
                    NodeData, Nodes = np.append(InNodeData_full, OutputNodeData_full_nudged)*Variabs.mag_factor, np.append(InNodes_full, OutputNodes_full)
                else:
                    OutputNodeData_full_nudged = State.p_nudge
                    NodeData, Nodes = np.append(InNodeData_full*train_sample, OutputNodeData_full_nudged)*Variabs.mag_factor, np.append(InNodes_full, OutputNodes_full)
            else:  # input nodes multiplied by pressure for train sample
                OutputNodeData_full_nudged = State.p_nudge
                NodeData, Nodes = np.append(InNodeData_full*train_sample, OutputNodeData_full_nudged)*Variabs.mag_factor, np.append(InNodes_full, OutputNodes_full)
            print('OutputNodeData_full_nudged', OutputNodeData_full_nudged)
        else:
            if task_type == 'Allostery_contrastive':
                NodeData, Nodes = InNodeData_full, InNodes_full
            else:  # input nodes multiplied by pressure for train sample
                NodeData, Nodes = InNodeData_full*train_sample, InNodes_full
        EdgeData, Edges, GroundNodes = array([EdgeData_full[0]]), Edges_full, GroundNodes_full
        # print('m=', m, '\nNodeData ', NodeData, ', Nodes ', Nodes)
        print('\ni=', i, '\nm=', m) 
        # print('InNodeData_full', InNodeData_full)
        # print('train_sample', train_sample)
        # print('InNodes_full', InNodes_full)
        # print('OutputNodes_full', OutputNodes_full)       
    elif task_type == 'Allostery_one_pair':
        # Determine boundary conditions - only first direction for 'Allostery_one_pair'
        NodeData, Nodes, EdgeData, Edges = array([InNodeData_full[0]]), array([InNodes_full[0]]),\
                                            array([EdgeData_full[0]]), array([Edges_full[0]])
        if sim_type == 'w marbles':
            GroundNodes = array([GroundNodes_full[0]])
        else:
            GroundNodes = array([GroundNodes_full[0] for i in range(len(GroundNodes_full))])
    elif task_type=='Allostery' or task_type=='Flow_clockwise':
        m = i % 2  # iterate between 1st and 2nd inputs 
        # Determine boundary conditions - modulus 2 for 'Allostery'
        EdgeData, Edges = array([EdgeData_full[m]]), Edges_full   
        if task_type=='Flow_clockwise' and sim_type!='w marbles':  # test flow direction, use both inputs
            NodeData = array([InNodeData_full[i][0] for i in range(len(InNodeData_full))])
            Nodes = array([InNodes_full[i][0] for i in range(len(InNodes_full))])
        elif flow_scheme == 'Constrastive' or flow_scheme == 'Hebbian_like':
            if m == 0:
                NodeData, Nodes = array([InNodeData_full[m]]), array([InNodes_full[m]]) 
            else:
                NodeData, Nodes = array([InNodeData_full[m]]), array([InNodes_full[m]])
            NodeData = array([InNodeData_full[i][0] for i in range(len(InNodeData_full))])
            Nodes = array([InNodes_full[i][0] for i in range(len(InNodes_full))])
        else:  # test flow - flow from 1 direction
            NodeData, Nodes = array([InNodeData_full[m]]), array([InNodes_full[m]]) 

        if sim_type=='w marbles':
            if flow_scheme=='unidir':
                GroundNodes = array([GroundNodes_full[m]])
            else:
                GroundNodes = array([GroundNodes_full[i] for i in range(len(GroundNodes_full))])
        else:
            GroundNodes = array([GroundNodes_full[i] for i in range(len(GroundNodes_full))])

        # Simulating normal direction of flow: from input to output

        # Otherwise, if true, switch ground and input nodes every 2nd iteration
        if i % 4 > 1 and flow_scheme == 'taktak':
            Nodes = array([GroundNodes_full[m]])
            GroundNodes = array([InNodes_full[m]])
    elif task_type == 'XOR':
        # select random input and output
        m = np.where([rand.randint(0, 2), rand.randint(0, 2)])[0]
        NodeData = np.append(InNodeData_full[m], OutputNodeData_full)
        Nodes = np.append(InNodes_full[m], OutputNodes_full)
        EdgeData = array([EdgeData_full[m]])
        Edges = array([Edges_full[m]])
        GroundNodes = array([GroundNodes_full[m]])
    elif task_type == 'Channeling_diag' or task_type == 'Channeling_straight':
        # iterate between 1st and 2nd inputs (1st comes at beginning and end and 2nd comes once in the middle)
        m=i%2           
        if m == 0:
            # Determine boundary conditions
            NodeData, Nodes, EdgeData, Edges, GroundNodes = array([InNodeData_full[0]]), array([InNodes_full[0]]),\
                                                            array([EdgeData_full[0]]), array([Edges_full[0]]),\
                                                            array([GroundNodes_full[0]])
        else:
            NodeData, Nodes, EdgeData, Edges = array([InNodeData_full[0]]), array([GroundNodes_full[0]]),\
                                               array([EdgeData_full[1]]), array([Edges_full[1]])
            GroundNodes = array([InNodes_full[1], GroundNodes_full[1]])
    elif task_type == 'Counter':
        print('task is counter')
        m = i % 2  # iterate between 1st and 2nd inputs 
        # Determine boundary conditions - modulus 2 for 'Counter'
        NodeData, Nodes, EdgeData, Edges = array([InNodeData_full[m]]), array([InNodes_full[m]]),\
                                           array([EdgeData_full[m]]), Edges_full
        if m == 0:  # push from bottom (Nodes[0]), open all sides (Output), block top (no Ground)    
            GroundNodes = OutputNodes_full
            print('m==0 GroundNodes', GroundNodes)
        else:  # Push from top (Nodes[1]), open bottom (Ground[1])
            GroundNodes = array([GroundNodes_full[m]])
            print('m==1 GroundNodes', GroundNodes)
    elif task_type == 'Memristor':
        # NodeData, Nodes, EdgeData, Edges = array([InNodeData_full[0]]), array([InNodes_full[0]]),\
        #                                    array([EdgeData_full[0]]), Edges_full
        # GroundNodes = array([GroundNodes_full[0]])
        NodeData, Nodes, EdgeData, Edges = InNodeData_full, InNodes_full, array([EdgeData_full[0]]), Edges_full
        GroundNodes = GroundNodes_full
    elif task_type == 'Iris':
        rand_int = random.randint(0,np.shape(train_data)[0])  # random integer to sample from training set
        NodeData = train_data[rand_int] # Iris inputs into nodes
        GroundNodes = array([GroundNodes_full[train_target[rand_int]]])  # single only ground node corresponding to correct iris
        Nodes, EdgeData, Edges = InNodes_full, EdgeData_full, Edges_full
    return NodeData, Nodes, EdgeData, Edges, GroundNodes
