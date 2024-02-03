import numpy as np
from numpy import zeros as zeros
from numpy import arange as arange

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
    # Cstr      = 2D array not including last column (which is f from Rocks and Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
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
