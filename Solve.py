import numpy as np
import numpy.random as rand
import copy
from numpy.linalg import inv as inv
import Matrixfuncs

def solve_flow_const_K(K, BigClass, u, Cstr, f, iters_same_BCs):
    """
    solve_flow_const_K solves the flow under given conductance configuration without changing Ks, until simulation converges

    inputs:
    K_max          - float, maximal conductance value
    NE             - int, # edges
    EI             - np.array, node number on 1st side of all edges
    u              - 1D array sized [NE + constraints, ], flow field at edges from previous solution iteration
    Cstr           - 2D array without last column, which is f from Rocks & Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116
    f              - constraint vector (from Rocks and Katifori 2018)
    iters_same_BSc - # iteration allowed under same boundary conditions (same constraints)

    outputs:
    p     - 1D array sized [NN + constraints, ], pressure at nodes at end of current iteration step
    u_nxt - 1D array sized [NE + constraints, ], flow velocity at edgses at end of current iteration step
    """

    u_nxt = copy.copy(u)

    for o in range(iters_same_BCs):	

        # create effective conductivities if they are flow dependent
        K_eff = copy.copy(K)
        if BigClass.Variabs.K_type == 'flow_dep':
            K_eff[u_nxt>0] = BigClass.Variabs.K_max
        K_eff_mat = np.eye(BigClass.Strctr.NE) * K_eff

        L, L_bar = BigClass.Solver.solve.buildL(BigClass, BigClass.Strctr.DM, K_eff_mat, Cstr, BigClass.Strctr.NN)  # Lagrangian

        p, u_nxt = BigClass.Solver.solve.Solve_flow(L_bar, BigClass.Strctr.EI, BigClass.Strctr.EJ, K_eff, f, round=10**-10)  # pressure and flow
        
        # NETfuncs.PlotNetwork(p, u_nxt, self.K, BigClass, BigClass.Strctr.EIEJ_plots, BigClass.Strctr.NN, 
        # 					 BigClass.Strctr.NE, nodes='no', edges='yes', savefig='no')

        # break the loop
        # since no further changes will be measured in flow and conductivities at end of next cycle
        if np.all(np.where(u_nxt>0)[0] == np.where(u>0)[0]):
        # if np.all(u_nxt == u):
            u = copy.copy(u_nxt)
            break
        else:
            # NETfuncs.PlotNetwork(p, u_nxt, self.K, BigClass, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE)
            u = copy.copy(u_nxt)

    return p, u_nxt

def buildL(BigClass, DM, K_mat, Cstr, NN):
    """
    Builds expanded Lagrangian with constraints 
    as in the Methods section of Rocks and Katifori 2018 (https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
    np.array cubic array sized [NNodes + Constraints]

    input:
    DM    - Incidence matrix np.array [NE, NN]
    K_mat - cubic np.array sized NE with flow conductivities on diagonal
    Cstr  - np.array sized [Constraints, NN + 1] of constraints 
    NN    - NNodes, ind

    output:
    L     - Shortened Lagrangian np.array cubic array sized [NNodes]
    L_bar - Full  augmented Lagrangian, np.array cubic array sized [NNodes + Constraints]
    """
    L = np.dot(DM.T, np.dot(K_mat, DM))
    L_bar = np.zeros([NN + len(Cstr), NN + len(Cstr)])
    L_bar[NN:,:NN] = Cstr  # the bottom most rows of augmented L are the constraints
    L_bar[:NN,NN:] = Cstr.T  # the rightmost columns of augmented L are the constraints
    L_bar[:NN,:NN] = L  # The topmost and leftmost part of augmented L are the basic L
    return L, L_bar

    

def Solve_flow(L_bar, EI, EJ, K, f, round=10**-10):
    """
    Solves for the pressure at nodes and flow at edges, given Lagrangian etc.
    flow at edge defined as difference in pressure between input and output nodes time conductivity at each edge
    
    input:
    L_bar  - Full augmented Lagrangian, 2D np.array sized [NNodes + constraints]
    EI, EJ - 1D np.arrays sized NEdges such that EI[i] is node connected to EJ[i] at certain edge
    K      - [NE] 2D cubic np.array of conductivities
    f      - constraint vector (from Rocks and Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
    round  - float, value below which the absolute value of u and p are rounded to 0. 
    
    output:
    p - hydrostatic pressure, 1D np.array sized NNodes 
    u - velocity through each edge 1D np.array sized len(EI) 
    """
    
    # Inverse Lagrangian
    IL_bar = inv(L_bar)
    
    # pressure p and velocity u
    p = np.dot(IL_bar,f)
    u = ((p[EI] - p[EJ]).T*K)[0]

    p, u = round_small(p, u)
    return p, u

def round_small(p, u):
    """
    round_small rounds values of u and p that are close to 0 to get rid of rounding problems
    """

    p[abs(p)<10**-10] = 0  # Correct for very low pressures
    u[abs(u)<10**-10] = 0  # Correct for very low velocities

    return p, u

def dot_triple(X, Y, Z):
    return np.dot(X, np.dot(Y, Z))

def create_randomized_u(BigClass, NE, frac_moved, u_thresh, noise):
    """
    create_randomized_u
    """
    u_zeros = np.zeros([NE])  # flow field of zeros for calculation of diffusive flow
    u_rand = (1+frac_moved) * u_thresh * 2 * (rand.random([NE])-1/2)  # fictional velocity field just to move marble randomly
    
    if noise == 'rand_K':  # add randomized flow to diffusive //TODO fix 
        # specific constraints for training step
        NodeData, Nodes, EdgeData, Edges, GroundNodes = BigClass.Strctr.Constraints_afo_task(BigClass, 'w marbles', 0)

        # BC and constraints as matrix
        Cstr_full, Cstr, f = Constraints.ConstraintMatrix(NodeData, Nodes, EdgeData, Edges, GroundNodes, 
                                                          BigClass.Strctr.NN, BigClass.Strctr.EI, BigClass.Strctr.EJ)  
        
        # calculate flow for no marbles
        p_diffusive, u_diffusive = self.solve_flow_const_K(BigClass, u_zeros, Cstr, f, BigClass.Variabs.iterations)
        
        # normalize randomized flow relative to diffusive amplitudes and add 
        u_rand_norm = u_rand / u_thresh * np.mean(np.abs(u_diffusive))
        u_rand = u_rand_norm + u_diffusive

    print('CPU is used')
    return u_rand

