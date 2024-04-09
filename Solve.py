import numpy as np
import numpy.random as rand
from numpy.linalg import inv as inv

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

def create_randomized_u(BigClass, NE, frac_moved, u_thresh, noise):
    """
    create_randomized_u
    """
    u_zeros = np.zeros([NE])  # flow field of zeros for calculation of diffusive flow
    u_rand = (1+frac_moved) * u_thresh * 2 * (rand.random([NE])-1/2)  # fictional velocity field just to move marble randomly
    if noise == 'rand_K':  # add randomized flow to diffusive
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

    return u_rand

