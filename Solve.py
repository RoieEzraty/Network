import numpy as np
from numpy.linalg import inv as inv

def Solve_flow(L_bar, EI, EJ, K, f):
    """
    Solves for the pressure at nodes and flow at edges, given Lagrangian etc.
    flow at edge defined as difference in pressure between input and output nodes time conductivity at each edge
    
    input:
    L_bar - Full augmented Lagrangian, 2D np.array sized [NNodes + constraints]
    EI, EJ - 1D np.arrays sized NEdges such that EI[i] is node connected to EJ[i] at certain edge
    K - NEdges] 2D cubic np.array of conductivities
    f - constraint vector (from Rocks and Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
    
    output:
    p - hydrostatic pressure, 1D np.array sized NNodes 
    u - velocity through each edge 1D np.array sized len(EI) 
    """
    
    # Inverse Lagrangian
    IL_bar = inv(L_bar)
    
    # pressure p and velocity u
    p = np.dot(IL_bar,f)
    u = ((p[EI] - p[EJ]).T*K)[0]
    
    return p, u
    
