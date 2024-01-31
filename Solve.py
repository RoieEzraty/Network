import numpy as np
from numpy.linalg import inv as inv

def Solve_flow(L_bar, EI, EJ, K, f):
    # Inverse Lagrangian
    IL_bar = inv(L_bar)
    
    # pressure p and velocity u
    p = np.dot(IL_bar,f)
    u = ((p[EI] - p[EJ]).T*K)[0]
    
    return p, u
    
