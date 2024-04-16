# import cupy as cp
import numpy as np
import copy
import Matrixfuncs

# def solve_flow_const_K(K, BigClass, u, Cstr, f, iters_same_BCs):
#     """
#     solve_flow_const_K solves the flow under given conductance configuration without changing Ks, until simulation converges

#     inputs:
#     K_max          - float, maximal conductance value
#     NE             - int, # edges
#     EI             - np.array, node number on 1st side of all edges
#     u              - 1D array sized [NE + constraints, ], flow field at edges from previous solution iteration
#     Cstr           - 2D array without last column, which is f from Rocks & Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116
#     f              - constraint vector (from Rocks and Katifori 2018)
#     iters_same_BSc - # iteration allowed under same boundary conditions (same constraints)

#     outputs:
#     p     - 1D array sized [NN + constraints, ], pressure at nodes at end of current iteration step
#     u_nxt - 1D array sized [NE + constraints, ], flow velocity at edgses at end of current iteration step
#     """

#     u_cp = cp.asarray(u)
#     u_nxt_cp = copy.copy(u_cp)
#     K_cp = cp.asarray(K)
#     DM_cp = cp.asarray(BigClass.Strctr.DM)
#     Cstr_cp = cp.asarray(Cstr)
#     f_cp = cp.asarray(f)
#     EI_cp = cp.asarray(BigClass.Strctr.EI)
#     EJ_cp = cp.asarray(BigClass.Strctr.EJ)
    

#     for o in range(iters_same_BCs):	

#         # create effective conductivities if they are flow dependent
        
#         K_eff_cp = copy.copy(K_cp)
#         if BigClass.Variabs.K_type == 'flow_dep':
#             K_eff_cp[u_nxt_cp>0] = BigClass.Variabs.K_max
#         K_eff_mat_cp = cp.eye(BigClass.Strctr.NE) * K_eff_cp

#         L_cp, L_bar_cp = BigClass.Solver.solve.buildL(BigClass, DM_cp, K_eff_mat_cp, Cstr_cp, BigClass.Strctr.NN)  # Lagrangian

#         p_cp, u_nxt_cp = BigClass.Solver.solve.Solve_flow(L_bar_cp, EI_cp, EJ_cp, K_eff_cp, f_cp, round=10**-10)  # pressure and flow
        
#         # NETfuncs.PlotNetwork(p, u_nxt, self.K, BigClass, BigClass.Strctr.EIEJ_plots, BigClass.Strctr.NN, 
#         # 					 BigClass.Strctr.NE, nodes='no', edges='yes', savefig='no')

#         # break the loop
#         # since no further changes will be measured in flow and conductivities at end of next cycle
#         # if cp.all(cp.where(u_nxt_cp>0)[0] == cp.where(u_cp>0)[0]):
#         if cp.all((u_nxt_cp>0)[0] == (u_cp>0)[0]):
#         # if np.all(u_nxt == u):
#             u_cp = copy.copy(u_nxt_cp)
#             break
#         else:
#             # NETfuncs.PlotNetwork(p, u_nxt, self.K, BigClass, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE)
#             u_cp = copy.copy(u_nxt_cp)

#     return cp.asnumpy(p_cp), cp.asnumpy(u_nxt_cp)

# def buildL(BigClass, DM, K_mat, Cstr, NN):
#     """
#     Builds expanded Lagrangian with constraints 
#     as in the Methods section of Rocks and Katifori 2018 (https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
#     np.array cubic array sized [NNodes + Constraints]

#     input:
#     DM    - Incidence matrix np.array [NE, NN]
#     K_mat - cubic np.array sized NE with flow conductivities on diagonal
#     Cstr  - np.array sized [Constraints, NN + 1] of constraints 
#     NN    - NNodes, ind

#     output:
#     L     - Shortened Lagrangian np.array cubic array sized [NNodes]
#     L_bar - Full  augmented Lagrangian, np.array cubic array sized [NNodes + Constraints]
#     """
#     # L = BigClass.Solver.solve.dot_triple(DM.T, K_mat, DM)
#     L = cp.dot(DM.T, cp.dot(K_mat, DM))
#     L_bar = cp.zeros([NN + len(Cstr), NN + len(Cstr)])
#     L_bar[NN:,:NN] = Cstr  # the bottom most rows of augmented L are the constraints
#     L_bar[:NN,NN:] = Cstr.T  # the rightmost columns of augmented L are the constraints
#     L_bar[:NN,:NN] = L  # The topmost and leftmost part of augmented L are the basic L
#     return L, L_bar

def Solve_flow(L_bar, EI, EJ, K, f, round=10**-10):
    """
    Solves for the pressure at nodes and flow at edges using CUDA.
    """
    # Inverse Lagrangian
    L_bar = cp.asarray(L_bar)
    EI = cp.asarray(EI)
    EJ = cp.asarray(EJ)
    K = cp.asarray(K)
    f = cp.asarray(f)

    IL_bar = cp.linalg.inv(L_bar)
    
    # pressure p and velocity u
    p_cp = cp.dot(IL_bar, f)
    u_cp = ((p_cp[EI] - p_cp[EJ]).T * K)[0]

    p = cp.asnumpy(p_cp)
    u = cp.asnumpy(u_cp)

    p, u = round_small(p, u, round)

    # IL_bar = cp.linalg.inv(L_bar)
    
    # # pressure p and velocity u
    # p_cp = cp.dot(IL_bar, f)
    # u_cp = ((p_cp[EI] - p_cp[EJ]).T * K)[0]
    
    return p, u

def round_small(p, u, threshold):
    """
    Rounds values of u and p that are close to 0 to get rid of rounding problems.
    """
    # p[cp.abs(p) < threshold] = 0
    # u[cp.abs(u) < threshold] = 0

    p[np.abs(p) < threshold] = 0
    u[np.abs(u) < threshold] = 0

    return p, u

def dot_triple(X, Y, Z):
    X_cp = cp.asarray(X)
    Y_cp = cp.asarray(Y)
    Z_cp = cp.asarray(Z)
    return cp.asnumpy(cp.dot(X_cp, cp.dot(Y_cp, Z_cp)))


def create_randomized_u(BigClass, NE, frac_moved, u_thresh, noise):
    """
    Generates a randomized velocity field using CuPy for GPU acceleration.
    Converts outputs back to NumPy arrays.

    Parameters:
    - NE: Number of elements in the velocity field (int).
    - frac_moved: Fraction to adjust the velocity field by (float).
    - u_thresh: Threshold for the velocity (float).
    - noise: Indicator for the type of noise to add (str, currently unused but placeholder for future functionality).

    Returns:
    - u_rand: Randomized velocity field (NumPy array).
    """
    # Note: Input conversion isn't necessary as parameters are simple types or the function generates CuPy arrays internally
    u_rand = (1 + frac_moved) * u_thresh * 2 * (cp.random.random(NE) - 0.5)
    
    # Assuming other steps that might use `noise` or other computations would also use CuPy for consistency
    
    # Convert the CuPy array back to a NumPy array before returning

    print('GPU is used')

    return cp.asnumpy(u_rand)