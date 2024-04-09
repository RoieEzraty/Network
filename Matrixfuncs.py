import numpy as np
import numpy.random as rand
import copy
from numpy import array as array
from numpy import arange as arange
from numpy import zeros as zeros
 
 
def buildL(DM, K_mat, Cstr, NN):
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
    L_bar = zeros([NN + len(Cstr), NN + len(Cstr)])
    L_bar[NN:,:NN] = Cstr  # the bottom most rows of augmented L are the constraints
    L_bar[:NN,NN:] = Cstr.T  # the rightmost columns of augmented L are the constraints
    L_bar[:NN,:NN] = L  # The topmost and leftmost part of augmented L are the basic L
    return L, L_bar
  
  
def build_incidence(Variabs):
    """
    Builds incidence matrix DM as np.array [NEdges, NNodes]
    its meaning is 1 at input node and -1 at outpus for every row which resembles one edge.

    input (extracted from Variabs input):
    a        - N cells in vertical direction of lattice, int
    b        - N cells in horizontal direction of lattice, int
    typ      - type of lattice (Nachi style or mine) str
    Periodic - 1 if lattice is periodic, 0 if not

    output:
    EI, EJ     - 1D np.arrays sized NEdges such that EI[i] is node connected to EJ[i] at certain edge
    EIEJ_plots - EI, EJ divided to pairs for ease of use
    DM         - Incidence matrix as np.array [NEdges, NNodes]
    NE         - NEdges, int
    NN         - NNodes, int
    """

    a = Variabs.NGrid
    b = Variabs.NGrid
    typ = Variabs.net_typ
    Periodic = Variabs.Periodic

    if typ == 'Cells':  # Roie style
        
        NN = 5*a*a  # 5 nodes in every cell 

        EI = []
        EJ = []

        # The cells individually
        for i in range(a*a):
            for j in range(4):
                EI.append(5*i+j)
                EJ.append(5*i+4)

        # Connecting them
        for i in range(a-1):
            for j in range(a-1):
                EI.append(5*i*a + 5*j + 2)
                EJ.append(5*i*a + 5*j + 5)
                EI.append(5*i*a + 5*j + 3)
                EJ.append(5*(i+1)*a + 5*j + 1)
            EI.append(5*(i+1)*a - 2)
            EJ.append(5*(i+2)*a - 4)
        for j in range(a-1):
            EI.append(5*(a-1)*a + 5*j + 2)
            EJ.append(5*(a-1)*a + 5*j + 5)


    elif typ == 'Nachi':  # Nachi style
        
        NN = a*b # + 1
        
        EI = []
        EJ = []
        for i in range(b-1):
            for j in range(a-1):
                EI.append(i*a + j)
                EJ.append(i*a + j + 1)
                EI.append(i*a + j)
                EJ.append((i+1)*a + j)
            EI.append(i*a + a-1)
            EJ.append((i+1)*a + a-1)
        for j in range(a-1):
                EI.append((b-1)*a + j)
                EJ.append((b-1)*a + j + 1)
        NE0 = len(EI)
                
        if Periodic:
            for j in range(a):
                EI.append(j)
                EJ.append((b-1)*a + j)     
                
            for i in range(b):
                EI.append(i*a)
                EJ.append(i*a + a-1) 
                
    EI = array(EI)
    EJ = array(EJ)
    NE = len(EI)

    # delete unrelated boundary edges and correct for nodes, only for Nachi style
    if typ == 'Nachi':
        # boundaries are not connected, remove them
        boundary_edges = arange(1, NE-a, 2*a-1)  # left boundary
        boundary_edges = np.append(boundary_edges, arange(0, 2*a-2, 2))  # top boundary
        boundary_edges = np.append(boundary_edges, arange(2*a-2, NE-a+1, 2*a-1))  # right boundary
        boundary_edges = np.append(boundary_edges, arange(NE-a+1, NE, 1))  # bottom boundary


        EI = np.delete(EI, boundary_edges)
        EJ = np.delete(EJ, boundary_edges)

        EI[EI>a*(b-1)] = EI[EI>a*(b-1)] - 1
        EJ[EJ>a*(b-1)] = EJ[EJ>a*(b-1)] - 1
        EI[EI>a-2] = EI[EI>a-2]-1
        EJ[EJ>a-2] = EJ[EJ>a-2]-1
        EI = EI - 1
        EJ = EJ - 1
            
        NE = len(EI)
        NN = NN - 4
            
    # for plots
    EIEJ_plots = [(EI[i], EJ[i]) for i in range(len(EI))]
    
    DM = zeros([NE, NN])  # Incidence matrix
    for i in range(NE):
        DM[i,int(EI[i])] = +1.
        DM[i,int(EJ[i])] = -1.
        
    return EI, EJ, EIEJ_plots, DM, NE, NN


def ChangeKFromFlow(u, thresh, K, NGrid, K_change_scheme='marbles_pressure', K_max=1, K_min=0.5, beta=0.0):
    """
    Change conductivities of full network given velocities
    This is done by dividing the network into cells
    and changing the K's cell by cell

    input:
    u               - [NEdges, 1] array of flow through edges
    thresh          - threshold of velocity that moves the marble and changes K, float
    K               - [NEdges] 2D cubic np.array of conductivities
    NGrid           - number of cells at each side of the network
    K_change_scheme - str, scheme for how to change conductivities due to u or p. default='marbles'
    K_max           - float, value of maximal conductivity, default=1
    K_min           - float, value of minimal conductivity, default=0.5
    beta            - float, vaule for conductivity change proportional to velocity squared, default=0.0
    
    output:
    K_nxt - [NEdges] 2D cubic np.array of conductivities for next iteration
    """
    K_nxt = copy.copy(K)

    if K_change_scheme == 'propto_current_squared':
        u_sqrd_mean = np.mean(u**2)
        R = K ** (-1)
        R_max = K_min ** (-1)
        R_nxt = R + beta * u ** 2 / u_sqrd_mean * (R_max - R) / R_max
        K_nxt = R_nxt ** (-1)
    elif K_change_scheme == 'marbles_u' or K_change_scheme == 'marbles_pressure':    
        NCells = NGrid*NGrid  # total number of cells in network
        for i in range(NCells):  # change K's in every cell separately
            u_sub = u[4*i:4*(i+1)]  # velocities at particular cell
            K_sub = K[4*i:4*(i+1)]  # conductivities at particular cell
            K_sub_nxt = ChangeKFromFlow_singleCell(u_sub, thresh, K_sub, K_max, K_min, K_change_scheme)  # change K's at particular cell
            K_nxt[4*i:4*(i+1)] = K_sub_nxt  # put them in the right place at K_nxt
    return K_nxt

def ChangeKFromFlow_singleCell(u, thresh, K, K_max, K_min, K_change_scheme):
    """
    Change conductivities of cell as a 2D cubic np.array sized 4
    u and K are sub vectors and matrices w/4 elements representing 4 edges of single cell

    input:
    u      - 1D np.array of flow through cell edges, 4 elements
    thresh - threshold of velocity that moves the marble and changes K, float
    K      - 2D cubic array of conductivities with 4 elements on diag
    K_max  - value of maximal conductivity
    K_min  - value of minimal conductivity
    K_change_scheme - str, scheme for how to change conductivities due to u or p. default='marbles_pressure'

    output:
    K_nxt - 2D cubic array of conductivities with 4 elements on diag for next iteration
    """

    if K_change_scheme == 'marbles_pressure':  # marbles move due to pressure difference delta_p
        delta_p = u / K  # pressure difference at edge
        p_thresh = thresh / K_max  # pressure difference threshold to move marble
        u_in_ind = np.where(delta_p>p_thresh)[0]  # all indices where u enters the cell at velocity greater than threshold to move marble
        u_out_ind = np.where(delta_p==min(delta_p.T))[0]  # indices of minimal flow, possibly exiting the cell
    elif K_change_scheme == 'marbles_u':  # marbles move due to flow u
        u_in_ind = np.where(u>thresh)[0]  # all indices where u enters the cell at velocity greater than threshold to move marble
        u_out_ind = np.where(u==min(u.T))[0]  # indices if minimal flow, possibly exiting the cell

    if all(u[u_out_ind]>0):  # no flow exits the cell, it is a ground, don't put marble inside
        K_nxt = K_max*np.ones([4])
    else:  # normal cell, not ground
        pick_u_out = [u_out_ind[rand.randint(0, len(u_out_ind))]]  # if two edges have exactly the same output flow, choose random one
        # check if there is flow inwards in edge where marble is at. then it has to move
        cond1 = len(list(set(np.where(K==K_min)[0]) & set(u_in_ind)))>0  
        cond2 = all(K == K_max)  # marble is in middle of cell (happens only at first simulation iteration)
        cond3 = len(u_in_ind) != 0  # inflow too weak to move marble
        if (cond1 or cond2) and cond3:  # if flow moves marble from one edge (1st cond) or middle (2nd cond), put lowest conductivity there
        # if np.where(K==K_min)[0] == u_in or all(K == K_max):
            K_nxt = K_max*np.ones([4])
            K_nxt[pick_u_out] = K_min 
        else:  # flow does not change conductivity
            K_nxt = copy.copy(K)
    return K_nxt

def K_by_cells(K, K_min, NGrid):
    """
    K_by_cells find location of marble at each cell

    inputs:
    K     - 1D array of conductivities
    K_min - float, minimal conductivity. This is where a marble is at.
    NGrid - number of cells in network is NGrid X NGrid

    outputs:
    K_cells - 2D array sized [# cells, iters] with positions of marbles in each cell (0=middle, 1=left, 2=bot, 3=right, 4=top)
    """
    K_cells = np.zeros([NGrid**2, ])
    for i in range(NGrid**2):
        cell = K[4*i:4*(i+1)]
        marble_place = np.where(cell==K_min)[0]
        if len(marble_place) == 0:
            K_cells[i] = 0
        else:
            K_cells[i] = marble_place + 1
    return K_cells