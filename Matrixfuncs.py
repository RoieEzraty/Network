import numpy as np
import numpy.random as rand
import copy
import itertools
import DatasetManipulations
from numpy import array as array
from numpy import arange as arange
from numpy import zeros as zeros
 
def buildL(BigClass, DM, K_mat, Cstr, NN):
    """
    Builds expanded Lagrangian with constraints 
    as in the Methods section of Rocks and Katifori 2018 (https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
    np.array cubic array sized [NNodes + Constraints]

    input:
    BigClass - class instance including the user variables (Variabs), network structure (Strctr) and networkx (NET) and network state (State) class instances
               I will not go into everything used from there to save space here.
    DM       - Incidence matrix np.array [NE, NN]
    K_mat    - cubic np.array sized NE with flow conductivities on diagonal
    Cstr     - np.array sized [Constraints, NN + 1] of constraints 
    NN       - NNodes, ind

    output:
    L     - Shortened Lagrangian np.array cubic array sized [NNodes]
    L_bar - Full  augmented Lagrangian, np.array cubic array sized [NNodes + Constraints]
    """
    L = BigClass.Solver.solve.dot_triple(DM.T, K_mat, DM)
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
    print('Periodic', Periodic)

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

    elif typ == 'oneCol':

        NN = 5*a  # 5 nodes in every cell 

        EI = []
        EJ = []

        # The cells individually
        for i in range(a):
            for j in range(4):
                EI.append(5*i+j)
                EJ.append(5*i+4)

        # Connecting them
    
        for j in range(a-1):
            EI.append(5*j + 3)
            EJ.append(5*(j+1) + 1)

    elif typ == 'Nachi':  # Nachi style
        
        NN = a*b # + 1
        
        EI = []
        EJ = []
        for i in range(b-1):
            for j in range(a-1):
                EI.append(i*a + j)
                EJ.append(i*a + j + 1)  # connecting to the right
                EI.append(i*a + j)
                EJ.append((i+1)*a + j)  # connecting down one row
            EI.append(i*a + a-1)  # rightmost nodes only down
            EJ.append((i+1)*a + a-1)
        for j in range(a-1):  # bottom row only to the right
                EI.append((b-1)*a + j)
                EJ.append((b-1)*a + j + 1)
        NE0 = len(EI)
                
        # if Variabs.Periodic==True:  # This one always returns true for some reason
        #     print('Periodic True')
        #     for j in range(a):
        #         EI.append(j)
        #         EJ.append((b-1)*a + j)     
        #     for i in range(b):
        #         print(f'under periodic i={i}, j={j}')
        #         print('EI', EI)
        #         print('EJ', EJ)
        #         EI.append(i*a)
        #         EJ.append(i*a + a-1) 

    elif typ == 'FC':

        NN = len(Variabs.input_nodes_lst) + len(Variabs.output_nodes) + 1
        ground_node = copy.copy(NN) - 1
        EI = []
        EJ = []

        # connect inputs to outputs
        for i, inNode in enumerate(Variabs.input_nodes_lst):
            for j, outNode in enumerate(Variabs.output_nodes):
                EI.append(inNode)
                EJ.append(outNode)

        # connect input to ground
        for i, inNode in enumerate(Variabs.input_nodes_lst):
            EI.append(inNode)
            EJ.append(ground_node)

        # connect output to ground
        for i, outNode in enumerate(Variabs.output_nodes):
            EI.append(outNode)
            EJ.append(ground_node)
                
    EI = array(EI)
    EJ = array(EJ)
    NE = len(EI)
    print('EI', EI)
    print('EJ', EJ)
    print('NE', NE)

    # delete unrelated boundary edges and correct for nodes, only for Nachi style
    # if typ == 'Nachi':
    #     # boundaries are not connected, remove them
    #     boundary_edges = arange(1, NE-a, 2*a-1)  # left boundary
    #     boundary_edges = np.append(boundary_edges, arange(0, 2*a-2, 2))  # top boundary
    #     boundary_edges = np.append(boundary_edges, arange(2*a-2, NE-a+1, 2*a-1))  # right boundary
    #     boundary_edges = np.append(boundary_edges, arange(NE-a+1, NE, 1))  # bottom boundary

    #     EI = np.delete(EI, boundary_edges)
    #     EJ = np.delete(EJ, boundary_edges)

    #     EI[EI>a*(b-1)] = EI[EI>a*(b-1)] - 1
    #     EJ[EJ>a*(b-1)] = EJ[EJ>a*(b-1)] - 1
    #     EI[EI>a-2] = EI[EI>a-2]-1
    #     EJ[EJ>a-2] = EJ[EJ>a-2]-1
    #     EI = EI - 1
    #     EJ = EJ - 1
            
    #     NE = len(EI)
    #     NN = NN - 4
            
    # for plots
    EIEJ_plots = [(EI[i], EJ[i]) for i in range(len(EI))]
    
    DM = zeros([NE, NN])  # Incidence matrix
    for i in range(NE):
        DM[i,int(EI[i])] = +1.
        DM[i,int(EJ[i])] = -1.
        
    return EI, EJ, EIEJ_plots, DM, NE, NN


def ChangeKFromFlow(u, thresh, K, K_backg, NGrid, K_change_scheme='marbles_pressure', allowed_cells=[], K_max=1, K_min=0.5, beta=0.0):
    """
    Change conductivities of full network given velocities
    This is done by dividing the network into cells
    and changing the K's cell by cell

    input:
    u               - [NEdges, 1] array of flow through edges
    thresh          - threshold of velocity that moves the marble and changes K, float
    K               - [NEdges] 2D cubic np.array of conductivities
    K_backg         - [NEdges] 2D cubic np.array of background conductivities, as if no marbles
    NGrid           - number of cells at each side of the network
    K_change_scheme - str, scheme for how to change conductivities due to u or p. default='marbles_pressure'
    allowed_cells   - np.array of ints denoting the cells whose K's are allowed to change
    K_max           - float, value of maximal conductivity, default=1
    K_min           - float, value of minimal conductivity, default=0.5
    beta            - float, vaule for conductivity change proportional to velocity squared, default=0.0
    
    output:
    K_nxt - [NEdges] 2D cubic np.array of conductivities for next iteration
    """
    K_nxt = copy.copy(K)

    if K_change_scheme == 'propto_current_squared':  # resistances (1/conductances) are proportional to Q^2
        u_sqrd_mean = np.mean(u**2)
        R = K ** (-1)
        R_max = K_min ** (-1)
        R_nxt = R + beta * u ** 2 / u_sqrd_mean * (R_max - R) / R_max
        K_nxt = R_nxt ** (-1)
    # if conductances change due to delta p or Q    
    elif K_change_scheme == 'marbles_u' or K_change_scheme == 'marbles_pressure' or K_change_scheme=='marbles_p_lower_l_half' or K_change_scheme == 'marbles_p_upper_l_half':  
        # NCells = NGrid*NGrid  # total number of cells in network
        NCells = int(len(K_nxt)/4)  # total number of cells in network
        for i in range(NCells):  # change K's in every cell separately
            if (K_change_scheme=='marbles_p_lower_l_half' or K_change_scheme=='marbles_p_upper_l_half') and i not in allowed_cells:  # skip update of the K's in that cell since it is not at lower left half of domain
                # print(f'cell #{i} skipped')
                pass
            else:  # update K's cell by cell
                u_sub = u[4*i:4*(i+1)]  # velocities at particular cell
                K_sub = K[4*i:4*(i+1)]  # conductivities at particular cell
                if type(thresh) == np.ndarray:
                    thresh_sub = thresh[4*i:4*(i+1)]
                else:
                    thresh_sub = copy.copy(thresh)
                K_backg_sub = K_backg[4*i:4*(i+1)]  # background conductivities at particular cell
                # change K's at particular cell
                K_sub_nxt = ChangeKFromFlow_singleCell(u_sub, thresh_sub, K_sub, K_backg_sub, K_max, K_min, K_change_scheme)
                K_nxt[4*i:4*(i+1)] = K_sub_nxt  # put them in the right place at K_nxt
    return K_nxt


def ChangeKFromFlow_singleCell(u, thresh, K, K_backg, K_max, K_min, K_change_scheme):
    """
    Change conductivities of cell as a 2D cubic np.array sized 4
    u and K are sub vectors and matrices w/4 elements representing 4 edges of single cell

    input:
    u               - 1D np.array of flow through cell edges, 4 elements
    thresh          - threshold of velocity that moves the marble and changes K, float
    K               - [2, 2] 2D cubic np.array of conductivities
    K_backg         - [2, 2] 2D cubic np.array of background conductivities, as if no marbles
    K_max           - value of maximal conductivity
    K_min           - value of minimal conductivity
    K_change_scheme - str, scheme for how to change conductivities due to u or p. default='marbles_pressure'

    output:
    K_nxt - 2D cubic array of conductivities with 4 elements on diag for next iteration
    """

    if K_change_scheme == 'marbles_pressure' or K_change_scheme == 'marbles_p_lower_l_half' or K_change_scheme == 'marbles_p_upper_l_half':  # marbles move due to pressure difference delta_p
        # u_out_ind = np.where(delta_p==min(delta_p.T))[0]  # indices of minimal flow, possibly exiting the cell
        delta_p = u / K  # pressure difference at edge
        p_thresh = thresh / K_max  # pressure difference threshold to move marble
        u_in_ind = np.where(delta_p>p_thresh)[0]  # all indices where u enters the cell at velocity 
                                                  # greater than threshold to move marble
        u_out_ind = np.where(u==min(u.T))[0]  # indices if minimal flow, possibly exiting the cell
    elif K_change_scheme == 'marbles_u':  # marbles move due to flow u
        u_in_ind = np.where(u>thresh)[0]  # all indices where u enters the cell at velocity 
                                          # greater than threshold to move marble
        u_out_ind = np.where(u==min(u.T))[0]  # indices if minimal flow, possibly exiting the cell

    K_nxt = copy.copy(K_backg)
    # no flow exits the cell, it is a ground, don't put marble inside, else
    if not(all(u[u_out_ind]>0)):  # normal cell, not ground
    #     continue
    # else:  # normal cell, not ground
        pick_u_out = [u_out_ind[rand.randint(0, len(u_out_ind))]]  # if two edges have exactly the same output flow, 
                                                                   # choose random one
        # check if there is flow inwards in edge where marble is at. then it has to move
        cond1 = len(list(set(np.where(K==K_min)[0]) & set(u_in_ind)))>0  
        # cond2 = all(K == K_max)  # marble is in middle of cell (happens only at first simulation iteration)
        cond2 = all(K > K_min)  # marble is in middle of cell (happens only at first simulation iteration)
        cond3 = len(u_in_ind) != 0  # inflow too weak to move marble
        if (cond1 or cond2) and cond3:  # if flow moves marble from edge (1st cond) or middle (2nd cond), put lowest K there
        # if np.where(K==K_min)[0] == u_in or all(K == K_max):
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


def create_regression_dataset(data_size_each_input, Nin, desired_p_frac, train_frac):
    """
    create_regression_dataset creates np.arrays that represent the input pressures (regression_data) and their desired output pressures
    (regression_target) for regression task with number of samples data_size_each_axis**Nin

    inputs:
    data_size_each_input - int, # of pressures in each input channel to sample from, ranging 0-1
    Nin                  - int, # of input nodes
    desred_p_frac        - np.array sizes [Nout, Nin] corresponding to regression coefficient matrix of the task, i.e. x = M p_in
    train_frac           - float ranging 0-1, fraction of datapoints out of data_size_each_axis**Nin data points used to train,
                           the rest are for test

    outputs:
    train_data   - np.array sized [data_size_each_axis**Nin*train_frac, Nin], all possible pressure input combinations,
                   i.e. 0-1 in each input, for train set only
    train_target - np.array sized [data_size_each_axis**Nin*train_frac, Nout], the output pressure of train_data, i.e. = M p_in
    test_data    - same as train_data but for test
    test_target  - same as train_target but for test
    """
    print('creating dataset using specific function')
    regression_data = np.array(list(itertools.product(range(1, data_size_each_input + 1), repeat=Nin)))/data_size_each_input
    regression_target = np.matmul(regression_data, desired_p_frac)
    train_data, train_target, test_data, test_target = DatasetManipulations.divide_train_test(regression_data, regression_target, train_frac)
    return train_data, train_target, test_data, test_target