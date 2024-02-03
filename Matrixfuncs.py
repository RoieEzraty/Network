import numpy as np
import numpy.random as rand
import copy
from numpy import array as array
from numpy import arange as arange
from numpy import zeros as zeros

def initiateK(NE):
    K = 2*np.ones([NE])
    # K[0] = 1
    K_mat = np.eye(NE) * K
    return K, K_mat
 
 
def buildL(DM, K_mat, Cstr, NN):
    L = np.dot(DM.T, np.dot(K_mat, DM))
    L_bar = zeros([NN + len(Cstr), NN + len(Cstr)])
    L_bar[NN:,:NN] = Cstr
    L_bar[:NN,NN:] = Cstr.T
    L_bar[:NN,:NN] = L
    return L, L_bar
  
  
def build_incidence(a, b, typ='Cells', Periodic=0):

    if typ == 'Cells':
        
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


    elif typ == 'Nachi':
        
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
    
    DM = zeros([NE, NN])
    for i in range(NE):
        DM[i,int(EI[i])] = +1.
        DM[i,int(EJ[i])] = -1.
        
    return EI, EJ, EIEJ_plots, DM, NE, NN


def ChangeKFromFlow(u, K, NGrid):
    """
    Change conductivities of full network given velocities
    This is done by dividing the network into cells
    and changing the K's cell by cell

    input:
    u     - NEdgesX1 array of flow through edges
    K     - NEdgesXNEdges array of conductivities
    NGrid - number of cells at each side of the network

    output:
    K_nxt - NEdgesXNEdges array of conductivities for next iteration
    """

    K_nxt = copy.copy(K)
    NCells = NGrid*NGrid  # total number of cells in network
    for i in range(NCells):  # change K's in every cell separately
        u_sub = u[4*i:4*(i+1)]  # velocities at particular cell
        K_sub = K[4*i:4*(i+1)]  # conductivities at particular cell
        K_sub_nxt = ChangeKFromFlow_singleCell(u_sub, K_sub)  # change K's at particular cell
        K_nxt[4*i:4*(i+1)] = K_sub_nxt  # put them in the right place at K_nxt
    return K_nxt


def ChangeKFromFlow_singleCell(u, K):
    """
    Change conductivities as a 4X4 matrix
    u and K are sub vectors and matrices w/4 elements representing 4 edges of single cell

    input:
    u - 4X1 array of flow through cell edges
    K - 4X4 array of conductivities

    output:
    K_nxt - 4X4 array of conductivities for next iteration
    """

    K_max = 2  # maximal conductivity
    K_min = 0.2  # minimal conductivity
    u_in_ind = np.where(u>0)[0]  # all indices where u enters the cell
    u_out_ind = np.where(u==min(u.T) )[0]  # indices if minimal flow, possibly exiting the cell
    if u[u_out_ind]>0:  # no flow exits the cell, it is a ground, don't put marble inside
        K_nxt = K_max*np.ones([4])
    else:  # normal cell, not ground
        pick_u_out = [u_out_ind[rand.randint(0, len(u_out_ind))]]  # if two edges have exactly the same output flow, choose random one
        cond1 = len(list(set(np.where(K==K_min)[0]) & set(u_in_ind)))>0  # check if there is flow inwards in edge where marble is at. then it has to move
        cond2 = all(K == K_max)  # marble is in middle of cell (happens only at first simulation iteration)
        if cond1 or cond2:  # if flow moves marble from one edge (1st cond) or middle (2nd cond), put lowest conductivity there
        # if np.where(K==K_min)[0] == u_in or all(K == K_max):
            K_nxt = K_max*np.ones([4])
            K_nxt[pick_u_out] = K_min 
        else:  # flow does not change conductivity
            K_nxt = copy.copy(K)
    return K_nxt