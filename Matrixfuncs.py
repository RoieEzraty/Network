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
    K_nxt = copy.copy(K)
    for i in range(NGrid*NGrid):
        u_sub = u[4*i:4*(i+1)]
        K_sub = K[4*i:4*(i+1)]
        K_sub_nxt = ChangeKFromFlow_singleCell(u_sub, K_sub)
        K_nxt[4*i:4*(i+1)] = K_sub_nxt
    return K_nxt


def ChangeKFromFlow_singleCell(u, K):
    # u and K are sub vectors and matrices containing 4 elements representing 4 edges of single cell
#     u_in = where(u==max(u.T))[1]
#     u_out = where(u==min(u.T))[1]
#     u_in = np.where(u==max(u.T))[0]
    K_max = 2
    K_min = 0.2
    u_in = np.where(u>0)[0]
    u_out = np.where(u==min(u.T))[0]
    rand_u_out = rand.randint(0, len(u_out))
    pick_u_out = [u_out[rand.randint(0, len(u_out))]]
    cond = len(list(set(np.where(K==K_min)[0]) & set(u_in)))>0  # check if there is flow inwards where the marble is at, so it has to move
    if cond or all(K == K_max):
    # if np.where(K==K_min)[0] == u_in or all(K == K_max):
        K = K_max*np.ones([4])
        K[pick_u_out] = K_min       
    return K