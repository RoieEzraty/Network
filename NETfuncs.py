import numpy as np
import copy
from numpy import zeros as zeros
from numpy import ones as ones
from numpy import array as array
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt

import Statistics

# def buildNetwork(EIEJ_plots):
#     """
#     Builds a networkx network using edges from EIEJ_plots which are built upon EI and EJ at "Matrixfuncs.py"
#     After this step, the order of edges at EIEJ_plots and in the networkx net is not the same which is shit
    
#     input:
#     EIEJ_plots - 2D np.array sized [NE, 2] - 
#                  EIEJ_plots[i,0] and EIEJ_plots[i,1] are input and output nodes to edge i, respectively
    
#     output:
#     NET - networkx network containing just the edges from EIEJ_plots
#     """
#     NET = nx.DiGraph()  # initiate graph object
#     NET.add_edges_from(EIEJ_plots)  # add edges 
#     return NET

 
def plotNetStructure(NET, NGrid, layout='Cells', plot='no', node_labels=True):
    """
    Plots the structure (nodes and edges) of networkx NET 
    
    input:
    NET - networkx net of nodes and edges
    layout - graph visual layout, string. Roie style is 'Cells'
    
    output:
    pos_lattice - dict of positions of nodes from NET.nodes
    matplotlib plot of network structure
    """
    
    if layout == 'Cells':  # Roie style of 2D array of connected crosses
        pos_lattice = {}  # initiate dictionary of node positions
        # NGrid = int(np.sqrt(len(NET.nodes)/5))  # number of cells in network, considering only cubic array of cells
        k=0  # dummy
        for i in range(NGrid):  # network rows
            for j in range(NGrid):  # network columns
                pos_lattice[5*i+5*j+5*k] = array([-2.2+5*j, 0+5*i])  # left node in cell
                pos_lattice[5*i+5*j+5*k+1] = array([0+5*j, -2.2+5*i])  # lower node in cell
                pos_lattice[5*i+5*j+5*k+2] = array([2.2+5*j, 0+5*i])  # right node
                pos_lattice[5*i+5*j+5*k+3] = array([0+5*j, 2.2+5*i])  # upper node
                pos_lattice[5*i+5*j+5*k+4] = array([0+5*j, 0+5*i])  # middle node
            k+=NGrid-1  # add to dummy index so skipping to next cell
    elif layout == 'oneCol':  # Roie style of single column of connected crosses
        pos_lattice = {}  # initiate dictionary of node positions
        # NGrid = int(len(NET.nodes)/5)  # number of cells in network, considering only cubic array of cells
        for i in range(NGrid):  # network columns
            pos_lattice[5*i] = array([-2.2, 0+5*i])  # left node in cell
            pos_lattice[5*i+1] = array([0, -2.2+5*i])  # lower node in cell
            pos_lattice[5*i+2] = array([2.2, 0+5*i])  # right node
            pos_lattice[5*i+3] = array([0, 2.2+5*i])  # upper node
            pos_lattice[5*i+4] = array([0, 0+5*i])  # middle node
    elif layout == 'spectral':
        pos_lattice = nx.spectral_layout(NET)
    elif layout == 'planar':
        pos_lattice = nx.planar_layout(NET)
    else:
        pos_lattice = nx.spectral_layout(NET)

    if plot == 'yes':
        nx.draw_networkx(NET, pos_lattice, edge_color='b', node_color='b', with_labels=node_labels)
        plt.show()
        
    print('NET is ready')
    return pos_lattice


def PlotNetwork(p, u, K, BigClass, EIEJ_plots, NN, NE, nodes='yes', edges='yes', pressureSurf='no', savefig='no'):
    """
    Plots the flow network structure alongside hydrostatic pressure, flows and conductivities
    pressure denoted by colors from purple (high) to cyan (low)
    flow velocity denoted by arrow direction and thickness
    conductivity denoted by arrow color - black (low) and blue (high)
    
    input:
    p - 1D np.array sized [NNodes], hydrostatic pressure at nodes
    u - 1D np.array sized [NEdges], flow velocities at edges (positive is into cell center)
    K - 1D np.array sized [NEdges], flow conductivities for every edge
    pos_lattice - dict from
    layout - graph visual layout, string. Roie style is 'Cells'
    
    output:
    matplotlib plot of network structure
    """
    NET = BigClass.NET.NET
    pos_lattice = BigClass.NET.pos_lattice

    # Preliminaries for the plot
    # node_sizes = 24
    node_sizes = 8*24
    # IO_node_sizes = 42
    u_rescale_factor = 5
    
    # p values at nodes - the same in EIEJ and in networkx's NET
    val_map = {i : p[i][0] for i in range(NN)}
    # purple_map = {i : (int(p[i][0] == np.max(p[:NN+1])) + 0.5*int(p[i][0] == np.min(p[:NN+1]))) for i in range(NN)}
    values = [val_map.get(node, 0.25) for node in NET.nodes()]
    # purple_values = [purple_map.get(node) for node in NET.nodes()]

    # velocity and conductivity values at edges - not the same in EIEJ and in networkx's NET
    NETEdges = list(NET.edges)
    
    # rearrange u and K for network
    # since NET.edges and EIEJ_plots are not the same, thanks networkx you idiot
    u_NET = zeros(NE)  # velocity field arranged as edges in NET.edges and not EIEJ_plots
    K_NET = zeros(NE)  # conductivity values at edges in NET.edges and not EIEJ_plots
    for i in range(NE):
        K_NET[i] = K[EIEJ_plots.index(NETEdges[i])]   
        u_NET[i] = u[EIEJ_plots.index(NETEdges[i])]   

    # red_edges = [NETEdges[i] for i in np.where(K_NET==min(K_NET))[0]]
#     edge_colours = ['black' if not edge in red_edges else 'red'
#                     for edge in NET.edges()]
    # black_edges = [edge for edge in NET.edges() if edge not in red_edges]
    
    # low_K_NET_inds = np.where(K_NET==min(K_NET))[0]  # indices of edges with low conductivity
    # high_K_NET_inds = np.where(K_NET!=min(K_NET))[0]  # indices of edges with higher conductivity
    # low_K_NET_inds = np.where(K_NET!=max(K_NET))[0]  # indices of edges with low conductivity
    # high_K_NET_inds = np.where(K_NET==max(K_NET))[0]  # indices of edges with higher conductivity
    low_K_NET_inds = np.where(K_NET==BigClass.Variabs.K_min)[0]  # indices of edges with low conductivity
    high_K_NET_inds = np.where(K_NET!=BigClass.Variabs.K_min)[0]  # indices of edges with higher conductivity
    positive_u_NET_inds = np.where(u_NET>10**-10)[0]  # indices of edges with positive flow vel
    negative_u_NET_inds = np.where(u_NET<-10**-10)[0]  # indices of edges with negative flow vel
    
    r_edges_positive = [NETEdges[i] for i in list(set(low_K_NET_inds) & set(positive_u_NET_inds))]  # edges with low conductivity, positive flow
    r_edges_negative = [NETEdges[i] for i in list(set(low_K_NET_inds) & set(negative_u_NET_inds))]  # edges with low conductivity, negative flow
    b_edges_positive = [NETEdges[i] for i in list(set(high_K_NET_inds) & set(positive_u_NET_inds))]  # edges with high conductivity, positive flow
    b_edges_negative = [NETEdges[i] for i in list(set(high_K_NET_inds) & set(negative_u_NET_inds))]  # edges with high conductivity, negative flow

    # save arrow sizes
    rescaled_u_NET = abs(u_NET)*u_rescale_factor/max(abs(u_NET))
#     edgewidths_b = rescaled_u_NET[np.where(K_NET!=min(K_NET))]
#     edgewidths_r = rescaled_u_NET[np.where(K_NET==min(K_NET))[0]]

    edgewidths_k_positive = rescaled_u_NET[list(set(low_K_NET_inds) & set(positive_u_NET_inds))]
    edgewidths_k_negative = rescaled_u_NET[list(set(low_K_NET_inds) & set(negative_u_NET_inds))]
    edgewidths_b_positive = rescaled_u_NET[list(set(high_K_NET_inds) & set(positive_u_NET_inds))]
    edgewidths_b_negative = rescaled_u_NET[list(set(high_K_NET_inds) & set(negative_u_NET_inds))]

    if pressureSurf == 'yes':
        p_mat = Statistics.p_mat(BigClass, p)
        figsizeX = 5*np.shape(p_mat)[0]
        figsizeY = 5*np.shape(p_mat)[1]
        X = np.arange(0, figsizeX, 5)
        Y = np.arange(0, figsizeY, 5)
        X, Y = np.meshgrid(X, Y)

        # figsize = 5*BigClass.Variabs.NGrid
        # p_mat = Statistics.p_mat(BigClass, p)
        # X = np.arange(-4, 5, 8)
        # Y = np.arange(-4, figsize+5, figsize/np.shape(p_mat)[1]+2)
        # X, Y = np.meshgrid(X, Y)
        print(X)
        print(Y)
        print(p_mat)

        # Plot the surface.
        plt.contourf(X, Y, p_mat.T, cmap=plt.cm.cool,
                     linewidth=0, antialiased=False)

    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    if nodes == 'yes':
        nx.draw_networkx_nodes(NET, pos_lattice, cmap=plt.get_cmap('cool'), 
                                node_color = values, node_size = node_sizes)
        # nx.draw_networkx_labels(NET, pos_lattice) 
    
    if edges == 'yes':
        # draw with right arrow widths ("width") and directions ("arrowstyle")
        # nx.draw_networkx_nodes(NET, pos_lattice, cmap=plt.get_cmap('binary'), 
        #                         node_color = purple_values, node_size = IO_node_sizes)
        # nx.draw_networkx_edges(NET, pos_lattice, edgelist=r_edges_positive, edge_color='r', arrows=True, width=edgewidths_k_positive,
        #                        arrowstyle='-|>')
        # nx.draw_networkx_edges(NET, pos_lattice, edgelist=r_edges_negative, edge_color='r', arrows=True, width=edgewidths_k_negative,
        #                        arrowstyle='<|-')
        # nx.draw_networkx_edges(NET, pos_lattice, edgelist=b_edges_positive, edge_color='b', arrows=True, width=edgewidths_b_positive,
        #                        arrowstyle='-|>')
        # nx.draw_networkx_edges(NET, pos_lattice, edgelist=b_edges_negative, edge_color='b', arrows=True, width=edgewidths_b_negative,
        #                        arrowstyle='<|-')
        nx.draw_networkx_edges(NET, pos_lattice, edgelist=r_edges_positive, edge_color='r', arrows=True, width=edgewidths_k_positive,
                               arrowstyle='->')
        nx.draw_networkx_edges(NET, pos_lattice, edgelist=r_edges_positive, edge_color='r', arrows=True, width=1,
                               arrowstyle='-[')
        nx.draw_networkx_edges(NET, pos_lattice, edgelist=r_edges_negative, edge_color='r', arrows=True, width=edgewidths_k_negative,
                               arrowstyle='<-')
        nx.draw_networkx_edges(NET, pos_lattice, edgelist=r_edges_negative, edge_color='r', arrows=True, width=1,
                               arrowstyle=']-')
        nx.draw_networkx_edges(NET, pos_lattice, edgelist=b_edges_positive, edge_color='k', arrows=True, width=edgewidths_b_positive,
                               arrowstyle='->')
        nx.draw_networkx_edges(NET, pos_lattice, edgelist=b_edges_negative, edge_color='k', arrows=True, width=edgewidths_b_negative,
                               arrowstyle='<-')
    if savefig=='yes':
        # prelims for figures
        comp_path = "C:\\Users\\SMR_Admin\\OneDrive - huji.ac.il\\PhD\\Network Simulation repo\\figs and data\\"
        # comp_path = "C:\\Users\\roiee\\OneDrive - huji.ac.il\\PhD\\Network Simulation repo\\figs and data\\"
        datenow = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        plt.savefig(comp_path + 'network_' + str(datenow) + '.png', bbox_s='tight')

    plt.show()

def PlotPressureContours(BigClass, p):
    p_mat = Statistics.p_mat(BigClass, p)
    return p_mat
