import numpy as np
import copy
from numpy import zeros as zeros
from numpy import ones as ones
from numpy import array as array
import networkx as nx
import matplotlib.pyplot as plt


def buildNetwork(EIEJ_plots):
    """
    Builds a networkx network using edges from EIEJ_plots which are built upon EI and EJ at "Matrixfuncs.py"
    After this step, the order of edges at EIEJ_plots and in the networkx net is not the same which is shit
    
    input:
    EIEJ_plots - 2D np.array sized [NE, 2] - 
                 EIEJ_plots[i,0] and EIEJ_plots[i,1] are input and output nodes to edge i, respectively
    
    output:
    NET - networkx network containing just the edges from EIEJ_plots
    """
    
    NET = nx.DiGraph()  # initiate graph object
    NET.add_edges_from(EIEJ_plots)  # add edges 
    return NET

 
def plotNetStructure(NET, layout='Cells'):
    """
    Plots the structure (nodes and edges) of networkx NET 
    
    input:
    NET - networkx net of nodes and edges
    layout - graph visual layout, string. Roie style is 'Cells'
    
    output:
    pos_lattice - dict of positions of nodes from NET.nodes
    matplotlib plot of network structure
    """
    
    if layout == 'Cells':  # Roie style of connected crosses
        pos_lattice = {}  # initiate dictionary of node positions
        NGrid = int(np.sqrt(len(NET.nodes)/5))  # number of cells in network, considering only cubic array of cells
        k=0  # dummy
        for i in range(NGrid):  # network rows
            for j in range(NGrid):  # network columns
                pos_lattice[5*i+5*j+5*k] = array([-2+5*j, 0+5*i])  # left node in cell
                pos_lattice[5*i+5*j+5*k+1] = array([0+5*j, -2+5*i])  # lower node in cell
                pos_lattice[5*i+5*j+5*k+2] = array([2+5*j, 0+5*i])  # right node
                pos_lattice[5*i+5*j+5*k+3] = array([0+5*j, 2+5*i])  # upper node
                pos_lattice[5*i+5*j+5*k+4] = array([0+5*j, 0+5*i])  # middle node
            k+=NGrid-1  # add to dummy index so skipping to next cell
    elif layout == 'spectral':
        pos_lattice = nx.spectral_layout(NET)
    elif layout == 'planar':
        pos_lattice = nx.planar_layout(NET)
    else:
        pos_lattice = nx.spectral_layout(NET)
    nx.draw_networkx(NET, pos_lattice)
    plt.show()
    print('NET is ready')
    return pos_lattice


def PlotNetwork(p, u, K, NET, pos_lattice, EIEJ_plots, NN, NE):
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

    # p values at nodes - the same in EIEJ and in networkx's NET
    val_map = {i : p[i][0] for i in range(NN)}
    values = [val_map.get(node, 0.25) for node in NET.nodes()]

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
    
    low_K_NET_inds = np.where(K_NET==min(K_NET))[0]  # indices of edges with low conductivity
    high_K_NET_inds = np.where(K_NET!=min(K_NET))[0]  # indices of edges with higher conductivity
    positive_u_NET_inds = np.where(u_NET>=0)[0]  # indices of edges with positive flow vel
    negative_u_NET_inds = np.where(u_NET<0)[0]  # indices of edges with negative flow vel
    
    k_edges_positive = [NETEdges[i] for i in list(set(low_K_NET_inds) & set(positive_u_NET_inds))]  # edges with low conductivity, positive flow
    k_edges_negative = [NETEdges[i] for i in list(set(low_K_NET_inds) & set(negative_u_NET_inds))]  # edges with low conductivity, negative flow
    b_edges_positive = [NETEdges[i] for i in list(set(high_K_NET_inds) & set(positive_u_NET_inds))]  # edges with high conductivity, positive flow
    b_edges_negative = [NETEdges[i] for i in list(set(high_K_NET_inds) & set(negative_u_NET_inds))]  # edges with high conductivity, negative flow

    # save arrow sizes
    rescaled_u_NET = abs(u_NET)*8/max(abs(u_NET))
#     edgewidths_b = rescaled_u_NET[np.where(K_NET!=min(K_NET))]
#     edgewidths_r = rescaled_u_NET[np.where(K_NET==min(K_NET))[0]]

    edgewidths_k_positive = rescaled_u_NET[list(set(low_K_NET_inds) & set(positive_u_NET_inds))]
    edgewidths_k_negative = rescaled_u_NET[list(set(low_K_NET_inds) & set(negative_u_NET_inds))]
    edgewidths_b_positive = rescaled_u_NET[list(set(high_K_NET_inds) & set(positive_u_NET_inds))]
    edgewidths_b_negative = rescaled_u_NET[list(set(high_K_NET_inds) & set(negative_u_NET_inds))]

    x=2
    
    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    nx.draw_networkx_nodes(NET, pos_lattice, cmap=plt.get_cmap('cool'), 
                           node_color = values, node_size = 300)
    nx.draw_networkx_labels(NET, pos_lattice)
    
    # draw with right arrow widths ("width") and directions ("arrowstyle")
    nx.draw_networkx_edges(NET, pos_lattice, edgelist=k_edges_positive, edge_color='k', arrows=True, width=edgewidths_k_positive,
                           arrowstyle='-|>')
    nx.draw_networkx_edges(NET, pos_lattice, edgelist=k_edges_negative, edge_color='k', arrows=True, width=edgewidths_k_negative,
                           arrowstyle='<|-')
    nx.draw_networkx_edges(NET, pos_lattice, edgelist=b_edges_positive, edge_color='b', arrows=True, width=edgewidths_b_positive,
                           arrowstyle='-|>')
    nx.draw_networkx_edges(NET, pos_lattice, edgelist=b_edges_negative, edge_color='b', arrows=True, width=edgewidths_b_negative,
                           arrowstyle='<|-')
    plt.show()