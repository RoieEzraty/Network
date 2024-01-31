import numpy as np
import copy
from numpy import zeros as zeros
from numpy import ones as ones
from numpy import array as array
import networkx as nx
import matplotlib.pyplot as plt


def buildNetwork(EIEJ_plots):
    NET = nx.DiGraph()    
    # for i, edge in enumerate(EI):
    #     NET.add_edge(EI[i], EJ[i])
    NET.add_edges_from(EIEJ_plots)
    print(NET.edges)
    return NET

 
def plotNetStructure(NET, layout='Cells'):

    if layout == 'Cells':
        pos_lattice = {}
        NGrid = int(np.sqrt(len(NET.nodes)/5))
        k=0
        for i in range(NGrid):
            for j in range(NGrid):
                pos_lattice[5*i+5*j+5*k] = array([-2+5*j, 0+5*i])
                pos_lattice[5*i+5*j+5*k+1] = array([0+5*j, -2+5*i])
                pos_lattice[5*i+5*j+5*k+2] = array([2+5*j, 0+5*i])
                pos_lattice[5*i+5*j+5*k+3] = array([0+5*j, 2+5*i])
                pos_lattice[5*i+5*j+5*k+4] = array([0+5*j, 0+5*i]) 
            k+=NGrid-1        
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

#     # positions of edges
#     pos_lattice = nx.spectral_layout(NET)
#     pos_lattice_2 = nx.planar_layout(NET)

    # p values at nodes - the same in EIEJ and in networkx's NET
    val_map = {i : p[i][0] for i in range(NN)}
    values = [val_map.get(node, 0.25) for node in NET.nodes()]

    # velocity and conductivity values at edges - not the same in EIEJ and in networkx's NET
    NETEdges = list(NET.edges)
    
#     # use previous u and K
#     u_NET = copy.copy(u)
#     K_NET = copy.copy(K)
    
    # rearrange u and K for network
    u_NET = zeros(NE)  # velocity field arranged as edges in NET.edges and not EIEJ_plots
    K_NET = zeros(NE)  # conductivity values at edges in NET.edges and not EIEJ_plots
    for i in range(NE):
        K_NET[i] = K[EIEJ_plots.index(NETEdges[i])]   
        u_NET[i] = u[EIEJ_plots.index(NETEdges[i])]   

    # Specify the edges you want here


    # red_edges = [NETEdges[i] for i in np.where(K_NET==min(K_NET))[0]]
#     edge_colours = ['black' if not edge in red_edges else 'red'
#                     for edge in NET.edges()]
    # black_edges = [edge for edge in NET.edges() if edge not in red_edges]
    
    low_K_NET_inds = np.where(K_NET==min(K_NET))[0]
    high_K_NET_inds = np.where(K_NET!=min(K_NET))[0]
    positive_u_NET_inds = np.where(u_NET>=0)[0]
    negative_u_NET_inds = np.where(u_NET<0)[0]
    
    k_edges_positive = [NETEdges[i] for i in list(set(low_K_NET_inds) & set(positive_u_NET_inds))]
    k_edges_negative = [NETEdges[i] for i in list(set(low_K_NET_inds) & set(negative_u_NET_inds))]
    b_edges_positive = [NETEdges[i] for i in list(set(high_K_NET_inds) & set(positive_u_NET_inds))]
    b_edges_negative = [NETEdges[i] for i in list(set(high_K_NET_inds) & set(negative_u_NET_inds))]

    # save arrow sizes
    rescaled_u_NET = abs(u_NET)*8/max(abs(u_NET))
#     edgewidths_b = rescaled_u_NET[np.where(K_NET!=min(K_NET))]
#     edgewidths_r = rescaled_u_NET[np.where(K_NET==min(K_NET))[0]]

    edgewidths_k_positive = rescaled_u_NET[list(set(low_K_NET_inds) & set(positive_u_NET_inds))]
    edgewidths_k_negative = rescaled_u_NET[list(set(low_K_NET_inds) & set(negative_u_NET_inds))]
    edgewidths_b_positive = rescaled_u_NET[list(set(high_K_NET_inds) & set(positive_u_NET_inds))]
    edgewidths_b_negative = rescaled_u_NET[list(set(high_K_NET_inds) & set(negative_u_NET_inds))]
    
    arrowlst = []
    for i in range(len(NET.edges)):
        arrowlst.append('<|-')

    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    # pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(NET, pos_lattice, cmap=plt.get_cmap('cool'), 
                           node_color = values, node_size = 500)
    nx.draw_networkx_labels(NET, pos_lattice)
#     nx.draw_networkx_edges(NET, pos_lattice, edgelist=red_edges, edge_color='r', arrows=True, width=edgewidths_r, arrowstyle=arrowlst)
#     nx.draw_networkx_edges(NET, pos_lattice, edgelist=black_edges, arrows=True, width=edgewidths_b)
    nx.draw_networkx_edges(NET, pos_lattice, edgelist=k_edges_positive, edge_color='k', arrows=True, width=edgewidths_k_positive,
                           arrowstyle='-|>')
    nx.draw_networkx_edges(NET, pos_lattice, edgelist=k_edges_negative, edge_color='k', arrows=True, width=edgewidths_k_negative,
                           arrowstyle='<|-')
    nx.draw_networkx_edges(NET, pos_lattice, edgelist=b_edges_positive, edge_color='b', arrows=True, width=edgewidths_b_positive,
                           arrowstyle='-|>')
    nx.draw_networkx_edges(NET, pos_lattice, edgelist=b_edges_negative, edge_color='b', arrows=True, width=edgewidths_b_negative,
                           arrowstyle='<|-')
    plt.show()

#     nx.draw_networkx_nodes(NET, pos_lattice_2, cmap=plt.get_cmap('cool'), 
#                            node_color = values, node_size = 500)
#     nx.draw_networkx_labels(NET, pos_lattice_2)
#     nx.draw_networkx_edges(NET, pos_lattice_2, edgelist=red_edges, edge_color='r', arrows=True, width=edgewidths_r)
#     nx.draw_networkx_edges(NET, pos_lattice_2, edgelist=black_edges, arrows=True, width=edgewidths_b)
#     plt.show()