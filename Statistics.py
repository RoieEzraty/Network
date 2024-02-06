## Statistics over the network etc.

import numpy as np

def flow_MSE(u, step, u_nxt=[]):
	"""
	flow_MSE calculates the MSE between different instances of u given iteration difference step

	input:
	u     - if u_nxt == []: 2D np.array [NEdges, iteration] of velocities at each edge (rows) and iteration step (cols)
			else:           1D np.array [Nedges] of velocities at early simulation step
	step  - calculate the MSE at step steps
	u_nxt - optional 1D np.array [Nedges] of velocities at later simulation step

	output:
	MSE - 1D np.array [iteration/step] of MSE between velocities at different simulation steps
	"""

	if len(u_nxt) == 0:
		MSE = np.zeros([np.shape(u)[1],])
		for i in range(np.shape(u)[1]-step):
			MSE[i] = np.sqrt(np.square(u[:, i+step] - u[:, i]).mean())
	else:
		MSE = np.sqrt(np.square(u_nxt - u).mean())
	return MSE


def K_Hamming(K_cells, step, K_cells_nxt=[]):
	"""
	K_Hamming calculates the Hamming between different conductivity constellations given iteration difference step

	input:
	K_cells       - if K_cells_nxt == []: 2D np.array [NGrids**2, iteration] of edge conductivities at each edge (rows) and iteration step (cols)
			        else:                 1D np.array [NGrids**2] of conductivities at early simulation step
	step          - calculate the Hamming distance at step steps
	K_cells_nxt   - optional 1D np.array [Nedges] of conductivities at later simulation step

	output:
	Hamming - 1D np.array [iteration/step] of Hamming distance between edge conductivity constellations at different simulation steps
	"""

	if len(K_cells_nxt) == 0:
		Hamming = np.zeros([np.shape(K_cells)[1],])
		for i in range(np.shape(K_cells)[1]-step):
			Hamming[i] = np.mean(K_cells[:, i+step] != K_cells[:, i])
	else:
		Hamming = np.mean(K_cells_nxt != K_cells)
	return Hamming


def power_dissip(u, K):
	"""
	power_dissip calculates the power dissipation in network given flow and conductivity constellation

	input:
	u - 1D np.array [Nedges] of velocities at edges
	K - 1D np.array [Nedges] of conductivities at edges

	output:
	P - float, power dissipation in network
	"""
	P = np.sum(u**2/K)
	return P

