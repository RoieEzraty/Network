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


def shear_type(u):
	"""
	shear_type calculates the type of shear the system obtains after training,
	defined as the two diagonla values of the normalized and demeaned matrix u_final
	containing the flows from both outputs (cols) given flow from each input (rows),
	u_final is calculated under flow_iteration with sim_type='allostery test'

	input:
	u = np.array [2, 2] of final flows from outputs (cols) given each input (rows)

	output:
	shear_type = np.array [2, 1] of shear type obstained after training under each input. 
	             -1 = thickening, 1 = thinning, 0 = no effect
	"""
	u_mean = np.mean(u, axis=1)
	u_demean = (u.T - u_mean).T 
	u_demean_norm = (u_demean.T / u_mean).T
	shear_type = (u_demean_norm[0,0]-u_demean_norm[1,0])/2
	return shear_type
    
    # shear_type = np.array([u_demean_norm[0,0], u_demean_norm[1,1]])

def curl_direction(u, NGrid):
	"""
	curl_direction calculates the curl of u, edges sorted as in Roie's network design
	curl>0 --> counter-clockwise

	inputs:
	u     - 1D array [NE+constaints, ] of flow field after flowing from both inputs to both outputs
	NGrid - int, # cells in each row (and col) of network
	
	outputs:
	curl (w) - float, the normalized vorticity of the flow in the whole network, indicative of amount of learning
			   curl>0 --> counter-clockwise
	"""
	u_by_cells = np.zeros([NGrid*NGrid,2])  # initialize array 2D value of velocity at every cell
	for i in range(NGrid*NGrid):
	    u_by_cells[i,0] = (u[4*i]+(-u[4*i+2]))/2  # ux
	    u_by_cells[i,1] = (u[4*i+1]+(-u[4*i+3]))/2  # uy

	u_by_cells = np.reshape(u_by_cells, [NGrid, NGrid, 2])  # reshape as net
	u_by_cells_mag = np.sqrt(u_by_cells[:,:,0]**2+u_by_cells[:,:,1]**2)

	dxuy = np.append(np.diff(u_by_cells[:,:,1],axis=1).T, [np.diff(u_by_cells[:,:,1],axis=1)[:,-1]], axis=0).T  # d/dx(uy)
	dyux = np.append(np.diff(u_by_cells[:,:,0],axis=0), [np.diff(u_by_cells[:,:,0],axis=0)[-1,:]], axis=0)  # d/dy(ux)
	curl = dxuy-dyux
	# u_mean = np.mean(u_by_cells_mag)
	u_mean = 1
	curl_norm = curl/u_mean
	return np.mean(np.mean(curl_norm))

def p_mat(p, NGrid):
	"""
	p_mat calculates the average p in each cell from State.p 1D array from nodes sorted as in Roie's network design.

	inputs:
	p     - 1D array [NN+constaints, ] of hydrostatic pressure values
	NGrid - int, # cells in each row (and col) of network
	
	outputs:
	p_mat - 2D array [NGrid, NGrid] pressure in each cell as matrix
	"""
	p_mat = np.zeros([NGrid*NGrid, ])  # initialize array 2D value of velocity at every cell
	for i in range(NGrid*NGrid):
	    p_mat[i] = np.mean(p[5*i:5*(i+1)])
	p_mat = np.reshape(p_mat, [NGrid, NGrid])  # reshape as net
	return p_mat