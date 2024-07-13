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


def calc_loss(output, target, function):
	"""
	calc_loss calculates the loss between output and target given the loss function "function"

	inputs:
	output   - array, output of flow through ground nodes
	target   - array, desired target output
	function - string, naming the method used to calculate the loss

	output:
	loss - float, loss as in machine learning
	"""
	if function == 'MSE':  # Mean Square error where output and taget are 1D arrays of same length.
		loss = np.mean(np.square(output-target))
	elif function == 'abs_diff':
		loss = np.mean(np.abs((output-target)/((output+target)/2)))  # mean for Allostery since output is 2dim. mean is redundant in regression.
		# loss = np.abs((output-target)/((output+target)/2))
	elif function == 'diff':
		loss = target-output  # mean for Allostery since output is 2dim. mean is redundant in regression.
	elif function == 'MSE_normalized':
		loss = np.mean(np.square(output-target)/((np.square(output)+np.square(target))/4))
	elif function == 'cross_entropy':  # as in Li & Mao 2024 https://arxiv.org/abs/2404.15471
		pc = array([np.exp(output[i])/np.sum(np.exp(output)) for i in range(len(output))])  # predicted probability of output
		loss = - np.dot(target, np.log(pc))
	return loss


def calc_ratio_loss(output, target, input_p):
	"""
	calc_ratio calculates the ratio between output and targets

	inputs:
	output   - array, output of flow through ground nodes
	target   - array, desired target output
	input_p  - array for Regression, float for rest (I think), pressure at input nodes

	output:
	ratio - float
	"""
	if np.size(output)>1:
		# ratio_loss = (target[0]/target[1]-output[0]/output[1])/(target[0]/target[1] - 1)
		ratio_loss = np.mean(np.abs((target - output)/(target)))
	else:
		# p_noBalls = np.sum(input_p)/(2+.64)  # numerically calculated, theoretical p at output as if no balls, large nets.
		# ratio_loss = (target - output)/(target - p_noBalls)
		ratio_loss = np.abs((target - output)/(target))
		R_tot = np.sum(input_p)/output
		print('R tot ', R_tot)
	return ratio_loss


def calculate_p_nudge(BigClass, State, error=0.0, p_in=0.0, error_prev=0.0, p_in_prev=0.0):
	"""
	calculate_p_nudge calculates the nudged pressure - whether in contrastive learning or dual problem

	inputs:
	BigClass  - class instance with all relevant data
	p_desired - 1d array of desired outputs at output nodes
	error     - 1d array of error from desired measure for the dual case, sized [2,]

	outputs:
	p_nudge - 1d array of pressure values to be assigned to output nodes, for the clamped stage
	"""
	if BigClass.Variabs.flow_scheme=='dual':
		print('p_nudge', State.p_nudge)
		print('error', error)
		if BigClass.Variabs.task_type=='Allostery_contrastive':
			p_nudge = State.p_nudge - np.dot(BigClass.Variabs.alpha, error)
			outputs_dual = State.outputs_dual + BigClass.Variabs.alpha * error
		elif BigClass.Variabs.task_type=='Regression_contrastive':
			# p_nudge = State.p_nudge - BigClass.Variabs.alpha * error
			print('p_in', p_in)
			print('error_prev', error_prev)
			print('p_in_prev', p_in_prev)
			p_nudge = State.p_nudge - BigClass.Variabs.alpha * (p_in - p_in_prev) * (error - error_prev)
			print('State.outputs_dual', State.outputs_dual)
			outputs_dual = State.outputs_dual + BigClass.Variabs.alpha * (error - error_prev)
			print('outputs_dual next', outputs_dual)
		return p_nudge, outputs_dual
	else:
		p_nudge = BigClass.Variabs.etta*BigClass.Variabs.p_desired + (1-BigClass.Variabs.etta)*State.p_outputs
		return p_nudge

# def calculate_outputs_dual(BigClass, outputs_dual_prev, error):
# 	"""
# 	calculate_output_dual calculates the nudged outputs - dual problem

# 	inputs:
# 	BigClass - class instance with all relevant data
# 	state    - 1d array of measured outputs at output nodes
# 	error    - 1d array of error from desired measure for the dual case, sized [2,]

# 	outputs:
# 	p_nudge - 1d array of pressure values to be assigned to output nodes, for the clamped stage
# 	"""
# 	if BigClass.Variabs.flow_scheme=='dual':
# 		outputs_dual = outputs_dual_prev + BigClass.Variabs.alpha * error
# 	return outputs_dual

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


def p_mat(BigClass, p):
	"""
	p_mat calculates the average p in each cell from State.p 1D array from nodes sorted as in Roie's network design.

	inputs:
	p     - 1D array [NN+constaints, ] of hydrostatic pressure values
	NGrid - int, # cells in each row (and col) of network
	
	outputs:
	p_mat - 2D array [NGrid, NGrid] pressure in each cell as matrix
	"""
	NGrid = BigClass.Variabs.NGrid

	if BigClass.Variabs.task_type == 'Counter':
		p_mat = np.zeros([NGrid, ])  # initialize array 2D value of velocity at every cell
		for i in range(NGrid):
		    p_mat[i] = np.mean(p[5*i:5*(i+1)])
		p_mat = np.reshape(np.array([p_mat, p_mat]), (2, NGrid))  # reshape as net
	else:
		p_mat = np.zeros([NGrid*NGrid, ])  # initialize array 2D value of velocity at every cell
		for i in range(NGrid*NGrid):
		    p_mat[i] = np.mean(p[5*i:5*(i+1)])
		p_mat = np.reshape(p_mat, [NGrid, NGrid])  # reshape as net
	return p_mat