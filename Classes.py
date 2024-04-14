import numpy as np
import copy

import numpy.random as rand
import pickle
import networkx as nx
import matplotlib.pyplot as plt

import NETfuncs, Matrixfuncs, Solve, Constraints, Statistics, FileFuncs


class User_variables:
	"""
	User_variables saves all user variables for network simulation

	inputs:
	NGrid              - int, lattice dimension is Ngrid X Ngrid
	input_p            - float, pressure at input node
	flow_scheme        - str, order of pressure appliance in training and test
						 'one_shot' = apply pressure drop from 1 output node and 1 output node, wait till convergence
						 'unidir'   = apply pressure drop only in the regular directions - constrained node = positive, ground = 0
                                      there are 2 input and output pairs, exchange between them
                         'taktak'   = apply pressure drop unidir once, meaning 1st input and output pair and then 2nd pair.
                                      then switch ground and constrained nodes to apply oposite dir.
	task_type          - str, task that is being simulated
						 'Allostery_one_pair' = 1 pair of input and outputs
						 'Allostery'          = 2 pairs of input and outputs
						 'XOR'                = 2 inputs and 2 outputs. difference between output nodes encodes the XOR result of the 2 inputs
						 'Channeling_diag'    = 1st from input to diagonal output, then from output to 2 perpindicular nodes. 
                                                test from input to output
                         'Channeling_straight' = 1st from input to output on same column, then from output to 2 perpindicular nodes. 
                                                 test from input to output (same as 1st)
                		 'Counter' = column of cells bottomost and topmost nodes are input/output (switching), 
                                     rightmost nodes (1 each row) ground. more about the task in "_counter.ipynb".
	K_scheme           - str, scheme to change conductivities
						 'propto_current_squared' = conductivity on edge changes due to squared on edge (use beta argument), no marbles involves
						 'marbles_pressure'       = conductivities in each cells change due to marbles moving due to pressure difference.
						                            binary values K_min, K_max
						 'marbles_u'              = conductivities in each cells change due to marbles moving due to flow velocity.
						                            binary values K_min, K_max
	K_type             - str, effect of flow on conductivity without changing marble positions
					     'bidir'    = conductivity is the same regardless of flow directions
					     'flow_dep' = conductivity depends on flow direction - if into cell then maximal, if out and there is a marble then lower
	iterations         - int, # iterations allowed under flow cycles / updating conductivities
	input_output_pairs - np.array of input and output node pairs. State.flow_iterate() will know how to handle them.
	Periodic           - bool, 'True'=lattice has periodic boundaries, default='False'
	net_typ            - str, layout for NETfuncs plotNetStructure(). 
	                     'Cells' is Roie's style of network and is default
	                     'Nachi' is Nachi style
	u_thresh           - float, threshold to move marbles, default=1
	fixed_nodes        - 1D array, numbers the nodes with fixed value assigned by fixed_node_p() function, for 'XOR' task, default=0
	K_max              - default=1
	K_min              - default=0.5
	beta               - default=0.0
	"""

	def __init__(self, NGrid, input_p, flow_scheme, task_type, K_scheme, K_type, iterations, input_output_pairs, 
		         Periodic='False', net_typ='Cells', u_thresh=1, fixed_nodes=0, K_max=1, K_min=0.5, beta=0.0):
		self.NGrid = NGrid		
		if len(input_p)==1:
			self.input_p = input_p

		self.flow_scheme = flow_scheme
		# # of iterations in every circle, for calculation of convergence
		if flow_scheme == 'taktak':
			self.circle_step = 4
		elif flow_scheme == 'unidir':
			self.circle_step = 2
		else:
			self.circle_step = 1

		self.task_type = task_type
		if task_type == 'XOR' or 'Counter':
			self.fixed_nodes = fixed_nodes
		else: 
			self.fixed_nodes = np.array([])

		self.iterations = iterations
		self.input_output_pairs = input_output_pairs
		self.Periodic = Periodic
		self.net_typ = net_typ
		self.u_thresh = u_thresh
		self.K_scheme = K_scheme
		self.K_type = K_type
		self.K_max = K_max
		self.K_min = K_min
		self.beta = beta

	def assign_input_p(self, p):
		"""
		assign_input_p assigns input pressure value to Variabs instance, input nodes will have this value
		"""
		self.input_p = p

	def assign_K_min(self, K_min):
		"""
		assign_K_min assigns minimal conductance value to Variabs instance, edges with marble will have this value
		"""
		self.K_min = K_min

	def assign_fixed_node_p(self, p):
		"""
		Assigns the values of the fixed nodes for XOR task
		values of fixed nodes are p/3 and 2*p/3, these could be modified...

		input:
		p = input pressure, float

		output:
		fixed_node_p = 1D numpy array [2, ], hydro. pressure on fixed nodes
		"""
		if self.task_type == 'XOR':
			self.fixed_node_p = np.array([p/3, 2*p/3])
		elif self.task_type == 'Counter':
			self.fixed_node_p = np.ones([self.NGrid, ])*p
		else:
			self.fixed_node_p = np.array([])
			print('no fixed nodes other than input')


class Net_structure:
	"""
	Net_structure class save the structure of the network
	"""
	
	def __init__(self):
		self.NE = 0
		self.NN = 0

	def build_incidence(self, Variabs):
		"""
		build_incidence builds the incidence matrix DM
		
		inputs:
		Variabs - variables class

		outputs:
		EI         - np.array, node number on 1st side of all edges
		EJ         - np.array, node number on 2nd side of all edges
		EIEJ_plots - np.array, combined EI and EJ, each line is two nodes of edge, for visual ease
		DM         - np.array, connectivity matrix NE X NN
		NE         - int, # edges in network
		NN         - int, # nodes in network
		"""
		self.EI, self.EJ, self.EIEJ_plots, self.DM, self.NE, self.NN = Matrixfuncs.build_incidence(Variabs)

	def Boundaries_and_connections(self, BigClass):
		"""
		Identify edges at connections of cells and at boundaries for ease of use. They will have zero gradient for p on them.

		inputs:
		NGrid - int, lattice dimension is Ngrid X Ngrid

		outputs:
		EdgesTotal - 2D array sized [???, 2], all edges connecting cells in network
		"""
		NGrid = BigClass.Variabs.NGrid
		if BigClass.Variabs.task_type == 'Counter':
			NConnections = int(NGrid-1)
			left_side = [0 + 4*i for i in range(NGrid)]  # enumerate leftmost edges in network
			bottom_side = [1]  # enumerate bottommost edge in network
			right_side = []  # right side is ground so no need to treat as boundary
			top_side = [4*NGrid-1]  # enumerate topmost edges in network
			EdgesBounaries = np.append(left_side, np.append(bottom_side, top_side))
		else:
			NConnections = int(NGrid*(NGrid-1)*2)  # connections between cells

			left_side = [0 + 4*NGrid*i for i in range(NGrid)]  # enumerate leftmost edges in network
			bottom_side = [1 + 4*i for i in range(NGrid)]  # enumerate bottommost edges in network
			right_side = [2 + 4*(NGrid-1) + 4*NGrid*i for i in range(NGrid)]  # enumerate rightmost edges in network
			top_side = [4*NGrid*(NGrid-1) + 3 + 4*i for i in range(NGrid)]  # enumerate topmost edges in network
			EdgesBounaries = np.append(left_side, np.append(bottom_side, np.append(right_side, top_side)))
		
		# connect them all
		EdgesConnections = [int(i) for i in range(self.NE-NConnections, self.NE)]  # enumerate all number from NConnections to NE
		
		self.EdgesTotal = np.append(EdgesConnections, EdgesBounaries)

	def Setup_constraints(self, BigClass):
		"""
		Setup_constraints sets up the constraints on the network for specific run, in form of 2D arrays

		inputs:
		input_output_pairs - np.array of input and output node pairs. State.flow_iterate() will know how to handle them.
		fixed_nodes   - 1D array, numbers the nodes with fixed value assigned by fixed_node_p() function, for 'XOR' task, default=0
		input_p            - float, B.C. pressure at input node

		outputs:
		output_edges               - 1D array, all output edge numbers in flow scheme
		InNodeData_full            - 1D array, all input node pressures in flow scheme
		InNodes_full               - 1D array, all input node numbers in flow scheme
		FixedNodeData_full         - 1D array, all fixed node pressures in flow scheme
		FixedNodes_full            - 1D array, all fixed node numbers in flow scheme
		GroundNodes_full           - 1D array, all node numbers with fixed value 0 for pressure in training stages
		GroundNodes_full_Allostery - 1D array, all node numbers with fixed value 0 for pressure in allostery test stage
		EdgeData_full              - 1D array, all fixed edge pressure drops in flow scheme
		Edges_full                 - 1D array, all fixed edge numbers in flow scheme for training stages
		Edges_full_Allostery       - 1D array, all fixed edge numbers in flow scheme for test stages
		"""
		# dummy variab for ease
		in_out_pairs = copy.copy(BigClass.Variabs.input_output_pairs)
		fixed_nodes = copy.copy(BigClass.Variabs.fixed_nodes)
		len_in_out_pairs = np.shape(in_out_pairs)[1]
		len_fixed = np.shape(fixed_nodes)[1]

		InNodeData_full = np.array([[BigClass.Variabs.input_p], [BigClass.Variabs.input_p]])  # input p value
		# input p node
		InNodes_full = np.array([[in_out_pairs[i, 0]] for i in range(len_in_out_pairs)])  
		# if fixed nodes exist, use them:
		if len(fixed_nodes)>0:
			FixedNodeData_full = np.array([BigClass.Variabs.fixed_node_p[i] for i in range(len_fixed)])  # input p value
			# input p node
			FixedNodes_full = np.array([fixed_nodes[0, i] for i in range(len_fixed)])  
		else:
			FixedNodeData_full = np.array([])
			FixedNodes_full = np.array([])

		# nodes with zero pressure
		GroundNodes_full = np.array([[in_out_pairs[i, 1]] for i in range(len_in_out_pairs)])  
		GroundNodes_full_Allostery = np.array([GroundNodes_full[i][0] for i in range(len(GroundNodes_full))])

		EdgeData_full = np.array([[0], [0]])  # pressure drop value on edges specified by Edges_full

		# Full list of edges that have specified pressure drop. it is different from EdgesTotal only if
		# there are non-zero boundary conditions at periphery of network.
		Edges_full = [self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[0])[0][0]) 
		                                    & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		                                    & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])], 
		                    self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[1])[0][0]) 
		                                    & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		                                    & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])]]

		# Same as Edges_full but for Allostery task, don't know why I have two here, should check later
		Edges_full_Allostery = copy.copy(Edges_full)

		# output edges which are ground node but don't have to be in middle of cell
		self.output_edges = np.array([np.where(np.append(self.EI, self.EJ)==GroundNodes_full[i])[0] % len(self.EI) 
		                              for i in range(len(GroundNodes_full))])

		# add all B.C.s to Strctr instance
		self.InNodeData_full = InNodeData_full
		self.InNodes_full = InNodes_full
		# If there are nodes with fixed pressure which are not input/output (for 'XOR' for example)
		if len(FixedNodeData_full)>0:
			self.FixedNodeData_full = FixedNodeData_full
			self.FixedNodes_full = FixedNodes_full
		else:
			self.FixedNodeData_full = np.array([])
			self.FixedNodes_full = np.array([])
		self.GroundNodes_full = GroundNodes_full
		self.GroundNodes_full_Allostery = GroundNodes_full_Allostery
		self.EdgeData_full = EdgeData_full
		self.Edges_full = Edges_full
		self.Edges_full_Allostery = Edges_full_Allostery

	def Constraints_afo_task(self, BigClass, sim_type, i):
		"""
		Constraints_afo_task sets up the constraints on nodes and edges for specific learning task, and for specific step.
		This comes after Setup_constraints which sets them for the whole task

		inputs:
		flow_scheme - str, order of pressure appliance in training and test
					  'one_shot' = apply pressure drop from 1 output node and 1 output node, wait till convergence
					  'unidir'   = apply pressure drop only in the regular directions - constrained node = positive, ground = 0
                                   there are 2 input and output pairs, exchange between them
                      'taktak'   = apply pressure drop unidir once, meaning 1st input and output pair and then 2nd pair.
                                   then switch ground and constrained nodes to apply oposite dir.
        sim_type    - simulation type: 'no marbles'     - flows from 1 input at a time and all outputs, all edges have high State.K
		                               'allostery test' - flows from 1 input at a time and all outputs, K's as in State.K
		                               'w marbles'      - (default) flows from 1 input and one output K's as in State.K, 
		                                                  update conductivities as due to flow u and calculate flow again so
		                                                  further update of conductivities will not change a thing
		i           - int, index from which to extract B.C.s, corresponding to some mod of the training / test step
		"""

		# dummy variab
		task_type = copy.copy(BigClass.Variabs.task_type)
		
		if task_type == 'Allostery_one_pair':
			# Determine boundary conditions - only first direction for 'Allostery_one_pair'
			NodeData = self.InNodeData_full[0]
			Nodes = self.InNodes_full[0] 
			EdgeData = self.EdgeData_full[0]
			Edges = self.Edges_full[0]
			if sim_type == 'w marbles':
				GroundNodes = self.GroundNodes_full[0]
			else:
				GroundNodes = np.array([self.GroundNodes_full[0][0] for i in range(len(self.GroundNodes_full))])

		elif task_type=='Allostery' or task_type=='Flow_clockwise':
			m = i % 2  # iterate between 1st and 2nd inputs 

			# Determine boundary conditions - modulus 2 for 'Allostery'
			EdgeData = self.EdgeData_full[m]
			Edges = self.Edges_full[m]
			if task_type=='Flow_clockwise' and sim_type!='w marbles':  # test flow direction, use both inputs
				NodeData = np.array([self.InNodeData_full[i][0] for i in range(len(self.InNodeData_full))])
				Nodes = np.array([self.InNodes_full[i][0] for i in range(len(self.InNodes_full))])
			else:  # test flow - flow from 1 direction
				NodeData = self.InNodeData_full[m]
				Nodes = self.InNodes_full[m] 

			if sim_type=='w marbles':
				GroundNodes = self.GroundNodes_full[m]
			else:
				GroundNodes = np.array([self.GroundNodes_full[i][0] for i in range(len(self.GroundNodes_full))])


			# Simulating normal direction of flow: from input to output

			# Otherwise, if true, switch ground and input nodes every 2nd iteration
			if i % 4 > 1 and BigClass.Variabs.flow_scheme == 'taktak':
				Nodes = self.GroundNodes_full[m]
				GroundNodes = self.InNodes_full[m]

		elif task_type == 'XOR':
			# select random input and output
			m = np.where([rand.randint(0, 2), rand.randint(0, 2)])[0]

			NodeData = np.append(self.InNodeData_full[m], self.FixedNodeData_full)
			Nodes = np.append(self.InNodes_full[m], self.FixedNodes_full)
			EdgeData = self.EdgeData_full[m]
			Edges = self.Edges_full[m]
			GroundNodes = self.GroundNodes_full[m]

		elif task_type == 'Channeling_diag' or task_type == 'Channeling_straight':
			# iterate between 1st and 2nd inputs (1st comes at beginning and end and 2nd comes once in the middle)

			m=i%2
			
			if m == 0:
				# Determine boundary conditions
				NodeData = self.InNodeData_full[0]
				Nodes = self.InNodes_full[0] 
				EdgeData = self.EdgeData_full[0]
				Edges = self.Edges_full[0]
				GroundNodes = self.GroundNodes_full[0]
			else:
				NodeData = self.InNodeData_full[0]
				Nodes = self.GroundNodes_full[0]
				EdgeData = self.EdgeData_full[1]
				Edges = self.Edges_full[1]
				GroundNodes = BigClass.Variabs.input_output_pairs[1, :]

		elif task_type == 'Counter':
			m = i % 2  # iterate between 1st and 2nd inputs 

			# Determine boundary conditions - modulus 2 for 'Counter'
			EdgeData = self.EdgeData_full[m]
			Edges = self.Edges_full[m]
			NodeData = np.append(self.InNodeData_full[m], self.FixedNodeData_full)
			Nodes = np.append(self.InNodes_full[m], self.FixedNodes_full)
			GroundNodes = self.GroundNodes_full[m]

		return NodeData, Nodes, EdgeData, Edges, GroundNodes


class Net_state:
	"""
	Net_state class stores internal info of network state for all time steps

	inputs:

	outputs:
	K     - 1D array sized [NE, ], conductances 
	K_mat - 2D array sized [NE, NE], conductances on diagonal of matrix
	u     - 1D array sized [NE, ], flows calculated as (p_i-p_j)/K_ij. >0 going inside cell. 
	p     - 1D array sized [NN, ], hydrostatic pressure on nodes.
	"""
	def __init__(self):
		super(Net_state, self).__init__()
		self.K = []
		self.K_mat = []
		self.u = []
		self.p = []

	def initiateArrays(self, BigClass, NE, iters):
		"""
		initiateArrays initializes arrays used in flow_iterate()

		inputs:
		NGrid - int, lattice dimension is Ngrid X Ngrid
		NE    - int, # edges
		iters - int, # iterations allowed under flow cycles / updating conductivities

		outputs:
		Hamming_i - Hamming distance in edge binary K space, at i'th iteration
		MSE_i     - MSE of flow at i'th iteration
		stop_bool - boolean for stopping the loop if simulation converged before reaching iters
		"""
		self.u_final = np.zeros([2, 2])  # velocity from each output (columns) under each input (rows), for Allostery solely
		self.u_all = np.zeros([NE, iters])  # flow velocity at every edge (rows) for every one of iterations (cols)
		self.K_all = np.zeros([NE, iters])  # conductivity of every edge (rows) for every one of iterations (cols)
		self.K_cells = np.zeros([BigClass.Variabs.NGrid**2, iters])  # 2D array [# cells, iters] with positions of marbles in each cell 
		                                                             # (0=middle, 1=left, 2=bot, 3=right, 4=top)
		self.power_dissip = np.NaN*np.zeros([BigClass.Variabs.iterations, ])  # total power dissipation 
																		 # defined: sum((p_i-p_j)**2*K_ij) over edges, for every iteration
		self.MSE =	np.zeros([BigClass.Variabs.iterations, ])  # total MSE of flow velocity for every iteration
		self.Hamming = np.zeros([BigClass.Variabs.iterations, ])  # total Hamming distance in edge binary K space for every iteration
		Hamming_i = 1  # dummy for to not break loop before assigned different value
		MSE_i = 1  # dummy for to not break loop before assigned different value
		self.convergence_time = np.NaN
		stop_bool = 0
		return Hamming_i, MSE_i, stop_bool

	def initiateK(self, BigClass, noise='no', noise_amp=0.):
		"""
		Builds initial conductivity matrix, simulation steps and updates are under Solve.py

		input:
		Variabs   - class, user variables
		Strctr    - class, network structure
		noise     - str, add noise to initial state of conductivities
				    'no'     = all marbles in middle
				    'rand_u' = take randomized flow field and move marbles accordingly
				    'rand_K' = take randomized flow field, add to solved velocity field under all marbles in middle, 
				               then move marbles accordingly
		noise_amp - float, amplitude of noise, 0.0-1.0 generally
		
		output:
		K     - 1D np.array sized [NEdges] with initial conductivity values for every edge
		K_mat - 2D cubic np.array sized [NEdges] with initial conductivity values on diagonal
		"""
		NE = BigClass.Strctr.NE  # # edges
		K_max = BigClass.Variabs.K_max  # 2D array sized [NE, NE], conductances on diagonal of matrix 
		frac_moved = noise_amp  # fraction of marbles moved, on average
		u_thresh = BigClass.Variabs.u_thresh  # float, threshold of flow above which marbles move

		self.K = K_max*np.ones([NE])

		if noise == 'rand_u' or noise == 'rand_K':
			u_rand = Solve.create_randomized_u(BigClass, NE, frac_moved, u_thresh, noise)
			self.K = Matrixfuncs.ChangeKFromFlow(u_rand, u_thresh, self.K, BigClass.Variabs.NGrid, 
												K_change_scheme=BigClass.Variabs.K_scheme, K_max=BigClass.Variabs.K_max, 
												K_min=BigClass.Variabs.K_min)  # move marble, change conductivities
		
		self.K_mat = np.eye(NE) * self.K  # save as matrix

	def fixAllK(self, BigClass):
		if BigClass.Variabs.task_type == 'Counter':  # for counter task, allow marble to get stuck at boundary edge (not input or output)
			bc_nodes_full = [BigClass.Strctr.InNodes_full,  BigClass.Strctr.GroundNodes_full]
		else:  # for all other tasks, no marbles to get stuck at boundary edge
			bc_nodes_full = [BigClass.Strctr.InNodes_full, BigClass.Strctr.FixedNodes_full, BigClass.Strctr.GroundNodes_full]
		for i in bc_nodes_full:
			length = len(i)
			self.fixK(BigClass, i, length)
		self.K_mat = np.eye(BigClass.Strctr.NE) * self.K

	def fixK(self, BigClass, nodes, length):
		for n in nodes.reshape(length):
			self.K[BigClass.Strctr.EJ == n] = BigClass.Variabs.K_max

	def solve_flow_const_K(self, BigClass, u, Cstr, f, iters_same_BCs):
		"""
		solve_flow_const_K solves the flow under given conductance configuration without changing Ks, until simulation converges

		inputs:
		K_max          - float, maximal conductance value
		NE             - int, # edges
		EI             - np.array, node number on 1st side of all edges
		u              - 1D array sized [NE + constraints, ], flow field at edges from previous solution iteration
		Cstr           - 2D array without last column, which is f from Rocks & Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116
		f              - constraint vector (from Rocks and Katifori 2018)
		iters_same_BSc - # iteration allowed under same boundary conditions (same constraints)

		outputs:
		p     - 1D array sized [NN + constraints, ], pressure at nodes at end of current iteration step
		u_nxt - 1D array sized [NE + constraints, ], flow velocity at edgses at end of current iteration step
		"""

		u_nxt = copy.copy(u)

		for o in range(iters_same_BCs):	

			# create effective conductivities if they are flow dependent
			K_eff = copy.copy(self.K)
			if BigClass.Variabs.K_type == 'flow_dep':
				K_eff[u_nxt>0] = BigClass.Variabs.K_max
			K_eff_mat = np.eye(BigClass.Strctr.NE) * K_eff

			L, L_bar = Matrixfuncs.buildL(BigClass.Strctr.DM, K_eff_mat, Cstr, BigClass.Strctr.NN)  # Lagrangian

			p, u_nxt = Solve.Solve_flow(L_bar, BigClass.Strctr.EI, BigClass.Strctr.EJ, K_eff, f, round=10**-10)  # pressure and flow
			
			# NETfuncs.PlotNetwork(p, u_nxt, self.K, BigClass, BigClass.Strctr.EIEJ_plots, BigClass.Strctr.NN, 
			# 					 BigClass.Strctr.NE, nodes='no', edges='yes', savefig='no')

			# print(np.where(u_nxt>0)[0])
			# print(np.where(u>0)[0])
			# print(np.where(u_nxt>0)[0] == np.where(u>0)[0])
			# print(np.all(np.where(u_nxt>0)[0] == np.where(u>0)[0]))
			# break the loop
			# since no further changes will be measured in flow and conductivities at end of next cycle
			if np.all(np.where(u_nxt>0)[0] == np.where(u>0)[0]):
			# if np.all(u_nxt == u):
				u = copy.copy(u_nxt)
				break
			else:
				# NETfuncs.PlotNetwork(p, u_nxt, self.K, BigClass, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE)
				u = copy.copy(u_nxt)

		return p, u_nxt

	def solve_flow_until_conv(self, BigClass, u, Cstr, f, iters_same_BCs, sim_type):
		"""
		solve_flow_until_conv solves the flow under same BCs while updating K until convergence. 
		used as part of flow_iterate()
		uses solve_flow_const_K()

		inputs:
		BigClass -       class instance including the user variables (Variabs), network structure (Strctr) and networkx (NET) class instances
				         I will not go into everything used from there to save space here.
	    u              - 1D array sized [NE + constraints, ], flow at each edge at beginning of iteration
        Cstr           - 2D array without last column (which is f from Rocks and Katifori 2018)
    	f              - constraint vector (from Rocks and Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
	    iters_same_BCs - int, maximal # iterations under same boundary condition
	    sim_type       - str, simulation type, see flow_iterate() function for descrip.

	    outputs:
	    p     - pressure at every node under the specific BC, after convergence while allowing conductivities to change
	    u_nxt - flow at every edge under the specific BC, after convergence while allowing conductivities to change
		"""

		for l in range(iters_same_BCs):

			if BigClass.Variabs.K_type == 'flow_dep':
				# print('solve flow under constant K for the %d time' %l)
				p, u_nxt = self.solve_flow_const_K(BigClass, u, Cstr, f, iters_same_BCs)
			else:
				L, L_bar = Matrixfuncs.buildL(BigClass.Strctr.DM, self.K_mat, Cstr, BigClass.Strctr.NN)  # Lagrangian

				p, u_nxt = Solve.Solve_flow(L_bar, BigClass.Strctr.EI, BigClass.Strctr.EJ, self.K, f)  # pressure and flow

			# NETfuncs.PlotNetwork(p, u_nxt, self.K, BigClass, BigClass.Strctr.EIEJ_plots, 
			# 	                 BigClass.Strctr.NN, BigClass.Strctr.NE, 
			# 					nodes='yes', edges='no', savefig='no')
			# NETfuncs.PlotNetwork(p, u_nxt, self.K, BigClass, BigClass.Strctr.EIEJ_plots, 
			# 	                 BigClass.Strctr.NN, BigClass.Strctr.NE, 
			# 					nodes='no', edges='yes', savefig='no')
			# plt.show()

			# if sim_type == 'w marbles' or sim_type == 'allostery test':  # Update conductivities
			if sim_type == 'w marbles':  # Update conductivities
				K_nxt = Matrixfuncs.ChangeKFromFlow(u_nxt, BigClass.Variabs.u_thresh, self.K, BigClass.Variabs.NGrid, 
													K_change_scheme=BigClass.Variabs.K_scheme, K_max=BigClass.Variabs.K_max, 
													K_min=BigClass.Variabs.K_min, beta=BigClass.Variabs.beta)
				
				self.K_mat = np.eye(BigClass.Strctr.NE) * K_nxt
				# print('difference in u %f ' %(np.mean(np.abs(u_nxt) - np.abs(u))/np.mean(np.abs(u_nxt))))
				K_old = copy.copy(self.K)
				self.K = copy.copy(K_nxt)

				# all input and output nodes have no marbles
				self.fixAllK(BigClass)
				# print('difference in K %d ' %int(np.sum(K_old != self.K)/2))

			# break the loop
			# since no further changes will be measured in flow and conductivities at end of next cycle
			if np.all(u_nxt == u):
				# print('converged, no further change in K')
				print('# iterations %d' %l)
				break
			else:
				# NETfuncs.PlotNetwork(p, u_nxt, self.K, BigClass, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE)
				u = copy.copy(u_nxt)

		return p, u_nxt

	def flow_iterate(self, BigClass, sim_type='w marbles', plot='yes', savefig='no'):
		"""
		Flow_iterate simulates the flow scheme as in Variabs.flow_scheme using network structure as in Strctr
		For explanation on flow scheme see the jupyter notebook "Flow Network Simulation...ipynb"
		Optionally plots the network as a networkx graph

		input:
		BigClass - class instance including the user variables (Variabs), network structure (Strctr) and networkx (NET) class instances
				   I will not go into everything used from there to save space here.
		sim_type - simulation type: 'no marbles'     - flows from 1 input at a time and all outputs, all edges have high State.K
		                            'allostery test' - flows from 1 input at a time and all outputs, K's as in State.K
		                            'w marbles'      - (default) flows from 1 input and one output K's as in State.K, 
		                                               update conductivities as due to flow u and calculate flow again so
		                                               further update of conductivities will not change a thing
		plot     - flag: 'no'   - do not plot anything
		                 'yes'  - plot every iteration in Variabs.iterations
		                 'last' - only last couple of steps, one for each input
		save_fig - str, 'yes' saves the figure plotted as png, 'no' doesn't. default='no'

		output:
		u_final - 2D np.array [2, 2] flow out of output nodes / edges, rows for different inputs, columns different outputs
		u_all   - 2D np.array [NE, iterations] of flows through all edges for all simulation steps
		K_all   - 2D np.array [NE, iterations] of conductivities on all edges for all simulation steps
		"""
		# dummy size for convenience
		circ_step = copy.copy(BigClass.Variabs.circle_step)
		iters = copy.copy(BigClass.Variabs.iterations)
		NE = copy.copy(BigClass.Strctr.NE)
		NN = copy.copy(BigClass.Strctr.NN)
		EI = copy.copy(BigClass.Strctr.EI)
		EJ = copy.copy(BigClass.Strctr.EJ)
		EIEJ_plots = copy.copy(BigClass.Strctr.EIEJ_plots)

		# Determine # iterations
		iters, iters_same_BCs = self.num_iterations(BigClass, sim_type)

		# Initiate data arrays
		Hamming_i, MSE_i, stop_bool = self.initiateArrays(BigClass, NE, iters)

		# Iterate - solve flow and optionally update conductivities
		for i in range(iters):
			cycle = int(np.floor(i/circ_step))  # task goes over multiple cycles of training, keep the cycle num.
			print('cycle # %d' %cycle)
			
			# specific constraints for training step
			NodeData, Nodes, EdgeData, Edges, GroundNodes = BigClass.Strctr.Constraints_afo_task(BigClass, sim_type, i)

			# BC and constraints as matrix
			Cstr_full, Cstr, f = Constraints.ConstraintMatrix(NodeData, Nodes, EdgeData, Edges, GroundNodes, 
															  NN, EI, EJ)  

			# Build Lagrangians and solve flow, optionally update conductivities and repeat.
			u = np.zeros([NE,])

			# solve for flow interatively until convergence under same BCs, optionally changing K due to flow
			p, u = self.solve_flow_until_conv(BigClass, u, Cstr, f, iters_same_BCs, sim_type)

			# Save data in assigned arrays
			if sim_type == 'allostery test' or sim_type == 'no marbles':
				self.u = u
				self.p = p
				self.u_final[i,:] = [np.sum(u[BigClass.Strctr.output_edges[0]]), np.sum(u[BigClass.Strctr.output_edges[1]])]
			else:
				self.p = p
				self.u_all[:, i] = u
				self.K_all[:, i] = self.K
				self.K_cells[:, i] = Matrixfuncs.K_by_cells(self.K, BigClass.Variabs.K_min, BigClass.Variabs.NGrid)
				self.power_dissip[i] = Statistics.power_dissip(u, self.K)
				if i >= circ_step:
					MSE_i = Statistics.flow_MSE(self.u_all[:, i-circ_step], circ_step, u)
					Hamming_i = Statistics.K_Hamming(self.K_cells[:, i-circ_step], circ_step, self.K_cells[:, i])
					self.MSE[i-circ_step] = MSE_i
					self.Hamming[i-circ_step] = Hamming_i

			# Optionally plot
			if plot == 'yes' or (plot == 'last' and (i == (iters - 1) or i == (iters - 2))):
				print('condition supplied')
				# NETfuncs.PlotNetwork(p, u, self.K, BigClass, EIEJ_plots, NN, NE, 
				# 					nodes='yes', edges='no', savefig=savefig)
				NETfuncs.PlotNetwork(p, u, self.K, BigClass, EIEJ_plots, NN, NE, 
									nodes='yes', edges='yes', pressureSurf='yes', savefig=savefig)
				plt.show()

			# plot if convergence reached before iters # of iterations
			if stop_bool == 1:
				NETfuncs.PlotNetwork(p, u, self.K, BigClass, EIEJ_plots, NN, NE, 
									nodes='yes', edges='no', savefig=savefig)
				NETfuncs.PlotNetwork(p, u, self.K, BigClass, EIEJ_plots, NN, NE, 
									nodes='no', edges='yes', savefig=savefig)
				break

			# obtain # of iterations until convergence of conductivities after full cycle of changing BCs and break the loop
			# since no further changes will be measured in flow and conductivities at end of next cycle
			if np.isnan(self.convergence_time) and MSE_i == 0.0:
				self.convergence_time = cycle
				print('loop break')
				if plot == 'last':
					stop_bool = 1  # go one more time, for plotting
					if BigClass.Variabs.flow_scheme == 'unidir' or BigClass.Variabs.flow_scheme == 'taktak':
						NETfuncs.PlotNetwork(p, u, self.K, BigClass, EIEJ_plots, NN, NE, 
											nodes='yes', edges='no', savefig=savefig)
						NETfuncs.PlotNetwork(p, u, self.K, BigClass, EIEJ_plots, NN, NE, 
											nodes='no', edges='yes', savefig=savefig)
						plt.show()
				else:
					break 

	def num_iterations(self, BigClass, sim_type):
		"""
		num_iterations determines maximal # iterations in total flow scheme (iters) and in same boundary conditions (BCs)

		inputs:
		task_type - str, task that is being simulated, see User_variables() class for descrip.
		K_type    - str, effect of flow on conductivity without changing marble positions, see User_variables() class for descrip.
		sim_type  - str, simulation type, see flow_iterate() function for descrip.

		outputs:
		iters          - int, maximal # iterations in total flow scheme
		iters_same_BCs - int, maximal # iterations under same boundary conditions
		"""
		if sim_type=='w marbles' and BigClass.Variabs.task_type!='Channeling_diag' and BigClass.Variabs.task_type!='Channeling_straight':
			iters = BigClass.Variabs.iterations  # total iteration #
			iters_same_BCs = iters  # iterations at every given B.C. - solve flow and update K, repeat 3 times.
		elif BigClass.Variabs.task_type=='Channeling_diag' or BigClass.Variabs.task_type=='Channeling_straight':
			iters = 3
			iters_same_BCs = iters 
		elif BigClass.Variabs.task_type=='Flow_clockwise' and sim_type!='w marbles':  # test stage, flow direction task
			iters = 1
			iters_same_BCs = BigClass.Variabs.iterations
		else:
			iters = 2  # 1 flow iteration for every Boundary Condition
			if BigClass.Variabs.K_type=='flow_dep':
				iters_same_BCs = iters  # still requires some time to converge due to flow feedback since K is not bidirectional
			else:
				iters_same_BCs = 1  # 1 iteration at every Boundary Condition, since conductivities do not change
		return iters, iters_same_BCs


class Networkx_net:
	"""
	Networkx_net contains networkx data for plots

	inputs:

	outputs:
	NET - networkx net object (initially empty)
	"""
	def __init__(self):
		super(Networkx_net, self).__init__()
		self.NET = []

	def buildNetwork(self, BigClass):
		"""
	    Builds a networkx network using edges from EIEJ_plots which are built upon EI and EJ at "Matrixfuncs.py"
	    After this step, the order of edges at EIEJ_plots and in the networkx net is not the same which is shit
	    
	    input:
	    EIEJ_plots - 2D np.array sized [NE, 2] - 
	                 EIEJ_plots[i,0] and EIEJ_plots[i,1] are input and output nodes to edge i, respectively
	    
	    output:
	    NET - networkx network containing just the edges from EIEJ_plots
	    """
		EIEJ_plots = BigClass.Strctr.EIEJ_plots
		NET = nx.DiGraph()  # initiate graph object
		NET.add_edges_from(EIEJ_plots)  # add edges 
		self.NET = NET

	def build_pos_lattice(self, BigClass, plot='no'):
		"""
		build_pos_lattice builds the lattice of positions of edges and nodes

		inputs:
		net_typ - str, layout for NETfuncs plotNetStructure(). 
	              'Cells' is Roie's style of network and is default
	              'Nachi' is Nachi style

	    outputs:
	    pos_lattice - dict, positions of nodes from NET.nodes
		"""
		self.pos_lattice = NETfuncs.plotNetStructure(self.NET, BigClass.Variabs.NGrid, layout=BigClass.Variabs.net_typ, plot=plot)