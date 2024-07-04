import numpy as np
import copy

import numpy.random as rand
import pickle
import time
import networkx as nx
import matplotlib.pyplot as plt

from numpy import array as array
from sklearn import datasets

import Solve_CUDA
import Solve

import NETfuncs, Matrixfuncs, Constraints, Statistics, FileFuncs, DatasetManipulations


class User_variables:
	"""
	User_variables saves all user variables for network simulation

	inputs:
	NGrid            - int, lattice dimension is Ngrid X Ngrid
	input_p          - float, pressure at input node
	flow_scheme      - str, order of pressure appliance in training and test
					   'one_shot' = apply pressure drop from 1 output node and 1 output node, wait till convergence
					   'unidir'   = apply pressure drop only in the regular directions - constrained node = positive, ground = 0
                                    there are 2 input and output pairs, exchange between them
                       'taktak'   = apply pressure drop unidir once, meaning 1st input and output pair and then 2nd pair.
                                    then switch ground and constrained nodes to apply oposite dir.
	task_type        - str, task that is being simulated
					   'Allostery_one_pair'  = 1 pair of input and outputs
					   'Allostery'           = 2 pairs of input and outputs
					   'XOR'                 = 2 inputs and 2 outputs. difference between output nodes encodes the XOR result of the 2 inputs
					   'Channeling_diag'     = 1st from input to diagonal output, then from output to 2 perpindicular nodes. 
                                               test from input to output
                       'Channeling_straight' = 1st from input to output on same column, then from output to 2 perpindicular nodes. 
                                               test from input to output (same as 1st)
                	   'Counter'             = column of cells bottomost and topmost nodes are input/output (switching), 
                                               rightmost nodes (1 each row) ground. more about the task in "_counter.ipynb".
                       'Iris'                = classification, sklearn's iris dataset, 4 inputs 3 output classes
	K_scheme         - str, scheme to change conductivities
					   'propto_current_squared' = conductivity on edge changes due to squared on edge (use beta argument), no marbles involves
					   'marbles_pressure'       = conductivities in each cells change due to marbles moving due to pressure difference.
					                              binary values K_min, K_max
					   'marbles_u'              = conductivities in each cells change due to marbles moving due to flow velocity.
					                              binary values K_min, K_max
	K_type           - str, effect of flow on conductivity without changing marble positions
					   'bidir'    = conductivity is the same regardless of flow directions
					   'flow_dep' = conductivity depends on flow direction - if into cell then maximal, if out and there is a marble then lower
	iterations       - int, # iterations allowed under flow cycles / updating conductivities
	input_nodes_lst  - array of all input nodes in task, even when they switch roles. State.flow_iterate() will handle them.
    output_nodes_lst - array of all output nodes in task, even when they switch roles. State.flow_iterate() will handle them.
	Periodic         - bool, 'True'=lattice has periodic boundaries, default='False'
	net_typ          - str, layout for NETfuncs plotNetStructure(). 
	                   'Cells' is Roie's style of network and is default
	                   'Nachi' is Nachi style
	u_thresh         - float, threshold to move marbles, default=1
	fixed_nodes      - 1D array, numbers the nodes with fixed value assigned by fixed_node_p() function, for 'XOR' task, default=0
	K_max            - default=1, maximal conductivity value, i.e. the value of edge without marble
	K_min            - default=0.5, minim conductivity value, i.e. the value of edge with marble
	beta             - default=0.0, multiplication factor in conductivity update scheme "propto_flow"
	train_frac       - default=0.0, fraction of data used for training set in classification task
	sub_task_type    - default='None', another specification of task, for regression whether 2output or not etc.
	flow_scheme      - str, if user knows flow scheme.
					   'unidir' - input is input, output is output
					   'taktak' - every 2nd cycle inputs and outputs switch. For "Allostery_constrastive" fixed nodes also switch
					              Reminder: for old allostery, the cycle is twice as long
	"""

	def __init__(self, NGrid, input_p, task_type, K_type, iterations, input_nodes_lst, output_nodes_lst, Periodic='False', net_typ='Cells', 
		         u_thresh=1, fixed_nodes=0, K_max=1, K_min=0.5, beta=0.0, train_frac=0.0, desired_p_frac=0.0, etta=0.0, mag_factor=1.0, sub_task_type='None',
		         flow_scheme='None'):
		self.NGrid = NGrid		
		if len(input_p)==1:
			self.input_p = input_p

		self.task_type = task_type
		# Schemes to change conductivities and order of pressure appliance in training and test, given task
		if task_type == 'Iris':
			iris = datasets.load_iris()
			iris_data = iris.data
			iris_data_norm = iris_data/iris_data.mean(axis=0)
			iris_target = iris.target
			self.train_data, self.train_target, self.test_data, self.test_target = DatasetManipulations.divide_train_test(iris_data, iris_target, train_frac)
			self.K_scheme = 'marbles_pressure'
			self.flow_scheme = 'unidir'  # apply pressure drop only in the regular directions - constrained node = positive, ground = 0
		                                 # there are 2 input and output pairs, exchange between them
			self.net_typ = 'Cells'
		elif task_type == 'Regression_contrastive':
			data_size=15
			# regression_data = DatasetManipulations.cartesian(([np.arange(1,data_size), np.arange(1,data_size)]))/data_size
			regression_data = np.array(np.meshgrid(np.arange(1,data_size), np.arange(1,data_size))).T.reshape(-1, 2)/data_size
			regression_target = np.matmul(regression_data, desired_p_frac)
			self.train_data, self.train_target, self.test_data, self.test_target = DatasetManipulations.divide_train_test(regression_data, regression_target, train_frac)
			self.K_scheme = 'marbles_pressure'
			self.flow_scheme = 'unidir'  # apply pressure drop only in the regular directions - constrained node = positive, ground = 0
		                                 # there are 2 input and output pairs, exchange between them
			self.net_typ = 'Cells'                             
		elif task_type == 'Allostery_one_pair':
		    self.K_scheme = 'propto_current_squared'
		    self.flow_scheme = 'one_shot'  # apply pressure drop from 1 output node and 1 output node, wait till convergence
		    self.net_typ = 'Cells'
		elif task_type == 'Channeling_straight' or task_type == 'Channeling_diag':
		    self.K_scheme = 'marbles_pressure'
		    self.flow_scheme = 'one_shot'  # apply pressure drop from 1 output node and 1 output node, wait till convergence
		    self.net_typ = 'Cells'
		elif task_type == 'Counter' or task_type == 'Memristor':
		    self.K_scheme = 'marbles_pressure'
		    self.flow_scheme = 'unidir'
		    self.net_typ = 'oneCol'
		elif task_type == 'Iris':
		    self.K_scheme = 'marbles_pressure'
		    self.flow_scheme = 'Contrastive'
		    self.net_typ = 'Cells'
		else:
		    self.K_scheme = 'marbles_pressure'
		    if flow_scheme == 'taktak':  # If user gave specific flow scheme
		    	self.flow_scheme = 'taktak'
		    else:
		    	self.flow_scheme = 'unidir'  # apply pressure drop only in the regular directions - constrained node = positive, ground = 0
		        	                         # there are 2 input and output pairs, exchange between them
		    # flow_scheme = 'taktak'  # apply pressure drop unidir once, meaning 1st input and output pair and then 2nd pair.
		                              # then switch ground and constrained nodes to apply oposite dir.
		    self.net_typ = 'Cells'
		# additional sizes given specific tasks
		if self.task_type == 'XOR' or self.task_type == 'Counter' or self.task_type == 'Allostery_contrastive' or self.task_type == 'Regression_contrastive':
			self.fixed_nodes = fixed_nodes
			if self.task_type == 'Allostery_contrastive' or self.task_type == 'Regression_contrastive':
				if np.size(desired_p_frac)>len(fixed_nodes)*len(input_nodes_lst):
					self.desired_p_frac = desired_p_frac[0]  # not in use for Regression, rather regression_target is used
				else:
					self.desired_p_frac = desired_p_frac  # not in use for Regression, rather regression_target is used
				self.etta = etta
				self.mag_factor = mag_factor	
			else:
				self.mag_factor = 1.0
		else: 
			self.fixed_nodes = array([])
			self.mag_factor = 1.0

		# num of iterations in every cycle, for calculation of convergence
		if self.flow_scheme == 'taktak':
			self.cycle_len = 4
		elif self.flow_scheme == 'unidir' or self.flow_scheme == 'contrastive' or self.flow_scheme == 'Hebbian_like' or self.flow_scheme == 'dual':
			self.cycle_len = 2
		else:
			self.cycle_len = 1

		self.iterations = iterations
		# self.input_output_pairs = input_output_pairs
		self.input_nodes_lst = input_nodes_lst
		self.output_nodes_lst = output_nodes_lst
		self.Periodic = Periodic
		self.u_thresh = u_thresh  # for task_type=='Memristor' this is not u_thresh used. It is updated in self.update_u_thresh
		self.K_type = K_type
		self.K_max = K_max
		self.K_min = K_min
		self.beta = beta

		self.sub_task_type = sub_task_type

	def assign_input_p(self, p):
		"""
		assign_input_p assigns input pressure value to Variabs instance, input nodes will have this value
		also assign the desired pressure if contrastive learning is used

		inputs:
		p - float, pressure for all inlets
		"""
		self.input_p = p
		if self.task_type == 'Allostery_contrastive':
			self.desired_p = self.desired_p_frac * p
		elif self.task_type == 'Regression_contrastive':
			self.input_p = self.input_p * np.ones(2)

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
			self.fixed_node_p = array([p/3, 2*p/3])
		elif self.task_type == 'Counter':
			self.fixed_node_p = np.ones([self.NGrid, ])*p
		else:
			self.fixed_node_p = array([])
			print('no fixed nodes other than input')

	def update_u_thresh(self, Strctr):
		if self.task_type == 'Memristor':
			u_thresh_updt = np.ones(Strctr.NE)
			u_thresh_updt[0:self.NGrid*4] = np.repeat(self.u_thresh, 4)
			self.u_thresh = u_thresh_updt
			print(self.u_thresh)


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
		EI         - array, node number on 1st side of all edges
		EJ         - array, node number on 2nd side of all edges
		EIEJ_plots - array, combined EI and EJ, each line is two nodes of edge, for visual ease
		DM         - array, connectivity matrix NE X NN
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

		# boundaries are zero flux
		if BigClass.Variabs.task_type == 'Counter' or BigClass.Variabs.task_type == 'Memristor':  # only a single column of cells
			NConnections = int(NGrid-1)  # connections between cells
			left_side = [0 + 4*i for i in range(NGrid)]  # enumerate leftmost edges in network
			bottom_side = [1]  # enumerate bottommost edge in network
			right_side = [2, 4*NGrid-2]  # right side is ground, other than topmost and bottommost cells, so no need to treat as boundary
			# right_side = [4*NGrid-2]  # right side is ground, other than topmost and bottommost cells, so no need to treat as boundary
			# right_side = np.empty([], dtype=int)  # right side is ground, other than topmost and bottommost cells, so no need to treat as boundary
			top_side = [4*NGrid-1]  # enumerate topmost edges in network
		else:  # N-by-N number of cells
			NConnections = int(NGrid*(NGrid-1)*2)  # connections between cells
			left_side = [0 + 4*NGrid*i for i in range(NGrid)]  # enumerate leftmost edges in network
			bottom_side = [1 + 4*i for i in range(NGrid)]  # enumerate bottommost edges in network
			right_side = [2 + 4*(NGrid-1) + 4*NGrid*i for i in range(NGrid)]  # enumerate rightmost edges in network
			top_side = [4*NGrid*(NGrid-1) + 3 + 4*i for i in range(NGrid)]  # enumerate topmost edges in network		
		# concatenate all boundaries
		EdgesBounaries = np.append(left_side, np.append(bottom_side, np.append(right_side, top_side)))

		# connection between cells are zero flux
		EdgesConnections = [int(i) for i in range(self.NE-NConnections, self.NE)]  # enumerate all number from NConnections to NE

		# connect them all and save in Strctr class instance
		self.EdgesTotal = np.append(EdgesConnections, EdgesBounaries)

	def Setup_constraints(self, BigClass):
		"""
		Setup_constraints sets up the constraints on the network for specific run, in form of 2D arrays

		inputs:
		input_output_pairs - array of input and output node pairs. State.flow_iterate() will know how to handle them.
		fixed_nodes        - 1D array, numbers the nodes with fixed value assigned by fixed_node_p() function, for 'XOR' task, default=0
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
		"""

		# dummy variab for ease
		InNodes_full = copy.copy(BigClass.Variabs.input_nodes_lst)  # input p node
		GroundNodes_full = copy.copy(BigClass.Variabs.output_nodes_lst)  # output node has 0 pressure
		# in_out_pairs = copy.copy(BigClass.Variabs.input_output_pairs)
		fixed_nodes = copy.copy(BigClass.Variabs.fixed_nodes)  # nodes with fixed p

		# lengths
		if np.size(fixed_nodes)>0:
			# len_fixed = np.shape(fixed_nodes)[1]
			len_fixed = np.shape(fixed_nodes)[0]
		else:
			len_fixed = 0

		# # create constraints
		# input p value
		if type(BigClass.Variabs.input_p)==np.int32 or type(BigClass.Variabs.input_p)==np.float64:  # if p is single value
			InNodeData_full = np.ones(np.shape(InNodes_full)[0])*BigClass.Variabs.input_p  
		else:  # if p is vector of many inputs
			InNodeData_full = BigClass.Variabs.input_p 

		# if fixed nodes exist, use them:
		if len_fixed>0:
			if 'BigClass.Variabs.fixed_node_p' in locals():
				FixedNodeData_full = array([BigClass.Variabs.fixed_node_p[i] for i in range(len_fixed)])  # input p value
			else:
				FixedNodeData_full = np.zeros(len_fixed)
			# input p node
			# FixedNodes_full = array([fixed_nodes[0, i] for i in range(len_fixed)])  
			FixedNodes_full = copy.copy(fixed_nodes)
		else:
			FixedNodeData_full = array([])
			FixedNodes_full = array([])
		# nodes with zero pressure
		# GroundNodes_full = array([[in_out_pairs[i, 1]] for i in range(len_in_out_pairs)])  
		# GroundNodes_full_Allostery = array([GroundNodes_full[i][0] for i in range(len(GroundNodes_full))])
		EdgeData_full = array([0, 0])  # pressure drop value on edges specified by Edges_full
		# Full list of edges that have specified pressure drop. it is different from EdgesTotal only if
		# there are non-zero boundary conditions at periphery of network.
		# Edges_full = [self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[0])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])], 
		#                     self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[1])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])]]
		Edges_full = copy.copy(self.EdgesTotal)

		# output edges which are ground node but don't have to be in middle of cell
		if BigClass.Variabs.task_type == 'Allostery_contrastive' or BigClass.Variabs.task_type == 'Regression_contrastive':  # for regression task in Nachi&Sam method, outputs are the fixed nodes. 
																    														 # They are fixed only every 2nd iteration, from "Constraints.constaints_afo_task".
			self.output_edges = array([np.where(np.append(self.EI, self.EJ)==FixedNodes_full[i])[0] % len(self.EI) 
		                               for i in range(len(FixedNodes_full))])
			self.input_edges = array([np.where(np.append(self.EI, self.EJ)==InNodes_full[i])[0] % len(self.EI) 
		                               for i in range(len(InNodes_full))])
			self.ground_edges = array([np.where(np.append(self.EI, self.EJ)==GroundNodes_full[i])[0] % len(self.EI) 
		                               for i in range(len(GroundNodes_full))]), 
		else:  # outputs are ground nodes
			self.output_edges = array([np.where(np.append(self.EI, self.EJ)==GroundNodes_full[i])[0] % len(self.EI) 
		                               for i in range(len(GroundNodes_full))])

		# add all B.C.s to Strctr instance
		self.InNodeData_full = InNodeData_full
		self.InNodes_full = InNodes_full
		# If there are nodes with fixed pressure which are not input/output (for 'XOR' for example)
		if len_fixed>0:
			self.FixedNodeData_full = FixedNodeData_full
			self.FixedNodes_full = FixedNodes_full
		else:
			self.FixedNodeData_full = array([])
			self.FixedNodes_full = array([])
		self.GroundNodes_full = GroundNodes_full
		# self.GroundNodes_full_Allostery = GroundNodes_full_Allostery
		self.EdgeData_full = EdgeData_full
		self.Edges_full = Edges_full
		# self.Edges_full_Allostery = Edges_full_Allostery


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
	def __init__(self, use_gpu=False):
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

	def initiateK(self, BigClass, noise='no', noise_amp=0., gpu=False):
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
		K     - 1D array sized [NEdges] with initial conductivity values for every edge
		K_mat - 2D cubic array sized [NEdges] with initial conductivity values on diagonal
		"""
		NE = BigClass.Strctr.NE  # # edges
		K_max = BigClass.Variabs.K_max  # 2D array sized [NE, NE], conductances on diagonal of matrix 
		frac_moved = noise_amp  # fraction of marbles moved, on average
		u_thresh = BigClass.Variabs.u_thresh  # float, threshold of flow above which marbles move

		self.K_backg = K_max*np.ones([NE])
		# print(self.K)
		if BigClass.Variabs.task_type == 'Counter':
			for i in range(BigClass.Variabs.NGrid-2):
				self.K_backg[(i+1)*4+2] = K_max / ((BigClass.Variabs.NGrid - i - 2) * 3 - 2)
			# print('background resistances')
			# print(1/self.K_backg)
			self.K = copy.copy(self.K_backg)
		elif BigClass.Variabs.task_type == 'Memristor':  # use previous values for K
			if len(self.K)>0:  # if a previous K exists
				print('pass is true')
				pass
			else:  # if not, create one
				self.K = copy.copy(self.K_backg)
		else:
			self.K_backg = K_max*np.ones([NE])
			self.K = copy.copy(self.K_backg)
			
		if noise == 'rand_u' or noise == 'rand_K':
			u_rand = BigClass.Solver.solve.create_randomized_u(BigClass, NE, frac_moved, u_thresh, noise)
			self.K = Matrixfuncs.ChangeKFromFlow(u_rand, u_thresh, self.K, self.K_backg, BigClass.Variabs.NGrid, 
												K_change_scheme=BigClass.Variabs.K_scheme, K_max=BigClass.Variabs.K_max, 
												K_min=BigClass.Variabs.K_min)  # move marble, change conductivities
		
		self.K_mat = np.eye(NE) * self.K  # save as matrix

	def fixAllK(self, BigClass):
		if BigClass.Variabs.task_type == 'Counter':  # for counter task, allow marble to get stuck at boundary edge (not input or output)
			bc_nodes_full = [BigClass.Strctr.InNodes_full,  BigClass.Strctr.GroundNodes_full]
		else:  # for all other tasks, no marbles to get stuck at boundary edge
			bc_nodes_full = [BigClass.Strctr.InNodes_full, BigClass.Strctr.FixedNodes_full, BigClass.Strctr.GroundNodes_full]
		for i in bc_nodes_full:
			self.fixK(BigClass, i)
		self.K_mat = np.eye(BigClass.Strctr.NE) * self.K

	def fixK(self, BigClass, nodes):
		length = len(nodes)
		for n in nodes.reshape(length):
			# self.K[BigClass.Strctr.EJ == n] = BigClass.Variabs.K_max
			self.K[BigClass.Strctr.EJ == n] = self.K_backg[BigClass.Strctr.EJ == n]

	def solve_flow_const_K(self, BigClass, u, Cstr, f, iters_same_BCs):
		"""
		solve_flow_const_K solves the flow under given conductance configuration without changing Ks, until simulation converges

		inputs:
		K_max          - float, maximal conductance value
		NE             - int, # edges
		EI             - array, node number on 1st side of all edges
		u              - 1D array sized [NE + constraints, ], flow field at edges from previous solution iteration
		Cstr           - 2D array without last column, which is f from Rocks & Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116
		f              - constraint vector (from Rocks and Katifori 2018)
		iters_same_BSc - # iteration allowed under same boundary conditions (same constraints)

		outputs:
		p     - 1D array sized [NN + constraints, ], pressure at nodes at end of current iteration step
		u_nxt - 1D array sized [NE + constraints, ], flow velocity at edgses at end of current iteration step
		"""

		u_nxt = copy.copy(u)

		for o in range(iters_same_BCs):	 # even though balls don't move, o>0 since the effective K matrix depends on direction of flow
										 # and flow changes due to effective K so need to iterate.

			# create effective conductivities if they are flow dependent
			K_eff = copy.copy(self.K)
			if BigClass.Variabs.K_type == 'flow_dep':
				# K_eff[u_nxt>0] = BigClass.Variabs.K_max
				K_eff[u_nxt>0] = self.K_backg[u_nxt>0]
			K_eff_mat = np.eye(BigClass.Strctr.NE) * K_eff
			L, L_bar = Matrixfuncs.buildL(BigClass, BigClass.Strctr.DM, K_eff_mat, Cstr, BigClass.Strctr.NN)  # Lagrangian

			p, u_nxt = BigClass.Solver.solve.Solve_flow(L_bar, BigClass.Strctr.EI, BigClass.Strctr.EJ, K_eff, f, round=10**-10)  # pressure and flow
			# NETfuncs.PlotNetwork(p, u_nxt, self.K, BigClass, BigClass.Strctr.EIEJ_plots, BigClass.Strctr.NN, 
			# 					 BigClass.Strctr.NE, nodes='yes', edges='yes', savefig='no')

			# break the loop
			# since no further changes will be measured in flow and conductivities at end of next cycle
			if np.all((u_nxt>0) == (u>0)):
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

		for l in range(iters_same_BCs):  # l>0 since the balls move, change K matrix which changes flow.
										 # flow then changes K so need to iterate over l>0.
			# print('Iteration # ', l, 'under the same BC')
			if BigClass.Variabs.K_type == 'flow_dep':  # update conductivities proportional to flow, no marbles, this is unphysical		
				p, u_nxt = self.solve_flow_const_K(BigClass, u, Cstr, f, iters_same_BCs)
				# print('calculated u, %d edges contain different values than before' % np.size(np.where(u_nxt-u)))
				# p, u_nxt = BigClass.Solver.solve.solve_flow_const_K(self.K, BigClass, u, Cstr, f, iters_same_BCs)
			else:    # update conductivities due to movement of marbles
				L, L_bar = Matrixfuncs.buildL(BigClass, BigClass.Strctr.DM, self.K_mat, Cstr, BigClass.Strctr.NN)  # Lagrangian
				p, u_nxt = BigClass.Solver.solve.Solve_flow(L_bar, BigClass.Strctr.EI, BigClass.Strctr.EJ, self.K, f)  # pressure and flow

			# if sim_type == 'w marbles' or sim_type == 'allostery test':  # Update conductivities
			if sim_type == 'w marbles':  # Update conductivities
				K_nxt = Matrixfuncs.ChangeKFromFlow(u_nxt, BigClass.Variabs.u_thresh, self.K, self.K_backg, BigClass.Variabs.NGrid, 
													K_change_scheme=BigClass.Variabs.K_scheme, K_max=BigClass.Variabs.K_max, 
													K_min=BigClass.Variabs.K_min, beta=BigClass.Variabs.beta)
				# print('calculated K, %d edges contain different values than before' % np.size(np.where(K_nxt-self.K)))
				self.K_mat = np.eye(BigClass.Strctr.NE) * K_nxt
				K_old = copy.copy(self.K)
				self.K = copy.copy(K_nxt)

				# all input and output nodes have no marbles
				self.fixAllK(BigClass)

			# break the loop
			# since no further changes will be measured in flow and conductivities at end of next cycle
			if np.all(u_nxt == u):
				break
			else:
				# NETfuncs.PlotNetwork(p, u_nxt, self.K, BigClass, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE)
				# NETfuncs.PlotNetwork(p, u_nxt, self.K, BigClass, BigClass.Strctr.EIEJ_plots, BigClass.Strctr.NN, 
				# 				 BigClass.Strctr.NE, nodes='yes', edges='yes', savefig='no')
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
		u_final - 2D array [2, 2] flow out of output nodes / edges, rows for different inputs, columns different outputs
		u_all   - 2D array [NE, iterations] of flows through all edges for all simulation steps
		K_all   - 2D array [NE, iterations] of conductivities on all edges for all simulation steps
		"""
		# dummy size for convenience
		cyc_len = copy.copy(BigClass.Variabs.cycle_len)
		iters = copy.copy(BigClass.Variabs.iterations)
		NE = copy.copy(BigClass.Strctr.NE)
		NN = copy.copy(BigClass.Strctr.NN)
		EI = copy.copy(BigClass.Strctr.EI)
		EJ = copy.copy(BigClass.Strctr.EJ)
		EIEJ_plots = copy.copy(BigClass.Strctr.EIEJ_plots)
		self.p_nudge = np.zeros(2)

		if BigClass.Variabs.task_type == 'Regression_contrastive' or BigClass.Variabs.task_type == 'Allostery_contrastive':
			if BigClass.Variabs.flow_scheme == 'dual':
				self.error_vec = np.zeros(int(np.ceil(BigClass.Variabs.iterations/2)), 2)  # error of desired vs. measured output pressure
				self.ratio_vec = np.zeros(int(np.ceil(BigClass.Variabs.iterations/2)), 2)  # ratio between the two pressure outputs
				if BigClass.Variabs.task_type == 'Allostery_contrastive':
					self.p_dual_vec = np.zeros(int(np.ceil(BigClass.Variabs.iterations/2)), 2)  # ratio between the two pressure outputs
					self.output_dual_vec = np.zeros(int(np.ceil(BigClass.Variabs.iterations/2)))  # ratio between the two pressure outputs
					self.p_vec = np.zeros(int(np.ceil(BigClass.Variabs.iterations/2)), 2)  # ratio between the two pressure outputs
					self.output_vec = np.zeros(int(np.ceil(BigClass.Variabs.iterations/2)))  # ratio between the two pressure outputs
				elif BigClass.Variabs.task_type == 'Allostery_contrastive':
					self.p_dual_vec = np.zeros(int(np.ceil(BigClass.Variabs.iterations/2)))  # ratio between the two pressure outputs
					self.output_dual_vec = np.zeros(int(np.ceil(BigClass.Variabs.iterations/2)), 2)  # ratio between the two pressure outputs
					self.p_vec = np.zeros(int(np.ceil(BigClass.Variabs.iterations/2)))  # ratio between the two pressure outputs
					self.output_vec = np.zeros(int(np.ceil(BigClass.Variabs.iterations/2)), 2)  # ratio between the two pressure outputs
			else:
				self.error_vec = np.zeros(int(np.ceil(BigClass.Variabs.iterations/2)))  # error of desired vs. measured output pressure
				self.ratio_vec = np.zeros(int(np.ceil(BigClass.Variabs.iterations/2)))  # ratio between the two pressure outputs
			self.u_final_vec = np.zeros([int(np.ceil(BigClass.Variabs.iterations)), 2])  # ratio between the two pressure outputs

		# Determine # iterations
		iters, iters_same_BCs = self.num_iterations(BigClass, sim_type)

		# Initiate data arrays
		Hamming_i, MSE_i, stop_bool = self.initiateArrays(BigClass, NE, iters)

		# Iterate - solve flow and optionally update conductivities
		for i in range(iters):
			cycle = int(np.floor(i/cyc_len))  # task goes over multiple cycles of training, keep the cycle num.
			print('cycle # %d' %cycle)
			
			# if regression task, pull a sample from training set
			if BigClass.Variabs.task_type == 'Regression_contrastive':
				# cycle=0
				train_sample = BigClass.Variabs.train_data[cycle]
				print('train_sample ', train_sample, ' train_target ', BigClass.Variabs.train_target[cycle])
			else:  # no training set needed
				train_sample = np.array([])			

			# specific constraints for training step	
			NodeData, Nodes, EdgeData, Edges, GroundNodes = Constraints.Constraints_afo_task(BigClass.Variabs.task_type, BigClass.Variabs.flow_scheme, 
				                                                                             sim_type, BigClass.Strctr.InNodeData_full, 
				                                                                             BigClass.Strctr.InNodes_full, 
				                                                                             BigClass.Strctr.FixedNodeData_full, 
				                                                                             BigClass.Strctr.FixedNodes_full, 
				                                                                             BigClass.Strctr.GroundNodes_full, 
				                                                                             BigClass.Strctr.EdgeData_full, 
				                                                                             BigClass.Strctr.Edges_full, i, train_sample, self.p_nudge,
				                                                                             BigClass.Variabs.mag_factor)

			# if BigClass.Variabs.task_type == 'Regression_contrastive':  # update input p for this loop step as calculated in Constraints_afo_task
			# 	BigClass.Variabs.assign_input_p(NodeData)
			# 	print(BigClass.Variabs.input_p)

			# BC and constraints as matrix
			Cstr_full, Cstr, f = Constraints.ConstraintMatrix(NodeData, Nodes, EdgeData, Edges, GroundNodes, 
															  NN, EI, EJ)  

			# initiate zeros for velocity field
			u = np.zeros([NE,])
			
			# Build Lagrangians and solve for flow iteratively until convergence under same BCs, 
			# optionally changing K due to flow
			self.p, self.u = self.solve_flow_until_conv(BigClass, u, Cstr, f, iters_same_BCs, sim_type)

			# Save data in assigned arrays
			self.save_state_statistics(BigClass, sim_type, cyc_len, cycle, i, NodeData)

			if BigClass.Variabs.flow_scheme == 'dual':
				if i%2!=0:  # training change resistance step
					if BigClass.Variabs.task_type == 'Allostery_contrastive':
						delta = BigClass.Variabs.alpha * (self.error_vec[cycle])
					elif BigClass.Variabs.task_type == 'Regression_contrastive':
						delta = BigClass.Variabs.alpha * (loss[0]-loss[1])
					self.p_dual_vec[cycle+1] = self.p_dual[cycle] - np.sum(delta)
					self.output_dual_vec[cycle+1] = output_dual_vec[cycle] + delta
				else:  # training measure step
					self.p_vec[cycle+1] = NodeData
					self.output_vec[cycle+1] = self.p_output  # this is the output
			
			# Optionally plot
			if plot == 'yes' or (plot == 'last' and (i == (iters - 1) or i == (iters - 2))):
				print('condition supplied')
				
				NETfuncs.PlotNetwork(self.p, self.u, self.K, BigClass, EIEJ_plots, NN, NE, 
									nodes='no', edges='yes', pressureSurf='yes', savefig=savefig)
				# NETfuncs.PlotNetwork(self.p, self.u, self.K, BigClass, EIEJ_plots, NN, NE, 
				# 					nodes='no', edges='yes', pressureSurf='no', savefig=savefig)
				# NETfuncs.PlotNetwork(self.p, self.u, self.K, BigClass, EIEJ_plots, NN, NE, 
				# 					nodes='yes', edges='no', pressureSurf='no', savefig=savefig)
				plt.show()

			# plot if convergence reached before iters # of iterations
			if stop_bool == 1:
				NETfuncs.PlotNetwork(self.p, self.u, self.K, BigClass, EIEJ_plots, NN, NE, 
									nodes='yes', edges='no', savefig=savefig)
				NETfuncs.PlotNetwork(self.p, self.u, self.K, BigClass, EIEJ_plots, NN, NE, 
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
						NETfuncs.PlotNetwork(self.p, self.u, self.K, BigClass, EIEJ_plots, NN, NE, 
											nodes='yes', edges='no', savefig=savefig)
						NETfuncs.PlotNetwork(self.p, self.u, self.K, BigClass, EIEJ_plots, NN, NE, 
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

	def save_state_statistics(self, BigClass, sim_type, cyc_len, cycle, i, NodeData):
		"""
		save_state_statistics saves relevant data regarding network state in State class instance, e.g. p, u, Power dissip.

		inputs:
		BigClass - class instance including the user variables (Variabs), network structure (Strctr) and networkx (NET) class instances
				    I will not go into everything used from there to save space here.
		sim_type - str, simulation type, see flow_iterate() function for descrip.
		cyc_len  - int, num of iterations in every circle, for calculation of convergence
		cycle    - int, task goes over multiple cycles of training, cycle num. 
		i        - int, index at which the flow iterate loop is at
		NodeData - 1D array at length as "Nodes" corresponding to pressures at each node from "Nodes"

		outputs:
		None, only self
		"""
		if sim_type == 'allostery test' or sim_type == 'no marbles':
			self.u_final[i,:] = [np.sum(self.u[BigClass.Strctr.output_edges[0]]), np.sum(self.u[BigClass.Strctr.output_edges[1]])]
		elif BigClass.Variabs.task_type == 'Memristor':
			self.u_final = np.sum(self.u[BigClass.Strctr.output_edges])
		else:
			self.u_all[:, i] = self.u
			self.K_all[:, i] = self.K
			self.K_cells[:, i] = Matrixfuncs.K_by_cells(self.K, BigClass.Variabs.K_min, BigClass.Variabs.NGrid)
			if BigClass.Variabs.task_type == 'Allostery_contrastive' or BigClass.Variabs.task_type == 'Regression_contrastive':
				self.p_output = array([self.p[k][0] for k in BigClass.Strctr.FixedNodes_full])
				if BigClass.Variabs.task_type == 'Allostery_contrastive':
					desired_p = BigClass.Variabs.desired_p
					self.u_out = array([np.sum(self.u[BigClass.Strctr.output_edges[0]]), np.sum(self.u[BigClass.Strctr.output_edges[1]])])
					self.u_final = array([np.sum(self.u[BigClass.Strctr.ground_edges]), np.sum(self.u[BigClass.Strctr.input_edges])])
					print('ground edges are ', BigClass.Strctr.ground_edges, ' flow through them is ', self.u[BigClass.Strctr.ground_edges],
				  		  ' resulting in total ', self.u_final/NodeData[0], ' flow normalized by input pressure')
					self.u_final_vec[i,:] = self.u_final
				elif BigClass.Variabs.task_type == 'Regression_contrastive':
					desired_p = BigClass.Variabs.train_target[cycle]
					self.u_out = array([np.sum(self.u[BigClass.Strctr.output_edges[0]])])
				# if i%2==1:  # if at end of cycle - calculate errors and print pressures
				if i%2==0:  # if at beginning of cycle - calculate errors and print pressures
					print('p at output:', self.p_output)
					if BigClass.Variabs.flow_scheme == 'taktak' and i%4>1:
						if np.size(desired_p)>1:
							print('desired_p ', np.flip(desired_p))
							print('p_nudge ', np.flip(self.p_nudge))
							self.error_vec[int(np.floor(i/2))] = Statistics.calc_loss(np.flip(self.p_output), desired_p, 'abs_diff')
							self.ratio_vec[int(np.floor(i/2))] = Statistics.calc_ratio_loss(np.flip(self.p_output), desired_p, NodeData)
					else:
						print('desired_p ', desired_p)
						print('p_nudge ', self.p_nudge)
						if BigClass.Variabs.flow_scheme == 'dual':
							self.error_vec[int(np.floor(i/2))] = Statistics.calc_loss(self.p_output, desired_p, 'diff')
						else:
							self.error_vec[int(np.floor(i/2))] = Statistics.calc_loss(self.p_output, desired_p, 'abs_diff')
						self.ratio_vec[int(np.floor(i/2))] = Statistics.calc_ratio_loss(self.p_output, desired_p, NodeData)
					print('error ', self.error_vec[int(np.floor(i/2))])
					print('ratio', self.ratio_vec[int(np.floor(i/2))])
				else:  # end of cycle - update dual pressure and output
					self.p_nudge = Statistics.calculate_p_nudge(BigClass, self, desired_p, self.error_vec[int(np.floor(i/2))])
					if BigClass.Variabs.flow_scheme == 'dual':
						self.output_dual_vec[int(np.floor(i/2))] = Statistics.calculate_output_dual(BigClass, self.output_dual_vec[int(np.floor(i/2))-1], 
																									self.error_vec[int(np.floor(i/2))])
			self.power_dissip[i] = Statistics.power_dissip(self.u, self.K)
			if i >= cyc_len:
				MSE_i = Statistics.flow_MSE(self.u_all[:, i-cyc_len], cyc_len, self.u)
				Hamming_i = Statistics.K_Hamming(self.K_cells[:, i-cyc_len], cyc_len, self.K_cells[:, i])
				self.MSE[i-cyc_len] = MSE_i
				self.Hamming[i-cyc_len] = Hamming_i


class Solver:
    def __init__(self, use_gpu=False):
        super(Solver, self).__init__()
        # Decide which solver to use based on the use_gpu flag
        self.solve = Solve_CUDA if use_gpu else Solve


class Networkx_net:
	"""
	Networkx_net contains networkx data for plots

	inputs:

	outputs:
	NET - networkx net object (initially empty)
	"""
	def __init__(self, scale, squish):
		super(Networkx_net, self).__init__()
		self.scale = scale
		self.squish = squish
		self.NET = []

	def buildNetwork(self, BigClass):
		"""

	    Builds a networkx network using edges from EIEJ_plots which are built upon EI and EJ at "Matrixfuncs.py"
	    After this step, the order of edges at EIEJ_plots and in the networkx net is not the same which is shit
	    
	    input:

	    EIEJ_plots - 2D array sized [NE, 2] - 
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
		self.pos_lattice = NETfuncs.plotNetStructure(self.NET, BigClass.Variabs.NGrid, self.scale, self.squish, layout=BigClass.Variabs.net_typ, plot=plot)