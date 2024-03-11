import numpy as np
import copy

import numpy.random as rand
import pickle
import networkx as nx
import matplotlib.pyplot as plt

import NETfuncs, Matrixfuncs, Solve, Constraints, Statistics


class User_variables:
	"""docstring for User_variables"""
	def __init__(self, NGrid, Periodic, net_typ, u_thresh, input_p, flow_scheme, task_type, K_scheme, K_type,
				iterations, input_output_pairs, fixed_node_pairs=0, K_max=1, K_min=0.5, beta=0.0):
		self.NGrid = NGrid
		self.Periodic = Periodic
		self.net_typ = net_typ
		self.u_thresh = u_thresh
		if len(input_p)==1:
			self.input_p = input_p

		self.flow_scheme = flow_scheme
		if flow_scheme == 'taktak':
			self.circle_step = 4
		elif flow_scheme == 'unidir':
			self.circle_step = 2
		else:
			self.circle_step = 1

		self.task_type = task_type
		if task_type == 'XOR':
			self.fixed_node_pairs = fixed_node_pairs
		else: 
			self.fixed_node_pairs = np.array([])

		self.K_scheme = K_scheme
		self.K_type = K_type
		self.K_max = K_max
		self.K_min = K_min
		self.beta = beta

		self.iterations = iterations
		self.input_output_pairs = input_output_pairs

	def assign_input_p(self, p):
		self.input_p = p

	def assign_K_min(self, K_min):
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
			self.fixed_node_p = np.array([[p/3], [2*p/3]])
		else:
			self.fixed_node_p = np.array([])
			print('no fixed nodes other than input')


class Net_structure:
	"""docstring for Net_Strctr"""
	# def __init__(self, EI, EJ, EIEJ_plots, DM, NE, NN):
	# 	self.EI = EI
	# 	self.EJ = EJ 
	# 	self.EIEJ_plots = EIEJ_plots
	# 	self.DM = DM
	# 	self.NE = NE 
	# 	self.NN = NN
	def __init__(self):
		self.NE = 0
		self.NN = 0

	def build_incidence(self, Variabs):
		"""
		"""
		self.EI, self.EJ, self.EIEJ_plots, self.DM, self.NE, self.NN = Matrixfuncs.build_incidence(Variabs)

	def Boundaries_and_connections(self, Variabs):
		"""
		Identify edges at connections of cells and at boundaries for ease of use
		"""

		NGrid = Variabs.NGrid
		NConncetions = int(NGrid*(NGrid-1)*2)
		EdgesConnections = [int(i) for i in range(self.NE-NConncetions, self.NE)]

		NBoundaries = NGrid*4
		left_side = [0 + 4*NGrid*i for i in range(NGrid)]
		bottom_side = [1 + 4*i for i in range(NGrid)]
		right_side = [2 + 4*(NGrid-1) + 4*NGrid*i for i in range(NGrid)]
		top_side = [4*NGrid*(NGrid-1) + 3 + 4*i for i in range(NGrid)]
		EdgesBounaries = np.append(left_side, np.append(bottom_side, np.append(right_side, top_side)))
		# EdgesBounaries = np.array([], int)
		self.EdgesTotal = np.append(EdgesConnections, EdgesBounaries)

	def Setup_constraints(self, Variabs):
		InNodeData_full = np.array([[Variabs.input_p], [Variabs.input_p]])  # input p value
		InNodes_full = np.array([[Variabs.input_output_pairs[i, 0]] for i in range(len(Variabs.input_output_pairs))])  # input p node
		# if fixed nodes exist, append them to nodes:
		if len(Variabs.fixed_node_pairs)>0:
			FixedNodeData_full = np.array([[Variabs.fixed_node_p[0]], [Variabs.fixed_node_p[1]]])  # input p value
			FixedNodes_full = np.array([[Variabs.fixed_node_pairs[i, 0]] for i in range(len(Variabs.fixed_node_pairs))])  # input p node
		else:
			FixedNodeData_full = np.array([])
			FixedNodes_full = np.array([])

		# nodes with zero pressure
		GroundNodes_full = np.array([[Variabs.input_output_pairs[i, 1]] for i in range(len(Variabs.input_output_pairs))])  
		GroundNodes_full_Allostery = np.array([GroundNodes_full[i][0] for i in range(len(GroundNodes_full))])

		EdgeData_full = np.array([[0], [0]])  # pressure drop value on edges specified by Edges_full

		# Edges_full = array([EdgesTotal[(EdgesTotal!=np.where(EI==InNodes_full[0])[0][0]) 
		#                                & (EdgesTotal!=np.where(EI==GroundNodes_full[0])[0][0]) 
		#                                & (EdgesTotal!=np.where(EI==GroundNodes_full[1])[0][0])], 
		#                     EdgesTotal[(EdgesTotal!=np.where(EI==InNodes_full[1])[0][0]) 
		#                                & (EdgesTotal!=np.where(EI==GroundNodes_full[0])[0][0]) 
		#                                & (EdgesTotal!=np.where(EI==GroundNodes_full[1])[0][0])]])

		# Edges_full_Allostery = array([EdgesTotal[(EdgesTotal!=np.where(EI==InNodes_full[0])[0][0]) 
		#                                          & (EdgesTotal!=np.where(EI==GroundNodes_full[0])[0][0]) 
		#                                          & (EdgesTotal!=np.where(EI==GroundNodes_full[1])[0][0])], 
		#                               EdgesTotal[(EdgesTotal!=np.where(EI==InNodes_full[1])[0][0]) 
		#                                          & (EdgesTotal!=np.where(EI==GroundNodes_full[0])[0][0]) 
		#                                          & (EdgesTotal!=np.where(EI==GroundNodes_full[1])[0][0])]])

		# Full list of edges that have specified pressure drop. it is different from EdgesTotal only if
		# there are non-zero boundary conditions at periphery of network.
		# print(self.EdgesTotal)
		# print(self.EJ)
		# print(InNodes_full)
		# print(GroundNodes_full)
		# print(self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[0])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])])
		# print(self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[1])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])])
		# print([self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[0])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])], 
		#                     self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[1])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])]])
		# Edges_full = np.array([self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[0])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])], 
		#                     self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[1])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		#                                     & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])]])
		Edges_full = [self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[0])[0][0]) 
		                                    & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		                                    & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])], 
		                    self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[1])[0][0]) 
		                                    & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		                                    & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])]]

		# Same as Edges_full but for Allostery task where both ground nodes are indeed grounded
		# Edges_full_Allostery = np.array([self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[0])[0][0]) 
		#                                               & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		#                                               & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])], 
		#                               self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[1])[0][0]) 
		#                                               & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		#                                               & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])]])
		Edges_full_Allostery = [self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[0])[0][0]) 
		                                              & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		                                              & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])], 
		                              self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==InNodes_full[1])[0][0]) 
		                                              & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		                                              & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])]]

		# Edges_full = array([EdgesConnections, EdgesConnections])
		# output_edges = [np.where(np.append(EI, EJ)==GroundNodes_full[i])[0][0] % len(EI) for i in range(len(GroundNodes_full))]
		self.output_edges = np.array([np.where(np.append(self.EI, self.EJ)==GroundNodes_full[i])[0] % len(self.EI) 
		                           for i in range(len(GroundNodes_full))])

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

	def Constraints_afo_task(self, Variabs, sim_type, i):
		"""
		Explanation?
		"""
		if Variabs.task_type == 'Allostery_one_pair':
			# Determine boundary conditions
			NodeData = self.InNodeData_full[0]
			Nodes = self.InNodes_full[0] 
			EdgeData = self.EdgeData_full[0]
			Edges = self.Edges_full[0]
			if sim_type == 'w marbles':
				GroundNodes = self.GroundNodes_full[0]
			else:
				GroundNodes = np.array([self.GroundNodes_full[0][0] for i in range(len(self.GroundNodes_full))])

		elif Variabs.task_type == 'Allostery':
			m = i % 2  # iterate between 1st and 2nd inputs 

			# Determine boundary conditions
			NodeData = self.InNodeData_full[m]
			Nodes = self.InNodes_full[m] 
			EdgeData = self.EdgeData_full[m]
			Edges = self.Edges_full[m]
			if sim_type == 'w marbles':
				GroundNodes = self.GroundNodes_full[m]
			else:
				GroundNodes = np.array([self.GroundNodes_full[i][0] for i in range(len(self.GroundNodes_full))])

			# Simulating normal direction of flow: from input to output

			# Otherwise, if true, switch ground and input nodes every 2nd iteration
			if i % 4 > 1 and Variabs.flow_scheme == 'taktak':
				Nodes = self.GroundNodes_full[m]
				GroundNodes = self.InNodes_full[m]

		elif Variabs.task_type == 'XOR':
			# m = i % 4  # iterate between 1st and 2nd inputs 
			# draw random input for XOR.  m is empty / [0] / [1] / [0, 1]
			m = np.where([rand.randint(0, 2), rand.randint(0, 2)])[0]

			NodeData = self.InNodeData_full[m]
			Nodes = self.InNodes_full[m] 
			EdgeData = self.EdgeData_full[m]
			Edges = self.Edges_full[m]

			# Determine boundary conditions
			NodeData = np.append(self.InNodeData_full, self.FixedNodeData_full)
			Nodes = np.append(self.InNodes_full, self.FixedNodes_full)
			EdgeData = self.EdgeData_full[0]
			Edges = self.Edges_full[0]
			GroundNodes = self.GroundNodes_full[m]

		return NodeData, Nodes, EdgeData, Edges, GroundNodes


class Net_state:
	"""docstring for Net_state"""
	def __init__(self):
		super(Net_state, self).__init__()
		self.K = []
		self.K_mat = []
		self.u = []
		self.p = []

	def initiateK(self, Variabs, Strctr, noise='no'):
		"""
		Builds initial conductivity matrix, simulation steps and updates are under Solve.py

		input:
		NE    - NEdges, int
		K_max - value of maximal conductivity (no marble)
		
		output:
		K     - 1D np.array sized [NEdges] with initial conductivity values for every edge
		K_mat - 2D cubic np.array sized [NEdges] with initial conductivity values on diagonal
		"""
		NE = Strctr.NE
		K_max = Variabs.K_max

		self.K = K_max*np.ones([NE])
		self.K_mat = np.eye(NE) * self.K

		if noise == 'yes':
			# fictional velocity field just to move marble randomly
			u_rand = 1.25 * Variabs.u_thresh * 2 * (rand.random([Strctr.NN])-1/2) 
			self.K = Matrixfuncs.ChangeKFromFlow(u_rand, Variabs.u_thresh, self.K, Variabs.NGrid, 
												K_change_scheme=Variabs.K_scheme, K_max=Variabs.K_max, 
												K_min=Variabs.K_min)  # move marble, change conductivities

	def fixAllK(self, Variabs, Strctr):
		for i in [Strctr.InNodes_full, Strctr.FixedNodes_full, Strctr.GroundNodes_full]:
			length = len(i)
			self.fixK(Variabs, Strctr, i, length)
		self.K_mat = np.eye(Strctr.NE) * self.K

	def fixK(self, Variabs, Strctr, nodes, length):
		for n in nodes.reshape(length):
			self.K[Strctr.EJ == n] = Variabs.K_max

	def solve_flow_const_K(self, Variabs, Strctr, NET, u, Cstr, f, iters_same_BCs):
		"""
		Explain function Roie, perhaps don't miss out on it this time
		"""

		u_nxt = copy.copy(u)

		for o in range(iters_same_BCs):	

			# print('iteration # %d with constant K' %o)
			# print('starting from u =')
			# print(u)

			K_eff = copy.copy(self.K)
			K_eff[u_nxt>0] = Variabs.K_max
			K_eff_mat = np.eye(Strctr.NE) * K_eff

			L, L_bar = Matrixfuncs.buildL(Strctr.DM, K_eff_mat, Cstr, Strctr.NN)  # Lagrangian

			p, u_nxt = Solve.Solve_flow(L_bar, Strctr.EI, Strctr.EJ, K_eff, f)  # pressure and flow

			u_nxt[abs(u_nxt)<10**-10] = 0  # Correct for very low velocities
			p[abs(p)<10**-10] = 0  # Correct for very low pressures

			# print('going to u_nxt =')
			# print(u_nxt)
			
			# NETfuncs.PlotNetwork(p, u_nxt, self.K, NET.NET, NET.pos_lattice, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE, 
			# 					nodes='no', edges='yes', savefig='no')

			# break the loop
			# since no further changes will be measured in flow and conductivities at end of next cycle
			if np.all(np.where(u_nxt>0)[0] == np.where(u>0)[0]):
			# if np.all(u_nxt == u):
				u = copy.copy(u_nxt)
				break
			else:
				# NETfuncs.PlotNetwork(p, u_nxt, self.K, NET.NET, NET.pos_lattice, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE)
				u = copy.copy(u_nxt)

		return p, u_nxt

	def solve_flow_until_conv(self, Variabs, Strctr, NET, u, Cstr, f, iters_same_BCs, sim_type):
		"""
		Explain function Roie, perhaps don't miss out on it this time
		"""

		for l in range(iters_same_BCs):

			if Variabs.K_type == 'flow_dep':
				# print('solve flow under constant K for the %d time' %l)
				p, u_nxt = self.solve_flow_const_K(Variabs, Strctr, NET, u, Cstr, f, iters_same_BCs)
			else:
				L, L_bar = Matrixfuncs.buildL(Strctr.DM, self.K_mat, Cstr, Strctr.NN)  # Lagrangian

				p, u_nxt = Solve.Solve_flow(L_bar, Strctr.EI, Strctr.EJ, self.K, f)  # pressure and flow

			# u_nxt[abs(u_nxt)<10**-10] = 0  # Correct for very low velocities
			# p[abs(p)<10**-10] = 0  # Correct for very low pressures

			if sim_type == 'w marbles' or sim_type == 'allostery test':  # Update conductivities
				K_nxt = Matrixfuncs.ChangeKFromFlow(u_nxt, Variabs.u_thresh, self.K, Variabs.NGrid, 
													K_change_scheme=Variabs.K_scheme, K_max=Variabs.K_max, 
													K_min=Variabs.K_min, beta=Variabs.beta)
				
				self.K_mat = np.eye(Strctr.NE) * K_nxt
				self.K = copy.copy(K_nxt)

				# all input and output nodes have no marbles
				self.fixAllK(Variabs, Strctr)

			# break the loop
			# since no further changes will be measured in flow and conductivities at end of next cycle
			if np.all(u_nxt == u):
				break
			else:
				# NETfuncs.PlotNetwork(p, u_nxt, self.K, NET.NET, NET.pos_lattice, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE)
				u = copy.copy(u_nxt)

		return p, u_nxt

	def flow_iterate(self, Variabs, Strctr, NET, sim_type='w marbles', plot='yes', savefig='no'):
		"""
		Flow_iterate simulates the flow scheme as in Variabs.flow_scheme using network structure as in Strctr
		For explanation on flow scheme see the jupyter notebook "Flow Network Simulation...ipynb"
		Optionally plots the network as a networkx graph

		input:
		Variabs - user variables class
		Strctr  - net structure class
		NET     - class containing networkx net and extras
		sim_type - simulation type: 'no marbles'     - flows from 1 input at a time and all outputs, all edges have high State.K
		                            'allostery test' - flows from 1 input at a time and all outputs, K's as in State.K
		                            'w marbles'      - (default) flows from 1 input and one output K's as in State.K, 
		                                               update conductivities as due to flow u and calculate flow again so
		                                               further update of conductivities will not change a thing
		plot     - flag: 'no'   - do not plot anything
		                 'yes'  - plot every iteration in Variabs.iterations
		                 'last' - only last couple of steps, one for each input
		
		output:
		u_final - 2D np.array [2, 2] flow out of output nodes / edges, rows for different inputs, columns different outputs
		u_all   - 2D np.array [NE, iterations] of flows through all edges for all simulation steps
		K_all   - 2D np.array [NE, iterations] of conductivities on all edges for all simulation steps
		"""

		# Initiate data arrays
		self.u_final = np.zeros([2, 2])
		self.u_all = np.zeros([Strctr.NE, Variabs.iterations])
		self.K_all = np.zeros([Strctr.NE, Variabs.iterations])
		self.K_cells = np.zeros([Variabs.NGrid**2, Variabs.iterations])
		self.power_dissip = np.NaN*np.zeros([Variabs.iterations, ])
		self.MSE =	np.zeros([Variabs.iterations, ])
		self.Hamming = np.zeros([Variabs.iterations, ])
		Hamming_i = 1  # dummy for to not break loop before assigned different value
		MSE_i = 1  # dummy for to not break loop before assigned different value
		self.convergence_time = np.NaN
		stop_bool = 0

		# Determine # iterations
		if sim_type=='w marbles':
			iters = Variabs.iterations  # total iteration #
			# iters_same_BCs = 5  # iterations at every given Boundary Conditions - solve flow and update K, repeat 3 times.
			iters_same_BCs = Variabs.iterations  # iterations at every given B.C. - solve flow and update K, repeat 3 times.
		else:
			iters = 2  # 1 flow iteration for every Boundary Condition
			if Variabs.K_type == 'flow_dep':
				iters_same_BCs = Variabs.iterations  # still requires some time to converge due to flow feedback since K is not bidirectional
			else:
				iters_same_BCs = 1  # 1 iteration at every Boundary Condition, since conductivities do not change

		# Iterate - solve flow and optionally update conductivities
		for i in range(iters):
			cycle = int(np.floor(i/Variabs.circle_step))  # task goes over multiple cycles of training, keep the cycle num.
			print('cycle # %d' %cycle)
			
			# specific constraints for training step
			NodeData, Nodes, EdgeData, Edges, GroundNodes = Strctr.Constraints_afo_task(Variabs, sim_type, i)

			# print(Variabs.input_p)

			# BC and constraints as matrix
			Cstr_full, Cstr, f = Constraints.ConstraintMatrix(NodeData, Nodes, EdgeData, Edges, GroundNodes, 
															Strctr.NN, Strctr.EI, Strctr.EJ)  

			# Build Lagrangians and solve flow, optionally update conductivities and repeat.
			u = np.zeros([Strctr.NE,])

			# solve for flow interatively until convergence under same BCs, optionally changing K due to flow
			p, u = self.solve_flow_until_conv(Variabs, Strctr, NET, u, Cstr, f, iters_same_BCs, sim_type)

			# Save data in assigned arrays
			if sim_type == 'allostery test' or sim_type == 'no marbles':
				self.u = u
				self.p = p
				self.u_final[i,:] = [np.sum(u[Strctr.output_edges[0]]), np.sum(u[Strctr.output_edges[1]])]
			else:
				# self.p_all[:, i] = p
				self.u_all[:, i] = u
				self.K_all[:, i] = self.K
				self.K_cells[:, i] = Matrixfuncs.K_by_cells(self.K, Variabs.K_min, Variabs.NGrid)
				self.power_dissip[i] = Statistics.power_dissip(u, self.K)
				if i >= Variabs.circle_step:
					MSE_i = Statistics.flow_MSE(self.u_all[:, i-Variabs.circle_step], Variabs.circle_step, u)
					Hamming_i = Statistics.K_Hamming(self.K_cells[:, i-Variabs.circle_step], Variabs.circle_step, self.K_cells[:, i])
					self.MSE[i-Variabs.circle_step] = MSE_i
					self.Hamming[i-Variabs.circle_step] = Hamming_i

			# Optionally plot
			if plot == 'yes' or (plot == 'last' and (i == (iters - 1) or i == (iters - 2))):
				print('condition supplied')
				NETfuncs.PlotNetwork(p, u, self.K, NET.NET, NET.pos_lattice, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE, 
									nodes='yes', edges='no', savefig=savefig)
				NETfuncs.PlotNetwork(p, u, self.K, NET.NET, NET.pos_lattice, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE, 
									nodes='no', edges='yes', savefig=savefig)
				plt.show()

			if stop_bool == 1:
				NETfuncs.PlotNetwork(p, u, self.K, NET.NET, NET.pos_lattice, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE, 
									nodes='yes', edges='no', savefig=savefig)
				NETfuncs.PlotNetwork(p, u, self.K, NET.NET, NET.pos_lattice, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE, 
									nodes='no', edges='yes', savefig=savefig)
				break

			# obtain # of iterations until convergence of conductivities after full cycle of changing BCs and break the loop
			# since no further changes will be measured in flow and conductivities at end of next cycle
			if np.isnan(self.convergence_time) and MSE_i == 0.0:
				self.convergence_time = cycle
				print('loop break')
				if plot == 'last':
					stop_bool = 1  # go one more time for plot
					if Variabs.flow_scheme == 'unidir' or Variabs.flow_scheme == 'taktak':
						NETfuncs.PlotNetwork(p, u, self.K, NET.NET, NET.pos_lattice, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE, 
											nodes='yes', edges='no', savefig=savefig)
						NETfuncs.PlotNetwork(p, u, self.K, NET.NET, NET.pos_lattice, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE, 
											nodes='no', edges='yes', savefig=savefig)
						plt.show()
				else:
					break 


class Networkx_net:
	"""docstring for Networkx_net"""
	def __init__(self):
		super(Networkx_net, self).__init__()
		self.NET = []

	def buildNetwork(self, Strctr):
		self.NET = NETfuncs.buildNetwork(Strctr.EIEJ_plots)
	
	def build_pos_lattice(self, Variabs):
		self.pos_lattice = NETfuncs.plotNetStructure(self.NET, Variabs.net_typ)