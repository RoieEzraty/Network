import numpy as np
import copy
from numpy import zeros as zeros
from numpy import ones as ones
from numpy import array as array
from numpy import arange as arange
from numpy import meshgrid as meshgrid
from numpy import dot as dot
from numpy.linalg import inv as inv

import numpy.linalg as la
import numpy.random as rand
import pickle
import networkx as nx
import matplotlib.pyplot as plt

import NETfuncs, Matrixfuncs, Solve, Constraints, Statistics


class User_variables:
	"""docstring for User_variables"""
	def __init__(self, NGrid, Periodic, net_typ, K_max, K_min, u_thresh, input_p, flow_scheme, iterations, input_output_pairs):
		self.NGrid = NGrid
		self.Periodic = Periodic
		self.net_typ = net_typ
		self.K_max = K_max
		self.K_min = K_min
		self.u_thresh = u_thresh
		if len(input_p)==1:
			self.input_p = input_p
		self.flow_scheme = flow_scheme
		if flow_scheme == 'taktak':
			self.circle_step = 4
		else:
			self.circle_step = 2
		self.iterations = iterations
		self.input_output_pairs = input_output_pairs

	def assign_input_p(self, p):
		self.input_p = p


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
		NodeData_full = array([[Variabs.input_p], [Variabs.input_p]])  # input p value
		Nodes_full = array([[Variabs.input_output_pairs[i, 0]] for i in range(len(Variabs.input_output_pairs))])  # input p node


		GroundNodes_full = array([[Variabs.input_output_pairs[i, 1]] for i in range(len(Variabs.input_output_pairs))])  # nodes with zero pressure
		GroundNodes_full_Allostery = array([GroundNodes_full[i][0] for i in range(len(GroundNodes_full))])

		EdgeData_full = array([[0], [0]])  # pressure drop value on edge

		# Edges_full = array([EdgesTotal[(EdgesTotal!=np.where(EI==Nodes_full[0])[0][0]) 
		#                                & (EdgesTotal!=np.where(EI==GroundNodes_full[0])[0][0]) 
		#                                & (EdgesTotal!=np.where(EI==GroundNodes_full[1])[0][0])], 
		#                     EdgesTotal[(EdgesTotal!=np.where(EI==Nodes_full[1])[0][0]) 
		#                                & (EdgesTotal!=np.where(EI==GroundNodes_full[0])[0][0]) 
		#                                & (EdgesTotal!=np.where(EI==GroundNodes_full[1])[0][0])]])

		# Edges_full_Allostery = array([EdgesTotal[(EdgesTotal!=np.where(EI==Nodes_full[0])[0][0]) 
		#                                          & (EdgesTotal!=np.where(EI==GroundNodes_full[0])[0][0]) 
		#                                          & (EdgesTotal!=np.where(EI==GroundNodes_full[1])[0][0])], 
		#                               EdgesTotal[(EdgesTotal!=np.where(EI==Nodes_full[1])[0][0]) 
		#                                          & (EdgesTotal!=np.where(EI==GroundNodes_full[0])[0][0]) 
		#                                          & (EdgesTotal!=np.where(EI==GroundNodes_full[1])[0][0])]])

		Edges_full = array([self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==Nodes_full[0])[0][0]) 
		                                    & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		                                    & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])], 
		                    self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==Nodes_full[1])[0][0]) 
		                                    & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		                                    & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])]])

		Edges_full_Allostery = array([self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==Nodes_full[0])[0][0]) 
		                                              & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		                                              & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])], 
		                              self.EdgesTotal[(self.EdgesTotal!=np.where(self.EJ==Nodes_full[1])[0][0]) 
		                                              & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[0])[0][0]) 
		                                              & (self.EdgesTotal!=np.where(self.EJ==GroundNodes_full[1])[0][0])]])

		# Edges_full = array([EdgesConnections, EdgesConnections])
		# output_edges = [np.where(np.append(EI, EJ)==GroundNodes_full[i])[0][0] % len(EI) for i in range(len(GroundNodes_full))]
		self.output_edges = array([np.where(np.append(self.EI, self.EJ)==GroundNodes_full[i])[0] % len(self.EI) 
		                           for i in range(len(GroundNodes_full))])

		self.NodeData_full = NodeData_full
		self.Nodes_full = Nodes_full
		self.GroundNodes_full = GroundNodes_full
		self.GroundNodes_full_Allostery = GroundNodes_full_Allostery
		self.EdgeData_full = EdgeData_full
		self.Edges_full = Edges_full
		self.Edges_full_Allostery = Edges_full_Allostery


class Net_state:
	"""docstring for Net_state"""
	def __init__(self):
		super(Net_state, self).__init__()
		self.K = []
		self.K_mat = []
		self.u = []
		self.p = []

	def initiateK(self, Variabs, Strctr):
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
		return self

	def flow_iterate(self, Variabs, Strctr, NET, sim_type='w marbles', plot='yes'):
		"""
		Flow_iterate simulates the flow scheme as in Variabs.flow_scheme using network structure as in Strctr
		For explanation on flow scheme see the jupyter notebook "Flow Network Simulation...ipynb"
		Optionally plots the network as a networkx graph

		input:
		Variabs - user variables class
		Strctr  - net structure class
		NET     - class containing networkx net and extras
		sim_type - simulation type: 'no marbles'     - flows from one input at a time and all outputs, all edges have high conductivity
		                            'allostery test' - flows from one input at a time and all outputs, conductivities as in State.K
		                            'w marbles'      - (default) flows from one input and one output conductivities as in State.K, 
		                                               update conductivities as due to flow u and calculate flow again so flow further
		                                               update of conductivities will not change a thing
		plot     - flag: 'no'   - do not plot anything
		                 'yes'  - plot every iteration in Variabs.iterations
		                 'last' - only last couple of steps, one for each input
		
		output:
		u_final - 2D np.array [2, 2] flow out of output nodes / edges, rows for different inputs, columns different outputs
		u_all   - 2D np.array [NE, iterations] of flows through all edges for all simulation steps
		K_all   - 2D np.array [NE, iterations] of conductivities on all edges for all simulation steps
		"""

		# Initiate data arrays
		self.u_final = zeros([2, 2])
		self.u_all = np.zeros([Strctr.NE, Variabs.iterations])
		self.K_all = np.zeros([Strctr.NE, Variabs.iterations])
		self.K_cells = np.zeros([Variabs.NGrid**2, Variabs.iterations])
		self.power_dissip = np.NaN*np.zeros([Variabs.iterations, ])
		self.MSE =	np.zeros([Variabs.iterations, ])
		self.Hamming = np.zeros([Variabs.iterations, ])
		Hamming_i = 1  # dummy for to not break loop before assigned different value
		MSE_i = 1  # dummy for to not break loop before assigned different value
		self.convergence_time = np.NaN

		# Determine # iterations
		if sim_type=='w marbles':
			iters = Variabs.iterations  # total iteration #
			# iters_same_BCs = 5  # iterations at every given Boundary Conditions - solve flow and update K, repeat 3 times.
			iters_same_BCs = Variabs.iterations  # iterations at every given Boundary Conditions - solve flow and update K, repeat 3 times.
		else:
			iters = 2  # 1 flow iteration for every Boundary Condition
			iters_same_BCs = 1  # 1 iteration at every Boundary Condition, since conductivities do not change

		# Iterate - solve flow and optionally update conductivities
		for i in range(iters):
			cycle = int(np.floor(i/Variabs.circle_step))
			m = i % 2  # iterate between 1st and 2nd inputs 

			# Determine boundary conditions
			NodeData = Strctr.NodeData_full[m]
			Nodes = Strctr.Nodes_full[m] 
			EdgeData = Strctr.EdgeData_full[m]
			Edges = Strctr.Edges_full[m]
			if sim_type == 'w marbles':
				GroundNodes = Strctr.GroundNodes_full[m]
			else:
				GroundNodes = array([Strctr.GroundNodes_full[i][0] for i in range(len(Strctr.GroundNodes_full))])

			# Simulating normal direction of flow: from input to output

			# Otherwise, if true, switch ground and input nodes every 2nd iteration
			if i % 4 > 1 and Variabs.flow_scheme == 'taktak':
				Nodes = Strctr.GroundNodes_full[m]
				GroundNodes = Strctr.Nodes_full[m]

			# BC and constraints as matrix
			Cstr_full, Cstr, f = Constraints.ConstraintMatrix(NodeData, Nodes, EdgeData, Edges, GroundNodes, Strctr.NN, Strctr.EI, Strctr.EJ)  

			# Build Lagrangians and solve flow, optionally update conductivities and repeat.
			u = np.zeros([Strctr.NE,])
			for l in range(iters_same_BCs):

				L, L_bar = Matrixfuncs.buildL(Strctr.DM, self.K_mat, Cstr, Strctr.NN)  # Lagrangian

				# p, u = Solve.Solve_flow(L_bar, Strctr.EI, Strctr.EJ, self.K, f)  # pressure and flow
				p, u_nxt = Solve.Solve_flow(L_bar, Strctr.EI, Strctr.EJ, self.K, f)  # pressure and flow

				u_nxt[abs(u_nxt)<10**-10] = 0  # Correct for very low velocities

				if sim_type == 'w marbles':  # Update conductivities
					# K_nxt = Matrixfuncs.ChangeKFromFlow(u, Variabs.u_thresh, self.K, Variabs.K_max, Variabs.K_min, Variabs.NGrid)
					K_nxt = Matrixfuncs.ChangeKFromFlow(u_nxt, Variabs.u_thresh, self.K, Variabs.K_max, Variabs.K_min, Variabs.NGrid)
					for n in Strctr.Nodes_full.reshape(2):
						K_nxt[Strctr.EJ == n] = Variabs.K_max
					for n in Strctr.GroundNodes_full.reshape(2):
						K_nxt[Strctr.EJ == n] = Variabs.K_max
					self.K_mat = np.eye(Strctr.NE) * K_nxt
					self.K = copy.copy(K_nxt)

				# break the loop
				# since no further changes will be measured in flow and conductivities at end of next cycle
				if np.all(u_nxt == u):
					break
				else:
					u = copy.copy(u_nxt)

			# Save data in assigned arrays
			if sim_type == 'allostery test' or sim_type == 'no marbles':
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
					# print('pressure is ' + str(Variabs.input_p) + ' Hamming is ' + str(Hamming_i) + ' flow MSE is ' + str(MSE_i))
					self.MSE[i-Variabs.circle_step] = MSE_i
					self.Hamming[i-Variabs.circle_step] = Hamming_i

			# Optionally plot
			if plot == 'yes' or plot == 'last' and (i == (iters - 1) or i == (iters - 2)):
				NETfuncs.PlotNetwork(p, u, self.K, NET.NET, NET.pos_lattice, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE)

			# obtain # of iterations until convergence of conductivities after full cycle of changing BCs and break the loop
			# since no further changes will be measured in flow and conductivities at end of next cycle
			if np.isnan(self.convergence_time) and Hamming_i == 0.0:
				self.convergence_time = cycle
				print('loop break')
				if plot == 'last':
					NETfuncs.PlotNetwork(p, u, self.K, NET.NET, NET.pos_lattice, Strctr.EIEJ_plots, Strctr.NN, Strctr.NE)
				break

			# # Break loop if simulation converged
			# if MSE_i == 0:
			# 	break


class Networkx_net:
	"""docstring for Networkx_net"""
	def __init__(self):
		super(Networkx_net, self).__init__()
		self.NET = []

	def buildNetwork(self, Strctr):
		self.NET = NETfuncs.buildNetwork(Strctr.EIEJ_plots)
	
	def build_pos_lattice(self, Variabs):
		self.pos_lattice = NETfuncs.plotNetStructure(self.NET, Variabs.net_typ)