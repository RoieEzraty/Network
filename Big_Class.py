import numpy as np

class Big_Class:
	"""
	Big_Class contains the main classes under Network Simulation
	"""
	
	def __init__(self, Variabs):
		self.Variabs = Variabs

	def assign_solver(self, solver):
		self.Solver = solver

	def add_Strctr(self, Strctr):
		self.Strctr = Strctr

	def add_State(self, State):
		self.State = State

	def add_NET(self, NET):
		self.NET = NET
