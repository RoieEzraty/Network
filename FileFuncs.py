import numpy as np 
import pandas as pd
from datetime import datetime

def save_csv_files_Net(BigClass, p, K_min, comp_path):
	"""

	"""
	datenow = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
	df_p = pd.DataFrame(np.array(BigClass.State.p))
	df_p.to_csv(comp_path + str(datenow) + "_grid=" + str(BigClass.Variabs.NGrid) + "_p=" + str(p) + "_K_ratio="  + str(K_min) + "_p_field.csv")
	if isempty(BigClass.State.u_all[:,-1]):
		df_u = pd.DataFrame(np.array(BigClass.State.u_all[:,-1]))
	else:
		df_u = pd.DataFrame(np.array(BigClass.State.u))
	df_u.to_csv(comp_path + str(datenow) + "_grid=" + str(BigClass.Variabs.NGrid) + "_p=" + str(p) + "_K_ratio=" + str(K_min) + "_u_field.csv")
	df_K = pd.DataFrame(np.array(BigClass.State.K))
	df_K.to_csv(comp_path + str(datenow) + "_grid=" + str(BigClass.Variabs.NGrid) + "_p=" + str(p) + "_K_ratio=" + str(K_min) + "_K.csv")
