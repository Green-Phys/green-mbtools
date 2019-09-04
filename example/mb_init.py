import MB_analysis.mb as many_body
import numpy as np
import matplotlib.pyplot as plt

sim = '/Users/CanonYeh/Projects/LiH/sim_4ew_last.h5'
inp = '/Users/CanonYeh/Projects/LiH/input_4ew.h5'

LiH = many_body.mb(sim,inp)
#print(LiH.mo_energy)
#print(np.shape(LiH.mo_energy))
k_occ = LiH.get_occ()
print(k_occ)
print(np.shape(k_occ))
gw0 = LiH.get_g_w0()
print(np.shape(gw0))
gw0 = LiH.g_orthogonal(gw0)

print(np.shape(gw0))
