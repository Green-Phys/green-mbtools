import h5py
import MB_analysis
import MB_analysis.src.quasiparticle as qs
import MB_analysis.src.orth as orth

sim_path = MB_analysis.__path__[0] + '/../data/H2_GW/sim.h5'
input_path = MB_analysis.__path__[0] + '/../data/H2_GW/input.h5'
manybody = MB_analysis.mb.initialize_MB_post(sim_path, input_path, '1e4')

'''
Quasiparticles have to be defined in an orthogonal orbitals basis.  
'''
F_sao = orth.sao_orth(manybody.fock, manybody.S, type='f')
F_sao = F_sao[:,0]
Sigma_sao = orth.sao_orth(manybody.sigma, manybody.S, type='f')
Sigma_sao = manybody.ir.tau_to_w(Sigma_sao[:, :, 0])

MB_path = MB_analysis.__path__[0] + '/../'
nevan_sigma_exe = MB_path + '/Nevanlinna/nevanlinna_sigma'
Zs = qs.Z_factor(F_sao, Sigma_sao, manybody.ir.wsample, nevan_sigma_exe, 'nevan_sigma')
print("Quasiparticle renormalization factor: {}", Zs)