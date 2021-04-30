import os
import subprocess
import shutil
import numpy as np

def maxent_run(gtau, tau_mesh, error=5e-3, params="green.param", exe_path='maxent', outdir="Maxent"):
  '''
  Run dim0 times maxent for gtau
  :param gtau: (nts, dim1), dim1 = (ns, nk, nao), (ns, nk), (ns) ... etc
  :param tau_mesh:
  :param exe_path: Maxent executable path
  :param outdir: output directory w.r.t. the current working directory
  :return:
  '''
  wkdir = os.path.abspath(os.getcwd())
  #outdir = os.path.abspath(wkdir+'/'+outdir)
  print("Maxent output:", os.path.abspath(wkdir+'/'+outdir))

  beta = tau_mesh[-1]
  nts  = tau_mesh.shape[0]
  ndim = len(gtau.shape)
  g_shape = gtau.shape
  assert nts == gtau.shape[0], "Number of imaginary time points mismatches."
  gtau = gtau.reshape(nts, -1)
  dim1 = gtau.shape[1]

  if not os.path.exists(outdir):
    os.mkdir(outdir)
  os.chdir(outdir)
  # Save dim1 for better understanding of output
  np.savetxt("dimensions.txt", np.asarray(g_shape[1:], dtype=int))
  # FIXME Can we have multiple layers of folders so that one can separate the whole job into chunks?
  for d1 in range(dim1):
    if not os.path.exists(str(d1)):
      os.mkdir(str(d1))
    np.savetxt("{}/G_tau.txt".format(d1), np.column_stack((tau_mesh, gtau[:, d1].real, np.array([error] * nts))))
  ## Start analytical continuation
  processes = []
  pp = 0
  for d1 in range(dim1):
    os.chdir(str(d1))
    shutil.copy(wkdir+"/green.param", "./")
    with open("log.txt", "w") as log:
      p = subprocess.Popen([exe_path, "./"+str(params), "--DATA=G_tau.txt", "--BETA=" + str(beta), "--NDAT=" + str(nts)], stdout=log,
            stderr=log)
      processes.append(p)
    pp += 1
    if pp % 64 == 0:
      for p in processes:
        p.communicate()
      processes = []
    os.chdir("..")

  for p in processes:
    p.communicate()
  os.chdir("..")
  gtau = gtau.reshape(g_shape)


def nevan_run(Gw, wsample, input_parser, nevan_exe="nevanlinna", outdir="Nevanlinna"):
  wkdir = os.path.abspath(os.getcwd())
  print("Nevanlinna output:", os.path.abspath(wkdir + '/' + outdir))

  nw   = wsample.shape[0]
  assert nw == Gw.shape[0], "Number of imaginary frequency points mismatches."
  ndim = len(Gw.shape)
  g_shape = Gw.shape
  Gw = Gw.reshape(nw, -1)
  dim1 = Gw.shape[1]

  if not os.path.exists(outdir):
    os.mkdir(outdir)
  os.chdir(outdir)
  # Save dim1 for better understanding of output
  np.savetxt("dimensions.txt", np.asarray(g_shape[1:], dtype=int))

  for d1 in range(dim1):
    if not os.path.exists(str(d1)):
      os.mkdir(str(d1))
    np.savetxt("{}/G_w.txt".format(d1), np.column_stack((wsample, Gw[:,d1].real, Gw[:,d1].imag)))
  ## Start analytical continuation
  processes = []
  pp = 0
  for d1 in range(dim1):
    os.chdir(str(d1))
    shutil.copy(wkdir + "/coeff", "./")
    with open("log.txt", "w") as log:
      p = subprocess.Popen([nevan_exe], stdin=subprocess.PIPE, stdout=log, stderr=log)
      p.stdin.write(str.encode(input_parser))
      processes.append(p)
    pp += 1
    if pp % 64 == 0:
      for p in processes:
        p.communicate()
      processes = []
    os.chdir("..")

  for p in processes:
    p.communicate()
  os.chdir("..")
  Gw = Gw.reshape(g_shape)

if __name__ == "__main__":
  import h5py
  import MB_analysis.mb as mb

  MB_path = MB_analysis.__path__[0]
  f = h5py.File(MB_path + '/data/H2_GW/sim.h5', 'r')
  Sr = f["S-k"][()].view(np.complex)
  Sr = Sr.reshape(Sr.shape[:-1])
  Fr = f["iter14/Fock-k"][()].view(np.complex)
  Fr = Fr.reshape(Fr.shape[:-1])
  Sigmar = f["iter14/Selfenergy/data"][()].view(np.complex)
  Sigmar = Sigmar.reshape(Sigmar.shape[:-1])
  Gr = f["iter14/G_tau/data"][()].view(np.complex)
  Gr = Gr.reshape(Gr.shape[:-1])
  mu = f["iter14/mu"][()]
  f.close()

  f = h5py.File(MB_path + '/data/H2_GW/input.h5', 'r')
  ir_list = f["/grid/ir_list"][()]
  weight = f["/grid/weight"][()]
  index = f["/grid/index"][()]
  conj_list = f["grid/conj_list"][()]
  f.close()

  ''' All k-dependent matrices should lie on a full Monkhorst-Pack grid. '''
  F = to_full_bz(Fr, conj_list, ir_list, index, 1)
  S = to_full_bz(Sr, conj_list, ir_list, index, 1)
  Sigma = to_full_bz(Sigmar, conj_list, ir_list, index, 2)
  G = to_full_bz(Gr, conj_list, ir_list, index, 2)

  # Standard way to initialize
  manybody = mb.MB_post(fock=F, sigma=Sigma, mu=mu, gtau=G, S=S, beta=1000, lamb='1e4')
  max      = Maxent(manybody, "maxent")
  # Run maxent for whatever target provided
  max.run(target = manybody.gtau)