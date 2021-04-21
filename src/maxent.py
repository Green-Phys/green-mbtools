import os
import subprocess
import shutil
import numpy as np

def run(gtau, tau_mesh, error=5e-3, params="green.param", exe_path, outdir="Maxent"):
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
  ndim = len(gtau.shape)
  g_shape = gtau.shape
  nts  = gtau.shape[0]
  assert nts == gtau.shape[0], "Number of imaginary time points mismatches."
  original_shape = gtau.shape
  gtau = gtau.reshape(nts, -1)
  dim1 = gtau.shape[1]

  if not os.path.exists(outdir):
    os.mkdir(outdir)
  os.chdir(outdir)
  # Save dim1 for better understanding of output
  np.savetxt("dimensions.txt", original_shape[1:])
  # FIXME Can we have multiple layers of folders so that one can separate the whole job into chunks?
  for d1 in range(dim1):
    if not os.path.exists(str(d1)):
      os.mkdir(str(d1))
    np.savetxt("{}/G_tau.txt".format(d1), np.column_stack((tau_mesh, gtau[:, d1], np.array([error] * nts))))
  ## Start analytical continuation
  processes = []
  pp = 0
  for d1 in range(dim1):
    os.chdir(str(d1))
    shutil.copy(wkdir+"/green.param", "./")
    with open("log.txt", "w") as log:
      p = subprocess.Popen([MAXENT, "./"+str(params), "--DATA=G_tau.txt", "--BETA=" + str(beta), "--NDAT=" + str(nts)], stdout=log,
            stderr=log)
      processes.append(p)
    pp += 1
    if pp % 64 == 0:
      for p in processes:
        p.wait()
      processes = []
    os.chdir("..")

  for p in processes:
    p.wait()
  os.chdir("..")
  gtau = gtau.reshape(original_shape)