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
    shutil.copy(str(params), "./green.param")
    with open("log.txt", "w") as log:
      p = subprocess.Popen([exe_path, "./green.param", "--DATA=G_tau.txt", "--BETA=" + str(beta), "--NDAT=" + str(nts)], stdout=log,
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
