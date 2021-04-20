from functools import reduce
import numpy as np

from MB_analysis.src import ft
from MB_analysis.src import dyson

# Only work for full_bz object
def interpolate(obj_k, kmesh, kpts_inter, hermi=False, debug=False):
  '''
  Interpolate obj_k[ns, nk, nao, nao] from kmesh to kpts_inter using Wannier interpolation
  '''
  ns = obj_k.shape[0]
  nk_cube = obj_k.shape[1]
  nk = int(np.cbrt(nk_cube))
  nao = obj_k.shape[-1]
  rmesh = ft.construct_rmesh(nk, nk, nk)
  fkr, frk = ft.compute_fourier_coefficients(kmesh, rmesh)
  weights = [1] * kmesh.shape[0]
  #obj_i = ft.k_to_real(frk, obj_k, [1]*kmesh.shape[0])
  obj_i = np.array([ft.k_to_real(frk, obj_k[s], weights) for s in range(ns)])
  if debug:
    center = (nk-1)//2
    obj_i = obj_i.reshape(ns, nk, nk, nk, nao, nao)
    for i in range(nk):
      print("obj_i[",i-center,", 0, 0] = ")
      print(np.diag(obj_i[0,i,center,center].real))
    obj_i = obj_i.reshape(ns, nk_cube, nao, nao)

  fkr_int, frk_int = ft.compute_fourier_coefficients(kpts_inter, rmesh)
  #obj_k_int = ft.real_to_k(fkr_int, obj_i)
  obj_k_int = np.array([ft.real_to_k(fkr_int, obj_i[s]) for s in range(ns)])

  if hermi:
    error = 0.0
    for s in range(ns):
      for ik in range(kpts_inter.shape[0]):
        obj = obj_k_int[s, ik]
        obj_sym = 0.5 * (obj + obj.conj().T)
        error = max(error, np.max(np.abs(obj_sym - obj)))
        obj_k_int[s, ik] = obj_sym
    print("The largest Hermitization error = ", error)

  return obj_k_int

# Only work for full_bz object
def interpolate_tk_object(obj_tk, kmesh, kpts_inter, hermi=False, debug=False):
  '''
  Interpolate dynamic obj_k[nts, ns, nk, nao, nao] from kmesh to kpts_inter using Wannier interpolation
  '''
  nts = obj_tk.shape[0]
  ns  = obj_tk.shape[1]
  nk_cube = obj_tk.shape[2]
  nk = int(np.cbrt(nk_cube))
  nao = obj_tk.shape[-1]
  rmesh = ft.construct_rmesh(nk, nk, nk)
  fkr, frk = ft.compute_fourier_coefficients(kmesh, rmesh)
  weights = [1]*kmesh.shape[0]
  obj_ti = np.array([ft.k_to_real(frk, obj_tk[it, s], weights) for it in range(nts) for s in range(ns)])
  
  if debug:
    center = (nk-1)//2
    obj_ti = obj_ti.reshape(nts*ns, nk, nk, nk, nao, nao)
    for i in range(nk):
      print("obj_i[",i-center,", 0, 0] = ")
      print(np.diag(obj_ti[0, i,center,center].real))
    obj_ti = obj_ti.reshape(nts, ns, nk_cube, nao, nao)

  fkr_int, frk_int = ft.compute_fourier_coefficients(kpts_inter, rmesh)
  obj_tk_int = np.array([ft.real_to_k(fkr_int, obj_ti[it, s]) for it in range(nts) for s in range(ns)])

  if hermi:
    error = 0.0
    for it in range(nts):
      for s in range(ns):
        for ik in range(kpts_inter.shape[0]):
          obj = obj_tk_int[it, s, ik]
          obj_sym = 0.5 * (obj + obj.conj().T)
          error = max(error, np.max(np.abs(obj_sym - obj)))
          obj_tk_int[it, s, ik] = obj_sym
    print("The largest Hermitization error = ", error)

  return obj_tk_int

# Only work for full_bz object
def interpolate_G(Fk, Sigma_tk, mu, Sk, kmesh, kpts_inter, ir, hermi=False, debug=False):
  ns = Fk.shape[0]
  nk_cube = Fk.shape[1]
  nk = int(np.cbrt(nk_cube))
  nao = Fk.shape[2]
  nts = ir.nts
  nw  = ir.nw

  if Sigma_tk is not None:
    assert nts == Sigma_tk.shape[0], "Number of imaginary time points mismatches."

  if Sk is None:
    print("Interpolating overlap...")
    Sk_int = interpolate(Sk, kmesh, kpts_inter, hermi, debug)
  else:
    Sk_int = np.array([[np.eye(nao)] * kpts_inter.shape[0]]*ns)

  print("Interpolating Fock...")
  Fk_int = interpolate(Fk, kmesh, kpts_inter, hermi, debug)
  # FIXME Too memory demanding and too slow as well.
  if Sigma_tk is not None:
    print("Interpolating self-energy...")
    Sigma_tk_int = interpolate_tk_object(Sigma_tk, kmesh, kpts_inter, hermi, debug)
  else:
    Sigma_tk_int = None

  # Optional: Orthogonalization before Dyson

  # Solve Dyson
  Gtk_int, dm_int = dyson.solve_dyson(Fk_int, Sk_int, Sigma_tk_int, mu, ir)

  return Gtk_int, Sigma_tk_int, Fk_int, Sk_int

if __name__ == '__main__':
  Sk10 = np.load("../data/winter/Sk10.npy")
  Sk10 = Sk10.reshape((1,) + Sk10.shape)
  kmesh_scaled_nk10 = np.load("../data/winter/kmesh_k10.npy")
  Sk  = np.load("../data/winter/Sk6.npy")
  Sk  = Sk.reshape((1,) + Sk.shape)
  kmesh_scaled  = np.load("../data/winter/kmesh_k6.npy")
  ns = Sk.shape[0]
  nk_cube = Sk.shape[1]
  nk = int(np.cbrt(nk_cube))

  # Wannier interpolation
  Sk10_inter = interpolate(Sk, kmesh_scaled, kmesh_scaled_nk10, hermi=True, debug=True)
  diff = Sk10_inter - Sk10
  print("Largest difference between the exact and the interpolated one: ", np.max(np.abs(diff))) 
  print("Reference value is ", 0.000385692904592791)
  
  # TODO Examples for interpolate_G
