import numpy as np

from MB_analysis.src import ir


'''
Fourier transform between real and reciprocal space
'''

def construct_rmesh(nkx, nky, nkz):
  Lx, Ly, Lz = (nkx-1)//2, (nky-1)//2, (nkz-1)//2 # nk=6, L=2
  leftx, lefty, leftz = (nkx-1)%2, (nky-1)%2, (nkz-1)%2 # left = 1

  rx = np.linspace(-Lx, Lx+leftx, nkx, endpoint=True) # -2,-1,0,1,2,3
  ry = np.linspace(-Ly, Ly+lefty, nky, endpoint=True)
  rz = np.linspace(-Lz, Lz+leftz, nkz, endpoint=True)
  RX, RY, RZ = np.meshgrid(rx, ry, rz)
  rmesh = np.array([RX.flatten(), RY.flatten(), RZ.flatten()]).T
  
  return rmesh


def construct_symmetric_rmesh(nkx, nky, nkz):
  Lx, Ly, Lz = (nkx - 1) // 2, (nky - 1) // 2, (nkz - 1) // 2  # nk=6, L=2
  leftx, lefty, leftz = (nkx - 1) % 2, (nky - 1) % 2, (nkz - 1) % 2  # left = 1

  rx = np.linspace(-Lx, Lx + leftx, nkx, endpoint=True)  # -2,-1,0,1,2,3
  ry = np.linspace(-Ly, Ly + lefty, nky, endpoint=True)
  rz = np.linspace(-Lz, Lz + leftz, nkz, endpoint=True)
  RX, RY, RZ = np.meshgrid(rx, ry, rz)
  rmesh = np.array([RX.flatten(), RY.flatten(), RZ.flatten()]).T
  pm_rmesh = np.append(rmesh, -rmesh, axis=0)
  rmesh = np.unique(pm_rmesh, axis=0)

  return rmesh

def compute_fourier_coefficients(kmesh, rmesh):
  '''
  Compute Fourier coefficients for direct and inverse Fourier transform
  ''' 
  fkr = np.zeros((kmesh.shape[0], rmesh.shape[0]), dtype=complex)
  frk = np.zeros((rmesh.shape[0], kmesh.shape[0]), dtype=complex)
  for ik, k in enumerate(kmesh):
    for ir, r in enumerate(rmesh):
      dp = 2.j*np.pi*np.dot(k,r)
      # coefficients from r to k
      fkr[ik, ir] = np.exp(-dp)
      # coefficients from k to r
      frk[ir, ik] = np.exp(dp)

  return fkr, frk

#def winter_fkr(kmesh, rmesh):


def k_to_real(frk, obj_k, weights):
  '''
  Perform Fourier transform from reciprocal to real space
  frk     - Fourier transform coefficients from reciprocal to real space 
  obj_k   - reciprocal space object
  weights - Fourier coefficient degenerate weights

  return obj_i - inverse Fourier transform of obj_k
  '''
  obj_i = np.einsum('k...,k,ki->i...', obj_k, weights, frk.conj().T)/np.sum(weights)
  return obj_i

def real_to_k(fkr, obj_i):
  '''
  Perform Fourier transform from reciprocal to real space
  fkr     - Fourier transform coefficients from real to reciprocal space
  obj_i   - real space object 

  return obj_k - Fourier transform of obj_i
  '''
  original_shape = obj_i.shape
  obj_k = np.dot(fkr.conj(), obj_i.reshape(obj_i.shape[0],-1))
  obj_k = obj_k.reshape((obj_k.shape[0],) + obj_i.shape[1:])
  return obj_k

if __name__ == '__main__':
  import MB_analysis
  MB_path = MB_analysis.__path__[0]
  Sk10 = np.load(MB_path + '/data/winter/Sk10.npy')
  kmesh_scaled_nk10 = np.load(MB_path + '/data/winter/kmesh_k10.npy')
  Sk  = np.load(MB_path + '/data/winter/Sk6.npy')
  kmesh_scaled  = np.load(MB_path + '/data/winter/kmesh_k6.npy')
  nk_cube = Sk.shape[0]
  nk = int(np.cbrt(nk_cube))

  print("Sk from nk = ", nk)
  print(np.diag(Sk[0].real))
  print("Sk from nk = 10")
  print(np.diag(Sk10[0].real))
  
  # Check Fourier transformation
  rmesh = construct_rmesh(nk, nk, nk)
  fkr, frk = compute_fourier_coefficients(kmesh_scaled, rmesh) 
  Si = k_to_real(frk, Sk, [1]*kmesh_scaled.shape[0])
  Sk_check = real_to_k(fkr, Si)
  diff = Sk_check - Sk
  print("Back transformation error: ", np.max(np.abs(diff)))     
