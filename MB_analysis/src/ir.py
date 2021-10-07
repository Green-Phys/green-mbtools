from functools import reduce
import numpy as np
import h5py
import os

'''
Fourier transform between imaginary time and Matsubara frequency using intermediate representation (IR)
'''

class IR_factory(object):
  def __init__(self, beta, lamb):

    self._ir_dict = {'1e3': '/../data/ir_grid/1e3_72.h5', '1e4': '/../data/ir_grid/1e4_104.h5', '1e5': '/../data/ir_grid/1e5_136.h5',
                '1e6': '/../data/ir_grid/1e6_168.h5', '1e7': '/../data/ir_grid/1e7_202.h5'}
    if lamb not in self._ir_dict.keys():
      raise ValueError("{} is not an acceptable lambda value.".format(lamb)
                       + " Acceptable lambdas are " + str(self._ir_dict.keys()))
    import MB_analysis
    self._MB_path = MB_analysis.__path__[0]
    for l in self._ir_dict:
      self._ir_dict[l] = self._MB_path + self._ir_dict[l]

    self.beta = beta
    self.lamb = lamb
    self.tau_mesh, self.wsample, self.Ttc, self.Tcn, self.Tnc, self.Tct = read_IR_matrices(os.path.abspath(self._ir_dict[lamb]), self.beta)
    self.nts = self.tau_mesh.shape[0]
    self.nw  = self.wsample.shape[0]

  def update(self, beta=None, lamb=None):
    if lamb not in self._ir_dict.keys():
      raise ValueError("{} is not an acceptable lambda value.".format(lamb)
                       + " Acceptable lambdas are " + str(self._ir_dict.keys()))
    if lamb is not None: self.lamb = lamb
    if beta is not None: self.beta = beta
    self.tau_mesh, self.wsample, self.Ttc, self.Tcn, self.Tnc, self.Tct = read_IR_matrices(self._ir_dict[self.lamb], self.beta)
    self.nts = self.tau_mesh.shape[0]
    self.nw = self.wsample.shape[0]

  def tau_to_w(self, X_t):
    X_w = np.zeros((self.nw,) + X_t.shape[1:], dtype=complex)
    original_shape = X_w.shape

    X_w, X_t = X_w.reshape(self.nw, -1), X_t.reshape(self.nts, -1)
    X_w = reduce(np.dot, (self.Tnc, self.Tct, X_t[1:-1]))
    X_w = X_w.reshape(original_shape)
    return X_w

  def w_to_tau(self, X_w):
    X_t = np.zeros((self.nts,) + X_w.shape[1:], dtype=complex)
    original_shape = X_t.shape

    X_w, X_t = X_w.reshape(self.nw, -1), X_t.reshape(self.nts, -1)
    X_t = reduce(np.dot, (self.Ttc, self.Tcn, X_w))
    X_t = X_t.reshape(original_shape)
    return X_t

  # TODO Specify the version of irbasis.
  def tau_to_w_other(self, X_t, wsample):
    nw = wsample.shape[0]
    X_w = np.zeros((nw,)+X_t.shape[1:], dtype=complex)
    original_shape = X_w.shape

    ir_factory = irbasis.load("F", float(self.lamb))
    tnc = ir_factory.compute_unl(wsample)
    tnc *= np.sqrt(self.beta)

    X_w, X_t = X_w.reshape(nw, -1), X_t.reshape(self.nts, -1)
    X_w = reduce(np.dot, (tnc, self.Tct, X_t[1:-1]))
    X_w = X_w.reshape(original_shape)
    return X_w

def read_IR_matrices(ir_path, beta):
  ir = h5py.File(ir_path,'r')
  wsample = ir["fermi/wsample"][()]
  xsample = ir["fermi/xsample"][()]

  Ttc_minus1 = ir["/fermi/ux1l_minus"][()]
  Ttc_tmp = ir["/fermi/uxl"][()]
  Ttc_1 = ir["fermi/ux1l"][()]
  Ttc = np.zeros((Ttc_tmp.shape[0]+2, Ttc_tmp.shape[1]))
  Ttc[0], Ttc[1:-1], Ttc[-1] = Ttc_minus1, Ttc_tmp, Ttc_1
  Tnc_re = ir["/fermi/uwl_re"][()]
  Tnc_im = ir["/fermi/uwl_im"][()]
  Tnc = Tnc_re + 1j*Tnc_im
  ir.close()

  wsample = (2*wsample+1)*np.pi/beta
  tau_mesh = np.zeros(xsample.shape[0]+2)
  tau_mesh[0], tau_mesh[1:-1], tau_mesh[-1] = 0, (xsample+1)*beta/2.0, beta

  Ttc *= np.sqrt(2.0/beta)
  Tnc *= np.sqrt(beta)
  Tct = np.linalg.inv(Ttc[1:-1])
  Tcn = np.linalg.inv(Tnc)

  return tau_mesh, wsample, Ttc, Tcn, Tnc, Tct


if __name__ == "__main__":
  ir = IR_factory(1000, '1e4')
  print(ir.nts)
  ir.update(1000, '1e5')
  print(ir.nts)