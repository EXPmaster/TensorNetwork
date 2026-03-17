# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""implementation of different Matrix Product Operators."""
import pickle
import numpy as np
from tensornetwork.backends import backend_factory
from tensornetwork.backend_contextmanager import get_default_backend
from tensornetwork.backends.abstract_backend import AbstractBackend
from typing import List, Union, Text, Optional, Any, Type
Tensor = Any


# TODO (mganahl): this class is very similar to BaseMPS. The two could probably
# be merged.
class BaseMPO:
  """
  Base class for MPOs.
  """

  def __init__(self,
               tensors: List[Tensor],
               backend: Optional[Union[AbstractBackend, Text]] = None,
               name: Optional[Text] = None) -> None:
    """
    Initialize a BaseMPO.
    Args:
      tensors: A list of `Tensor` objects.
      backend: The name of the backend that should be used to perform 
        contractions. 
      name: A name for the MPO.
    """
    if backend is None:
      backend = get_default_backend()
    if isinstance(backend, AbstractBackend):
      self.backend = backend
    else:
      self.backend = backend_factory.get_backend(backend)
    self.tensors = [self.backend.convert_to_tensor(t) for t in tensors]
    if len(self.tensors) > 0:
      if not all(
          self.tensors[0].dtype == tensor.dtype for tensor in self.tensors):
        raise TypeError('not all dtypes in BaseMPO.tensors are the same')

    self.name = name

  def __iter__(self):
    return iter(self.tensors)

  def __len__(self) -> int:
    return len(self.tensors)

  @property
  def dtype(self) -> Type[np.number]:
    if not all(
        self.tensors[0].dtype == tensor.dtype for tensor in self.tensors):
      raise TypeError('not all dtypes in BaseMPO.tensors are the same')
    return self.tensors[0].dtype

  @property
  def bond_dimensions(self) -> List[int]:
    """Returns a vector of all bond dimensions.
        The vector will have length `N+1`, where `N == num_sites`."""
    return [self.tensors[0].shape[0]
           ] + [tensor.shape[1] for tensor in self.tensors]
  
  def save(self, path: Text) -> None:
    """
    Save the MPO to disk.
    Args:
      path: The path to save the MPO to.
    """
    tensors = [self.backend.convert_to_numpy(tensor) for tensor in self.tensors]
    data = {'tensors': tensors, 'name': self.name}
    with open(path, 'wb') as f:
      pickle.dump(data, f)
  
  @classmethod
  def load(cls, path: Text, backend: Optional[Union[Text, AbstractBackend]] = None):
    with open(path, 'rb') as f:
      data = pickle.load(f)
    data['backend'] = backend
    return cls(**data)


class InfiniteMPO(BaseMPO):
  """
  Base class for implementation of infinite MPOs. Users should implement 
  specific infinite MPOs by deriving from InfiniteMPO.
  """

  def __init__(self,
               tensors: List[Tensor],
               backend: Optional[Union[AbstractBackend, Text]] = None,
               name: Optional[Text] = None) -> None:
    """
    Initialize an infinite MPO object
    Args:
      tensors: The mpo tensors.
      backend: An optional backend. Defaults to the defaulf backend  
        of TensorNetwork.
      name: An optional name for the MPO.
    """
    super().__init__(tensors=tensors, backend=backend, name=name)
    if self.bond_dimensions[0] != self.bond_dimensions[-1]:
      raise ValueError('left and right MPO ancillary dimension have to match')

  def roll(self, num_sites) -> None:
    tensors = [self.tensors[n] for n in range(num_sites, len(self.tensors))
              ] + [self.tensors[n] for n in range(num_sites)]
    self.tensors = tensors


class FiniteMPO(BaseMPO):
  """
  Base class for implementation of finite MPOs. Users should implement 
  specific finite MPOs by deriving from FiniteMPO
  """

  def __init__(self,
               tensors: List[Tensor],
               backend: Optional[Union[AbstractBackend, Text]] = None,
               name: Optional[Text] = None) -> None:
    """
    Initialize a finite MPO object
    Args:
      tensors: The mpo tensors.
      backend: An optional backend. Defaults to the defaulf backend  
        of TensorNetwork.
      name: An optional name for the MPO.
    """

    super().__init__(tensors=tensors, backend=backend, name=name)
    if (self.bond_dimensions[0] != 1) or (self.bond_dimensions[-1] != 1):
      raise ValueError('left and right MPO ancillary dimensions have to be 1')


class FiniteXXZ(FiniteMPO):
  """
  The Heisenberg Hamiltonian.
  """

  def __init__(self,
               Jz: np.ndarray,
               Jxy: np.ndarray,
               Bz: np.ndarray,
               dtype: Type[np.number],
               backend: Optional[Union[AbstractBackend, Text]] = None,
               name: Text = 'XXZ_MPO') -> None:
    """
    Returns the MPO of the finite XXZ model.
    Args:
      Jz:  The Sz*Sz coupling strength between nearest neighbor lattice sites.
      Jxy: The (Sx*Sx + Sy*Sy) coupling strength between nearest neighbor.
        lattice sites
      Bz: Magnetic field on each lattice site.
      dtype: The dtype of the MPO.
      backend: An optional backend.
      name: A name for the MPO.
    Returns:
      FiniteXXZ: The mpo of the finite XXZ model.
    """
    self.Jz = Jz
    self.Jxy = Jxy
    self.Bz = Bz
    N = len(Bz)
    mpo = []
    temp = np.zeros((1, 5, 2, 2), dtype=dtype)
    #BSz
    temp[0, 0, 0, 0] = -0.5 * Bz[0]
    temp[0, 0, 1, 1] = 0.5 * Bz[0]

    #Sm
    temp[0, 1, 0, 1] = Jxy[0] / 2.0 * 1.0
    #Sp
    temp[0, 2, 1, 0] = Jxy[0] / 2.0 * 1.0
    #Sz
    temp[0, 3, 0, 0] = Jz[0] * (-0.5)
    temp[0, 3, 1, 1] = Jz[0] * 0.5

    #11
    temp[0, 4, 0, 0] = 1.0
    temp[0, 4, 1, 1] = 1.0
    mpo.append(temp)
    for n in range(1, N - 1):
      temp = np.zeros((5, 5, 2, 2), dtype=dtype)
      #11
      temp[0, 0, 0, 0] = 1.0
      temp[0, 0, 1, 1] = 1.0
      #Sp
      temp[1, 0, 1, 0] = 1.0
      #Sm
      temp[2, 0, 0, 1] = 1.0
      #Sz
      temp[3, 0, 0, 0] = -0.5
      temp[3, 0, 1, 1] = 0.5
      #BSz
      temp[4, 0, 0, 0] = -0.5 * Bz[n]
      temp[4, 0, 1, 1] = 0.5 * Bz[n]

      #Sm
      temp[4, 1, 0, 1] = Jxy[n] / 2.0 * 1.0
      #Sp
      temp[4, 2, 1, 0] = Jxy[n] / 2.0 * 1.0
      #Sz
      temp[4, 3, 0, 0] = Jz[n] * (-0.5)
      temp[4, 3, 1, 1] = Jz[n] * 0.5
      #11
      temp[4, 4, 0, 0] = 1.0
      temp[4, 4, 1, 1] = 1.0

      mpo.append(temp)
    temp = np.zeros((5, 1, 2, 2), dtype=dtype)
    #11
    temp[0, 0, 0, 0] = 1.0
    temp[0, 0, 1, 1] = 1.0
    #Sp
    temp[1, 0, 1, 0] = 1.0
    #Sm
    temp[2, 0, 0, 1] = 1.0
    #Sz
    temp[3, 0, 0, 0] = -0.5
    temp[3, 0, 1, 1] = 0.5
    #BSz
    temp[4, 0, 0, 0] = -0.5 * Bz[-1]
    temp[4, 0, 1, 1] = 0.5 * Bz[-1]

    mpo.append(temp)
    super().__init__(tensors=mpo, backend=backend, name=name)


class FiniteHeisenberg(FiniteMPO):
  """
  The Heisenberg Hamiltonian.
  H = Jx * Sx*Sx + Jy * Sy*Sy + Jz * Sz*Sz + h * Sz
  """

  def __init__(self,
               Jx: np.ndarray,
               Jy: np.ndarray,
               Jz: np.ndarray,
               Bz: np.ndarray,
               dtype: Type[np.number],
               backend: Optional[Union[AbstractBackend, Text]] = None,
               name: Text = 'Heisenberg_MPO') -> None:
    """
    Returns the MPO of the finite Heisenberg model.
    Args:
      Jx: The Sx*Sx coupling strength between nearest neighbor lattice sites.
      Jy: The Sy*Sy coupling strength between nearest neighbor lattice sites.
      Jz: The Sz*Sz coupling strength between nearest neighbor lattice sites.
      Bz: Magnetic field on each lattice site.
      dtype: The dtype of the MPO.
      backend: An optional backend.
      name: A name for the MPO.
    Returns:
      FiniteHeisenberg: The mpo of the finite Heisenberg model.
    """
    self.Jz = Jz
    self.Jx = Jx
    self.Jy = Jy
    self.Bz = Bz
    sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
    sigma_y = np.array([[0, -1j], [1j, 0]]).astype(dtype)
    sigma_z = np.diag([1, -1]).astype(dtype)
    sigma_i = np.eye(2).astype(dtype)
    N = len(Bz)
    mpo = []
    temp = np.zeros((1, 5, 2, 2), dtype=dtype)
    #BSz
    temp[0, 0, :, :] = Bz[0] * sigma_z

    #Sx
    temp[0, 1, :, :] = Jx[0] * sigma_x
    #Sy
    temp[0, 2, :, :] = Jy[0] * sigma_y
    #Sz
    temp[0, 3, :, :] = Jz[0] * sigma_z

    #11
    temp[0, 4, :, :] = sigma_i
    mpo.append(temp)

    for n in range(1, N - 1):
      temp = np.zeros((5, 5, 2, 2), dtype=dtype)
      #11
      temp[0, 0, :, :] = sigma_i
      #Sx
      temp[1, 0, :, :] = sigma_x
      #Sy
      temp[2, 0, :, :] = sigma_y
      #Sz
      temp[3, 0, :, :] = sigma_z
      #BSz
      temp[4, 0, :, :] = Bz[n] * sigma_z

      #Sx
      temp[4, 1, :, :] = Jx[n] * sigma_x
      #Sy
      temp[4, 2, :, :] = Jy[n] * sigma_y
      #Sz
      temp[4, 3, :, :] = Jz[n] * sigma_z
      #11
      temp[4, 4, :, :] = sigma_i

      mpo.append(temp)
    temp = np.zeros((5, 1, 2, 2), dtype=dtype)
    #11
    temp[0, 0, :, :] = sigma_i
    #Sx
    temp[1, 0, :, :] = sigma_x
    #Sy
    temp[2, 0, :, :] = sigma_y
    #Sz
    temp[3, 0, :, :] = sigma_z
    #BSz
    temp[4, 0, :, :] = Bz[-1] * sigma_z

    mpo.append(temp)
    super().__init__(tensors=mpo, backend=backend, name=name)


class FiniteTFI(FiniteMPO):
  """
  The famous transverse field Ising Hamiltonian.
  The ground state energy of the infinite system at criticality is -4/pi.

  Convention: sigma_z=diag([-1,1])
  """

  def __init__(self,
               Jx: np.ndarray,
               Bz: np.ndarray,
               dtype: Type[np.number],
               backend: Optional[Union[AbstractBackend, Text]] = None,
               name: Text = 'TFI_MPO') -> None:
    """
    Returns the MPO of the finite TFI model.
    Args:
      Jx:  The Sx*Sx coupling strength between nearest neighbor lattice sites.
      Bz:  Magnetic field on each lattice site.
      dtype: The dtype of the MPO.
      backend: An optional backend.
      name: A name for the MPO.
    Returns:
      FiniteTFI: The mpo of the infinite TFI model.
    """
    self.Jx = Jx.astype(dtype)
    self.Bz = Bz.astype(dtype)
    N = len(Bz)
    sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
    sigma_z = np.diag([1, -1]).astype(dtype)
    mpo = []
    temp = np.zeros(shape=[1, 3, 2, 2], dtype=dtype)
    #Bsigma_z
    temp[0, 0, :, :] = self.Bz[0] * sigma_x
    #sigma_x
    temp[0, 1, :, :] = self.Jx[0] * sigma_z
    #11
    temp[0, 2, 0, 0] = 1.0
    temp[0, 2, 1, 1] = 1.0
    mpo.append(temp)
    for n in range(1, N - 1):
      temp = np.zeros(shape=[3, 3, 2, 2], dtype=dtype)
      #11
      temp[0, 0, 0, 0] = 1.0
      temp[0, 0, 1, 1] = 1.0
      #sigma_x
      temp[1, 0, :, :] = sigma_z
      #Bsigma_z
      temp[2, 0, :, :] = self.Bz[n] * sigma_x
      #sigma_x
      temp[2, 1, :, :] = self.Jx[n] * sigma_z
      #11
      temp[2, 2, 0, 0] = 1.0
      temp[2, 2, 1, 1] = 1.0
      mpo.append(temp)

    temp = np.zeros([3, 1, 2, 2], dtype=dtype)
    #11
    temp[0, 0, 0, 0] = 1.0
    temp[0, 0, 1, 1] = 1.0
    #sigma_x
    temp[1, 0, :, :] = sigma_z
    #Bsigma_z
    temp[2, 0, :, :] = self.Bz[-1] * sigma_x
    mpo.append(temp)
    super().__init__(tensors=mpo, backend=backend, name=name)


class FiniteFreeFermion2D(FiniteMPO):
  """
  Free fermions on a 2d grid
  """

  def __init__(self,
               t1: float,
               t2: float,
               v: float,
               N1: int,
               N2: int,
               dtype: Type[np.number],
               backend: Optional[Union[AbstractBackend, Text]] = None,
               name: Text = '2DTFI_MPO'):
    """
    Returns the MPO of the free fermions on
    an N1 by N2 grid. The MPO is snaked
    along the vertical direction.

    Args:
      t1: The hopping amplitude along vertical direction.
      t2: The hopping amplitude along horizontal direction.
      v:  local potential strength.
      N1: The vertical size of the lattice.
      N2: The horizontal size of the lattice.
      dtype: The dtype of the MPO.
      backend: An optional backend.
      name: A name for the MPO.

    Returns:
      FiniteFreeFermion2D: The mpo of the TFI model on an N1 x N2 grid.
    """

    self.t1 = t1
    self.t2 = t2
    self.N1 = N1
    self.N2 = N2
    self.v = v

    eye = np.eye(2).astype(dtype)
    c = np.array([[0, 1], [0, 0]]).astype(dtype)
    particle_number = np.diag([0, 1]).astype(dtype)
    sigma_z = np.diag([1, -1]).astype(dtype)
    cdag = c.T.conj()

    mpo_dim = 2 * N1 + 2
    mpo_matrix = np.zeros((1, mpo_dim, 2, 2), dtype=dtype)
    mpo_matrix[0, 0, :, :] = v * particle_number

    mpo_matrix[0, 1, :, :] = t1 * cdag
    mpo_matrix[0, N1, :, :] = t2 * cdag
    mpo_matrix[0, N1 + 1, :, :] = t1 * c # c @ sigma_z == -c
    mpo_matrix[0, 2 * N1, :, :] = t2 * c # c @ sigma_z == -c

    mpo_matrix[0, 2 * N1 + 1, :, :] = eye
    mpo = [mpo_matrix]

    n2 = 0
    for n in range(1, N1 * N2 - 1):
      if (n + 1) % N1 == 0:
        _t1 = 0
      else:
        _t1 = t1

      if n < N1 * (N2 - 1):
        _t2 = t2
      else:
        _t2 = 0
      if n % N1 == 0:
        n2 += 1
      mpo_matrix = np.zeros((mpo_dim, mpo_dim, 2, 2), dtype=dtype)
      mpo_matrix[0, 0, :, :] = eye
      mpo_matrix[1, 0, :, :] = c
      for n1 in range(2, N1 + 1):
        mpo_matrix[n1, n1 - 1, :, :] = sigma_z

      mpo_matrix[N1 + 1, 0, :, :] = cdag
      for n1 in range(N1 + 2, 2 * N1 + 1):
        mpo_matrix[n1, n1 - 1, :, :] = sigma_z

      mpo_matrix[2 * N1 + 1, 0, :, :] = v * particle_number

      mpo_matrix[2 * N1 + 1, 1, :, :] = _t1 * cdag # cdag @ sigma_z == cdag
      mpo_matrix[2 * N1 + 1, N1, :, :] = _t2 * cdag
      mpo_matrix[2 * N1 + 1, N1+1, :, :] = _t1 * c # c @ sigma_z == -c
      mpo_matrix[2 * N1 + 1, 2*N1, :, :] = _t2 * c # c @ sigma_z == -c

      mpo_matrix[2 * N1 + 1, 2 * N1 + 1, :, :] = eye
      mpo.append(mpo_matrix)

    mpo_matrix = np.zeros((mpo_dim, 1, 2, 2), dtype=dtype)
    mpo_matrix[0, 0, :, :] = eye
    mpo_matrix[1, 0, :, :] = c
    mpo_matrix[N1 + 1, 0, :, :] = cdag
    mpo_matrix[2 * N1 + 1, 0, :, :] = v * particle_number
    mpo.append(mpo_matrix)
    super().__init__(tensors=mpo, backend=backend, name=name)


class Finite2DTFI(FiniteMPO):
  """
  Transverse-field Ising model on a 2D N1 x N2 lattice.
  H = Jv * sum_{c,r} Z_(c,r)*Z_(c,r+1)  (vertical NN bonds within columns)
    + Jh * sum_{r,c} Z_(r,c)*Z_(r,c+1)  (horizontal NN bonds between columns)
    + sum_n Bz[n] * X_n                  (site-dependent transverse field)

  Sites are ordered in column-major (snake) order: (row r, col c) -> n = c*N1 + r.
  This makes vertical bonds nearest-neighbor in 1D (distance 1) and horizontal
  bonds long-range (distance N1), handled via MPO carriers.

  Virtual bond dimension: D = N1 + 3.
    State 0:      H accumulator.
    State 1:      Vertical Z carrier (active for 1 bond within a column).
    States 2+j:   Horizontal Z carrier for row j (j=0..N1-1), travels N1 steps.
    State N1+2:   Incoming identity (I right).
  """

  def __init__(self,
               Jv: float,
               Jh: float,
               Bz: np.ndarray,
               N1: int,
               N2: int,
               dtype: Type[np.number],
               backend: Optional[Union[AbstractBackend, Text]] = None,
               name: Text = '2DTFI_MPO') -> None:
    """
    Returns the MPO of the 2D TFI model on an N1 x N2 grid.

    Args:
      Jv:  Vertical ZZ coupling (scalar, uniform across all vertical bonds).
      Jh:  Horizontal ZZ coupling (scalar, uniform across all horizontal bonds).
      Bz:  Transverse field per site, shape (N1*N2,) in column-major order.
      N1:  Number of rows.
      N2:  Number of columns.
      dtype: The dtype of the MPO.
      backend: An optional backend.
      name: A name for the MPO.

    Returns:
      Finite2DTFI: The MPO of the 2D TFI model.
    """
    self.N1 = N1
    self.N2 = N2
    N = N1 * N2

    Bz = np.asarray(Bz, dtype=dtype)
    assert len(Bz) == N, f'Bz must have length N1*N2={N}, got {len(Bz)}'

    eye = np.eye(2).astype(dtype)
    sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
    sigma_z = np.diag([1, -1]).astype(dtype)

    D = N1 + 3  # virtual bond dimension
    mpo = []

    # First tensor: site n=0, r=0, c=0, shape (1, D, 2, 2)
    W = np.zeros((1, D, 2, 2), dtype=dtype)
    W[0, 0] = Bz[0] * sigma_x          # transverse field at site 0 -> H
    if N1 > 1:
      W[0, 1] = Jv * sigma_z           # launch vertical Z carrier for bond (r=0,c=0)-(r=1,c=0)
    if N2 > 1:
      W[0, 2 + 0] = Jh * sigma_z       # launch horizontal Z carrier for row 0
    W[0, N1 + 2] = eye                 # pass identity right
    mpo.append(W)

    # Interior tensors: sites n = 1 .. N-2
    for n in range(1, N - 1):
      r = n % N1   # row index in column-major order
      c = n // N1  # column index

      W = np.zeros((D, D, 2, 2), dtype=dtype)
      W[0, 0] = eye                    # H accumulator passes through

      if r > 0:
        W[1, 0] = sigma_z              # close vertical Z carrier: Z_n closes ZZ with site n-1

      if c > 0:
        W[2 + r, 0] = sigma_z          # close horizontal Z carrier for row r: ZZ with site n-N1

      # Propagate horizontal carriers for all rows j != r (they stay active)
      for j in range(N1):
        if j != r:
          W[2 + j, 2 + j] = eye

      W[N1 + 2, 0] = Bz[n] * sigma_x  # transverse field at site n -> H

      if r < N1 - 1:
        W[N1 + 2, 1] = Jv * sigma_z   # launch vertical Z carrier for bond n-(n+1)

      if c < N2 - 1:
        W[N1 + 2, 2 + r] = Jh * sigma_z  # launch horizontal Z carrier for row r

      W[N1 + 2, N1 + 2] = eye         # pass identity right
      mpo.append(W)

    # Last tensor: site n=N-1, r=N1-1, c=N2-1, shape (D, 1, 2, 2)
    r = N1 - 1
    c = N2 - 1
    W = np.zeros((D, 1, 2, 2), dtype=dtype)
    W[0, 0] = eye                      # H accumulator
    if r > 0:
      W[1, 0] = sigma_z                # close vertical Z carrier (r=N1-1 > 0 always)
    if c > 0:
      W[2 + r, 0] = sigma_z            # close horizontal Z carrier for last row
    W[N1 + 2, 0] = Bz[-1] * sigma_x   # transverse field at last site
    mpo.append(W)

    super().__init__(tensors=mpo, backend=backend, name=name)
