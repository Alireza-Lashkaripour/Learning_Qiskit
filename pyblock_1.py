import numpy as np
import matplotlib.pyplot as plt
from pyblock3.algebra.mpe import MPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, SparsePauliOp
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

fd = 'H2O.STO3G.FCIDUMP'
hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
mpo = hamil.build_qc_mpo()
mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)

bond_dim = 200
mps = hamil.build_mps(bond_dim)
mps = mps.canonicalize(center=0)
mps /= mps.norm()

dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0],
    dav_thrds=[1E-3], iprint=2, n_sweeps=10)
ener = dmrg.energies[-1]
print("Energy(Ground State) = %20.12f" % ener)

np.save("h2o_energy.npy", ener)

print('---------------------Dense Tensor Reconstruction----------------------')
target_tensor_index = 1
if target_tensor_index < len(mps.tensors):
    sparse_tensor_object = mps[target_tensor_index]
    try:
        dense_tensor = sparse_tensor_object.to_dense()
        print(f"Reconstructed Dense Tensor {target_tensor_index} Shape: {dense_tensor.shape}")
        np.save(f"h2o_dense_tensor_{target_tensor_index}.npy", dense_tensor)
        print(f"Dense Tensor {target_tensor_index} saved to h2o_dense_tensor_{target_tensor_index}.npy")
    except AttributeError:
        print(f"Error: Failed to find a '.to_dense()' method (or similar) for the tensor object at index {target_tensor_index}.")
        print("Please check the PyBlock3 documentation for the correct method to convert sparse tensors to dense NumPy arrays.")
    except Exception as e:
        print(f"An error occurred during dense tensor reconstruction: {e}")
else:
    print(f"Error: Index {target_tensor_index} is out of bounds for MPS tensors (Length: {len(mps.tensors)}).")


print('---------------------Save_MPS----------------------')

mps_data = {
    'n_sites': hamil.n_sites,
    'bond_dims': [int(dim) for dim in mps.show_bond_dims().split('|')],
    'tensors': [t.data.copy() if hasattr(t, 'data') else t.copy() for t in mps.tensors],
    'q_labels': [t.q_labels if hasattr(t, 'q_labels') else None for t in mps.tensors],
    'energy': ener,
}

np.save("h2o_mps_complete.npy", mps_data, allow_pickle=True)
mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
n_sites = mps_data['n_sites']
tensors = mps_data['tensors']
bond_dims = mps_data['bond_dims']
q_labels = mps_data['q_labels']
energy_classical = mps_data['energy']
