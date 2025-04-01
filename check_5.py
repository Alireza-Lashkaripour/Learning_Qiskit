import numpy as np 
import math
from scipy.linalg import sqrtm, polar
from pyblock3.algebra.mpe import MPE 
from pyblock3.hamiltonian import Hamiltonian 
from pyblock3.fcidump import FCIDUMP 
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, Operator
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
#from qiskit_nature.second_q.drivers.fcidump import FCIDumpDriver
#from qiskit_nature.second_q.convertors import QubitConverter
from qiskit.primitives import Estimator
from qiskit.circuit.library import UnitaryGate
import matplotlib.pyplot as plt

# Part 1: Classical MPS calculation with DMRG
# ------------------------------------------

fd = 'H2O.STO3G.FCIDUMP'
hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
mpo = hamil.build_qc_mpo()
mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
print('MPO (compressed) = ', mpo.show_bond_dims())

hamil_data = {
    'n_sites': hamil.n_sites,
    'orb_sym': hamil.orb_sym if hasattr(hamil, 'orb_sym') else None,
    'isym': hamil.isym if hasattr(hamil, 'isym') else None,
    'h1e': hamil.h1e.copy() if hasattr(hamil, 'h1e') else None,
    'g2e': hamil.g2e.copy() if hasattr(hamil, 'g2e') else None,
    'ecore': hamil.ecore if hasattr(hamil, 'ecore') else None,
    'pg': hamil.pg if hasattr(hamil, 'pg') else None
}
np.save("h2o_hamiltonian.npy", hamil_data, allow_pickle=True)
print("Hamiltonian saved to h2o_hamiltonian.npy")
# Construct MPS
bond_dim = 200
mps = hamil.build_mps(bond_dim)

# Canonicalize MPS
mps = mps.canonicalize(center=0)
mps /= mps.norm()

# DMRG optimization
dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0],
    dav_thrds=[1E-3], iprint=2, n_sweeps=10)
ener = dmrg.energies[-1]
print("Energy(Ground State) = %20.12f" % ener)
print('MPS energy = ', np.dot(mps, mpo @ mps))
print('MPS = ', mps.show_bond_dims())
print('MPS norm = ', mps.norm())
print('DMRG: ', dmrg)
np.save("h2o_energy.npy", ener)
print("MPS after(bond dim): ", mps.show_bond_dims())
print(mps[0])


# Save the complete MPS information
mps_data = {
    'n_sites': hamil.n_sites,
    'bond_dims': [int(dim) for dim in mps.show_bond_dims().split('|')],
    'tensors': [t.data.copy() if hasattr(t, 'data') else t.copy() for t in mps.tensors],
    'q_labels': [t.q_labels if hasattr(t, 'q_labels') else None for t in mps.tensors],
    'energy': ener
}

np.save("h2o_mps_complete.npy", mps_data, allow_pickle=True)
mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
n_sites = mps_data['n_sites']
tensors = mps_data['tensors']
bond_dims = mps_data['bond_dims']
q_labels = mps_data['q_labels']
energy_classical = mps_data['energy']
physical_dim = 4

print([n_sites for bond_dims in tensors])
print("Number of sites:", n_sites)
print("Bond dimensions:", bond_dims)
print("Checking tensor shapes...")
for i, tensor in enumerate(tensors):
    print(f"Tensor {i} shape:", tensor.shape if hasattr(tensor, 'shape') else "Complex structure")
for i in range(n_sites):
    print(f"Tensor {i} expected shape: (4, {bond_dims[i]}, {bond_dims[i+1]})")




def next_power_of_2(x):
    return 2**math.ceil(math.log2(x))

def pad_to_unitary(V, dim):
    m, n = V.shape
    if m < dim:
        V = np.pad(V, ((0, dim - m), (0, 0)), mode='constant')
    elif m > dim:
        V = V[:dim, :]
    Q, _ = np.linalg.qr(V)
    if Q.shape[1] < dim:
        Q_rand, _ = np.linalg.qr(np.random.randn(dim, dim - Q.shape[1]))
        Q = np.hstack([Q, Q_rand])
    U, _ = polar(Q)
    return U

def get_full_tensor(tensor, expected_shape):
    expected_size = np.prod(expected_shape)
    if tensor.size == expected_size:
        return tensor.reshape(expected_shape)
    elif tensor.ndim == 1:
        full_tensor = np.zeros(expected_shape, dtype=tensor.dtype)
        flat_tensor = tensor.flatten()
        num_to_fill = min(full_tensor.size, flat_tensor.size)
        for idx in range(num_to_fill):
            full_tensor.flat[idx] = flat_tensor[idx]
        return full_tensor
    else:
        raise ValueError("Cannot expand tensor of shape {} to expected shape {}".format(tensor.shape, expected_shape))

print("next_power_of_2(7):", next_power_of_2(7))
V_sample = np.array([[1, 2], [3, 4]], dtype=complex)
unitary_sample = pad_to_unitary(V_sample, 4)
print("Sample unitary from pad_to_unitary:")
print(unitary_sample)
print("Shape of sample unitary:", unitary_sample.shape)
tensor_sample = np.array([1, 2, 3, 4])
full_tensor_sample = get_full_tensor(tensor_sample, (2, 2))
print("Full tensor from get_full_tensor:")
print(full_tensor_sample)

# Process each MPS tensor to build corresponding unitaries
unitary_gates = []
for k in range(n_sites):
    D_in = bond_dims[k]
    D_out = bond_dims[k+1]
    expected_shape = (physical_dim, D_in, D_out)
    print("Site", k, "expected tensor shape:", expected_shape)
    full_tensor = get_full_tensor(tensors[k], expected_shape)
    print("Site", k, "full tensor shape:", full_tensor.shape)
    # Reshape tensor into a matrix T_mat of shape (4 * D_out, D_in)
    T_mat = full_tensor.reshape(physical_dim * D_out, D_in)
    print("Site", k, "reshaped matrix T_mat shape:", T_mat.shape)
    # For the input, use the bond dimension D_in; for the output, use 4 * D_out.
    N_in = next_power_of_2(D_in)
    N_out = next_power_of_2(physical_dim * D_out)
    dim = max(N_in, N_out)
    print("Site", k, "N_in:", N_in, "N_out:", N_out, "final dim:", dim)
    # Create W with shape (dim, N_in) and embed T_mat into its upper-left block
    W = np.zeros((dim, N_in), dtype=complex)
    W[:physical_dim * D_out, :D_in] = T_mat
    U_site = pad_to_unitary(W, dim)
    print("Site", k, "unitary shape:", U_site.shape)
    print("Unitary (first 2 rows):", U_site[:2])
    print("-" * 50)
    unitary_gates.append(U_site)

print("Completed processing unitaries for all", n_sites, "sites.")

# Build Qiskit circuits for each site based on the computed unitaries (for inspection)
circuits = []
for k, U in enumerate(unitary_gates):
    dim = U.shape[0]
    num_q = int(math.log2(dim))
    qr = QuantumRegister(num_q, f'q{k}')
    qc = QuantumCircuit(qr)
    gate = UnitaryGate(U, label=f"U{k}")
    qc.append(gate, qr)
    circuits.append(qc)
    print(f"Created circuit for site {k} with {num_q} qubits (unitary dim: {dim}).")
    print(qc.draw())
    print("-" * 50)

# ==============================
# Final Part: Sequential Composition and Energy Calculation
# ==============================

# Instead of taking a tensor product (which creates a huge Hilbert space),
# we combine the unitaries sequentially on a common register.
# We first determine a common dimension (the maximum dimension among our unitaries).
common_dim = max(U.shape[0] for U in unitary_gates)
print("Common register dimension:", common_dim)

# Function to pad a unitary to the common_dim x common_dim space
def pad_unitary_to_common(U, common_dim):
    dim = U.shape[0]
    if dim < common_dim:
        U_padded = np.eye(common_dim, dtype=complex)
        U_padded[:dim, :dim] = U
        return U_padded
    else:
        return U

# Pad each unitary to the common dimension
unitaries_padded = [pad_unitary_to_common(U, common_dim) for U in unitary_gates]

# Compute the sequential product: U_total = U_n ... U_1
U_total = np.eye(common_dim, dtype=complex)
for U in unitaries_padded:
    U_total = U @ U_total

# Compute the final state: |psi_qc> = U_total |0>
initial_state = np.zeros((common_dim,), dtype=complex)
initial_state[0] = 1.0
psi_qc = U_total @ initial_state

print("Final statevector |psi_qc>:")
print(psi_qc)

# Density matrix of the final state
rho = np.outer(psi_qc, np.conjugate(psi_qc))
print("Density matrix of the prepared state:")
print(rho)


