import numpy as np 
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
from qiskit.primitives import Estimator
import matplotlib.pyplot as plt

# Part 1: Classical MPS calculation with DMRG
# ------------------------------------------

fd = 'H2O.STO3G.FCIDUMP'
hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
mpo = hamil.build_qc_mpo()
mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
print('MPO (compressed) = ', mpo.show_bond_dims())

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

# Verify results
print('MPS energy = ', np.dot(mps, mpo @ mps))
print('MPS norm = ', mps.norm())
print('DMRG: ', dmrg)

# Save energy
np.save("h2o_energy.npy", ener)

print('---------------------Save_MPS----------------------')
print("MPS after(bond dim): ", mps.show_bond_dims())
print(mps[0])

# Calculate one-particle density matrix (1PDM)
pdm1 = np.zeros((hamil.n_sites, hamil.n_sites))
for i in range(hamil.n_sites):
    diop = OpElement(OpNames.D, (i, 0), q_label=SZ(-1, -1, hamil.orb_sym[i]))
    di = hamil.build_site_mpo(diop)
    for j in range(hamil.n_sites):
        djop = OpElement(OpNames.D, (j, 0), q_label=SZ(-1, -1, hamil.orb_sym[j]))
        dj = hamil.build_site_mpo(djop)
        # factor 2 due to alpha + beta spins
        pdm1[i, j] = 2 * np.dot((di @ mps).conj(), dj @ mps)

print("1PDM calculated from classical MPS:")
print(pdm1)
print("MPS after(bond dim): ", mps.show_bond_dims())
np.save("h2o_pdm1.npy", pdm1)

# Save the complete MPS information
mps_data = {
    'n_sites': hamil.n_sites,
    'bond_dims': [int(dim) for dim in mps.show_bond_dims().split('|')],
    'tensors': [t.data.copy() if hasattr(t, 'data') else t.copy() for t in mps.tensors],
    'q_labels': [t.q_labels if hasattr(t, 'q_labels') else None for t in mps.tensors],
    'energy': ener,
    'pdm1': pdm1
}

np.save("h2o_mps_complete.npy", mps_data, allow_pickle=True)
mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
n_sites = mps_data['n_sites']
tensors = mps_data['tensors']
bond_dims = mps_data['bond_dims']
q_labels = mps_data['q_labels']
pdm1 = mps_data['pdm1']
energy_classical = mps_data['energy']


print("Starting to save MPS information")
print(f"Created mps_data dictionary with keys: {list(mps_data.keys())}")
print(f"n_sites: {mps_data['n_sites']}")
print(f"bond_dims: {mps_data['bond_dims']}")
print(f"Number of tensors: {len(mps_data['tensors'])}")
print(f"Energy: {mps_data['energy']}")
print(f"PDM1 shape: {mps_data['pdm1'].shape if hasattr(mps_data['pdm1'], 'shape') else 'No shape attribute'}")
print("Saved mps_data to h2o_mps_complete.npy")
print("Loaded mps_data from h2o_mps_complete.npy")
print(f"Loaded data has keys: {list(mps_data.keys())}")
print(f"Extracted n_sites: {n_sites}")
print(f"Extracted tensors, count: {len(tensors)}")
print(f"Extracted bond_dims: {bond_dims}")
print(f"Extracted q_labels, count: {len(q_labels)}")
print(f"Extracted pdm1 with shape: {pdm1.shape if hasattr(pdm1, 'shape') else 'No shape attribute'}")
print(f"Extracted energy_classical: {energy_classical}")

print([n_sites for bond_dims in tensors])
print("Number of sites:", n_sites)
print("Bond dimensions:", bond_dims)
print("Checking tensor shapes...")
for i, tensor in enumerate(tensors):
    print(f"Tensor {i} shape:", tensor.shape if hasattr(tensor, 'shape') else "Complex structure")
for i in range(n_sites):
    print(f"Tensor {i} expected shape: (4, {bond_dims[i]}, {bond_dims[i+1]})")


from qiskit import QuantumCircuit, QuantumRegister
import numpy as np

# Load MPS data
mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
tensors = mps_data['tensors']
n_sites = mps_data['n_sites']
bond_dims = mps_data['bond_dims']

# Ancilla qubits = log2(bond_dim) for each bond
anc_per_site = int(np.log2(bond_dims[0]))  # Assuming uniform bond dims
total_anc = anc_per_site * (n_sites - 1)
phys_qubits = QuantumRegister(n_sites, 'p')
anc_qubits = QuantumRegister(total_anc, 'a')
qc = QuantumCircuit(phys_qubits, anc_qubits)

# Decompose MPS tensors into unitaries
unitaries = []
for i in range(n_sites):
    A = tensors[i]
    bond_in, phys_dim, bond_out = A.shape
    A_flat = A.reshape(bond_in * phys_dim, bond_out)
    Q, R = np.linalg.qr(A_flat)  # Q is unitary
    unitaries.append(Q)
    # Absorb R into next tensor (not shown)

# Add gates to circuit (simplified)
for i in range(n_sites):
    q_anc_start = i * anc_per_site
    q_anc = anc_qubits[q_anc_start: q_anc_start + anc_per_site]
    q_target = [phys_qubits[i]] + q_anc  # Combine phys + anc qubits
    qc.unitary(unitaries[i], q_target)
