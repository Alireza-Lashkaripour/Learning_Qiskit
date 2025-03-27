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
print('MPS energy = ', np.dot(mps, mpo @ mps))
print('MPS = ', mps.show_bond_dims())
print('MPS norm = ', mps.norm())
print('DMRG: ', dmrg)
np.save("h2o_energy.npy", ener)
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

def map_mps_to_quantum_circuit(mps_data):
    n_sites = mps_data['n_sites']
    tensors = mps_data['tensors']
    bond_dims = mps_data['bond_dims']
    
    qc = QuantumCircuit(n_sites)
    
    # Start with all qubits in |0⟩
    for i in range(n_sites):
        tensor = tensors[i]
        if hasattr(tensor, 'shape'):
            norm_factor = np.sqrt(np.sum(np.abs(tensor)**2))
            if norm_factor > 0:
                theta = 2 * np.arccos(min(1.0, max(0.0, 
                                     np.abs(tensor.flatten()[0])/norm_factor if len(tensor.flatten()) > 0 else 0.5)))
                qc.ry(theta, i)
        if i < n_sites - 1:
            qc.cx(i, i+1)
            
            if bond_dims[i+1] > 2:  # If bond dimension requires more entanglement
                qc.ry(np.pi/bond_dims[i+1], i+1)
                qc.cx(i, i+1)
    
    return qc

qc = map_mps_to_quantum_circuit(mps_data)
print(f"Created MPS-mapped quantum circuit with {n_sites} qubits")
print(f"Circuit depth: {qc.depth()}")
print(qc)
