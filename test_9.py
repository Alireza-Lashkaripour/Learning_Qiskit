import numpy as np 
from pyblock3.algebra.mpe import MPE 
from pyblock3.hamiltonian import Hamiltonian 
from pyblock3.fcidump import FCIDUMP 
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

fd = 'H2O.STO3G.FCIDUMP'
hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
mpo = hamil.build_qc_mpo()
mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
print('MPO (compressed) = ', mpo.show_bond_dims())

# Construct MPS: 
bond_dim = 200
mps = hamil.build_mps(bond_dim)
# Check that ground-state MPS is normalized:
#print('MPS = ', mps.show_bond_dims())

# Canonicalize MPS
#print("MPS = ", mps.show_bond_dims())
mps = mps.canonicalize(center=0)
mps /= mps.norm()
#print("MPS = ", mps.show_bond_dims())


# DMRG
dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0],
    dav_thrds=[1E-3], iprint=2, n_sweeps=10) # ==> number of opt. sweeps 
ener = dmrg.energies[-1]
print("Energy(Ground State) = %20.12f" % ener)
# Check ground-state energy: 
print('MPS energy = ', np.dot(mps, mpo @ mps))
print('MPS norm = ', mps.norm())
print('DMRG: ', dmrg)
############
# This part is for 'One Particle Density Matrix', based on the exaplin in github
np.save("h2o_energy.npy", ener)

print('---------------------Save_MPS----------------------')
print("MPS after(bond dim): ", mps.show_bond_dims())
#print(mps[0].__class__)
print(mps[0])

#np.save("MPS_tensors_saved.npy", mps[0])
#print('---------------------Save_MPS----------------------')
#print('TensorDot Product of MPS and MPO:', np.tensordot(mps[0], mps[1], axes=1))
# Now saving the 1PDM and MPS details( like in example)
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


print('----------------------Quantum_Circuit_Mapping------------------------')

# Critical parameters for H2O.STO3G
phys_dim = 4  # 4 possible states per site (0, ↑, ↓, ↑↓)
n_qubits = n_sites  # Should be 7 for H2O.STO3G

# Correct tensor shapes using bond_dims[i] (left) and bond_dims[i+1] (right)
shapes = [(bond_dims[i], phys_dim, bond_dims[i+1]) for i in range(n_sites)]


# Validate tensor shapes
print("Bond dimensions:", bond_dims)
tensors_3d = []
for i in range(n_sites):
    expected_size = np.prod(shapes[i])
    print(f"Tensor {i} size: {tensors[i].size}, expected shape: {shapes[i]} (size {expected_size})")
    # Handle sparse tensors by padding with zeros if necessary
    if tensors[i].size < expected_size:
        # Pad with zeros to match expected size (if sparse)
        padded_tensor = np.zeros(expected_size)
        padded_tensor[:tensors[i].size] = tensors[i]
        tensors_3d.append(padded_tensor.reshape(shapes[i]))
    else:
        tensors_3d.append(tensors[i].reshape(shapes[i]))

# Contract MPS to state vector
def mps_to_state(tensors):
    state = tensors[0][0, :, :]  # First site shape (1, 4, 4)
    for i in range(1, len(tensors)):
        state = np.einsum('...b,bpd->...pd', state, tensors[i])
        state = state.reshape(-1, tensors[i].shape[-1])
    return state.ravel()

state_vector = mps_to_state(tensors_3d)
state_vector = state_vector / np.linalg.norm(state_vector)  # Normalize

# Initialize quantum circuit
qc = QuantumCircuit(n_qubits)
qc.initialize(state_vector, range(n_qubits))

# Simulate and compute 1PDM
sv = Statevector.from_instruction(qc)
pdm1_quantum = np.zeros((n_qubits, n_qubits))

# Diagonal elements (simplified)
for i in range(n_qubits):
    occupation = np.diag([(x >> i) & 1 for x in range(2**n_qubits)])
    pdm1_quantum[i, i] = np.vdot(sv.data, occupation @ sv.data)

print("\nQuantum 1PDM (diagonal):\n", pdm1_quantum.diagonal())
print("Classical 1PDM (diagonal):\n", pdm1.diagonal())

# Compare energies
h1e = hamil.h1e
energy_quantum = np.sum(pdm1_quantum * h1e) + hamil.ecore
print(f"\nClassical Energy: {energy_classical:.12f}")
print(f"Quantum Energy (from 1PDM): {energy_quantum:.12f}")
print(f"Difference: {energy_classical - energy_quantum:.4e}")
