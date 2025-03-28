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



import matplotlib.pyplot as plt

# Step 2: Function to convert MPS tensors to a quantum circuit
def mps_to_circuit(tensors, n_sites):
    n_physical_qubits = n_sites
    max_bond_dim = max(bond_dims)
    n_ancilla = int(np.ceil(np.log2(max_bond_dim)))
    total_qubits = n_physical_qubits + n_ancilla

    print(f"Using {n_physical_qubits} physical qubits and {n_ancilla} ancilla qubits")

    # Initialize quantum circuit
    qc = QuantumCircuit(total_qubits)

    # Prepare initial ancilla state (|0...0>)
    ancilla_state = np.zeros(2**n_ancilla, dtype=complex)
    ancilla_state[0] = 1.0
    qc.initialize(ancilla_state, range(n_ancilla))

    # Process each tensor
    for i, tensor in enumerate(tensors):
        # Handle pyblock3 tensor objects
        if hasattr(tensor, 'data'):
            tensor_data = tensor.data  # Extract numerical data
        else:
            tensor_data = tensor  # Assume it's already a NumPy array

        tensor_data = np.array(tensor_data, dtype=complex)
        print(f"Site {i}: Tensor shape = {tensor_data.shape}")

        # Determine tensor rank and reshape accordingly
        rank = len(tensor_data.shape)
        if rank == 3:  # Middle sites: (phys_dim, left_bond, right_bond)
            phys_dim, left_bond_dim, right_bond_dim = tensor_data.shape
            matrix = tensor_data.reshape(phys_dim * left_bond_dim, right_bond_dim)
        elif rank == 2:  # Edge sites: (phys_dim, bond_dim)
            if i == 0:  # Left edge
                phys_dim, right_bond_dim = tensor_data.shape
                matrix = tensor_data.reshape(phys_dim, right_bond_dim)
                left_bond_dim = 1
            else:  # Right edge
                left_bond_dim, phys_dim = tensor_data.shape
                matrix = tensor_data.reshape(left_bond_dim * phys_dim, 1)
                right_bond_dim = 1
        else:
            raise ValueError(f"Unexpected tensor rank {rank} at site {i}")

        # Pad to square matrix for unitary
        max_dim = max(phys_dim * left_bond_dim, right_bond_dim)
        unitary = np.zeros((max_dim, max_dim), dtype=complex)
        unitary[:phys_dim * left_bond_dim, :right_bond_dim] = matrix

        # QR decomposition for unitary gate
        q, r = np.linalg.qr(unitary, mode='complete')
        gate = Operator(q)

        # Qubit mapping
        if i == 0:
            target_qubits = [n_ancilla + i] + list(range(n_ancilla))
        elif i == n_sites - 1:
            target_qubits = list(range(n_ancilla)) + [n_ancilla + i]
        else:
            target_qubits = [n_ancilla + i] + list(range(n_ancilla))

        # Apply gate (simplify to fit qubit count)
        gate_qubits = min(len(target_qubits), int(np.log2(gate.shape[0])))
        qc.unitary(gate, target_qubits[:gate_qubits], label=f'U_{i}')

    return qc

# Step 3: Generate the circuit
qc = mps_to_circuit(tensors, n_sites)
print("\nQuantum Circuit:")
print(qc)

# Step 4: Simulate the circuit
simulator = AerSimulator(method='statevector')
job = simulator.run(qc)
result = job.result()
statevector = result.get_statevector()

# Truncate to physical qubits
phys_statevector = statevector.data[:2**n_sites]
print("\nResulting statevector (physical qubits only):")
print(phys_statevector)
print(f"Norm of physical statevector: {np.linalg.norm(phys_statevector)}")

# Optional: Visualize
qc.draw(output='mpl')
plt.show()
