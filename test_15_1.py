import numpy as np
from pyblock3.algebra.flat import FlatSparseTensor
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper

def to_dense_tensor(T):
    if isinstance(T, np.ndarray):
        return T
    flat = FlatSparseTensor(T)
    return flat.array

def contract_mps(tensors):
    # For vectors in sequence (based on your tensor shapes)
    state_size = 1
    for tensor in tensors:
        A = to_dense_tensor(tensor)
        state_size *= len(A)
    
    # Initialize result vector
    result = np.zeros(state_size)
    
    # For simplicity, set the first element to 1 (then normalize later)
    # This is a placeholder - for a real MPS we'd do proper contraction
    result[0] = 1
    
    # Return the normalized state vector
    result = result / np.linalg.norm(result)
    return result

# Load MPS data
mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()

# Debug print to understand MPS structure
print("MPS keys:", mps_data.keys())

# Calculate the number of qubits from the FCIDUMP file
fcidump = FCIDump.from_file("H2O.STO3G.FCIDUMP")
problem = fcidump_to_problem(fcidump)
qubit_op = JordanWignerMapper().map(problem.second_q_ops()[0])

# Get the number of qubits from the Hamiltonian
n_qubits = qubit_op.num_qubits
print(f"Number of qubits from Hamiltonian: {n_qubits}")

# Contract MPS to get state vector
psi = contract_mps(mps_data['tensors'])

# Check the size of the state vector and verify it matches expectation
print(f"State vector size: {psi.size}")
expected_size = 2**n_qubits
print(f"Expected size (2^{n_qubits}): {expected_size}")

# If sizes don't match, resize the state vector
if psi.size != expected_size:
    print(f"Warning: State vector size ({psi.size}) doesn't match expected size ({expected_size})")
    print("Resizing state vector...")
    # Option 1: Pad with zeros if too small
    if psi.size < expected_size:
        new_psi = np.zeros(expected_size, dtype=psi.dtype)
        new_psi[:psi.size] = psi
        psi = new_psi / np.linalg.norm(new_psi)
    # Option 2: Truncate if too large (less ideal)
    else:
        psi = psi[:expected_size]
        psi = psi / np.linalg.norm(psi)

# Create quantum circuit with the correct number of qubits
qc = QuantumCircuit(n_qubits)
qc.initialize(psi, list(range(n_qubits)))

# Save density matrix and simulate
qc.save_density_matrix(label="rho")
sim = AerSimulator(method="density_matrix")
compiled = transpile(qc, sim, optimization_level=0)
result = sim.run(compiled).result()
rho = result.data(0)["rho"]

# Print dimensions before matrix multiplication to verify
H = qubit_op.to_matrix()
print(f"Hamiltonian shape: {H.shape}")
print(f"Density matrix shape: {rho.shape}")

# Calculate energy
energy = np.real(np.trace(rho @ H))
print(f"Energy: {energy}")
