import numpy as np
from pyblock3.algebra.flat import FlatSparseTensor
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
# from qiskit.quantum_info import DensityMatrix # DensityMatrix class not strictly needed for simulation result
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper
import sys # For sys.exit()

def to_dense_tensor(T):
    """Converts a potential PyBlock3 sparse tensor to a dense NumPy array."""
    if isinstance(T, np.ndarray):
        return T
    # Assuming T is a PyBlock3 tensor object that can be converted via FlatSparseTensor
    try:
        flat = FlatSparseTensor(T)
        return flat.array
    except Exception as e:
        print(f"Error converting tensor to dense: {e}")
        print("Make sure the tensor object is compatible with FlatSparseTensor or already a NumPy array.")
        sys.exit(1)

def mps_to_statevector(mps_tensors_maybe_sparse):
    """
    Contracts MPS tensors (OBC) to reconstruct the full statevector.
    Handles dense conversion internally.

    Args:
        mps_tensors_maybe_sparse: A list of tensors (can be PyBlock3 sparse or numpy)
                                  representing the MPS tensors [A^{(1)}, ..., A^{(N)}].
                                  Expected Shapes (after dense conversion):
                                  A^{(1)}: (d, chi_1) or (1, d, chi_1)
                                  A^{(k)}: (chi_{k-1}, d, chi_k) for 1 < k < N
                                  A^{(N)}: (chi_{N-1}, d) or (chi_{N-1}, d, 1)

    Returns:
        numpy.ndarray: The full statevector reshaped to a 1D array (complex type).
    """
    if not mps_tensors_maybe_sparse:
        return np.array([1.0], dtype=complex)

    # Convert all tensors to dense numpy arrays first
    mps_tensors = [to_dense_tensor(T) for T in mps_tensors_maybe_sparse]

    num_sites = len(mps_tensors)
    if num_sites == 0:
         return np.array([1.0], dtype=complex)

    # --- Start Contracting ---
    current_tensor = mps_tensors[0].astype(complex) # Start with complex type

    # Handle first tensor shape: (1, d, chi_1) -> (d, chi_1) or just (d, chi_1)
    if current_tensor.ndim == 3 and current_tensor.shape[0] == 1:
        current_tensor = current_tensor.reshape(current_tensor.shape[1:])
    elif current_tensor.ndim != 2:
        raise ValueError(f"Unexpected shape for first MPS tensor: {mps_tensors[0].shape}. Expected (d, chi_1) or (1, d, chi_1).")

    physical_dim = current_tensor.shape[0] # d

    for i in range(1, num_sites):
        next_tensor = mps_tensors[i].astype(complex)
        bond_dim_prev = current_tensor.shape[-1]

        # --- Prepare next_tensor ---
        # Intermediate tensor: (chi_{k-1}, d, chi_k)
        if i < num_sites - 1:
             if next_tensor.ndim != 3:
                 raise ValueError(f"Unexpected shape for intermediate MPS tensor {i+1}: {next_tensor.shape}. Expected (chi_{i-1}, d, chi_i).")
             if next_tensor.shape[0] != bond_dim_prev:
                  raise ValueError(f"Bond dimension mismatch between tensor {i} ({bond_dim_prev}) and tensor {i+1} ({next_tensor.shape[0]})")
             if next_tensor.shape[1] != physical_dim:
                 raise ValueError(f"Physical dimension mismatch at tensor {i+1}. Expected {physical_dim}, got {next_tensor.shape[1]}.")

        # Last tensor: (chi_{N-1}, d, 1) -> (chi_{N-1}, d) or just (chi_{N-1}, d)
        else: # i == num_sites - 1
            if next_tensor.ndim == 3 and next_tensor.shape[2] == 1:
                next_tensor = next_tensor.reshape(next_tensor.shape[:2])
            elif next_tensor.ndim != 2:
                 raise ValueError(f"Unexpected shape for last MPS tensor: {next_tensor.shape}. Expected (chi_{N-1}, d) or (chi_{N-1}, d, 1).")

            if next_tensor.shape[0] != bond_dim_prev:
                 raise ValueError(f"Bond dimension mismatch between tensor {i} ({bond_dim_prev}) and last tensor ({next_tensor.shape[0]})")
            if next_tensor.shape[1] != physical_dim:
                 raise ValueError(f"Physical dimension mismatch at last tensor. Expected {physical_dim}, got {next_tensor.shape[1]}.")
        # --- End Prepare next_tensor ---

        # Contract the last axis of current_tensor with the first axis of next_tensor
        # tensordot axes=([-1],[0]) means sum over last index of first tensor and first index of second tensor
        # Shape Evolution Example (N=3, d=2):
        # i=1: current(d, c1) @ next(c1, d, c2) -> result(d, d, c2)
        # i=2: current(d, d, c2) @ next(c2, d) -> result(d, d, d)
        current_tensor = np.tensordot(current_tensor, next_tensor, axes=([-1], [0]))

    # Final shape: (d, d, ..., d)
    # Flatten to statevector: C_{00..0}, C_{00..1}, ..., C_{11..1}
    statevector = current_tensor.flatten()

    # Normalize
    norm = np.linalg.norm(statevector)
    if not np.isclose(norm, 1.0):
        print(f"Warning: Statevector norm is {norm}, renormalizing.")
        if norm == 0:
            print("Error: Statevector norm is zero.")
            # Handle appropriately, maybe return zero vector or raise error
            return np.zeros_like(statevector)
        statevector = statevector / norm
    else:
        # Ensure normalization even if close, avoids potential minor downstream issues
         statevector = statevector / norm


    return statevector

# --- Main Script ---

# Load MPS data
try:
    mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
except FileNotFoundError:
    print("Error: h2o_mps_complete.npy not found. Make sure the file is in the correct directory.")
    sys.exit(1)


# Debug print to understand MPS structure
print("MPS keys:", mps_data.keys())
print(f"Number of sites from MPS data: {mps_data.get('n_sites', 'N/A')}") # PyBlock3 might store n_sites

# Calculate the number of qubits from the FCIDUMP file
try:
    fcidump = FCIDump.from_file("H2O.STO3G.FCIDUMP")
except FileNotFoundError:
     print("Error: H2O.STO3G.FCIDUMP not found. Make sure the file is in the correct directory.")
     sys.exit(1)

problem = fcidump_to_problem(fcidump)
qubit_op = JordanWignerMapper().map(problem.second_q_ops()[0])

# Get the number of qubits from the Hamiltonian
n_qubits = qubit_op.num_qubits
print(f"Number of qubits from Hamiltonian: {n_qubits}")

# --- Crucial Check ---
num_tensors_in_mps = len(mps_data['tensors'])
if num_tensors_in_mps != n_qubits:
    print(f"Error: Mismatch between number of MPS tensors ({num_tensors_in_mps}) and number of qubits from Hamiltonian ({n_qubits}).")
    print("Ensure the MPS file corresponds to the FCIDUMP file.")
    sys.exit(1)
# --------------------

print("Contracting MPS to statevector...")
try:
    # Use the corrected function
    psi = mps_to_statevector(mps_data['tensors'])
except ValueError as e:
    print(f"Error during MPS contraction: {e}")
    sys.exit(1)
except MemoryError as e:
     print(f"MemoryError during MPS contraction: {e}")
     print(f"Even with correct logic, contracting to a {n_qubits}-qubit statevector requires significant memory for the intermediate tensors.")
     print("Consider methods that avoid full statevector construction for larger systems.")
     sys.exit(1)


# Check the size of the state vector and verify it matches expectation
print(f"State vector size: {psi.size}")
expected_size = 2**n_qubits
print(f"Expected size (2^{n_qubits}): {expected_size}")
assert psi.size == expected_size, "Statevector size after contraction does not match expected size!"

# Create quantum circuit with the correct number of qubits
print("Initializing Qiskit circuit...")
qc = QuantumCircuit(n_qubits)
try:
    qc.initialize(psi, list(range(n_qubits)))
except Exception as e:
    print(f"Error during Qiskit circuit initialization: {e}")
    sys.exit(1)


# --- Simulate Density Matrix ---
print("Simulating density matrix...")
# !! Warning: Simulating the full density matrix can be memory-intensive for > ~10 qubits !!
# !! The matrix size is 2^N x 2^N !!
qc.save_density_matrix(label="rho")
sim = AerSimulator(method="density_matrix")
try:
    # Optimization level 0 is fine here as initialize dominates
    compiled = transpile(qc, sim, optimization_level=0)
    result = sim.run(compiled).result()
    rho = result.data(0)["rho"] # This rho is a DensityMatrix object from Qiskit >= 0.46
                                # For older Qiskit Aer, it might be a numpy array directly.
                                # Let's ensure it's a NumPy array for the trace calculation.
    if not isinstance(rho, np.ndarray):
         rho_matrix = rho.data # Extract numpy array from DensityMatrix object
    else:
         rho_matrix = rho

except Exception as e:
    print(f"Error during density matrix simulation: {e}")
    print("This step can fail due to memory limits.")
    sys.exit(1)


# --- Calculate Energy ---
print("Calculating energy...")
# !! Warning: Creating the full Hamiltonian matrix is also memory-intensive !!
# !! Matrix size is 2^N x 2^N !!
try:
    H = qubit_op.to_matrix()
except MemoryError as e:
    print(f"MemoryError when creating dense Hamiltonian matrix: {e}")
    print(f"The {n_qubits}-qubit Hamiltonian matrix is too large ({2**n_qubits}x{2**n_qubits}) to fit in memory.")
    print("Consider calculating expectation values using sparse methods or directly with MPS/MPO techniques.")
    sys.exit(1)
except Exception as e:
     print(f"Error creating dense Hamiltonian matrix: {e}")
     sys.exit(1)


print(f"Hamiltonian shape: {H.shape}")
print(f"Density matrix shape: {rho_matrix.shape}") # Use the extracted numpy array

# Calculate energy: E = Tr(rho * H)
try:
    energy = np.real(np.trace(rho_matrix @ H))
    print(f"\nCalculated Energy via Quantum Circuit: {energy}")
except MemoryError as e:
     print(f"MemoryError during trace calculation (rho @ H): {e}")
     print("The intermediate matrix product might be too large.")
     sys.exit(1)


# Compare with energy possibly stored in MPS data (if available)
if 'energy' in mps_data:
    print(f"Energy stored in loaded MPS data:   {mps_data['energy']}")

print("\nScript finished.")
