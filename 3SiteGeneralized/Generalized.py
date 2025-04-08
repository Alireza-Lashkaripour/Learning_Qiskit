import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
import matplotlib.pyplot as plt

def complete_unitary_from_rows(rows, dim):
    given = np.array(rows, dtype=complex)
    U = list(given)
    for i in range(dim):
        e = np.zeros(dim, dtype=complex)
        e[i] = 1
        for v in U:
            e = e - np.vdot(v, e) * v
        norm_e = np.linalg.norm(e)
        if norm_e > 1e-12:
            e = e / norm_e
            U.append(e)
        if len(U) == dim:
            break
    return np.array(U)

def enforce_unitarity(U):
    u, s, vh = np.linalg.svd(U)
    return u @ vh

def next_power_of_2(n):
    return 1 if n == 0 else 2**(n-1).bit_length()

def embed_in_power_of_2(matrix):
    """Embed a matrix into a matrix with dimensions of power of 2."""
    n = matrix.shape[0]
    target_size = next_power_of_2(n)
    
    if n == target_size:
        return matrix
    
    result = np.zeros((target_size, target_size), dtype=complex)
    result[:n, :n] = matrix
    
    # Make sure the remaining part is identity-like to ensure unitarity
    for i in range(n, target_size):
        result[i, i] = 1.0
        
    return result

def generate_random_mps_tensors(n_sites, d=2, chi=2):
    tensors = []
    
    # First tensor: shape (d, chi)
    A1 = np.random.rand(d, chi) + 1j * np.random.rand(d, chi)
    A1 = A1 / np.linalg.norm(A1)
    tensors.append(A1)
    
    # Middle tensors: shape (d, chi, chi)
    for _ in range(n_sites-2):
        A = np.random.rand(d, chi, chi) + 1j * np.random.rand(d, chi, chi)
        A = A / np.linalg.norm(A.reshape(-1))
        tensors.append(A)
    
    # Last tensor: shape (d, chi)
    An = np.random.rand(d, chi) + 1j * np.random.rand(d, chi)
    An = An / np.linalg.norm(An)
    tensors.append(An)
    
    return tensors

def build_mps_circuit(tensors, d=2, chi=2):
    n_sites = len(tensors)
    
    # Build local gates from MPS tensors
    gates = []
    
    # First gate: embedding the first tensor
    G1_row_00 = tensors[0].flatten()
    G1_dim = d * chi  # Dimension based on tensor size
    G1 = complete_unitary_from_rows([G1_row_00], G1_dim)
    G1 = enforce_unitarity(G1)
    
    # Embed into 2^n x 2^n matrix for Qiskit
    G1_embedded = embed_in_power_of_2(G1)
    gates.append(G1_embedded)
    
    # Middle gates: 2-qubit gates from tensors A[1] to A[n-2]
    for i in range(1, n_sites-1):
        rows = []
        for s in range(d):
            rows.append(tensors[i][s].flatten())
        G_dim = chi * chi  # Dimension for middle gates
        G = complete_unitary_from_rows(rows, G_dim)
        G = enforce_unitarity(G)
        
        # Embed into 2^n x 2^n matrix for Qiskit
        G_embedded = embed_in_power_of_2(G)
        gates.append(G_embedded)
    
    # Last gate: 1-qubit gate from tensor A[n-1]
    # For the last tensor (shape d, chi), we need a d×d unitary
    last_tensor = tensors[-1]
    if chi <= d:
        # Pad with zeros to make it d×d
        padded = np.zeros((d, d), dtype=complex)
        padded[:, :chi] = last_tensor
        Gn = padded
    else:
        # Take only first d columns to make it d×d
        Gn = last_tensor[:, :d]
    
    # Ensure it's unitary
    Gn = enforce_unitarity(Gn)
    
    # For last gate, we assume d is already a power of 2 (like d=2 for qubits)
    # But if not, we embed it
    if d & (d-1) != 0:  # Check if d is not a power of 2
        Gn = embed_in_power_of_2(Gn)
        
    gates.append(Gn)
    
    # Calculate qubit counts for each gate
    qubit_counts = []
    for g in gates:
        n_qubits = int(np.log2(g.shape[0]))
        qubit_counts.append(n_qubits)
    
    # Create quantum circuit with max qubits needed
    max_qubits = max(n_sites, max(qubit_counts))
    qc = QuantumCircuit(max_qubits)
    
    # Add gates to circuit (using dagger of each gate)
    # First gate
    n_qubits_1 = qubit_counts[0]
    gate_U1 = UnitaryGate(gates[0].conj().T, label=f"U1†")
    qc.append(gate_U1, list(range(n_qubits_1)))
    
    # Middle gates 
    for i in range(1, n_sites-1):
        n_qubits_i = qubit_counts[i]
        start_idx = max(0, i - (n_qubits_i // 2))
        gate = UnitaryGate(gates[i].conj().T, label=f"U{i+1}†")
        qc.append(gate, list(range(start_idx, start_idx + n_qubits_i)))
    
    # Last gate
    n_qubits_n = qubit_counts[-1]
    gate_Un = UnitaryGate(gates[-1].conj().T, label=f"U{n_sites}†")
    qc.append(gate_Un, list(range(n_sites - n_qubits_n, n_sites)))
    
    return qc, gates

def create_mps_circuit(n_sites, d=2, chi=2, custom_tensors=None):
    if custom_tensors:
        tensors = custom_tensors
    else:
        tensors = generate_random_mps_tensors(n_sites, d, chi)
    
    circuit, gates = build_mps_circuit(tensors, d, chi)
    return circuit, tensors, gates

if __name__ == "__main__":
    # Example usage
    n_sites = 12  # Number of sites
    d = 2        # Physical dimension
    chi = 4      # Bond dimension
    
    # Generate random MPS tensors and build circuit
    circuit, tensors, gates = create_mps_circuit(n_sites, d, chi)
    
    # Draw circuit
    print(circuit.draw(output='text'))
    circuit.draw(output='mpl')
    plt.show()
