import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

def create_simple_mps(N=3, d=2, D=2):
    """Create a simple MPS for N qubits in the GHZ state"""
    mps = []
    
    # First site: |0⟩ → |00⟩, |1⟩ → |11⟩
    A1 = np.zeros((d, 1, d))
    for i in range(d):
        A1[i, 0, i] = 1.0
    mps.append(A1)
    
    # Middle sites: pass through the bond index
    for i in range(1, N-1):
        Ai = np.zeros((d, d, d))
        for j in range(d):
            Ai[j, j, j] = 1.0
        mps.append(Ai)
    
    # Last site: close the chain
    AN = np.zeros((d, d, 1))
    for i in range(d):
        AN[i, i, 0] = 1/np.sqrt(d)  # Normalization
    mps.append(AN)
    
    return mps

def mps_to_state_vector(mps):
    """Convert MPS to full state vector"""
    N = len(mps)
    d = mps[0].shape[0]
    
    # Initialize state vector as the first tensor
    state = mps[0].reshape(-1, mps[0].shape[2])
    
    # Contract with remaining tensors
    for i in range(1, N):
        # Reshape state for contraction
        state = np.tensordot(state, mps[i], axes=(1, 1))
        # Reshape to (current_states, next_bond)
        state = state.transpose(0, 2, 1).reshape(-1, mps[i].shape[2])
    
    # Final reshape to vector
    return state.flatten()

def mps_to_quantum_circuit_simplified(mps):
    """
    Create a quantum circuit that approximately prepares the state described by the MPS.
    This is a simplified implementation focusing on the GHZ state.
    """
    N = len(mps)
    qr = QuantumRegister(N, 'q')
    cr = ClassicalRegister(N, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # For GHZ state: H on first qubit, then CNOTs
    qc.h(qr[0])
    for i in range(N-1):
        qc.cx(qr[i], qr[i+1])
    
    # Add measurement operations
    qc.measure(qr, cr)
    
    return qc

def compare_representations(N=3):
    """Compare classical MPS and quantum circuit representations"""
    # Create MPS for GHZ state
    mps = create_simple_mps(N)
    
    # Convert to state vector
    state_vector = mps_to_state_vector(mps)
    print("State vector from MPS:")
    print(state_vector)
    
    # Create equivalent quantum circuit
    qc = mps_to_quantum_circuit_simplified(mps)
    
    # Simulate quantum circuit
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(qc.remove_final_measurements(inplace=False), simulator).result()
    state_vector_qc = result.get_statevector()
    
    print("\nState vector from quantum circuit:")
    print(state_vector_qc.data)
    
    # Visual comparison
    print("\nQuantum Circuit:")
    print(qc)
    
    # Compare probabilities
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=8192).result()
    counts = result.get_counts()
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plot_histogram(counts)
    plt.title('Measurement Results from Quantum Circuit')
    plt.tight_layout()
    
    return mps, qc, state_vector, state_vector_qc

# Run the comparison
mps, qc, state_vector_mps, state_vector_qc = compare_representations(N=3)

# Display specific tensor from MPS
print("\nFirst tensor in MPS:")
print(mps[0])
