import numpy as np
from scipy.linalg import qr
from qutip import tensor, basis

def construct_mps_gates(A1, A2, A3):
    """Construct unitary gates from 3-site MPS tensors"""
    # Site 3 (single-qubit gate)
    G3 = A3.reshape(2,1)
    
    # Site 2 (two-qubit gate)
    A2_reshaped = A2.reshape(2,4)
    Q2, R2 = qr(A2_reshaped.T, mode='economic')
    G2 = Q2.conj().T.reshape(2,2,2,2)
    
    # Site 1 (two-qubit gate)
    A1_reshaped = A1.reshape(1,4)
    Q1, R1 = qr(A1_reshaped.T, mode='economic')
    G1 = Q1.conj().T.reshape(2,2,2,2)
    
    return G1, G2, G3

def validate_unitarity(G):
    """Verify gate unitarity"""
    dim = int(np.sqrt(G.size))
    U = G.reshape(dim, dim)
    return np.allclose(U @ U.T.conj(), np.eye(dim))

# Example usage with random left-orthogonal MPS
# Site 1 tensor (1x2)
A1 = np.random.randn(1,2)
A1 /= np.linalg.norm(A1)

# Site 2 tensor (2x2x2)
A2 = np.random.randn(2,2,2)
A2 = A2.reshape(2,4)
Q2, R2 = qr(A2.T, mode='economic')
A2 = Q2.T.conj().reshape(2,2,2)

# Site 3 tensor (2x1)
A3 = np.random.randn(2,1)
A3 /= np.linalg.norm(A3)

# Construct gates
G1, G2, G3 = construct_mps_gates(A1, A2, A3)

# Verify unitarity
print(f"G1 unitary: {validate_unitarity(G1)}")
print(f"G2 unitary: {validate_unitarity(G2)}") 
print(f"G3 unitary: {validate_unitarity(G3.reshape(2,1))}")

# Build quantum circuit
def mps_circuit(state, G1, G2, G3):
    """Apply MPS encoding circuit to |0> state"""
    # Apply first layer
    state = tensor(G1, G2).data.dot(state)
    # Apply CNOT gates
    state = tensor([qeye(2), sigmax()]).data.dot(state)
    # Apply second layer
    state = tensor(G3, qeye(2)).data.dot(state)
    return state

# Initialize |0> state
psi0 = tensor(basis(2,0), basis(2,0), basis(2,0))
psi = mps_circuit(psi0, G1, G2, G3)

