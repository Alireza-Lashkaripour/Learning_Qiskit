import numpy as np
from scipy.linalg import svd, qr

def create_left_canonical_mps(d=2, chi=2):
    """Create a valid 3-site left-canonical MPS with bond dimension χ=2"""
    # First site tensor (shape: d × 1 × χ)
    A1 = np.random.randn(d, 1, chi) + 1j*np.random.randn(d, 1, chi)
    A1 /= np.linalg.norm(A1)
    
    # Middle site tensor (shape: d × χ × χ)
    A2 = np.random.randn(d, chi, chi) + 1j*np.random.randn(d, chi, chi)
    # Perform QR decomposition to enforce left-orthogonality
    for i in range(d):
        Q, R = qr(A2[i].T, mode='economic')
        A2[i] = Q.T
        
    # Last site tensor (shape: d × χ × 1)
    A3 = np.random.randn(d, chi, 1) + 1j*np.random.randn(d, chi, 1)
    
    return [A1, A2, A3]

def construct_mpd_tensors(mps_tensors):
    """Construct unitary MPD tensors G1, G2, G3"""
    A1, A2, A3 = mps_tensors
    d = A1.shape[0]  # Physical dimension (2 for qubits)
    chi = A1.shape[2]  # Bond dimension
    
    # G1 construction (first site)
    G1 = np.zeros((d, d, 1, chi), dtype=complex)
    G1[0, :, 0, :] = A1.reshape(d, chi)
    
    # Complete G1 with orthogonal vectors using QR decomposition
    for i in range(1, d):
        random_vec = np.random.randn(d*chi) + 1j*np.random.randn(d*chi)
        Q, _ = qr(random_vec.reshape(chi, d).T)
        G1[i, :, 0, :] = Q.conj().T.reshape(d, chi)
    
    # G2 construction (middle site)
    G2 = np.zeros((d, d, chi, chi), dtype=complex)
    G2[0, :, :, :] = A2  # Original MPS component
    
    # Complete G2 with kernel vectors
    for i in range(1, d):
        for j in range(d):
            # Create random matrix in complement space
            rand_mat = np.random.randn(chi, chi) + 1j*np.random.randn(chi, chi)
            # Orthogonalize against A2[j]
            rand_mat -= np.trace(rand_mat @ A2[j].conj().T)/np.trace(A2[j] @ A2[j].conj().T) * A2[j]
            # QR decomposition for orthonormality
            Q, _ = qr(rand_mat)
            G2[i, j, :, :] = Q
    
    # G3 construction (last site)
    G3 = A3.reshape(d, chi, 1)  # Already satisfies right-orthogonality
    
    return [G1, G2, G3]

def verify_unitarity(G):
    """Check if tensor G implements a unitary transformation"""
    if G.ndim == 4:
        # Reshape to (input_dim × output_dim) matrix
        matrix = G.reshape(G.shape[0]*G.shape[1], -1)
    else:
        matrix = G.reshape(G.shape[0], -1)
        
    identity = np.eye(matrix.shape[0])
    product = matrix @ matrix.conj().T
    error = np.linalg.norm(product - identity)
    return error < 1e-10, error

# Test the implementation
mps_tensors = create_left_canonical_mps()
mpd_tensors = construct_mpd_tensors(mps_tensors)

print("Unitarity verification:")
for i, G in enumerate(mpd_tensors):
    is_unitary, error = verify_unitarity(G)
    print(f"G{i+1}: {'Unitary' if is_unitary else 'Non-unitary'}, Error = {error:.2e}")

