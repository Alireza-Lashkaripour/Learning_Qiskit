import numpy as np
from scipy.linalg import svd

def random_mps(N, d, D):
    """
    Generate a random MPS with N sites, local dimension d, and max bond dimension D.
    Returns a list of tensors A[i][s, a, b] where:
    - i: site index
    - s: physical index (0 to d-1)
    - a, b: left and right bond indices
    """
    mps = []
    
    # First site
    A = np.random.random((d, 1, min(d, D))) + 1j * np.random.random((d, 1, min(d, D)))
    mps.append(A)
    
    # Middle sites
    for i in range(1, N-1):
        left_dim = min(d**i, D)
        right_dim = min(d**(N-i-1), D)
        A = np.random.random((d, left_dim, right_dim)) + 1j * np.random.random((d, left_dim, right_dim))
        mps.append(A)
    
    # Last site
    if N > 1:
        A = np.random.random((d, min(d**(N-1), D), 1)) + 1j * np.random.random((d, min(d**(N-1), D), 1))
        mps.append(A)
    
    # Normalize
    normalize_mps(mps)
    print(mps)    
    return mps

def normalize_mps(mps):
    """Normalize an MPS using SVD"""
    N = len(mps)
    
    # Left-canonicalize: sweep from left to right
    for i in range(N-1):
        s, a, b = mps[i].shape
        A = mps[i].reshape(s * a, b)
        
        # SVD decomposition
        U, S, V = svd(A, full_matrices=False)
        
        # Update current tensor
        mps[i] = U.reshape(s, a, -1)
        
        # Update next tensor
        B = np.diag(S) @ V
        s_next, a_next, b_next = mps[i+1].shape
        mps[i+1] = np.tensordot(B, mps[i+1], axes=(1, 1))
        mps[i+1] = mps[i+1].transpose(1, 0, 2).reshape(s_next, -1, b_next)
    
    # Normalize the last tensor
    mps[-1] /= np.linalg.norm(mps[-1])
    print(mps)
