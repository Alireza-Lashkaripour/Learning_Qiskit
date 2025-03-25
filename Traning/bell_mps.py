import numpy as np

# Define the Bell state in computational basis
# |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])

# Reshape into a 2x2 matrix (for singular value decomposition)
bell_matrix = bell_state.reshape(2, 2)

# Perform SVD to get the MPS representation
U, S, Vh = np.linalg.svd(bell_matrix)

# Extract the A and B matrices
# A matrices correspond to the first qubit
A0 = U[:, 0] * np.sqrt(S[0])  # A[0]
A1 = U[:, 1] * np.sqrt(S[1]) if len(S) > 1 else np.zeros(2)  # A[1]

# B matrices correspond to the second qubit
B0 = Vh[0, :]  # B[0]
B1 = Vh[1, :] if Vh.shape[0] > 1 else np.zeros(2)  # B[1]

print("A0:", A0)
print("A1:", A1)
print("B0:", B0)
print("B1:", B1)

# Verify the MPS representation
reconstructed_state = np.zeros(4)
reconstructed_state[0] = np.dot(A0, B0)  # |00⟩ coefficient
reconstructed_state[1] = np.dot(A0, B1)  # |01⟩ coefficient
reconstructed_state[2] = np.dot(A1, B0)  # |10⟩ coefficient
reconstructed_state[3] = np.dot(A1, B1)  # |11⟩ coefficient

print("Original state:", bell_state)
print("Reconstructed state:", reconstructed_state)
