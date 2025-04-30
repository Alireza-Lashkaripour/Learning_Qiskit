import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Helper function: complete a unitary from given row(s)

# ------------------------------------------------------------------------------
# Helper: complete a unitary from given row(s)
def complete_unitary_from_rows(rows, dim):
    """
    Given a list of row vectors (each of length 'dim') that are assumed to be 
    (approximately) linearly independent, complete them to a full matrix of shape
    (dim, dim) by adding additional rows using Gram–Schmidt.
    """
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

# ------------------------------------------------------------------------------
# Helper: enforce unitarity using SVD.
def enforce_unitarity(U):
    """Given an (approximately) unitary matrix U, reunitarize it using SVD."""
    u, s, vh = np.linalg.svd(U)
    return u @ vh

# ------------------------------------------------------------------------------
# Step 1: Define the three–site MPS tensors.
# Dimensions: physical d=2, bond dimension chi=2.
#
# A[1]: shape (2,2)
A1 = np.array([[0.8,  0.6],
               [0.6, -0.8]], dtype=complex)

# A[2]: shape (2,2,2)
A2 = np.zeros((2,2,2), dtype=complex)
A2[0] = np.array([[0.9,  0.1],
                  [0.2,  0.8]], dtype=complex)
A2[1] = np.array([[0.3,  0.7],
                  [0.7, -0.3]], dtype=complex)

# A[3]: shape (2,2)
A3 = np.array([[0.85,  0.53],
               [0.53, -0.85]], dtype=complex)

print("MPS Tensor A[1]:")
print(A1)
print("\nMPS Tensor A[2] (for s=0 and s=1):")
print("A2[0] =\n", A2[0])
print("A2[1] =\n", A2[1])
print("\nMPS Tensor A[3]:")
print(A3)

# ------------------------------------------------------------------------------
# Step 2: Build the local gates from the MPS tensors.
#
# For a two–qubit gate we use a 4x4 unitary.
#
# For G[1]: embed A1 (flattened) into the row corresponding to |00⟩.
G1_row_00 = A1.flatten()  # 4 numbers from A1
G1 = complete_unitary_from_rows([G1_row_00], 4)
G1 = enforce_unitarity(G1)  # enforce unitarity
print("\nLocal Gate G[1] (4x4 unitary from A[1]):")
print(G1)

# For G[2]: embed A2 by assigning:
#   - row corresponding to |00⟩ ← flatten(A2[0])  (for s2 = 0)
#   - row corresponding to |01⟩ ← flatten(A2[1])  (for s2 = 1)
G2_row_00 = A2[0].flatten()  # for s2 = 0
G2_row_01 = A2[1].flatten()  # for s2 = 1
G2 = complete_unitary_from_rows([G2_row_00, G2_row_01], 4)
G2 = enforce_unitarity(G2)  # enforce unitarity
print("\nLocal Gate G[2] (4x4 unitary from A[2]):")
print(G2)

# For G[3]: since A3 is 2x2, we "unitarize" it.
u, s, vh = np.linalg.svd(A3)
G3 = u @ vh  # a 2x2 unitary
G3 = enforce_unitarity(G3)
print("\nLocal Gate G[3] (2x2 unitary from A[3]):")
print(G3)

# ------------------------------------------------------------------------------
# Step 3: Embed the local gates into the full 3–qubit Hilbert space.
#
# We assume the following:
# - G[1] acts on qubits 0 and 1 (indices [0,1]).
# - G[2] acts on qubits 1 and 2 (indices [1,2]).
# - G[3] acts on qubit 2.
#
# In our Qiskit circuit, we will add the dagger (conjugate transpose) of each gate.
U1_dag = G1.conj().T  # 4x4 unitary on qubits 0,1
U2_dag = G2.conj().T  # 4x4 unitary on qubits 1,2
U3_dag = G3.conj().T  # 2x2 unitary on qubit 2

print("\nDagger of G[1] (U1†):")
print(U1_dag)
print("\nDagger of G[2] (U2†):")
print(U2_dag)
print("\nDagger of G[3] (U3†):")
print(U3_dag)

# ------------------------------------------------------------------------------
# Step 4: Build the overall quantum circuit using Qiskit.
#
# The state preparation is:
#   |ψ⟩ = U† |000⟩,  with U† = U1† followed by U2† followed by U3†.
#
# In Qiskit, we append:
#   - U1† as a custom 2-qubit gate on qubits [0,1].
#   - U2† as a custom 2-qubit gate on qubits [1,2].
#   - U3† as a custom 1-qubit gate on qubit [2].
qc = QuantumCircuit(3, name="MPS Circuit")

# Append U1† on qubits [0,1]
gate_U1 = UnitaryGate(U1_dag, label="U1†")
qc.append(gate_U1, [0, 1])

# Append U2† on qubits [1,2]
gate_U2 = UnitaryGate(U2_dag, label="U2†")
qc.append(gate_U2, [1, 2])

# Append U3† on qubit [2]
gate_U3 = UnitaryGate(U3_dag, label="U3†")
qc.append(gate_U3, [2])

print("\nFinal Quantum Circuit:")
print(qc.draw(output='text'))

# Optionally, draw the circuit using matplotlib.
qc.draw(output='mpl')
plt.show()
