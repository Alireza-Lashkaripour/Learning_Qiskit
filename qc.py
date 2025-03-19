import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_circuit_layout
import matplotlib.pyplot as plt
from copy import deepcopy

# Define number of sites
n_sites = 5

# Define rotation parameters (theta values for each site)
# Creating random angles for demonstration
theta = np.random.uniform(0, 2*np.pi, n_sites)

# Create quantum registers for physical and bond qubits
physical_qr = QuantumRegister(n_sites, 'p')
bond_qr = QuantumRegister(n_sites - 1, 'b')  # For bond dimension = 2
circuit = QuantumCircuit(physical_qr, bond_qr)

# Often start with a product state and build entanglement
# You might apply Hadamard gates to create superpositions
for i in range(n_sites):
    circuit.h(physical_qr[i])

# Connect each physical qubit to its bond qubit, then to the next physical qubit
for i in range(n_sites - 1):
    # Connect physical qubit i to bond qubit i
    circuit.cx(physical_qr[i], bond_qr[i])
    
    # Connect bond qubit i to next physical qubit
    circuit.cx(bond_qr[i], physical_qr[i+1])

# Add rotations or other operations to each site
for i in range(n_sites):
    circuit.ry(theta[i], physical_qr[i])  # Parameterized rotation

# Add measurement capabilities
cr = ClassicalRegister(n_sites, 'c')
circuit.add_register(cr)
for i in range(n_sites):
    circuit.measure(physical_qr[i], cr[i])

# Draw the circuit
circuit.draw('mpl', fold=120, scale=0.7)
plt.title('5-Site Matrix Product State (MPS) Quantum Circuit')
plt.tight_layout()
plt.show()

# Print circuit information
print(f"Circuit for {n_sites}-site MPS:")
print(f"Total qubits: {circuit.num_qubits}")
print(f"Total classical bits: {circuit.num_clbits}")
print(f"Circuit depth: {circuit.depth()}")
print(f"Total number of operations: {circuit.size()}")
