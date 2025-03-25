from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import numpy as np
from qiskit.quantum_info import Statevector

# Create a circuit with 2 qubits
qc = QuantumCircuit(2)

# Apply Hadamard gate to the first qubit
qc.h(0)

# Apply CNOT with qubit 0 as control and qubit 1 as target
qc.cx(0, 1)

# Draw the circuit
print(qc.draw())

# Simulate the circuit
simulator = Aer.get_backend('statevector_simulator')
result = execute(qc, simulator).result()
statevector = result.get_statevector()

# Print the resulting state vector
print("State vector:")
print(statevector)

# Visualize the state
plot_bloch_multivector(statevector)
