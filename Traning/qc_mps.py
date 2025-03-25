from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
import numpy as np

# Create a circuit with 2 qubits
qc_mps = QuantumCircuit(2)

# Option 1: Use the StatePreparation class to prepare the Bell state
bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
qc_mps.append(StatePreparation(bell_state), [0, 1])

# Draw the circuit
print("MPS Circuit using StatePreparation:")
print(qc_mps.draw())

# Option 2: More explicit construction showing MPS structure
qc_explicit = QuantumCircuit(2)

# For this simple case, the standard Bell state circuit is equivalent
qc_explicit.h(0)
qc_explicit.cx(0, 1)

print("Equivalent explicit circuit:")
print(qc_explicit.draw())
