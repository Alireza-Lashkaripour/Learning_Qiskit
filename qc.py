from qiskit import QuantumCircuit
import numpy as np 

qc = QuantumCircuit(2)
# For Hadamard Gate:
qc.h(0) # on the qubit 0

# Control qubit 0, traget 1 (apply X gate)
qc.cx(0, 1)

# Rotation Gates 
# X rotation
#qc.rx(np.pi/2, 0)

# Y rotation
#qc.ry(np.pi/4, 1)

# Z rotation
#qc.rz(np.pi/3, 0)

# Now how to measure: 
#qc.measure([0,1], [0,1])

qc.draw("mpl")
print(qc) 
