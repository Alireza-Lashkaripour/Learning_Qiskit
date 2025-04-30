import numpy as np 

from qiskit import QuantumCircuit, transpile 
from qiskit_aer import AerSimulator 

circ = QuantumCircuit(2, 2)
circ.h(0)
circ.cx(0, 1)
circ.measure([0, 1], [0, 1])
print(circ) 

simulator = AerSimulator(method='matrix_product_state')

# Compiles the circuit for the simulator, so it converts the circ into the traget format
tcirc = transpile(circ, simulator)
# Runs the transpiled circ on the simulator 
result = simulator.run(tcirc).result()
# Gets the result 
counts = result.get_counts(0)
print(counts) 



# We can also save the internal state vector and also the full internal MPS. 

# Define a snapshot that shows the current state vector
circ.save_statevector(label='my_sv')
circ.save_matrix_product_state(label='my_mps')
circ.measure([0,1], [0,1])

# Execute and get saved data
tcirc = transpile(circ, simulator)
result = simulator.run(tcirc).result()
data = result.data(0)

print(data)

