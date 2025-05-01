from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, SparsePauliOp

circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0,1)
sv = Statevector.from_instruction(circ)
dm = DensityMatrix(sv)
print(circ.draw('text'))
print(sv)
print("Density Matrix:\n", dm.data)
#print("Entropy subsystem 0:", entropy(dm, [0]))
#correlated
print("ZZ expectation:", sv.expectation_value(SparsePauliOp.from_list([("ZZ", 1)])))
print("XX expectation:", sv.expectation_value(SparsePauliOp.from_list([("XX", 1)])))

bell_states = {}
circ1 = QuantumCircuit(2); circ1.h(0); circ1.cx(0,1); bell_states["phi+"] = circ1
circ2 = QuantumCircuit(2); circ2.h(0); circ2.cx(0,1); circ2.z(0); bell_states["phi-"] = circ2
circ3 = QuantumCircuit(2); circ3.h(0); circ3.cx(0,1); circ3.x(1); bell_states["psi+"] = circ3
circ4 = QuantumCircuit(2); circ4.h(0); circ4.cx(0,1); circ4.z(0); circ4.x(1); bell_states["psi-"] = circ4

for name, c in bell_states.items():
    sv = Statevector.from_instruction(c)
    print(name, "circuit:\n", c.draw('text'), sep='')
    print(name, "statevector:", sv, sep='\n')
