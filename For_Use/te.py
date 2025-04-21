from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper
import numpy as np
import prepare_MPS as mps
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

# 1. Load the FCIDUMP and build the problem
fcidump = FCIDump.from_file("H2O.STO3G.FCIDUMP")
problem = fcidump_to_problem(fcidump)
print("Problem:", problem)

# 2. Build fermionic and qubit Hamiltonians
ham_op = problem.hamiltonian.second_q_op()
print("Fermionic Hamiltonian:", ham_op)
mapper = JordanWignerMapper()
qubit_hamiltonian = mapper.map(ham_op)
print("Qubit Hamiltonian:", qubit_hamiltonian)

# 3. Create a random MPS and map it to a circuit
num_qubits = qubit_hamiltonian.num_qubits
bond_dim   = 4
A          = mps.create_random_tensors(num_qubits, bond_dim, 2)
phi_i      = np.ones(bond_dim)
phi_f      = np.ones(bond_dim)
qc, reg    = mps.MPS_to_circuit(A, phi_i, phi_f)
print("MPS preparation circuit:\n", qc)

# 4. Simulate that circuit with the MPS backend and print the statevector
sim = AerSimulator(method="matrix_product_state")
qc_t = transpile(qc, sim)
res  = sim.run(qc_t).result()
psi_mps = Statevector(res.get_statevector(qc_t))
print("Statevector from MPS circuit:\n", psi_mps)

# 5. Compute ⟨ψ|H|ψ⟩ for the MPS state
energy_mps = psi_mps.expectation_value(qubit_hamiltonian).real
print("Energy from MPS circuit:", energy_mps)

# 6. Exact ground state via NumPyMinimumEigensolver
solver = NumPyMinimumEigensolver()
ground_solver = GroundStateEigensolver(mapper, solver)
result = ground_solver.solve(problem)
print("Exact ground state energy:", result.total_energies[0].real)

