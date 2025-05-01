from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

file_path = "H2O.STO3G.FCIDUMP"  
fcidump = FCIDump.from_file(file_path)

problem = fcidump_to_problem(fcidump)
print(problem)

hamiltonian_op = problem.hamiltonian.second_q_op()
print("Fermionic Hamiltonian:", hamiltonian_op)

integrals = problem.hamiltonian.electronic_integrals
one_body_alpha = integrals.alpha["+-"]
print("One-electron integrals (alpha):")
print(one_body_alpha)

mapper = JordanWignerMapper()
qubit_hamiltonian = mapper.map(hamiltonian_op)
print("Qubit Hamiltonian:", qubit_hamiltonian)

# Use NumPyMinimumEigensolver as a substitute for DMRG
solver = NumPyMinimumEigensolver()
ground_state_solver = GroundStateEigensolver(mapper, solver)
result = ground_state_solver.solve(problem)

# Print the ground state energy
print("Ground state energy from exact solver:", result.total_energies[0].real)
