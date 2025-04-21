from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

# 1. Load the FCIDump file
file_path = "H2O.STO3G.FCIDUMP"
fcidump    = FCIDump.from_file(file_path)
problem    = fcidump_to_problem(fcidump)
print("ElectronicStructureProblem:", problem)

# 2. Extract the Fermionic Hamiltonian
ham_op = problem.hamiltonian.second_q_op()
print("Fermionic Hamiltonian:", ham_op)

# 3. One-electron integrals (alpha spin)
integrals      = problem.hamiltonian.electronic_integrals
one_body_alpha = integrals.alpha["+-"]
print("One-electron integrals (alpha):", one_body_alpha)

# 4. Map to qubits via Jordan–Wigner
mapper          = JordanWignerMapper()
qubit_hamiltonian = mapper.map(ham_op)
print("Qubit Hamiltonian:", qubit_hamiltonian)

# 5. Solve with a classical eigensolver
solver               = NumPyMinimumEigensolver()
ground_state_solver  = GroundStateEigensolver(mapper, solver)
result               = ground_state_solver.solve(problem)
print("Ground state energy from exact solver:", result.total_energies[0].real)

