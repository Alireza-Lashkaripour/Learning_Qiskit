import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..','..','MPS-in-Qiskit')))

from qiskit import transpile
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
import prepare_MPS as mps
import numpy as np
from qiskit.quantum_info import Statevector

# 1) Load and map the Hamiltonian
file_path = "H2O.STO3G.FCIDUMP"
fcidump = FCIDump.from_file(file_path)
problem = fcidump_to_problem(fcidump)
print(problem)

ham_op = problem.hamiltonian.second_q_op()
print("Fermionic Hamiltonian:", ham_op)

mapper = JordanWignerMapper()
qubit_hamiltonian = mapper.map(ham_op)
print("Qubit Hamiltonian:", qubit_hamiltonian)

# 2) Prepare a random MPS circuit
num_qubits = qubit_hamiltonian.num_qubits
A = mps.create_random_tensors(num_qubits, chi=2, d=2)
phi_i = np.random.rand(2); phi_f = np.random.rand(2)
phi_i /= np.linalg.norm(phi_i); phi_f /= np.linalg.norm(phi_f)

qc_mps, reg = mps.MPS_to_circuit(A, phi_i, phi_f)
print("MPS preparation circuit:", qc_mps)
print("MPS register:", reg)

# 3) Build the statevector directly from the circuit
psi = Statevector.from_instruction(qc_mps)
print("Statevector from MPS circuit:", psi.data)

# 4) Compute its expectation value of the Hamiltonian
energy_mps = psi.expectation_value(qubit_hamiltonian).real
print("Energy expectation from MPS:", energy_mps)

# 5) Exact ground-state solve for comparison
solver = NumPyMinimumEigensolver()
gs_solver = GroundStateEigensolver(mapper, solver)
result    = gs_solver.solve(problem)
print("Ground state energy from exact solver:", result.total_energies[0].real)

