import os
import sys
import numpy as np

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_algorithms import NumPyMinimumEigensolver

from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
# Corrected Imports: Split QubitConverter import
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.mappers.second_quantization import QubitConverter # Moved import
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock

# --- Configuration ---
file_path = "H2O.STO3G.FCIDUMP" # Ensure this FCIDUMP file exists
use_mps_simulator = False # Set True for 'matrix_product_state', False for 'statevector'

# --- 1. Load Molecular Problem and Map Hamiltonian ---
try:
    fcidump = FCIDump.from_file(file_path)
except FileNotFoundError:
    print(f"Error: FCIDUMP file not found at '{file_path}'")
    sys.exit(1)

problem = fcidump_to_problem(fcidump)
num_spatial_orbitals = problem.num_spatial_orbitals
num_particles = problem.num_particles

mapper = JordanWignerMapper()
# Pass only the mapper instance to QubitConverter
qubit_converter = QubitConverter(mapper=mapper, two_qubit_reduction=False)

qubit_hamiltonian = qubit_converter.convert(
    problem.hamiltonian.second_q_op(),
    num_particles=num_particles
)
num_qubits = qubit_hamiltonian.num_qubits

print(f"Problem: H2O STO-3G")
print(f"Number of spatial orbitals: {num_spatial_orbitals}")
print(f"Number of particles: {num_particles}")
print(f"Number of qubits: {num_qubits}\n")


# --- 2. Prepare Hartree-Fock Initial State Circuit ---
initial_state_hf = HartreeFock(
    num_spatial_orbitals=num_spatial_orbitals,
    num_particles=num_particles,
    qubit_converter=qubit_converter # Pass the converter instance
)
hf_circuit = initial_state_hf.construct_circuit(name="HF_State")


# --- 3. Simulate Hartree-Fock Circuit ---
if use_mps_simulator:
    simulator = AerSimulator(method='matrix_product_state')
    print("Using MPS Simulator")
else:
    simulator = AerSimulator(method='statevector')
    print("Using Statevector Simulator")

hf_circuit.save_statevector(label="final_hf_state")
transpiled_circuit = transpile(hf_circuit, simulator)

simulation_result = simulator.run(transpiled_circuit).result()
statevector_data = simulation_result.data()['final_hf_state']
psi_hf = Statevector(statevector_data)


# --- 4. Compute Energy Expectation Value for Hartree-Fock State ---
if not isinstance(qubit_hamiltonian, SparsePauliOp):
     try:
         # Attempt conversion if not already SparsePauliOp
         qubit_hamiltonian_op = SparsePauliOp(qubit_hamiltonian.paulis, qubit_hamiltonian.coeffs)
     except AttributeError:
         # Handle cases where conversion isn't straightforward (depends on actual type)
         print("Could not convert qubit Hamiltonian to SparsePauliOp for expectation value.")
         # Fallback or specific handling might be needed depending on the actual type
         # For now, exiting if conversion fails.
         sys.exit(1)
     except Exception as e:
        print(f"An unexpected error occurred during Hamiltonian conversion: {e}")
        sys.exit(1)
else:
    qubit_hamiltonian_op = qubit_hamiltonian

energy_hf_state = psi_hf.expectation_value(qubit_hamiltonian_op).real
print(f"\nEnergy from simulated Hartree-Fock state: {energy_hf_state:.8f}")


# --- 5. Exact Ground State Calculation (FCI) for Comparison ---
exact_solver = NumPyMinimumEigensolver()
# Pass the mapper instance directly to GroundStateEigensolver
gs_solver = GroundStateEigensolver(mapper, exact_solver)
exact_result = gs_solver.solve(problem)
ground_state_energy_exact = exact_result.total_energies[0].real

print(f"Exact Ground State Energy (FCI):          {ground_state_energy_exact:.8f}")


# --- 6. Comparison ---
energy_difference = abs(energy_hf_state - ground_state_energy_exact)
print(f"\nEnergy Difference (HF State - FCI):      {energy_difference:.8f}")

print("\nNote: The Hartree-Fock energy neglects electron correlation and is")
print("typically higher than the true ground state (FCI) energy.")

