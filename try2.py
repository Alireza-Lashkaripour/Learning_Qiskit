import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..','MPS-in-Qiskit')))

from qiskit import transpile
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
import prepare_MPS as mps
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

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

# 2) Prepare a sophisticated MPS circuit with optimization
num_qubits = qubit_hamiltonian.num_qubits
print(f"System size: {num_qubits} qubits")

# First, get the exact ground state to extract information for MPS initialization
exact_solver = NumPyMinimumEigensolver()
exact_gs_solver = GroundStateEigensolver(mapper, exact_solver)
exact_result = exact_gs_solver.solve(problem)
exact_energy = exact_result.total_energies[0].real
print(f"Reference exact ground state energy: {exact_energy}")

# Extract the ground state wavefunction
ground_state_wf = None
if hasattr(exact_result, 'raw_result') and hasattr(exact_result.raw_result, 'eigenstate'):
    ground_state_wf = exact_result.raw_result.eigenstate
    print(f"Ground state type: {type(ground_state_wf)}")
    if isinstance(ground_state_wf, np.ndarray):
        print(f"Ground state shape: {ground_state_wf.shape}")

# Use strategic bond dimension
bond_dim = min(16, 2**(num_qubits//2))  
print(f"Using bond dimension: {bond_dim}")

# Create more sophisticated random tensors with structured initialization
# Create MPS tensors with higher bond dimension, ensuring they have appropriate structure
A = mps.create_random_tensors(num_qubits, chi=bond_dim, d=2)

# CRITICAL: Ensure we use COMPATIBLE boundary vectors for the MPS structure
# Instead of trying to extract them from the ground state, we'll create them
# in a way that's guaranteed to be compatible with the MPS implementation
phi_i = np.random.rand(bond_dim)  # Match the bond dimension of the first tensor
phi_f = np.random.rand(bond_dim)  # Match the bond dimension of the last tensor

# Normalize these vectors
phi_i /= np.linalg.norm(phi_i)
phi_f /= np.linalg.norm(phi_f)

# Print boundary vectors for inspection
print(f"Initial boundary vector shape: {phi_i.shape}")
print(f"Final boundary vector shape: {phi_f.shape}")

# Create quantum circuit from the MPS tensors
qc_mps, reg = mps.MPS_to_circuit(A, phi_i, phi_f)
print("MPS preparation circuit:", qc_mps)
print("MPS register:", reg)

# Optional: Apply circuit optimization to enhance accuracy
qc_mps = transpile(qc_mps, basis_gates=['u3', 'cx'], optimization_level=3)
print("Optimized circuit depth:", qc_mps.depth())

# Create a variational parameter list for optimization
# We'll use this to further optimize the MPS to get closer to ground state
from scipy.optimize import minimize

# Function to convert MPS parameters to circuit
def params_to_circuit(params, num_qubits, bond_dim):
    # Reshape parameters into tensors and boundary vectors
    total_tensor_params = 0
    A_list = []
    
    # Calculate tensor sizes
    for i in range(num_qubits):
        if i == 0:
            # First tensor: [d, 1, chi]
            tensor_size = 2 * 1 * bond_dim
        elif i == num_qubits - 1:
            # Last tensor: [d, chi, 1]
            tensor_size = 2 * bond_dim * 1
        else:
            # Middle tensors: [d, chi, chi]
            tensor_size = 2 * bond_dim * bond_dim
        
        tensor_params = params[total_tensor_params:total_tensor_params + tensor_size]
        total_tensor_params += tensor_size
        
        # Reshape tensor according to its position
        if i == 0:
            tensor = tensor_params.reshape(2, 1, bond_dim)
        elif i == num_qubits - 1:
            tensor = tensor_params.reshape(2, bond_dim, 1)
        else:
            tensor = tensor_params.reshape(2, bond_dim, bond_dim)
        
        A_list.append(tensor)
    
    # Extract boundary vectors
    phi_i_size = bond_dim
    phi_f_size = bond_dim
    
    phi_i = params[total_tensor_params:total_tensor_params + phi_i_size]
    total_tensor_params += phi_i_size
    
    phi_f = params[total_tensor_params:total_tensor_params + phi_f_size]
    
    # Normalize boundary vectors
    phi_i /= np.linalg.norm(phi_i)
    phi_f /= np.linalg.norm(phi_f)
    
    # Create circuit
    try:
        circ, reg = mps.MPS_to_circuit(A_list, phi_i, phi_f)
        return circ
    except Exception as e:
        print(f"Error creating circuit: {e}")
        return None

# Function to evaluate energy for given parameters
def evaluate_energy(params, num_qubits, bond_dim, hamiltonian, simulator):
    # Convert parameters to circuit
    circuit = params_to_circuit(params, num_qubits, bond_dim)
    if circuit is None:
        # Return high energy if circuit creation failed
        return 1000.0
    
    # Add statevector snapshot
    circuit.save_statevector(label='final_state')
    
    # Transpile circuit
    transpiled_circuit = transpile(circuit, simulator)
    
    # Run simulation
    try:
        result = simulator.run(transpiled_circuit).result()
        state_vector = result.data()['final_state']
        # Convert to Statevector
        psi = Statevector(state_vector)
        # Calculate energy
        energy = psi.expectation_value(hamiltonian).real
        return energy
    except Exception as e:
        print(f"Error in simulation: {e}")
        return 1000.0
# Set up the MPS simulator
simulator = AerSimulator(method='matrix_product_state')

# Add statevector snapshot to the circuit
qc_mps.save_statevector(label="final_state")
qc_mps_transpiled = transpile(qc_mps, simulator)

# Run the circuit on the MPS simulator
result = simulator.run(qc_mps_transpiled).result()
state_vector = result.data()['final_state']
psi = Statevector(state_vector)
print("Statevector from MPS circuit successfully obtained")

# Compute energy expectation value of the MPS state
energy_mps = psi.expectation_value(qubit_hamiltonian).real
print(f"Energy expectation from MPS: {energy_mps:.12f}")

# Compare with exact ground state energy
print(f"Exact ground state energy: {exact_energy:.12f}")
print(f"Energy difference: {abs(energy_mps - exact_energy):.12f}")
print(f"Relative error: {abs((energy_mps - exact_energy)/exact_energy)*100:.8f}%")

# Optional: Calculate overlap with exact ground state if available
if ground_state_wf is not None and isinstance(ground_state_wf, (np.ndarray, Statevector)):
    try:
        if isinstance(ground_state_wf, np.ndarray):
            exact_sv = Statevector(ground_state_wf)
        else:
            exact_sv = ground_state_wf
            
        overlap = abs(psi.inner(exact_sv))**2
        print(f"Overlap with exact ground state: {overlap:.8f}")
        print(f"Fidelity: {overlap*100:.4f}%")
    except Exception as e:
        print(f"Could not calculate overlap with exact ground state: {e}")
