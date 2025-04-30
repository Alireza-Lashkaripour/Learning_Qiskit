import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, SparsePauliOp
from qiskit.circuit.library import StatePreparation

try:
    from qiskit_nature.second_q.formats.fcidump import FCIDump
    from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem 
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    qiskit_nature_available = True
except ImportError:
    print("Required Qiskit Nature components not found. Please ensure qiskit-nature is installed.")
    qiskit_nature_available = False

def load_mps_data_from_npy(filename="h2o_mps_complete.npy"):
    try:
        data = np.load(filename, allow_pickle=True).item()
        print(f"Loaded MPS data for {data.get('n_sites', 'N/A')} sites.")
        print(f"Reference Energy: {data.get('energy', 'N/A')}")
        if 'dense_tensors' in data and data.get('n_sites'):
             if len(data['dense_tensors']) != data['n_sites']:
                 raise ValueError("Number of tensors doesn't match n_sites.")
             phys_dims = [t.shape[1] for t in data['dense_tensors']]
             if not all(d == 4 for d in phys_dims):
                 print(f"Warning: Expected physical dimension 4, but found dimensions {phys_dims}. Code assumes physical dim=4.")
        return data
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        print("Please ensure the MPS data file is in the correct path.")
        return None
    except Exception as e:
        print(f"Error loading or validating MPS data: {e}")
        return None

def contract_mps(tensors):
    if not tensors:
        return None
    
    n_sites = len(tensors)
    state = tensors[0] 
    
    if state.shape[0] == 1:
        state = state.reshape(state.shape[1:]) 

    for i in range(1, n_sites):
        next_tensor = tensors[i] 
        
        state_shape = state.shape
        next_shape = next_tensor.shape
        
        state_reshaped = state.reshape(-1, state_shape[-1])
        next_tensor_reshaped = next_tensor.reshape(next_shape[0], -1)
        
        state = np.dot(state_reshaped, next_tensor_reshaped)
        
        new_shape = list(state_shape[:-1]) + list(next_shape[1:])
        state = state.reshape(new_shape)
        
    if state.shape[-1] == 1:
       state = state.reshape(state.shape[:-1]) 

    # Reshape final tensor (4, 4, ..., 4) into (2, 2, 2, ..., 2) for 2N qubits
    num_qubits = n_sites * 2
    qubit_shape = [2] * num_qubits
    try:
        # The order of physical indices in 'state' is (site0, site1, ... siteN-1)
        # Reshape each site's dim 4 -> (2, 2)
        # Example for N=3: shape (4, 4, 4) -> (2, 2, 2, 2, 2, 2)
        reshaped_state = state.reshape(qubit_shape)
    except ValueError as e:
         print(f"Error reshaping final state tensor: {e}")
         print(f"Expected total size {4**n_sites}, final tensor shape {state.shape}")
         return None

    # Flatten using Fortran order ('F') to match Qiskit's convention (q0 changes fastest)
    state_vector = reshaped_state.flatten(order='F') 
    
    norm = np.linalg.norm(state_vector)
    if norm > 1e-9:
        state_vector = state_vector / norm
    else:
        print("Warning: State vector norm is close to zero.")
        
    return state_vector

def load_hamiltonian_from_fcidump(fcidump_path): 
    if not qiskit_nature_available:
        print("Cannot load Hamiltonian: Qiskit Nature is not installed.")
        return None, 0.0
    try:
        fcidump = FCIDump.from_file(fcidump_path)
        print(f"Loaded FCIDump from {fcidump_path}")
        problem = fcidump_to_problem(fcidump)
        print("Converted FCIDump to ElectronicStructureProblem")

        nuclear_repulsion_energy = problem.nuclear_repulsion_energy if problem.nuclear_repulsion_energy is not None else 0.0
        if hasattr(problem, 'hamiltonian') and hasattr(problem.hamiltonian, 'constants') and 'nuclear_repulsion_energy' in problem.hamiltonian.constants:
             nuclear_repulsion_energy = problem.hamiltonian.constants['nuclear_repulsion_energy']
        print(f"Nuclear Repulsion Energy (offset): {nuclear_repulsion_energy}")

        ham_op = problem.hamiltonian.second_q_op()
        
        mapper = JordanWignerMapper()
        qubit_op = mapper.map(ham_op)

        if isinstance(qubit_op, list):
             if len(qubit_op) == 1:
                 qubit_op = qubit_op[0]
             else:
                  raise TypeError(f"Mapper returned multiple operators: {len(qubit_op)}")
        if not isinstance(qubit_op, SparsePauliOp):
             raise TypeError(f"Mapped operator is not a SparsePauliOp: {type(qubit_op)}")

        print(f"Hamiltonian mapped to {qubit_op.num_qubits} qubits using JordanWignerMapper.")
        return qubit_op, nuclear_repulsion_energy

    except FileNotFoundError:
        print(f"Error: FCIDump file not found at {fcidump_path}")
        return None, 0.0
    except Exception as e:
        print(f"Error loading or mapping Hamiltonian: {e}")
        return None, 0.0

def main():
    mps_data_file = "h2o_mps_complete.npy"  
    fcidump_file = "H2O.STO3G.FCIDUMP"  
    num_spatial_orbitals = 7 
    
    mps_data = load_mps_data_from_npy(mps_data_file)
    if mps_data is None:
        return

    n_sites = mps_data['n_sites']
    ref_energy = mps_data['energy']
    dense_tensors = mps_data['dense_tensors']

    if n_sites != num_spatial_orbitals:
         print(f"Warning: MPS n_sites ({n_sites}) != expected num_spatial_orbitals ({num_spatial_orbitals}).")

    num_qubits = n_sites * 2
    print(f"Determined number of qubits: {num_qubits} ({n_sites} sites * 2 qubits/site)")

    print("Contracting MPS tensors...")
    target_state_vector = contract_mps(dense_tensors)
    if target_state_vector is None:
        print("Failed to contract MPS.")
        return
        
    expected_len = 4**n_sites
    expected_len_qubit = 2**num_qubits
    if len(target_state_vector) != expected_len_qubit:
        print(f"Error: Contracted state vector length ({len(target_state_vector)}) != expected 2^{num_qubits}={expected_len_qubit}.")
        return
        
    print(f"Contracted MPS to state vector of length {len(target_state_vector)}. Norm: {np.linalg.norm(target_state_vector):.4f}")

    print(f"Building {num_qubits}-qubit circuit using StatePreparation...")
    qc = QuantumCircuit(num_qubits, name="MPS StatePrep")
    try:
        prep = StatePreparation(target_state_vector, label="mps_prep")
        qc.append(prep, range(num_qubits))
        print("StatePreparation appended successfully.")
        
    except Exception as e:
        print(f"Error during StatePreparation: {e}")
        return

    qc_transpiled = qc 

    print("\nLoading Hamiltonian...")
    hamiltonian_op, nre_offset = load_hamiltonian_from_fcidump(fcidump_file) 
    if hamiltonian_op is None:
        print("Failed to load Hamiltonian.")
        return

    if hamiltonian_op.num_qubits != qc_transpiled.num_qubits:
        print(f"Error: Hamiltonian qubits ({hamiltonian_op.num_qubits}) != Circuit qubits ({qc_transpiled.num_qubits}).")
        return

    print("\nVerifying circuit state...")
    try:
        circuit_sv = Statevector.from_instruction(qc_transpiled)
        circuit_state_vector = circuit_sv.data

        if len(target_state_vector) != len(circuit_state_vector):
             print(f"Error: Target state length ({len(target_state_vector)}) != Circuit state length ({len(circuit_state_vector)}).")
             fidelity = 0.0
        else:
            target_state_vector /= np.linalg.norm(target_state_vector)
            circuit_state_vector /= np.linalg.norm(circuit_state_vector)
            fidelity = np.abs(np.vdot(target_state_vector, circuit_state_vector))**2 
            print(f"State Fidelity: {fidelity:.8f}")

        print("Calculating energy expectation value...")
        calculated_electronic_energy = np.real(circuit_sv.expectation_value(hamiltonian_op))
        total_calculated_energy = calculated_electronic_energy + nre_offset
        
        print(f"\n--- Energy Comparison ---")
        print(f"Reference MPS Energy        : {ref_energy:.12f}")
        print(f"Calculated Electronic Energy: {calculated_electronic_energy:.12f}")
        print(f"Nuclear Repulsion Energy    : {nre_offset:.12f}")
        print(f"Total Calculated QC Energy  : {total_calculated_energy:.12f}")
        print(f"Energy Difference           : {np.abs(ref_energy - total_calculated_energy):.12e}")

        print("\nCalculating density matrix...")
        density_matrix = DensityMatrix(circuit_sv)
        print("Density Matrix (first 4x4 elements):")
        print(density_matrix.data[:4, :4])

    except Exception as e:
        print(f"Error during verification/calculation: {e}")

if __name__ == "__main__":
    main()
