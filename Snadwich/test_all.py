import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import scipy.linalg

# Import pyblock3 dependencies (assuming the previous code runs in the same environment)
from pyblock3.algebra.mpe import MPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ


def extract_mps_tensors(mps):
    """
    Extract tensors from pyblock3 MPS format and convert them to a format usable by PennyLane.
    
    Args:
        mps: The pyblock3 MPS object
        
    Returns:
        list: List of numpy arrays representing the MPS tensors
    """
    # Get the tensors from the MPS object
    tensors = []
    
    # Extract tensor data from each site in the MPS
    for i in range(len(mps)):
        # Get the tensor at site i
        tensor = mps[i].data
        # Convert to numpy array and append to list
        tensors.append(np.array(tensor))
        
    return tensors


def canonicalize_tensors(tensors):
    """
    Bring the MPS tensors into a canonical form.
    
    Args:
        tensors (list): List of MPS tensors
        
    Returns:
        list: Canonicalized MPS tensors
    """
    n_sites = len(tensors)
    # Left-canonicalize
    for i in range(n_sites - 1):
        tensor = tensors[i]
        shape = tensor.shape
        # Reshape for SVD
        tensor = tensor.reshape(shape[0] * shape[1], -1)
        # SVD
        u, s, vh = scipy.linalg.svd(tensor, full_matrices=False)
        # Update current tensor
        tensors[i] = u.reshape(shape[0], shape[1], -1)
        # Update next tensor
        tensors[i+1] = np.tensordot(np.diag(s) @ vh, tensors[i+1], axes=([1], [0]))
        
    return tensors


def mps_to_state_vector(tensors):
    """
    Convert MPS tensors to a full state vector.
    
    Args:
        tensors (list): List of MPS tensors
        
    Returns:
        array: Full state vector
    """
    # Initialize with the first tensor
    result = tensors[0]
    
    # Contract the remaining tensors
    for i in range(1, len(tensors)):
        # Contract result with the next tensor
        result = np.tensordot(result, tensors[i], axes=(-1, 0))
    
    # Reshape to a vector
    return result.reshape(-1)


def create_quantum_circuit_from_mps(tensors, dev=None):
    """
    Create a quantum circuit that prepares a state corresponding to the MPS.
    
    Args:
        tensors (list): List of MPS tensors
        dev (pennylane.Device): Optional PennyLane device
        
    Returns:
        function: PennyLane QNode that prepares the MPS state
    """
    n_qubits = len(tensors)
    
    if dev is None:
        dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit():
        # Convert MPS to state vector
        state_vector = mps_to_state_vector(tensors)
        
        # Normalize the state vector
        state_vector = state_vector / np.sqrt(np.sum(np.abs(state_vector)**2))
        
        # Prepare the state using StatePrep
        qml.StatePrep(state_vector, wires=range(n_qubits))
        
        # Return the state
        return qml.state()
    
    return circuit


def calculate_1rdm_quantum(circuit, n_sites):
    """
    Calculate the one-electron reduced density matrix (1-RDM) from a quantum circuit.
    
    Args:
        circuit: PennyLane QNode representing the quantum state
        n_sites: Number of sites/orbitals
        
    Returns:
        array: One-electron reduced density matrix
    """
    # Initialize the 1-RDM
    rdm1 = np.zeros((n_sites, n_sites), dtype=complex)
    
    # Get the state vector from the circuit
    state = circuit()
    
    # Create a new device and QNode for expectation values
    dev = qml.device("default.qubit", wires=n_sites)
    
    # Define a function to calculate expectation value of a^†_i a_j
    @qml.qnode(dev)
    def fermion_expectation(i, j, state_vector):
        # Prepare the state
        qml.StatePrep(state_vector, wires=range(n_sites))
        
        # For one-electron RDM in second quantization: <ψ|a^†_i a_j|ψ>
        # We need to use the Jordan-Wigner transformation to map fermionic operators to qubit operators
        
        # Create the operator a^†_i a_j using Jordan-Wigner transformation
        # This is a simplified version - in practice, you'd use PennyLane's built-in functions
        if i == j:
            # For diagonal elements, we measure number operator (1-Z_i)/2
            return qml.expval(qml.PauliZ(i)) * (-0.5) + 0.5
        else:
            # For off-diagonal elements, we need Jordan-Wigner strings
            # This is a simplified approach
            return 0.25 * (
                qml.expval(qml.PauliX(i) @ qml.PauliX(j)) + 
                qml.expval(qml.PauliY(i) @ qml.PauliY(j))
            ) + 0.25j * (
                qml.expval(qml.PauliX(i) @ qml.PauliY(j)) - 
                qml.expval(qml.PauliY(i) @ qml.PauliX(j))
            )
    
    # Calculate each element of the 1-RDM
    for i in range(n_sites):
        for j in range(n_sites):
            # Factor 2 to account for spin (assuming spin-restricted calculation)
            rdm1[i, j] = 2 * fermion_expectation(i, j, state)
    
    return rdm1


def map_pyblock3_mps_to_quantum_circuit(mps_filename, fcidump_filename):
    """
    Main function to map pyblock3 MPS to a quantum circuit and calculate 1-RDM.
    
    Args:
        mps_filename: Filename to save/load the MPS data
        fcidump_filename: FCIDUMP file for the Hamiltonian
        
    Returns:
        tuple: (quantum_circuit, classical_1rdm, quantum_1rdm)
    """
    # Load the Hamiltonian and build MPS as in your code
    hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fcidump_filename), flat=True)
    
    # Load pre-computed MPS if available, otherwise compute it
    try:
        # Assume mps_data contains the saved MPS object
        mps_data = np.load(mps_filename, allow_pickle=True)
        mps = mps_data.item()
        print("Loaded pre-computed MPS from file")
    except (FileNotFoundError, ValueError):
        print("Computing MPS from scratch...")
        # Build and optimize MPS as in your code
        bond_dim = 200
        mps = hamil.build_mps(bond_dim)
        mps = mps.canonicalize(center=0)
        mps /= mps.norm()
        
        # Run DMRG optimization
        mpo = hamil.build_qc_mpo()
        mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
        dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0],
                                       dav_thrds=[1E-3], iprint=2, n_sweeps=10)
        
        # Save MPS for future use
        np.save(mps_filename, mps)
    
    # Calculate classical 1-RDM
    pdm1_classical = np.zeros((hamil.n_sites, hamil.n_sites))
    for i in range(hamil.n_sites):
        diop = OpElement(OpNames.D, (i, 0), q_label=SZ(-1, -1, hamil.orb_sym[i]))
        di = hamil.build_site_mpo(diop)
        for j in range(hamil.n_sites):
            djop = OpElement(OpNames.D, (j, 0), q_label=SZ(-1, -1, hamil.orb_sym[j]))
            dj = hamil.build_site_mpo(djop)
            # factor 2 due to alpha + beta spins
            pdm1_classical[i, j] = 2 * np.dot((di @ mps).conj(), dj @ mps)
    
    print("Classical 1-RDM calculated:")
    print(pdm1_classical)
    
    # Extract MPS tensors and convert to a format usable by PennyLane
    tensors = extract_mps_tensors(mps)
    
    # Ensure tensors are in canonical form
    tensors = canonicalize_tensors(tensors)
    
    # Create quantum circuit from MPS
    circuit = create_quantum_circuit_from_mps(tensors)
    
    # Calculate 1-RDM from quantum circuit
    pdm1_quantum = calculate_1rdm_quantum(circuit, hamil.n_sites)
    
    print("Quantum 1-RDM calculated:")
    print(pdm1_quantum)
    
    # Compare the two 1-RDMs
    error = np.linalg.norm(pdm1_classical - pdm1_quantum)
    print(f"Frobenius norm of the difference between classical and quantum 1-RDMs: {error:.6e}")
    
    return circuit, pdm1_classical, pdm1_quantum


# Advanced implementation: Matrix product operator (MPO) to Quantum Circuit
def decompose_mps_tensor_to_gates(tensor, qubit_idx, next_qubit_idx, dev):
    """
    Decompose an MPS tensor into quantum gates using SVD.
    
    Args:
        tensor: MPS tensor to decompose
        qubit_idx: Index of the current qubit
        next_qubit_idx: Index of the next qubit
        dev: PennyLane device
        
    Returns:
        function: QNode with the gate sequence
    """
    # This is a more advanced implementation that would decompose
    # each MPS tensor into a sequence of gates
    
    @qml.qnode(dev)
    def gate_sequence():
        # Reshape tensor for SVD
        shape = tensor.shape
        reshaped = tensor.reshape(shape[0] * shape[1], -1)
        
        # SVD decomposition
        u, s, vh = scipy.linalg.svd(reshaped, full_matrices=False)
        
        # Map U to single-qubit gates on current qubit
        # This is a simplified approach - in practice would require more careful decomposition
        theta = 2 * np.arccos(np.clip(np.abs(u[0, 0]), 0, 1))
        phi = np.angle(u[0, 1]) - np.angle(u[0, 0])
        qml.RY(theta, wires=qubit_idx)
        qml.RZ(phi, wires=qubit_idx)
        
        # Map singular values and V† to entangling gates between current and next qubit
        # For simplicity, using controlled rotations
        for j in range(min(3, len(s))):  # Limit to top 3 singular values for simplicity
            angle = np.arcsin(np.clip(s[j] / np.sum(s), 0, 1))
            qml.CRY(angle, wires=[qubit_idx, next_qubit_idx])
        
        # Return state for testing
        return qml.state()
    
    return gate_sequence


# Example usage
if __name__ == "__main__":
    # File paths
    fd = 'H2O.STO3G.FCIDUMP'
    mps_file = "h2o_mps.npy"
    
    # Map MPS to quantum circuit and calculate 1-RDMs
    circuit, pdm1_classical, pdm1_quantum = map_pyblock3_mps_to_quantum_circuit(mps_file, fd)
    
    # Save results
    np.save("h2o_quantum_1rdm.npy", pdm1_quantum)
    np.save("h2o_classical_1rdm.npy", pdm1_classical)
    
    print("Quantum circuit successfully created from MPS, and 1-RDMs calculated!")
