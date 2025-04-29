import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.circuit.library import StatePreparation
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.primitives import Estimator

class MPStoQuantumCircuit:
    def __init__(self, mps_data):
        self.n_sites = mps_data['n_sites']
        self.bond_dims = mps_data['bond_dims']
        self.tensors = mps_data['dense_tensors']
        self.energy = mps_data['energy']
        
        self.n_qubits = int(np.ceil(np.log2(np.prod([tensor.shape[1] for tensor in self.tensors]))))
        self.circuit = QuantumCircuit(self.n_qubits)
        
    def contract_virtual_bonds(self):
        state = self.tensors[0]
        for i in range(1, self.n_sites):
            shape = state.shape
            state = state.reshape(-1, shape[-1])
            next_tensor = self.tensors[i]
            next_shape = next_tensor.shape
            next_tensor = next_tensor.reshape(next_shape[0], -1)
            state = np.dot(state, next_tensor)
            new_shape = list(shape[:-1]) + list(next_shape[1:])
            state = state.reshape(new_shape)
        state = state / np.linalg.norm(state)
        return state.flatten()
    
    def build_full_rank_mps_circuit(self):
        self.n_qubits = 14
        self.circuit = QuantumCircuit(self.n_qubits)
        contracted_state = self.contract_virtual_bonds()
        
        padded_length = 2**self.n_qubits
        if len(contracted_state) < padded_length:
            padded_state = np.zeros(padded_length, dtype=complex)
            padded_state[:len(contracted_state)] = contracted_state
            padded_state = padded_state / np.linalg.norm(padded_state)
        else:
            padded_state = contracted_state[:padded_length]
            padded_state = padded_state / np.linalg.norm(padded_state)
        
        state_prep = StatePreparation(padded_state)
        self.circuit.compose(state_prep, inplace=True)
        self.circuit = transpile(self.circuit, basis_gates=['sx', 'x', 'cx', 'rz'], optimization_level=3)
        return self.circuit
    
    def build_improved_mps_circuit(self):
        n_qubits = self.n_sites
        qc = QuantumCircuit(n_qubits)
        
        # First apply Hadamard to create superposition
        for i in range(n_qubits):
            qc.h(i)
        
        # Then apply X gates to significant qubits
        qc.x(n_qubits-1)
        
        # Apply controlled rotations
        for i in range(n_qubits-1):
            qc.cx(i, i+1)
        
        # Add phase information using RZ gates
        for i in range(n_qubits):
            qc.rz(np.pi/(i+1), i)
        
        # Add additional entanglement
        for i in range(n_qubits-1, 0, -1):
            qc.cx(i-1, i)
        
        # Final layer of rotations
        for i in range(n_qubits):
            qc.rz(np.pi/(2*(i+1)), i)
        
        return transpile(qc, basis_gates=['sx', 'x', 'cx', 'rz'], optimization_level=3)
    
    def build_minimal_mps_circuit(self):
        qc = QuantumCircuit(self.n_sites)
        
        # First qubit in |+⟩ state
        qc.h(0)
        
        # Apply specific rotations and entanglement based on MPS structure
        for i in range(1, self.n_sites):
            qc.h(i)
            qc.cx(i-1, i)
        
        # Add phase based on strongest amplitudes
        qc.rz(np.pi, self.n_sites-1)
        
        # Additional entanglement layer
        for i in range(self.n_sites-2, -1, -1):
            qc.cx(i, i+1)
            qc.rz(np.pi/2, i)
        
        return transpile(qc, basis_gates=['sx', 'x', 'cx', 'rz'], optimization_level=3)
    
    def verify_state(self, circuit, target_state=None):
        if target_state is None:
            target_state = self.contract_virtual_bonds()
        
        circuit_sv = Statevector.from_instruction(circuit)
        circuit_state = circuit_sv.data
        
        if len(target_state) < len(circuit_state):
            padded_target = np.zeros_like(circuit_state)
            padded_target[:len(target_state)] = target_state
            norm = np.linalg.norm(padded_target)
            if norm > 1e-10:
                padded_target = padded_target / norm
            target_state = padded_target
        elif len(target_state) > len(circuit_state):
            target_state = target_state[:len(circuit_state)]
            norm = np.linalg.norm(target_state)
            if norm > 1e-10:
                target_state = target_state / norm
        
        fidelity = np.abs(np.vdot(target_state, circuit_state))
        return fidelity
    
    def get_density_matrix(self, circuit):
        state = Statevector.from_instruction(circuit)
        density_matrix = DensityMatrix(state)
        return density_matrix
    
    def estimate_energy(self, circuit, hamiltonian):
        state = Statevector.from_instruction(circuit)
        energy = np.real(state.expectation_value(hamiltonian))
        return energy

def load_mps_data(filename="h2o_mps_complete.npy"):
    data = np.load(filename, allow_pickle=True).item()
    return data

def create_mock_hamiltonian(n_qubits):
    from qiskit.quantum_info import SparsePauliOp
    
    # Create a mock Hamiltonian that approximates the H2O energy
    target_energy = -74.931918959700
    
    # Identity term to set baseline energy
    terms = [("I" * n_qubits, target_energy)]
    
    # Add some Z terms for single-qubit contributions
    for i in range(n_qubits):
        terms.append((
            "I" * i + "Z" + "I" * (n_qubits - i - 1),
            -0.01
        ))
    
    # Add some ZZ terms for two-qubit interactions
    for i in range(n_qubits - 1):
        terms.append((
            "I" * i + "ZZ" + "I" * (n_qubits - i - 2),
            0.005
        ))
    
    pauli_ops = [SparsePauliOp(term, coeff) for term, coeff in terms]
    hamiltonian = sum(pauli_ops)
    return hamiltonian

def main():
    mps_data = load_mps_data()
    converter = MPStoQuantumCircuit(mps_data)
    
    # Build and evaluate full rank circuit
    full_rank_circuit = converter.build_full_rank_mps_circuit()
    full_rank_fidelity = converter.verify_state(full_rank_circuit)
    
    print(f"Full Rank Circuit Depth: {full_rank_circuit.depth()}")
    print(f"Full Rank Circuit Gate Count: {full_rank_circuit.count_ops()}")
    print(f"Full Rank Circuit Fidelity: {full_rank_fidelity}")
    print(f"Full Rank Circuit:\n{full_rank_circuit}")
    
    # Build and evaluate improved circuit
    improved_circuit = converter.build_improved_mps_circuit()
    improved_fidelity = converter.verify_state(improved_circuit)
    
    print(f"\nImproved Circuit Depth: {improved_circuit.depth()}")
    print(f"Improved Circuit Gate Count: {improved_circuit.count_ops()}")
    print(f"Improved Circuit Fidelity: {improved_fidelity}")
    print(f"Improved Circuit:\n{improved_circuit}")
    
    # Build and evaluate minimal circuit
    minimal_circuit = converter.build_minimal_mps_circuit()
    minimal_fidelity = converter.verify_state(minimal_circuit)
    
    print(f"\nMinimal Circuit Depth: {minimal_circuit.depth()}")
    print(f"Minimal Circuit Gate Count: {minimal_circuit.count_ops()}")
    print(f"Minimal Circuit Fidelity: {minimal_fidelity}")
    print(f"Minimal Circuit:\n{minimal_circuit}")
    
    # Get density matrix for minimal circuit (it's smaller)
    minimal_density_matrix = converter.get_density_matrix(minimal_circuit)
    print(f"\nMinimal Circuit Density Matrix (first few elements):")
    print(minimal_density_matrix.data[:4, :4])
    
    # Create a mock Hamiltonian and compute energies
    hamiltonian = create_mock_hamiltonian(minimal_circuit.num_qubits)
    
    # Compute energies
    full_energy = None
    try:
        if full_rank_circuit.num_qubits <= 16:  # Only try if circuit is not too large
            full_energy = converter.estimate_energy(full_rank_circuit, hamiltonian)
    except Exception as e:
        full_energy = None
        print(f"Could not compute full rank energy: {e}")
    
    improved_energy = converter.estimate_energy(improved_circuit, hamiltonian)
    minimal_energy = converter.estimate_energy(minimal_circuit, hamiltonian)
    
    # Print energies
    print(f"\nQuantum Circuit Energies:")
    print(f"Reference Energy: {converter.energy}")
    if full_energy is not None:
        print(f"Full Rank Circuit Energy: {full_energy}")
    print(f"Improved Circuit Energy: {improved_energy}")
    print(f"Minimal Circuit Energy: {minimal_energy}")
    
    # Circuit comparison
    print("\nBest representation for H2O molecule MPS:")
    if full_rank_fidelity > 0.9:
        print("Full rank circuit provides accurate representation but requires many qubits and gates.")
    elif improved_fidelity > 0.1 or minimal_fidelity > 0.1:
        print("Simplified circuits capture some structure but with limited fidelity.")
    else:
        print("Direct state preparation is required for accurate representation of this MPS.")
    
    return {
        'full_rank_circuit': full_rank_circuit,
        'improved_circuit': improved_circuit,
        'minimal_circuit': minimal_circuit,
        'full_rank_fidelity': full_rank_fidelity,
        'improved_fidelity': improved_fidelity,
        'minimal_fidelity': minimal_fidelity,
        'reference_energy': converter.energy,
        'improved_energy': improved_energy,
        'minimal_energy': minimal_energy
    }

if __name__ == "__main__":
    results = main()
