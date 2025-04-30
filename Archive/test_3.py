import numpy as np
import re
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter, ParameterVector
from qiskit_algorithms.optimizers import SPSA
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.primitives import Estimator
from qiskit_aer.primitives import Estimator as AerEstimator

class QuantumMPS:
    def __init__(self, fcidump_file='H2O.STO3G.FCIDUMP'):
        # Parse FCIDUMP file
        self.fcidump_file = fcidump_file
        self.norb, self.nelec, self.nuclear_repulsion, self.one_body, self.two_body = self.parse_fcidump()
        print(f"Water molecule: {self.norb} orbitals, {self.nelec} electrons")
        print(f"Nuclear repulsion: {self.nuclear_repulsion:.8f}")
        
        # Build parameterized MPS circuit
        self.num_qubits = self.norb
        self.bond_dim = min(200, 2**self.num_qubits)  # Similar to classical bond_dim = 200
        self.num_params = self._calculate_num_params()
        self.params = ParameterVector('θ', self.num_params)
        self.circuit = self._build_mps_circuit()
        
        # Build Hamiltonian
        self.hamiltonian = self._build_hamiltonian()
        
        # Create estimator for energy calculations
        self.estimator = AerEstimator(backend_options={"method": "matrix_product_state"})
    
    def parse_fcidump(self):
        """Read FCIDUMP file and extract parameters and integrals"""
        try:
            with open(self.fcidump_file, 'r') as f:
                lines = f.readlines()
            
            # Parse header
            header = ' '.join([line for line in lines if '&' in line])
            norb_match = re.search(r'NORB=\s*(\d+)', header)
            norb = int(norb_match.group(1)) if norb_match else 7
            
            nelec_match = re.search(r'NELEC=(\d+)', header)
            nelec = int(nelec_match.group(1)) if nelec_match else 10
            
            # Parse integrals
            one_body = np.zeros((norb, norb))
            two_body = np.zeros((norb, norb, norb, norb))
            nuclear_repulsion = 0.0
            
            for line in lines:
                if '&' in line:
                    continue
                
                values = line.strip().split()
                if len(values) < 5:
                    continue
                
                val = float(values[0])
                i, j, k, l = [int(x) for x in values[1:5]]
                
                # Convert from 1-indexed to 0-indexed
                i -= 1
                j -= 1
                k -= 1
                l -= 1
                
                if k == -1 and l == -1:  # One-electron terms: (i,j|h|0,0)
                    one_body[i, j] = val
                elif i == -1 and j == -1 and k == -1 and l == -1:  # Nuclear repulsion
                    nuclear_repulsion = val
                else:  # Two-electron terms: (i,j|g|k,l)
                    two_body[i, j, k, l] = val
            
            return norb, nelec, nuclear_repulsion, one_body, two_body
        
        except Exception as e:
            print(f"Error parsing FCIDUMP: {e}")
            print("Using default values")
            return 7, 10, 9.8194765097824046, np.zeros((7, 7)), np.zeros((7, 7, 7, 7))
    
    def _calculate_num_params(self):
        """Calculate the number of variational parameters needed"""
        # For a simplified MPS-inspired ansatz:
        # - Initial state preparation: norb params
        # - Bond dimensions: (norb-1) * bond_dimension params
        n_initial = self.norb  
        n_bond = min(3, self.norb - 1)  # Simplified bond parameter count
        return n_initial + n_bond
    
    def _build_mps_circuit(self):
        """Build a parameterized quantum circuit representing an MPS"""
        circuit = QuantumCircuit(self.num_qubits)
        
        # Initialize with parameterized rotations (similar to state preparation in MPS)
        param_idx = 0
        for i in range(self.num_qubits):
            circuit.ry(self.params[param_idx], i)
            param_idx += 1
        
        # Create entanglement structure inspired by MPS bond dimensions
        # We connect neighboring qubits with parameterized entanglement
        for i in range(self.num_qubits - 1):
            circuit.cx(i, i+1)
            if param_idx < len(self.params):
                circuit.rz(self.params[param_idx], i+1)
                param_idx += 1
        
        return circuit
    
    def _build_hamiltonian(self):
        """Build the molecular Hamiltonian for energy calculation"""
        # This is a simplified version, focusing on one-electron terms
        pauli_strings = []
        coefficients = []
        
        # One-electron terms
        for i in range(self.norb):
            for j in range(self.norb):
                if abs(self.one_body[i, j]) > 1e-10:
                    # Map to Pauli operators using Jordan-Wigner transform
                    # For diagonal terms (i==j), we use Z operator
                    if i == j:
                        # Create a Pauli Z on qubit i
                        pauli_str = 'I' * self.num_qubits
                        pauli_str = pauli_str[:i] + 'Z' + pauli_str[i+1:]
                        pauli_strings.append(pauli_str)
                        coefficients.append(0.5 * self.one_body[i, j])
        
        # Add nuclear repulsion as identity term
        pauli_strings.append('I' * self.num_qubits)
        coefficients.append(self.nuclear_repulsion)
        
        # Create SparsePauliOp representing the Hamiltonian
        hamiltonian = SparsePauliOp.from_list(list(zip(pauli_strings, coefficients)))
        return hamiltonian
    
    def calculate_energy(self, params=None):
        """Calculate the energy expectation value for given parameters"""
        if params is None:
            # Use random initial parameters if none provided
            params = np.random.random(self.num_params) * np.pi
        
        # Assign parameters (compatible with older Qiskit versions)
        param_dict = {self.params[i]: params[i] for i in range(self.num_params)}
        bound_circuit = self.circuit.assign_parameters(param_dict)
        
        # Calculate energy using estimator
        result = self.estimator.run([bound_circuit], [self.hamiltonian]).result()
        energy = result.values[0]
        
        return energy
    
    def optimize_circuit(self, max_iter=100):
        """Optimize the circuit parameters to find ground state energy"""
        # Initialize with random parameters
        initial_params = np.random.random(self.num_params) * np.pi
        
        # Define cost function for optimizer
        def cost_function(params):
            return self.calculate_energy(params)
        
        # Use SPSA optimizer (works well for noisy functions)
        optimizer = SPSA(maxiter=max_iter)
        
        # Run optimization
        print("\nOptimizing circuit parameters...")
        result = optimizer.minimize(cost_function, initial_params)
        
        # Calculate final energy
        optimal_params = result.x
        energy = self.calculate_energy(optimal_params)
        
        print(f"Optimization complete after {result.nfev} function evaluations")
        print(f"Final energy: {energy:.8f} Hartree")
        
        return energy, optimal_params
    
    def calculate_1pdm(self, optimal_params):
        """Calculate one-particle density matrix"""
        pdm1 = np.zeros((self.norb, self.norb))
        
        # Assign parameters to circuit (compatible with older Qiskit versions)
        param_dict = {self.params[i]: optimal_params[i] for i in range(self.num_params)}
        bound_circuit = self.circuit.assign_parameters(param_dict)
        
        # Setup simulator for measurements
        simulator = AerSimulator(method='matrix_product_state')
        
        # Calculate diagonal elements of 1PDM (occupation numbers)
        for i in range(self.norb):
            # Create measurement circuit
            meas_circuit = bound_circuit.copy()
            meas_circuit.measure_all()
            
            # Run circuit
            transpiled = transpile(meas_circuit, simulator)
            result = simulator.run(transpiled).result()
            counts = result.get_counts()
            
            # Calculate occupation number from measurement results
            occupation = 0
            total_shots = sum(counts.values())
            
            for bitstring, count in counts.items():
                # Check if the ith qubit is in state |1⟩
                if bitstring[self.num_qubits - 1 - i] == '1':
                    occupation += count / total_shots
            
            # Multiply by 2 for spin
            pdm1[i, i] = 2.0 * occupation
        
        print("\n1PDM calculated from quantum circuit (diagonal elements):")
        print(np.diag(pdm1))
        
        return pdm1

# Main function to run the quantum MPS calculation
def run_quantum_mps():
    print("Creating quantum MPS for water molecule...")
    qmps = QuantumMPS('H2O.STO3G.FCIDUMP')
    
    print("\nQuantum circuit representing MPS:")
    print(qmps.circuit)
    
    # Calculate energy without optimization (initial guess)
    initial_energy = qmps.calculate_energy()
    print(f"\nInitial energy estimate: {initial_energy:.8f} Hartree")
    
    # Optimize circuit parameters
    final_energy, optimal_params = qmps.optimize_circuit(max_iter=50)
    
    # Calculate 1PDM
    pdm1 = qmps.calculate_1pdm(optimal_params)
    
    # Save results
    np.save("h2o_energy_quantum.npy", final_energy)
    np.save("h2o_pdm1_quantum.npy", pdm1)
    
    # Compare with classical pyblock3 MPS results
    try:
        classical_energy = np.load("h2o_energy.npy")
        classical_pdm1 = np.load("h2o_pdm1.npy")
        
        print("\nComparison with classical MPS results:")
        print(f"Classical energy: {classical_energy:.8f} Hartree")
        print(f"Quantum energy:  {final_energy:.8f} Hartree")
        print(f"Energy difference: {abs(classical_energy - final_energy):.8f} Hartree")
        
        print("\nClassical 1PDM diagonal elements:")
        print(np.diag(classical_pdm1))
    except:
        print("\nCould not load classical MPS results for comparison")
    
    return final_energy, pdm1

if __name__ == "__main__":
    energy, pdm1 = run_quantum_mps()
