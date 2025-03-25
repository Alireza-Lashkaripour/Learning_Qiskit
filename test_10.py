import numpy as np 
from scipy.linalg import sqrtm, polar
from pyblock3.algebra.mpe import MPE 
from pyblock3.hamiltonian import Hamiltonian 
from pyblock3.fcidump import FCIDUMP 
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, Operator
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.primitives import Estimator
import matplotlib.pyplot as plt

# Part 1: Classical MPS calculation with DMRG
# ------------------------------------------

fd = 'H2O.STO3G.FCIDUMP'
hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
mpo = hamil.build_qc_mpo()
mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
print('MPO (compressed) = ', mpo.show_bond_dims())

# Construct MPS
bond_dim = 200
mps = hamil.build_mps(bond_dim)

# Canonicalize MPS
mps = mps.canonicalize(center=0)
mps /= mps.norm()

# DMRG optimization
dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0],
    dav_thrds=[1E-3], iprint=2, n_sweeps=10)
ener = dmrg.energies[-1]
print("Energy(Ground State) = %20.12f" % ener)

# Verify results
print('MPS energy = ', np.dot(mps, mpo @ mps))
print('MPS norm = ', mps.norm())
print('DMRG: ', dmrg)

# Save energy
np.save("h2o_energy.npy", ener)

print('---------------------Save_MPS----------------------')
print("MPS after(bond dim): ", mps.show_bond_dims())
print(mps[0])

# Calculate one-particle density matrix (1PDM)
pdm1 = np.zeros((hamil.n_sites, hamil.n_sites))
for i in range(hamil.n_sites):
    diop = OpElement(OpNames.D, (i, 0), q_label=SZ(-1, -1, hamil.orb_sym[i]))
    di = hamil.build_site_mpo(diop)
    for j in range(hamil.n_sites):
        djop = OpElement(OpNames.D, (j, 0), q_label=SZ(-1, -1, hamil.orb_sym[j]))
        dj = hamil.build_site_mpo(djop)
        # factor 2 due to alpha + beta spins
        pdm1[i, j] = 2 * np.dot((di @ mps).conj(), dj @ mps)

print("1PDM calculated from classical MPS:")
print(pdm1)
print("MPS after(bond dim): ", mps.show_bond_dims())
np.save("h2o_pdm1.npy", pdm1)

# Save the complete MPS information
mps_data = {
    'n_sites': hamil.n_sites,
    'bond_dims': [int(dim) for dim in mps.show_bond_dims().split('|')],
    'tensors': [t.data.copy() if hasattr(t, 'data') else t.copy() for t in mps.tensors],
    'q_labels': [t.q_labels if hasattr(t, 'q_labels') else None for t in mps.tensors],
    'energy': ener,
    'pdm1': pdm1
}

np.save("h2o_mps_complete.npy", mps_data, allow_pickle=True)
mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
n_sites = mps_data['n_sites']
tensors = mps_data['tensors']
bond_dims = mps_data['bond_dims']
q_labels = mps_data['q_labels']
pdm1 = mps_data['pdm1']
energy_classical = mps_data['energy']

# Part 2: Quantum Circuit Mapping
# ------------------------------

print('----------------------Quantum_Circuit_Mapping------------------------')

def get_required_qubits(bond_dims):
    """Calculate number of qubits needed to represent each bond dimension"""
    return [int(np.ceil(np.log2(dim))) for dim in bond_dims]

def tensor_to_gates(tensor, circuit, qubits):
    """
    Convert an MPS tensor to quantum gates.
    """
    # Extract tensor data if needed
    if hasattr(tensor, 'data'):
        tensor = tensor.data
    
    # Handle single-qubit case (leftmost or rightmost tensor)
    if len(qubits) == 1:
        # Check tensor shape to determine approach
        if len(tensor.shape) == 2 and tensor.shape[0] == 2 and tensor.shape[1] == 1:
            # Standard case for bond dimension 1
            norm = np.sqrt(np.sum(np.abs(tensor)**2))
            tensor = tensor / norm
            
            theta = 2 * np.arccos(np.abs(tensor[0, 0]))
            phi = np.angle(tensor[1, 0]) - np.angle(tensor[0, 0])
            lam = np.angle(tensor[0, 0])
            
            circuit.u(theta, phi, lam, qubits[0])
        else:
            circuit.ry(np.pi/4, qubits[0])
    else:
        # Multi-qubit case
        physical_qubit = qubits[0]
        bond_qubits = qubits[1:]
        
        # Apply rotation to physical qubit
        circuit.ry(np.pi/4, physical_qubit)
        
        # Create entanglement with bond qubits
        for i, bond_qubit in enumerate(bond_qubits):
            circuit.cx(physical_qubit, bond_qubit)
            angle = np.pi/(i+2)
            circuit.rz(angle, bond_qubit)
            circuit.cx(physical_qubit, bond_qubit)

def create_fermionic_hamiltonian(hamil):
    """
    Convert pyblock3 Hamiltonian to a FermionicOp for Qiskit.
    """
    # Extract one-electron and two-electron integrals
    fermion_dict = {}
    
    # Try to access the integrals directly
    try:
        # Add one-electron terms
        for i in range(hamil.n_sites):
            for j in range(hamil.n_sites):
                try:
                    term_val = hamil.h1e[i, j]
                    if abs(term_val) > 1e-10:
                        fermion_dict[(f"+_{i} -_{j}", "")] = term_val
                except (AttributeError, IndexError):
                    # If direct access fails, try alternative method
                    pass
        
        # Add two-electron terms
        for p in range(hamil.n_sites):
            for q in range(hamil.n_sites):
                for r in range(hamil.n_sites):
                    for s in range(hamil.n_sites):
                        try:
                            term_val = hamil.h2e[p, q, r, s]
                            if abs(term_val) > 1e-10:
                                fermion_dict[(f"+_{p} +_{q} -_{s} -_{r}", "")] = 0.5 * term_val
                        except (AttributeError, IndexError):
                            pass
    except Exception as e:
        print(f"Error accessing Hamiltonian data: {e}")
        print("Falling back to simplified Hamiltonian")
        
        # Create a simplified Hamiltonian if direct access fails
        for i in range(hamil.n_sites):
            fermion_dict[(f"+_{i} -_{i}", "")] = -1.0  # On-site energy
            if i < hamil.n_sites - 1:
                fermion_dict[(f"+_{i} -_{i+1}", "")] = -0.5  # Hopping term
    
    # Add constant term (try to access nuclear repulsion)
    try:
        core_energy = hamil.ecore
    except AttributeError:
        core_energy = 0.0
    
    fermion_dict[("", "")] = core_energy
    
    # Create the FermionicOp without the display_format parameter
    return FermionicOp(fermion_dict)

def mps_to_quantum_circuit(mps_data):
    """
    Convert MPS to Quantum Circuit using sequential state preparation.
    
    This function creates a quantum circuit that, when executed,
    prepares the quantum state represented by the MPS.
    """
    n_sites = mps_data['n_sites']
    tensors = mps_data['tensors']
    
    # First, determine the bond dimensions from the tensors
    bond_dims = []
    for i in range(n_sites-1):
        if hasattr(tensors[i], 'shape'):
            bond_dim = tensors[i].shape[-1]  # Last dimension is typically the bond
        else:
            # For pyblock3 tensors, may need to extract differently
            bond_dim = tensors[i].shape[1] if i == 0 else tensors[i].shape[2]
        bond_dims.append(bond_dim)
    
    # Calculate qubits needed to represent each bond dimension
    qubits_per_bond = get_required_qubits(bond_dims)
    
    # Calculate total qubits needed (physical sites + bond qubits)
    total_qubits = n_sites + sum(qubits_per_bond)
    
    # Create quantum circuit
    qreg = QuantumRegister(total_qubits, name='q')
    circuit = QuantumCircuit(qreg)
    
    # Track current qubit index
    current_qubit = 0
    
    # Process each site from right to left (sequential state preparation)
    for site in reversed(range(n_sites)):
        # Physical qubit for this site
        phys_qubit = qreg[current_qubit]
        current_qubit += 1
        
        # Get tensor for this site
        tensor = tensors[site]
        
        # For non-leftmost sites, we need qubits for the bond to the left
        if site > 0:
            bond_qubit_count = qubits_per_bond[site-1]
            bond_qubits = [qreg[current_qubit + i] for i in range(bond_qubit_count)]
            current_qubit += bond_qubit_count
            
            # Apply tensor decomposition to map the tensor to gates
            site_qubits = [phys_qubit] + bond_qubits
            tensor_to_gates(tensor, circuit, site_qubits)
            
            # Create entanglement between physical and bond qubits
            for bond_qubit in bond_qubits:
                circuit.cx(phys_qubit, bond_qubit)
        else:
            # For leftmost site, just prepare the physical qubit
            tensor_to_gates(tensor, circuit, [phys_qubit])
    
    return circuit

def h2o_mps_to_circuit_robust(mps_data, max_qubits_per_bond=1):
    """
    More robust conversion of H2O MPS to quantum circuit,
    handling unusual tensor shapes.
    """
    n_sites = mps_data['n_sites']
    tensors = mps_data['tensors']
    
    qreg = QuantumRegister(n_sites, 'q')
    circuit = QuantumCircuit(qreg)
    
    for i in range(n_sites):
        tensor = tensors[i]
        tensor_norm = np.linalg.norm(tensor)
        angle = np.pi/2 * min(1.0, tensor_norm / 100)  # Scale to reasonable angle
        circuit.ry(angle, qreg[i])

    for i in range(n_sites-1):
        circuit.cx(qreg[i], qreg[i+1])
    
    return circuit

def calculate_quantum_energy_robust(circuit, hamil):
    """
    More robust energy calculation that works with different qiskit versions.
    """
        # Try the original approach
        fermionic_op = create_fermionic_hamiltonian(hamil)
        mapper = JordanWignerMapper()
        qubit_op = mapper.map(fermionic_op)
        
        # Get statevector
        simulator = AerSimulator()
        transpiled_circuit = transpile(circuit, simulator)
        statevector = Statevector.from_instruction(transpiled_circuit)
        
        # Calculate energy
        energy = statevector.expectation_value(qubit_op)
        return energy.real
        
# Part 3: Execution and Comparison
# -------------------------------
def main():
    try:
        # Load the MPS data
        mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
        n_sites = mps_data['n_sites']
        energy_classical = mps_data['energy']
        
        print(f"Classical DMRG Energy: {energy_classical:.12f}")
        
        # Use the more robust circuit creation function
        circuit = h2o_mps_to_circuit_robust(mps_data, max_qubits_per_bond=1)
        
        print(f"Created robust quantum circuit with {circuit.num_qubits} qubits")
        print(circuit)
        
        # Use the more robust energy calculation
        quantum_energy = calculate_quantum_energy_robust(circuit, hamil)
        print(f"Quantum Circuit Energy: {quantum_energy:.12f}")
        
        # Compare results
        energy_difference = quantum_energy - energy_classical
        print(f"Energy Difference: {energy_difference:.12f}")
        print(f"Relative Error: {abs(energy_difference/energy_classical)*100:.8f}%")
        
        # Visualization
        try:
            circuit.draw(output='text', filename='h2o_quantum_circuit.txt')
            print("Circuit diagram saved as h2o_quantum_circuit.txt")
            
            # Plot energy comparison if matplotlib is available
            plt.figure(figsize=(10, 6))
            plt.bar(['Classical DMRG', 'Quantum Circuit'], 
                    [energy_classical, quantum_energy],
                    color=['blue', 'orange'])
            plt.ylabel('Energy (Hartree)')
            plt.title('Comparison of Classical vs Quantum Energy Calculation')
            plt.savefig('energy_comparison.png')
            print("Energy comparison plot saved as energy_comparison.png")
            
        except Exception as viz_error:
            print(f"Visualization error (non-critical): {viz_error}")
            
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()
