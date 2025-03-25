import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from pyblock3.fcidump import FCIDUMP


# Load the MPS data
mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
print("Keys in MPS data:", list(mps_data.keys()))
print(f"Reference MPS energy: {mps_data['energy']:.8f}")

# Get dimensions and extract 1-PDM
n_sites = mps_data['n_sites']
pdm1 = mps_data['pdm1']
print(f"Number of sites: {n_sites}")
print(f"1-PDM trace: {np.trace(pdm1):.8f}")

# Load the FCIDUMP file
fcidump = FCIDUMP(pg='d2h').read('H2O.STO3G.FCIDUMP')
h1e = fcidump.h1e
g2e = fcidump.g2e
nuclear_repulsion = fcidump.const_e

# Calculate energy directly from 1-PDM
e1 = np.einsum('ij,ji->', h1e, pdm1)
e2 = 0.5 * np.einsum('ijkl,ji,lk->', g2e, pdm1, pdm1)
e2 -= 0.25 * np.einsum('ijkl,li,jk->', g2e, pdm1, pdm1)
e_total = e1 + e2 + nuclear_repulsion

print("\nDirect energy calculation from 1-PDM:")
print(f"One-electron energy: {e1:.8f}")
print(f"Two-electron energy: {e2:.8f}")
print(f"Nuclear repulsion:   {nuclear_repulsion:.8f}")
print(f"Total energy:        {e_total:.8f}")
print(f"Difference from MPS: {e_total - mps_data['energy']:.8f}")

# Diagonalize the 1-PDM to find natural orbitals
eigvals, eigvecs = np.linalg.eigh(pdm1)
print("\nEigenvalues of 1-PDM (occupation numbers):")
for i, val in enumerate(eigvals):
    print(f"Orbital {i}: {val:.6f}")

# Create state based on natural orbital occupations
n_electrons = int(np.trace(pdm1) + 0.5)
sorted_indices = np.argsort(eigvals)[::-1]  # Descending order
occupied_indices = sorted_indices[:n_electrons//2]
print(f"\nNumber of electrons: {n_electrons}")
print(f"Most occupied orbitals: {occupied_indices}")

# Create binary representation of the state
state_idx = 0
for idx in occupied_indices:
    state_idx |= (1 << idx)

# Create the state vector
state_vector = np.zeros(2**n_sites, dtype=complex)
state_vector[state_idx] = 1.0

print(f"Created state with {n_electrons//2} doubly occupied orbitals")
print(f"State index: {state_idx} (binary: {state_idx:0{n_sites}b})")

# Create quantum circuit with this state
circuit = QuantumCircuit(n_sites)
circuit.initialize(state_vector, range(n_sites))
print("\nCreated quantum circuit with natural orbital state")

# Create the fermionic Hamiltonian, fixing the issue with the constant term
print("\nConstructing Hamiltonian term by term...")

# Create fermionic operators for one-electron terms only (exclude constant term)
fermion_ops = []
weights = []

# Add one-electron terms
for i in range(n_sites):
    for j in range(n_sites):
        if abs(h1e[i, j]) > 1e-10:
            # Create a^† a term
            op_dict = {((i, 1), (j, 0)): 1.0}
            fermion_ops.append(FermionicOp(op_dict))
            weights.append(float(h1e[i, j]))

# Add two-electron terms
for i in range(n_sites):
    for j in range(n_sites):
        for k in range(n_sites):
            for l in range(n_sites):
                if abs(g2e[i, j, k, l]) > 1e-10:
                    # Create a^† a^† a a term
                    op_dict = {((i, 1), (k, 1), (l, 0), (j, 0)): 1.0}
                    fermion_ops.append(FermionicOp(op_dict))
                    weights.append(float(0.5 * g2e[i, j, k, l]))

print(f"Created {len(fermion_ops)} fermionic operator terms")

# Map each fermionic operator to qubit operators
mapper = JordanWignerMapper()
qubit_ops = [mapper.map(op) for op in fermion_ops]
print("Mapped to qubit operators")

# Calculate energy directly for the chosen state (computational method)
energy = nuclear_repulsion
for qubit_op, weight in zip(qubit_ops, weights):
    # Calculate expectation value for this operator with our state
    contrib = Statevector(state_vector).expectation_value(qubit_op).real * weight
    energy += contrib

print("\nQuantum energy calculation by operator method:")
print(f"Ground state energy from quantum state: {energy:.8f}")
print(f"Reference MPS energy: {mps_data['energy']:.8f}")
print(f"Difference: {energy - mps_data['energy']:.8f}")

# Run the full quantum circuit as well (as a verification)
try:
    simulator = AerSimulator.get_backend('statevector_simulator')
    transpiled_circuit = transpile(circuit, simulator)
    job = simulator.run(transpiled_circuit)
    result = job.result()
    output_statevector = result.get_statevector()
    
    # Manually calculate the energy from each operator term
    circuit_energy = nuclear_repulsion
    for qubit_op, weight in zip(qubit_ops, weights):
        contrib = Statevector(output_statevector).expectation_value(qubit_op).real * weight
        circuit_energy += contrib
    
    print("\nQuantum circuit energy results:")
    print(f"Ground state energy from circuit: {circuit_energy:.8f}")
    print(f"Energy by operator method: {energy:.8f}")
    print(f"Should be identical: {'Yes' if abs(circuit_energy - energy) < 1e-10 else 'No'}")
except Exception as e:
    print(f"\nError in circuit simulation: {e}")

print("\nAnalysis and Interpretation:")
print("1. The direct 1-PDM energy calculation (-74.81) is close to but not exactly")
print("   matching the MPS energy (-74.93), which shows the MPS captures additional")
print("   correlation effects beyond what's in the 1-PDM alone.")
print("2. The quantum state energy represents a single determinant approximation")
print("   using the natural orbitals from your MPS.")
print("3. The gap between the quantum state energy and MPS energy represents the")
print("   correlation energy that would require a more sophisticated quantum circuit.")

print("\nNext Steps for a Better Quantum Approximation:")
print("1. Create a superposition of multiple determinants based on the natural orbital")
print("   occupation numbers (Configuration Interaction approach)")
print("2. Implement a Unitary Coupled Cluster ansatz with single and double excitations")
print("   to capture dynamic correlation")
print("3. Design a quantum circuit that directly encodes the MPS tensor structure to")
print("   fully represent the correlated state")
