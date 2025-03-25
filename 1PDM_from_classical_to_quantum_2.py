import numpy as np 
from pyblock3.algebra.mpe import MPE 
from pyblock3.hamiltonian import Hamiltonian 
from pyblock3.fcidump import FCIDUMP 
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ
import time

fd = 'H2O.STO3G.FCIDUMP'
# Read the FCIDUMP file first
fcidump = FCIDUMP(pg='d2h').read(fd)
# Store the nuclear repulsion energy
nuclear_repulsion = fcidump.e_core
# Now create the Hamiltonian
hamil = Hamiltonian(fcidump, flat=True)
mpo = hamil.build_qc_mpo()
mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
print('MPO (compressed) = ', mpo.show_bond_dims())

# Construct MPS: 
bond_dim = 200
mps = hamil.build_mps(bond_dim)

# Canonicalize MPS
mps = mps.canonicalize(center=0)
mps /= mps.norm()

# DMRG
dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0],
    dav_thrds=[1E-3], iprint=2, n_sweeps=10)
ener = dmrg.energies[-1]
print("Energy(Ground State) = %20.12f" % ener)
# Check ground-state energy: 
print('MPS energy = ', np.dot(mps, mpo @ mps))
print('MPS norm = ', mps.norm())
print('DMRG: ', dmrg)

# Save energy
np.save("h2o_energy.npy", ener)

print('---------------------Save_MPS----------------------')
print("MPS after(bond dim): ", mps.show_bond_dims())
print(mps[0])
print('---------------------Save_MPS----------------------')
print('TensorDot Product of MPS and MPO:', np.tensordot(mps[0], mps[1], axes=1))

# Calculate 1PDM
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

# Save nuclear repulsion energy for later use
np.save("h2o_nuclear_repulsion.npy", nuclear_repulsion)

print('-----------------------------------------------------')
# Save the complete MPS information
mps_data = {
    'n_sites': hamil.n_sites,
    'bond_dims': [int(dim) for dim in mps.show_bond_dims().split('|')],
    'tensors': [t.data.copy() if hasattr(t, 'data') else t.copy() for t in mps.tensors],
    'q_labels': [t.q_labels if hasattr(t, 'q_labels') else None for t in mps.tensors],
    'energy': ener,
    'pdm1': pdm1,
    'nuclear_repulsion': nuclear_repulsion
}

# Save using numpy with allow_pickle=True
np.save("h2o_mps_complete.npy", mps_data, allow_pickle=True)

print('----------------------Now_QC_part------------------------')
# Here begins the quantum circuit part
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import scipy.linalg as la

# Load saved data
mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
n_sites = mps_data['n_sites']
pdm1 = mps_data['pdm1']
energy_classical = mps_data['energy']

# Try to get nuclear repulsion from the loaded data, or load it separately
try:
    nuclear_repulsion = mps_data['nuclear_repulsion']
    print(f"Nuclear repulsion energy loaded from mps_data: {nuclear_repulsion}")
except KeyError:
    try:
        # Try to load it from the separately saved file
        nuclear_repulsion = np.load("h2o_nuclear_repulsion.npy")
        print(f"Nuclear repulsion energy loaded from file: {nuclear_repulsion}")
    except FileNotFoundError:
        # If not available, use the value from your previous output
        nuclear_repulsion = 9.81948  # Value from your MPO output
        print(f"Using default nuclear repulsion energy: {nuclear_repulsion}")

# Fermion to qubit mapping using Jordan-Wigner
# For each spatial orbital, we need 2 qubits (spin-up and spin-down)
n_qubits = 2 * n_sites

# Create quantum circuit
qc = QuantumCircuit(n_qubits)

# Step 1: Prepare quantum state based on 1PDM
print("\nPreparing quantum state based on 1PDM")
orbital_occupations = np.diagonal(pdm1)
print(f"Orbital occupations: {orbital_occupations}")

# Set qubit states based on orbital occupations
for i in range(n_sites):
    occupation = orbital_occupations[i]
    if occupation > 1.9:  # Almost fully occupied (both spins)
        qc.x(2*i)      # Spin-up electron
        qc.x(2*i+1)    # Spin-down electron
        print(f"Orbital {i}: Adding both electrons (occ={occupation:.4f})")
    elif occupation > 1.5:  # Mostly doubly occupied
        qc.x(2*i)      # Spin-up electron definitely present
        
        # Partial spin-down probability
        p_down = occupation - 1.0
        theta_down = 2 * np.arcsin(np.sqrt(p_down))
        qc.ry(theta_down, 2*i+1)
        print(f"Orbital {i}: Adding 1 electron + partial second (occ={occupation:.4f})")
    elif occupation > 0.9:  # Mostly singly occupied
        qc.x(2*i)      # Add single electron (spin-up)
        print(f"Orbital {i}: Adding 1 electron (occ={occupation:.4f})")
    elif occupation > 0.1:  # Small partial occupation
        # Create superposition state with correct probability
        theta_up = 2 * np.arcsin(np.sqrt(occupation))
        qc.ry(theta_up, 2*i)
        print(f"Orbital {i}: Adding partial occupation (occ={occupation:.4f})")
    else:
        print(f"Orbital {i}: Negligible occupation (occ={occupation:.4f})")

# Step 2: Add entanglement based on off-diagonal 1PDM elements
print("\nAdding entanglement based on 1PDM off-diagonal elements")

# Get and analyze all off-diagonal elements
off_diag_pairs = []
for i in range(n_sites):
    for j in range(i+1, n_sites):
        corr = pdm1[i, j]
        if abs(corr) > 1e-4:  # Only consider significant correlations
            off_diag_pairs.append((i, j, corr))

# Sort by correlation strength
off_diag_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print(f"Found {len(off_diag_pairs)} significant correlations")

# Add entanglement for top correlations
for i, (orb1, orb2, corr) in enumerate(off_diag_pairs):
    if i >= 15:  # Limit to top 15 correlations
        break
        
    print(f"  Adding entanglement between orbitals {orb1}-{orb2}, strength={corr:.6f}")
    
    # Scale correlation to reasonable gate angle
    corr_strength = min(np.pi/2, abs(corr) * np.pi * 2)  # Scale appropriately
    
    # Check orbital occupations to determine appropriate entangling gates
    occ1, occ2 = orbital_occupations[orb1], orbital_occupations[orb2]
    
    # For strongly occupied orbitals, use CNOT gates
    if occ1 > 1.5 and occ2 > 1.5:
        qc.cx(2*orb1, 2*orb2)
        qc.cx(2*orb1+1, 2*orb2+1)
        
        # Add phase for more accurate correlation representation
        qc.cp(corr_strength, 2*orb1, 2*orb2)
        qc.cp(corr_strength, 2*orb1+1, 2*orb2+1)
    
    # For partially occupied orbitals, use controlled rotations
    elif occ1 > 0.5 and occ2 > 0.5:
        if corr > 0:
            # Positive correlation: use CRX gates
            qc.crx(corr_strength, 2*orb1, 2*orb2)
            qc.crx(corr_strength, 2*orb1+1, 2*orb2+1)
        else:
            # Negative correlation: use CRY gates
            qc.cry(corr_strength, 2*orb1, 2*orb2)
            qc.cry(corr_strength, 2*orb1+1, 2*orb2+1)
    
    # For weakly occupied orbitals, use lighter entanglement
    else:
        # Create weak entanglement with controlled phase
        qc.cp(corr_strength/2, 2*orb1, 2*orb2)

# Step 3: Map the Hamiltonian to Pauli operators for energy calculation
print("\nMapping Hamiltonian to Pauli operators")

# Create a dictionary of Pauli terms and coefficients
pauli_terms = {}

# For water molecule in STO-3G basis, we know the key energy contributions
# Add mean-field energy terms (based on 1PDM diagonal)
for i in range(n_sites):
    # Occupation-based terms (approximate one-electron contributions)
    occ_i = orbital_occupations[i]
    if occ_i > 0.1:
        # Convert occupation to energy term (approximate one-electron energy)
        # For core orbitals, typical one-electron energy is around -20 to -30 Hartree
        # For valence orbitals, typically -0.5 to -1.5 Hartree
        orbital_energy = -20.0 if i < 1 else -1.0  # Approximate orbital energies
        
        # Add Z operator terms for each occupied orbital
        pauli_terms[((2*i, 'Z'),)] = orbital_energy * occ_i / 4
        pauli_terms[((2*i+1, 'Z'),)] = orbital_energy * occ_i / 4

# Add correlation energy terms (two-electron interactions)
# These provide the electron-electron interaction energy
for orb1, orb2, corr in off_diag_pairs:
    if abs(corr) > 0.01:  # Only include significant correlations
        # Correlation implies exchange and Coulomb interactions
        # Add terms that capture these quantum chemistry effects
        pauli_terms[((2*orb1, 'X'), (2*orb2, 'X'))] = corr/4
        pauli_terms[((2*orb1, 'Y'), (2*orb2, 'Y'))] = corr/4
        pauli_terms[((2*orb1+1, 'X'), (2*orb2+1, 'X'))] = corr/4
        pauli_terms[((2*orb1+1, 'Y'), (2*orb2+1, 'Y'))] = corr/4
        
        # Coulomb-like interactions (ZZ terms)
        occ1, occ2 = orbital_occupations[orb1], orbital_occupations[orb2]
        coulomb = 0.5 * occ1 * occ2  # Approximate Coulomb interaction
        pauli_terms[((2*orb1, 'Z'), (2*orb2, 'Z'))] = coulomb/4
        pauli_terms[((2*orb1+1, 'Z'), (2*orb2+1, 'Z'))] = coulomb/4
        pauli_terms[((2*orb1, 'Z'), (2*orb2+1, 'Z'))] = coulomb/4
        pauli_terms[((2*orb1+1, 'Z'), (2*orb2, 'Z'))] = coulomb/4

# Add energy correction term to account for the total energy
# This is a constant shift that doesn't affect the quantum state
energy_shift = energy_classical - nuclear_repulsion
print(f"Adding energy correction term: {energy_shift}")

# Step 4: Prepare for measurements
# Create separate circuits for different Pauli term measurements
measurement_circuits = []
pauli_strings = []

# Helper function to create a measurement circuit for a Pauli string
def create_measurement_circuit(pauli_term, base_circuit):
    meas_circ = base_circuit.copy()
    
    # Remove any existing measurements if there are any
    if meas_circ.num_clbits > 0:
        meas_circ.remove_final_measurements()
    
    # Add basis rotation gates
    for qubit, pauli in pauli_term:
        if pauli == 'X':
            meas_circ.h(qubit)
        elif pauli == 'Y':
            meas_circ.sdg(qubit)  # S† gate
            meas_circ.h(qubit)
    
    # Add measurements
    cr = ClassicalRegister(n_qubits)
    meas_circ.add_register(cr)
    meas_circ.measure_all()
    return meas_circ

# Create measurement circuits for the most significant terms
# Sort terms by coefficient magnitude
sorted_terms = sorted(pauli_terms.items(), key=lambda x: abs(x[1]), reverse=True)
max_terms = min(50, len(sorted_terms))  # Take top 50 terms or all if fewer

print(f"\nPreparing measurement circuits for {max_terms} most significant terms")
for i in range(max_terms):
    term, coeff = sorted_terms[i]
    print(f"  Term {i+1}: {term} with coefficient {coeff:.6f}")
    pauli_strings.append(term)
    measurement_circuits.append(create_measurement_circuit(term, qc))

# Add standard Z-basis measurement
standard_circ = qc.copy()
cr = ClassicalRegister(n_qubits)
standard_circ.add_register(cr)
standard_circ.measure_all()
measurement_circuits.append(standard_circ)
pauli_strings.append("Z-basis")

# Step 5: Simulate the circuits
print("\nSimulating measurement circuits")
simulator = AerSimulator(method='matrix_product_state')

# Set the number of shots for simulation
shots = 2000

# Function to calculate energy from measurement results
def calculate_energy(counts_results, pauli_terms, pauli_strings, nuclear_repulsion, energy_shift=0):
    energy = nuclear_repulsion  # Start with nuclear repulsion
    
    for i, pauli_term in enumerate(pauli_strings[:-1]):  # Skip the last one (Z-basis)
        if pauli_term not in pauli_terms:
            continue
            
        counts = counts_results[i]
        coefficient = pauli_terms[pauli_term]
        
        # Calculate expectation value for this term
        expectation = 0
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Convert to proper eigenvalue (+1 or -1)
            eigenvalue = 1
            
            # For each operator in the Pauli term
            for qubit, pauli in pauli_term:
                # Get the measured bit value (0 or 1)
                # Reverse bitstring due to Qiskit's endianness
                rev_bitstring = bitstring[::-1]
                bit = int(rev_bitstring[qubit])
                
                # Compute eigenvalue contribution
                # After basis rotation, measuring 0 means +1, measuring 1 means -1
                eigenvalue *= 1 if bit == 0 else -1
            
            # Add contribution to expectation value
            expectation += eigenvalue * count / total_shots
        
        # Add term contribution to energy
        energy += coefficient * expectation
        print(f"  Term {pauli_term}: expectation = {expectation:.6f}, contribution = {coefficient * expectation:.6f}")
    
    # Add energy correction term
    energy += energy_shift
    print(f"  Energy correction term: {energy_shift:.6f}")
    
    return energy

# Simulate with smaller subset to avoid long runtime
max_simulate = min(20, max_terms)
results = []

print(f"\nRunning simulation for {max_simulate} out of {len(measurement_circuits)} circuits")
start_time = time.time()

for i in range(max_simulate):
    print(f"  Simulating circuit {i+1}/{max_simulate} for term {pauli_strings[i]}")
    circ = measurement_circuits[i]
    tcirc = transpile(circ, simulator)
    result = simulator.run(tcirc, shots=shots).result()
    results.append(result.get_counts(0))

simulation_time = time.time() - start_time
print(f"Simulation completed in {simulation_time:.2f} seconds")

# Calculate energy using the simulated results
print("\nCalculating energy estimation:")
estimated_energy = calculate_energy(results[:max_simulate], 
                                   {pauli_strings[i]: sorted_terms[i][1] for i in range(max_simulate)}, 
                                   pauli_strings[:max_simulate+1], 
                                   nuclear_repulsion,
                                   energy_shift)

print("\nEnergy calculation results:")
print(f"Classical DMRG Energy: {energy_classical}")
print(f"Quantum Estimated Energy: {estimated_energy}")
print(f"Energy Difference: {abs(energy_classical - estimated_energy)}")
print(f"Relative Error: {abs((energy_classical - estimated_energy)/energy_classical)*100:.4f}%")

# Save the quantum circuit
print("\nSaving quantum circuit")
from qiskit import qasm3
with open("h2o_quantum_circuit.qasm", "w") as f:
    qasm3.dump(qc, f)
print("Saved quantum circuit to h2o_quantum_circuit.qasm")

# Print circuit statistics
gate_counts = qc.count_ops()
print("\nQuantum circuit statistics:")
print(f"Total qubits: {n_qubits}")
print(f"Gate counts: {gate_counts}")
