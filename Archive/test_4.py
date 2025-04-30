import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

# Load the saved MPS data
mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()

# Extract the necessary information
n_sites = mps_data['n_sites']
tensors = mps_data['tensors']
bond_dims = mps_data['bond_dims']
energy_classical = mps_data['energy']
pdm1 = mps_data['pdm1']

print(f"Loaded MPS with {n_sites} sites")
print(f"Bond dimensions: {bond_dims}")
print(f"Classical energy: {energy_classical}")

# Create a simulator that uses MPS representation
simulator = AerSimulator(method='matrix_product_state')

# Create a circuit to prepare an approximation of the MPS state
# using the 1-particle density matrix (pdm1)
approx_circuit = QuantumCircuit(n_sites)

# Apply rotations based on the diagonal elements of the 1PDM
# The diagonal elements give the occupation probability
for i in range(n_sites):
    occupation = pdm1[i, i] / 2.0  # Divide by 2 because pdm1 includes both spins
    theta = 2 * np.arcsin(np.sqrt(min(max(occupation, 0), 1)))  # Ensure value is between 0 and 1
    approx_circuit.ry(theta, i)

# Add some entanglement based on off-diagonal elements
for i in range(n_sites-1):
    for j in range(i+1, n_sites):
        if abs(pdm1[i, j]) > 0.1:  # Only consider significant correlations
            # Use the magnitude of the off-diagonal element to determine entanglement
            approx_circuit.cx(i, j)
            # Add a phase based on the sign of the off-diagonal element
            if pdm1[i, j] < 0:
                approx_circuit.z(j)

# Create a new circuit just for getting the statevector
statevector_circuit = QuantumCircuit(n_sites)
# Copy the state preparation part
for i in range(n_sites):
    occupation = pdm1[i, i] / 2.0
    theta = 2 * np.arcsin(np.sqrt(min(max(occupation, 0), 1)))
    statevector_circuit.ry(theta, i)

for i in range(n_sites-1):
    for j in range(i+1, n_sites):
        if abs(pdm1[i, j]) > 0.1:
            statevector_circuit.cx(i, j)
            if pdm1[i, j] < 0:
                statevector_circuit.z(j)

# Get the statevector directly instead of using snapshots
from qiskit.quantum_info import Statevector
statevector = Statevector.from_instruction(statevector_circuit)
statevector_array = statevector.data

print("Statevector dimension:", len(statevector_array))

# Now, let's calculate the 1PDM from our quantum state
quantum_pdm1 = np.zeros((n_sites, n_sites), dtype=complex)

# Calculate the diagonal elements (site occupations)
print("\nCalculating quantum 1PDM...")
for i in range(n_sites):
    # We can calculate occupations directly from the statevector
    # This is more efficient than running a separate circuit for each site
    occupation = 0
    for k, amplitude in enumerate(statevector_array):
        # Check if site i is occupied in basis state k
        if (k & (1 << i)) != 0:
            occupation += abs(amplitude)**2
    
    quantum_pdm1[i, i] = occupation

# Calculate off-diagonal elements (correlations)
for i in range(n_sites):
    for j in range(i+1, n_sites):
        # For off-diagonal elements, we need to calculate <a^†_i a_j>
        # This can be split into real and imaginary parts
        correlation = 0
        for k, amplitude_k in enumerate(statevector_array):
            for l, amplitude_l in enumerate(statevector_array):
                # Check if states k and l differ only by an electron moving from j to i
                if (k & (1 << i)) == 0 and (k & (1 << j)) != 0:
                    l_should_be = k ^ (1 << i) ^ (1 << j)  # Flip bits i and j
                    if l == l_should_be:
                        correlation += amplitude_k.conjugate() * amplitude_l
        
        quantum_pdm1[i, j] = correlation
        quantum_pdm1[j, i] = correlation.conjugate()

print("Quantum 1PDM calculated from statevector")

# Scale the quantum 1PDM to account for both alpha and beta spins
# In the classical calculation, the factor of 2 was applied
quantum_pdm1_scaled = quantum_pdm1 * 2

# Compare classical and quantum 1PDMs
print("\nComparison of 1-Particle Density Matrices:")
print("Classical 1PDM:")
print(pdm1)
print("\nQuantum 1PDM (from circuit approximation):")
print(quantum_pdm1_scaled.real)  # Just show the real part for simplicity

# Calculate the Frobenius norm of the difference as a measure of error
pdm_diff = pdm1 - quantum_pdm1_scaled.real
pdm_error = np.linalg.norm(pdm_diff, 'fro')
print(f"\n1PDM Difference (Frobenius norm): {pdm_error:.6f}")

# Visualize the 1PDMs side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot classical 1PDM
im1 = ax1.imshow(pdm1, cmap='viridis')
ax1.set_title('Classical 1PDM')
ax1.set_xlabel('Site Index')
ax1.set_ylabel('Site Index')
plt.colorbar(im1, ax=ax1)

# Plot quantum 1PDM
im2 = ax2.imshow(quantum_pdm1_scaled.real, cmap='viridis')
ax2.set_title('Quantum 1PDM')
ax2.set_xlabel('Site Index')
ax2.set_ylabel('Site Index')
plt.colorbar(im2, ax=ax2)

# Plot difference
im3 = ax3.imshow(pdm_diff, cmap='RdBu', vmin=-1, vmax=1)
ax3.set_title('Difference (Classical - Quantum)')
ax3.set_xlabel('Site Index')
ax3.set_ylabel('Site Index')
plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.savefig('pdm_comparison.png')
print("PDM comparison plot saved as 'pdm_comparison.png'")

# Additional code to extract Hamiltonian terms from FCIDUMP and calculate energy
try:
    # This would require having pyblock3 or another library that can read FCIDUMP files
    from pyblock3.fcidump import FCIDUMP
    
    # Load the FCIDUMP file
    fd = 'H2O.STO3G.FCIDUMP'
    fcidump = FCIDUMP(pg='d2h').read(fd)
    
    # Parse FCIDUMP file to extract one-electron and two-electron integrals
    # The FCIDUMP format stores integrals in a specific way
    
    # Initialize one-electron and two-electron integrals
    n_orb = fcidump.n_sites
    h1e = np.zeros((n_orb, n_orb))
    h2e = np.zeros((n_orb, n_orb, n_orb, n_orb))
    
    # Extract integrals from the FCIDUMP object
    # This depends on how the FCIDUMP object stores the integrals
    # Let's try some common attribute names
    
    print("\nExtracting Hamiltonian integrals from FCIDUMP...")
    
    # Get one-electron integrals
    if hasattr(fcidump, 'h1e_matrix'):
        h1e = fcidump.h1e_matrix
    elif hasattr(fcidump, 'h1e'):
        h1e = fcidump.h1e
    elif hasattr(fcidump, 'get_h1e'):
        h1e = fcidump.get_h1e()
    else:
        # Extract one-electron integrals from the raw data
        # Format is typically value i j 0 0 for one-electron integrals
        print("Extracting one-electron integrals manually...")
        for line in open(fd).readlines():
            if '&' in line or len(line.strip()) == 0:
                continue
            data = line.split()
            if len(data) == 5:
                val = float(data[0])
                i, j, k, l = [int(x) for x in data[1:]]
                if k == 0 and l == 0 and i > 0 and j > 0:
                    # One-electron integral
                    h1e[i-1, j-1] = val
                    h1e[j-1, i-1] = val  # Symmetry
    
    # Get two-electron integrals
    if hasattr(fcidump, 'h2e_matrix'):
        h2e = fcidump.h2e_matrix
    elif hasattr(fcidump, 'h2e'):
        h2e = fcidump.h2e
    elif hasattr(fcidump, 'get_h2e'):
        h2e = fcidump.get_h2e()
    else:
        # Extract two-electron integrals from the raw data
        # Format is typically value i j k l for two-electron integrals
        print("Extracting two-electron integrals manually...")
        for line in open(fd).readlines():
            if '&' in line or len(line.strip()) == 0:
                continue
            data = line.split()
            if len(data) == 5:
                val = float(data[0])
                i, j, k, l = [int(x) for x in data[1:]]
                if i > 0 and j > 0 and k > 0 and l > 0:
                    # Two-electron integral (physicist's notation: (ij|kl))
                    # Note: FCIDUMP indices are 1-based, so we need to subtract 1
                    h2e[i-1, j-1, k-1, l-1] = val
                    # Add all symmetry-related elements
                    h2e[j-1, i-1, k-1, l-1] = val
                    h2e[i-1, j-1, l-1, k-1] = val
                    h2e[j-1, i-1, l-1, k-1] = val
                    h2e[k-1, l-1, i-1, j-1] = val
                    h2e[l-1, k-1, i-1, j-1] = val
                    h2e[k-1, l-1, j-1, i-1] = val
                    h2e[l-1, k-1, j-1, i-1] = val
    
    # Get the nuclear repulsion energy
    nuc_repulsion = 0.0
    if hasattr(fcidump, 'e_core'):
        nuc_repulsion = fcidump.e_core
    elif hasattr(fcidump, 'ecore'):
        nuc_repulsion = fcidump.ecore
    else:
        # Extract nuclear repulsion energy from the raw data
        # Format is typically value 0 0 0 0
        for line in open(fd).readlines():
            if '&' in line or len(line.strip()) == 0:
                continue
            data = line.split()
            if len(data) == 5:
                val = float(data[0])
                i, j, k, l = [int(x) for x in data[1:]]
                if i == 0 and j == 0 and k == 0 and l == 0:
                    nuc_repulsion = val
    
    print(f"Nuclear repulsion energy: {nuc_repulsion:.8f}")
    
    # Calculate the one-electron energy contribution using the quantum 1PDM
    one_e_energy = 0.0
    for i in range(n_orb):
        for j in range(n_orb):
            one_e_energy += h1e[i, j] * quantum_pdm1_scaled.real[j, i]
    
    print(f"One-electron energy contribution: {one_e_energy:.8f}")
    
    # For a complete energy calculation, we would need the two-electron energy contribution
    # This would require the 2-RDM, which we don't have from our quantum circuit
    # However, we can make a mean-field approximation using the 1-RDM
    two_e_energy = 0.0
    for i in range(n_orb):
        for j in range(n_orb):
            for k in range(n_orb):
                for l in range(n_orb):
                    # Mean-field approximation: Γ_ijkl ≈ ρ_ik * ρ_jl - ρ_il * ρ_jk
                    # This is the Hartree-Fock approximation to the 2-RDM
                    gamma_ijkl = (quantum_pdm1_scaled.real[i, k] * quantum_pdm1_scaled.real[j, l] - 
                                 quantum_pdm1_scaled.real[i, l] * quantum_pdm1_scaled.real[j, k]) / 2.0
                    two_e_energy += h2e[i, j, k, l] * gamma_ijkl
    
    print(f"Two-electron energy contribution (mean-field approx): {two_e_energy:.8f}")
    
    # Total energy
    total_energy = one_e_energy + two_e_energy + nuc_repulsion
    print(f"Total quantum energy estimate: {total_energy:.8f}")
    print(f"Classical energy for reference: {energy_classical:.8f}")
    print(f"Energy difference: {abs(total_energy - energy_classical):.8f}")
    
except Exception as e:
    print(f"\nError processing FCIDUMP: {e}")
    print("Skipping detailed energy calculation.")
    
    # Fall back to a simple energy estimate
    estimated_h = np.eye(n_sites) * (-10)  # Just a placeholder
    quantum_energy_estimate = np.sum(estimated_h * quantum_pdm1_scaled.real.T)
    print(f"\nRough quantum energy estimate (fallback): {quantum_energy_estimate:.6f}")
    print(f"Classical energy for reference: {energy_classical:.6f}")

print("\nNote: The energy calculation demonstrates how we can compare")
print("the classical and quantum representations of the same state.")
print("A more accurate calculation would require implementing the full")
print("electronic structure Hamiltonian in the quantum circuit.")
