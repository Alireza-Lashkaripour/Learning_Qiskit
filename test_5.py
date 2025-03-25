import numpy as np
from pyblock3.fcidump import FCIDUMP

# Load the FCIDUMP file
fd = 'H2O.STO3G.FCIDUMP'
fcidump = FCIDUMP(pg='d2h').read(fd)

# Get the one-electron integrals
h1e = fcidump.h1e  # One-electron integrals
print("One-electron integrals shape:", h1e.shape)

# Create a dummy density matrix for testing purposes
# For a proper density matrix, all eigenvalues should be between 0 and 1
# and the trace should equal the number of electrons
n_sites = 7
n_electrons = 10  # From NELEC=10 in the FCIDUMP

# Create a simple diagonal density matrix (representing occupied orbitals)
# For water with 10 electrons, we typically have 5 doubly occupied orbitals
pdm1 = np.zeros((n_sites, n_sites))
for i in range(n_electrons // 2):  # Divide by 2 for spin
    pdm1[i, i] = 2.0  # Double occupation (alpha and beta electrons)

print("\nTest density matrix:")
print(pdm1)
print("Trace (should equal electron count):", np.trace(pdm1))

# Calculate one-electron energy contribution
energy_1e = np.einsum('ij,ij->', h1e, pdm1)

# Get the two-electron integrals (if needed for full energy calculation)
g2e = fcidump.g2e
print("Two-electron integrals shape:", g2e.shape)

# Calculate two-electron energy contribution 
# Using physicist's notation (ij|kl) format for two-electron integrals
energy_2e = 0.5 * np.einsum('ijkl,ij,kl->', g2e, pdm1, pdm1) - 0.5 * np.einsum('ijkl,il,kj->', g2e, pdm1, pdm1)

# Total electronic energy
energy_total = energy_1e + energy_2e + fcidump.const_e

print("\nEnergy components:")
print(f"One-electron energy: {energy_1e:.8f}")
print(f"Two-electron energy: {energy_2e:.8f}")
print(f"Nuclear repulsion:   {fcidump.const_e:.8f}")
print(f"Total energy:        {energy_total:.8f}")

# Now, let's try to extract information from the MPS data
try:
    mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
    print("\nKeys in MPS data:", list(mps_data.keys()))
    
    # Check if there's RDM data stored in the MPS file
    if 'rdm1' in mps_data:
        rdm1_from_mps = mps_data['rdm1']
        print("\nUsing 1-RDM from MPS data:")
        print(rdm1_from_mps)
        
        # Calculate energy using the 1-RDM from MPS
        energy_1e_mps = np.einsum('ij,ij->', h1e, rdm1_from_mps)
        energy_2e_mps = 0.5 * np.einsum('ijkl,ij,kl->', g2e, rdm1_from_mps, rdm1_from_mps) - 0.5 * np.einsum('ijkl,il,kj->', g2e, rdm1_from_mps, rdm1_from_mps)
        energy_total_mps = energy_1e_mps + energy_2e_mps + fcidump.const_e
        
        print("\nEnergy from MPS 1-RDM:")
        print(f"One-electron energy: {energy_1e_mps:.8f}")
        print(f"Two-electron energy: {energy_2e_mps:.8f}")
        print(f"Total energy:        {energy_total_mps:.8f}")
except Exception as e:
    print("\nCould not extract 1-RDM from MPS data:", e)
    
# Attempt to read the MPS tensors in a more direct way
try:
    tensors = mps_data['tensors']
    print("\nMPS tensor shapes:")
    for i, t in enumerate(tensors):
        print(f"Tensor {i}: {t.shape}")
        
    # Try to determine the bond dimensions and local dimensions
    if len(tensors) == n_sites:
        print("\nMPS appears to have one tensor per site")
    else:
        print(f"\nMPS has {len(tensors)} tensors for {n_sites} sites")
except Exception as e:
    print("\nError analyzing MPS tensors:", e)
