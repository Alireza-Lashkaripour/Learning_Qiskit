import numpy as np
from pyblock3.fcidump import FCIDUMP

# Load the FCIDUMP file
fd = 'H2O.STO3G.FCIDUMP'
fcidump = FCIDUMP(pg='d2h').read(fd)

# Get the one-electron integrals and other data
h1e = fcidump.h1e
n_sites = h1e.shape[0]
g2e = fcidump.g2e

# Load the MPS data with the actual density matrix
mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
print("Keys in MPS data:", list(mps_data.keys()))

# Extract the stored energy value from MPS for reference
mps_energy = mps_data['energy']
print(f"\nStored MPS energy: {mps_energy:.8f}")

# Extract the 1-particle density matrix from MPS data
pdm1_mps = mps_data['pdm1']
print(f"\n1-PDM from MPS data (shape: {pdm1_mps.shape}):")
print(pdm1_mps)
print(f"Trace of 1-PDM: {np.trace(pdm1_mps):.6f}")

# Calculate one-electron energy contribution using MPS density matrix
energy_1e = np.einsum('ij,ji->', h1e, pdm1_mps)

# Calculate two-electron energy contribution
# For two-electron integrals, we need to be careful about the format
# FCIDUMP typically uses chemist's notation (ij|kl) which is g2e[i,j,k,l]
# The contraction for energy uses physicist's notation <ij|kl> = (ik|jl)
# So we need to transpose indices appropriately
energy_2e = 0.5 * np.einsum('ijkl,ji,lk->', g2e, pdm1_mps, pdm1_mps)
energy_2e -= 0.25 * np.einsum('ijkl,jk,li->', g2e, pdm1_mps, pdm1_mps)

# Total electronic energy
energy_total = energy_1e + energy_2e + fcidump.const_e

# Print the energy components
print("\nEnergy components calculated from MPS 1-PDM:")
print(f"One-electron energy: {energy_1e:.8f}")
print(f"Two-electron energy: {energy_2e:.8f}")
print(f"Nuclear repulsion:   {fcidump.const_e:.8f}")
print(f"Total energy:        {energy_total:.8f}")
print(f"Difference from MPS energy: {energy_total - mps_energy:.8f}")

# Try alternative einsum contractions for the two-electron part
# Sometimes the convention for handling two-electron integrals varies
print("\nTrying alternative two-electron energy calculations:")

# Version 1 - Standard direct and exchange terms
e2_v1 = 0.5 * np.einsum('ijkl,ji,lk->', g2e, pdm1_mps, pdm1_mps)
e2_v1 -= 0.25 * np.einsum('ijkl,jk,li->', g2e, pdm1_mps, pdm1_mps)
print(f"Version 1: {e2_v1:.8f}")

# Version 2 - With different index ordering
e2_v2 = 0.5 * np.einsum('ijkl,ij,kl->', g2e, pdm1_mps, pdm1_mps)
e2_v2 -= 0.25 * np.einsum('ijkl,il,kj->', g2e, pdm1_mps, pdm1_mps)
print(f"Version 2: {e2_v2:.8f}")

# Version 3 - With transposition of indices in g2e
g2e_phys = np.transpose(g2e, (0, 2, 1, 3))  # Convert to physicist notation
e2_v3 = 0.5 * np.einsum('ikjl,ij,kl->', g2e_phys, pdm1_mps, pdm1_mps)
e2_v3 -= 0.25 * np.einsum('ikjl,il,kj->', g2e_phys, pdm1_mps, pdm1_mps)
print(f"Version 3: {e2_v3:.8f}")

# Try pyblock3's built-in energy calculation if available
try:
    # Check if pyblock3 has a direct energy calculation function
    if hasattr(fcidump, 'energy') and callable(getattr(fcidump, 'energy')):
        energy_pyblock = fcidump.energy(pdm1_mps)
        print(f"\nEnergy calculated by pyblock3: {energy_pyblock:.8f}")
        print(f"Difference from MPS energy: {energy_pyblock - mps_energy:.8f}")
except Exception as e:
    print(f"\nCould not use pyblock3's energy calculation: {e}")

# Check if we need to apply any symmetry considerations
print("\nChecking for symmetry information:")
if hasattr(fcidump, 'orbsym'):
    print(f"Orbital symmetry: {fcidump.orbsym}")
if hasattr(fcidump, 'isym'):
    print(f"Total symmetry: {fcidump.isym}")

# Final energy calculation based on best contract method
best_e2 = e2_v2  # Adjust based on results
best_energy = energy_1e + best_e2 + fcidump.const_e
print(f"\nBest energy estimate: {best_energy:.8f}")
print(f"Difference from MPS energy: {best_energy - mps_energy:.8f}")
