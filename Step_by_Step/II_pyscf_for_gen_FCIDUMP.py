import numpy as np
from pyscf import gto, scf, tools

# Define helium atom
mol = gto.Mole()
mol.atom = 'He 0 0 0'
mol.basis = 'cc-pvqz'  # Using a high-quality basis set
mol.spin = 0  # Singlet state (paired electrons)
mol.charge = 0
mol.verbose = 4  # More detailed output
mol.build()

# Run Hartree-Fock calculation
myhf = scf.RHF(mol)
hf_energy = myhf.kernel()

print(f"Hartree-Fock energy for He atom: {hf_energy} Hartree")
print(f"Theoretical exact energy for He atom: -2.903724 Hartree")
print(f"HF energy differs from exact due to electron correlation")

# Generate FCIDUMP file
tools.fcidump.from_scf(myhf, 'he_atom_fcidump.txt')

print("FCIDUMP file for He atom has been created as 'he_atom_fcidump.txt'")

# If you want to view the content of the file:
with open('he_atom_fcidump.txt', 'r') as f:
    print("\nFCIDUMP file content:")
    print(f.read())

# Optional: Run a more accurate calculation using FCI
try:
    from pyscf import fci
    cisolver = fci.FCI(myhf)
    fci_energy = cisolver.kernel()[0]
    print(f"\nFCI energy for He atom: {fci_energy} Hartree")
    print(f"This should be much closer to the exact energy within the basis set limit")
except ImportError:
    print("\nFCI module not available. Only Hartree-Fock energy calculated.")
