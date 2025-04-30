import numpy as np
from pyscf import gto, scf, tools

# Define hydrogen atom
mol = gto.Mole()
mol.atom = 'H 0 0 0'
mol.basis = 'cc-pvqz'  # Using a more complete basis for better accuracy
mol.spin = 1  # Doublet state (one unpaired electron)
mol.charge = 0
mol.verbose = 4  # More detailed output
mol.build()

# Run Hartree-Fock calculation
myhf = scf.RHF(mol)
energy = myhf.kernel()

print(f"Hartree-Fock energy: {energy} Hartree")
print(f"For H atom, HF energy should equal exact energy: {energy} Hartree")
print(f"The theoretical exact energy for H atom is -0.5 Hartree")

# Generate FCIDUMP file
tools.fcidump.from_scf(myhf, 'h_atom_fcidump.txt')

print("FCIDUMP file for H atom has been created as 'h_atom_fcidump.txt'")

# If you want to view the content of the file:
with open('h_atom_fcidump.txt', 'r') as f:
    print("\nFCIDUMP file content:")
    print(f.read())
