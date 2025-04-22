
import numpy as np
from pyscf import gto, scf, tools

# Define H2 molecule - simple hydrogen molecule with 0.74 Angstrom bond length
mol = gto.Mole()
mol.atom = '''
H 0 0 0
H 0 0 0.74
'''
mol.basis = 'sto-3g'  # Using STO-3G basis set
mol.spin = 0  # Singlet state
mol.build()

# Run Hartree-Fock calculation
myhf = scf.RHF(mol)
myhf.kernel()

# Generate FCIDUMP file
tools.fcidump.from_scf(myhf, 'h2_fcidump.txt')

print("FCIDUMP file for H2 has been created as 'h2_fcidump.txt'")

# If you want to view the content of the file:
with open('h2_fcidump.txt', 'r') as f:
    print("\nFCIDUMP file content:")
    print(f.read())
