import numpy as np 
from pyblock3.algebra.mpe import MPE 
from pyblock3.hamiltonian import Hamiltonian 
from pyblock3.fcidump import FCIDUMP 


fd = 'H2O.STO3G.FCIDUMP'
hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
mpo = hamil.build_qc_mpo()
mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
print('MPO (compressed) = ', mpo.show_bond_dims())

# Construct MPS: 
bond_dim = 200
mps = hamil.build_mps(bond_dim)

# Calculate the Expection Value of Hamil , MPS
ExpV = np.dot(mps, mpo @ mps)
print('Expecation Value = ', ExpV) # Just Cheking for myself

# MPS canonicalization 
mps = mps.canonicalize(center=0)
mps /= mps.norm() # Makes it unit vector 
InnerP = np.dot(mps, mps)
print('Inner Product = ', InnerP) # Just checking 

# DMRG
dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0],
    dav_thrds=[1E-3], iprint=2, n_sweeps=10) # ==> number of opt. sweeps 
ener = dmrg.energies[-1]
print("Energy(Ground State) = %20.12f" % ener)
# Check ground-state energy: 
print('MPS energy = ', np.dot(mps, mpo @ mps))
# Check that ground-state MPS is normalized:
print('MPS = ', mps.show_bond_dims())
print('MPS norm = ', mps.norm())


