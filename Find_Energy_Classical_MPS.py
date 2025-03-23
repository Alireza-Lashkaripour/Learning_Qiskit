import numpy as np 
from pyblock3.algebra.mpe import MPE 
from pyblock3.hamiltonian import Hamiltonian 
from pyblock3.fcidump import FCIDUMP 
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ

fd = 'H2O.STO3G.FCIDUMP'
hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
mpo = hamil.build_qc_mpo()
mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
print('MPO (compressed) = ', mpo.show_bond_dims())

# Construct MPS: 
bond_dim = 200
mps = hamil.build_mps(bond_dim)
# Check that ground-state MPS is normalized:
print('MPS = ', mps.show_bond_dims())

# ---------------------------------------------
# Calculate the Expection Value of Hamil , MPS
#ExpV = np.dot(mps, mpo @ mps)
#print('Expecation Value = ', ExpV) # Just Cheking for myself
# MPS canonicalization 
#mps = mps.canonicalize(center=0)
#mps /= mps.norm() # Makes it unit vector 
#InnerP = np.dot(mps, mps)
#print('Inner Product = ', InnerP) # Just checking 
# -----------------------------------------------

# Canonicalize MPS
print("MPS = ", mps.show_bond_dims())
mps = mps.canonicalize(center=0)
mps /= mps.norm()
print("MPS = ", mps.show_bond_dims())


# DMRG
dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0],
    dav_thrds=[1E-3], iprint=2, n_sweeps=10) # ==> number of opt. sweeps 
ener = dmrg.energies[-1]
print("Energy(Ground State) = %20.12f" % ener)
# Check ground-state energy: 
print('MPS energy = ', np.dot(mps, mpo @ mps))
print('MPS norm = ', mps.norm())
print('DMRG: ', dmrg)
############
# This part is for 'One Particle Density Matrix', based on the exaplin in github
np.save("h2o_energy.npy", ener)

# Now saving the 1PDM and MPS details( like in example)
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

