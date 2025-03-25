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
#print('MPS = ', mps.show_bond_dims())

# Canonicalize MPS
#print("MPS = ", mps.show_bond_dims())
mps = mps.canonicalize(center=0)
mps /= mps.norm()
#print("MPS = ", mps.show_bond_dims())


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

print('---------------------Save_MPS----------------------')
print("MPS after(bond dim): ", mps.show_bond_dims())
#print(mps[0].__class__)
print(mps[0])

#np.save("MPS_tensors_saved.npy", mps[0])
#print('---------------------Save_MPS----------------------')
#print('TensorDot Product of MPS and MPO:', np.tensordot(mps[0], mps[1], axes=1))
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



# Save the complete MPS information
mps_data = {
    'n_sites': hamil.n_sites,
    'bond_dims': [int(dim) for dim in mps.show_bond_dims().split('|')],
    'tensors': [t.data.copy() if hasattr(t, 'data') else t.copy() for t in mps.tensors],
    'q_labels': [t.q_labels if hasattr(t, 'q_labels') else None for t in mps.tensors],
    'energy': ener,
    'pdm1': pdm1
}

np.save("h2o_mps_complete.npy", mps_data, allow_pickle=True)
print('----------------------Now_QC_part------------------------')
mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
n_sites = mps_data['n_sites']
tensors = mps_data['tensors']
bond_dims = mps_data['bond_dims']
q_labels = mps_data['q_labels']
pdm1 = mps_data['pdm1']
energy_classical = mps_data['energy']



print("-----------------------Quantum_Circuit----------------------")


fcidump = FCIDUMP(pg='d2h').read(fd)

h1e = fcidump.h1e
n_sites = h1e.shape[0]
g2e = fcidump.g2e

mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
print("Keys in MPS data:", list(mps_data.keys()))

mps_energy = mps_data['energy']
print(f"\nStored MPS energy: {mps_energy:.8f}")

pdm1_mps = mps_data['pdm1']
print(f"\n1-PDM from MPS data (shape: {pdm1_mps.shape}):")
print(pdm1_mps)
print(f"Trace of 1-PDM: {np.trace(pdm1_mps):.6f}")

energy_1e = np.einsum('ij,ji->', h1e, pdm1_mps)

energy_2e = 0.5 * np.einsum('ijkl,ji,lk->', g2e, pdm1_mps, pdm1_mps)
energy_2e -= 0.25 * np.einsum('ijkl,jk,li->', g2e, pdm1_mps, pdm1_mps)

energy_total = energy_1e + energy_2e + fcidump.const_e

# Print the energy components
print("\nEnergy components calculated from MPS 1-PDM:")
print(f"One-electron energy: {energy_1e:.8f}")
print(f"Two-electron energy: {energy_2e:.8f}")
print(f"Nuclear repulsion:   {fcidump.const_e:.8f}")
print(f"Total energy:        {energy_total:.8f}")
print(f"Difference from MPS energy: {energy_total - mps_energy:.8f}")

