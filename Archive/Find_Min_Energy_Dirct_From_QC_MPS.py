import numpy as np 
from pyblock3.algebra.mpe import MPE 
from pyblock3.hamiltonian import Hamiltonian 
from pyblock3.fcidump import FCIDUMP 
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ



print("Start Pyblock3 calcualtion for classical MPS:")

fd = 'H2O.STO3G.FCIDUMP'
hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
mpo = hamil.build_qc_mpo()
mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
print('MPO (compressed) = ', mpo.show_bond_dims())

bond_dim = 200
mps = hamil.build_mps(bond_dim)
print('MPS = ', mps.show_bond_dims())

print("MPS = ", mps.show_bond_dims())
mps = mps.canonicalize(center=0)
mps /= mps.norm()
print("MPS = ", mps.show_bond_dims())


dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0],
    dav_thrds=[1E-3], iprint=2, n_sweeps=10) # ==> number of opt. sweeps 
ener = dmrg.energies[-1]
print("Energy(Ground State) = %20.12f" % ener)
print('MPS energy = ', np.dot(mps, mpo @ mps))
print('MPS norm = ', mps.norm())
print('DMRG: ', dmrg)
np.save("h2o_energy.npy", ener)

pdm1 = np.zeros((hamil.n_sites, hamil.n_sites))
for i in range(hamil.n_sites):
    diop = OpElement(OpNames.D, (i, 0), q_label=SZ(-1, -1, hamil.orb_sym[i]))
    di = hamil.build_site_mpo(diop)
    for j in range(hamil.n_sites):
        djop = OpElement(OpNames.D, (j, 0), q_label=SZ(-1, -1, hamil.orb_sym[j]))
        dj = hamil.build_site_mpo(djop)
        pdm1[i, j] = 2 * np.dot((di @ mps).conj(), dj @ mps)

print("1PDM calculated from classical MPS:")
print(pdm1)
print("MPS after(bond dim): ", mps.show_bond_dims())
np.save("h2o_pdm1.npy", pdm1)


print("End of PyBlock3 calculation")
print("---------------------------------------------------------------------------")
print("Start Qiskit Calculation: ")


from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

file_path = "H2O.STO3G.FCIDUMP"  
fcidump = FCIDump.from_file(file_path)

problem = fcidump_to_problem(fcidump)
print(problem)

hamiltonian_op = problem.hamiltonian.second_q_op()
print("Fermionic Hamiltonian:", hamiltonian_op)

integrals = problem.hamiltonian.electronic_integrals
one_body_alpha = integrals.alpha["+-"]
print("One-electron integrals (alpha):")
print(one_body_alpha)

mapper = JordanWignerMapper()
qubit_hamiltonian = mapper.map(hamiltonian_op)
print("Qubit Hamiltonian:", qubit_hamiltonian)

# Use NumPyMinimumEigensolver as a substitute for DMRG
solver = NumPyMinimumEigensolver()
ground_state_solver = GroundStateEigensolver(mapper, solver)
result = ground_state_solver.solve(problem)

# Print the ground state energy
print("Ground state energy from exact solver:", result.total_energies[0].real)
