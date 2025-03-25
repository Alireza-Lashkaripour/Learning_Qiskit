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
mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
n_sites = mps_data['n_sites']
tensors = mps_data['tensors']
bond_dims = mps_data['bond_dims']
q_labels = mps_data['q_labels']
pdm1 = mps_data['pdm1']
energy_classical = mps_data['energy']


# Continue from where your code left off
print('----------------------Quantum_Circuit_Mapping------------------------')

# Import necessary libraries
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from pyblock3.fcidump import FCIDUMP
import numpy as np

fd = 'H2O.STO3G.FCIDUMP'
fcidump = FCIDUMP(pg='d2h').read(fd)

n_qubits = n_sites
qc = QuantumCircuit(n_qubits)

print("Analyzing orbital occupations from 1PDM...")
occupations = np.diag(pdm1) / 2.0  # Divide by 2 since each spatial orbital holds 2 electrons
print("Orbital occupations:", occupations)

sorted_indices = np.argsort(occupations)[::-1]  # Descending order
print("Orbitals sorted by occupation:", sorted_indices)
print("Sorted occupations:", occupations[sorted_indices])

n_electrons = 7  # Number of spatial orbitals that should be occupied
occupied_orbitals = sorted_indices[:n_electrons]
print(f"Occupied orbitals (top {n_electrons}):", occupied_orbitals)

print("Preparing Hartree-Fock-like reference state...")
for orbital in occupied_orbitals:
    qc.x(orbital)  

print("Adding variational correlation layer...")
for i in range(n_qubits):
    if i in occupied_orbitals:
        angle = np.pi * (1.0 - occupations[i]) * 0.2
    else:
        angle = np.pi * occupations[i] * 0.2
    
    qc.ry(angle, i)

for i in range(n_qubits):
    for j in range(i+1, n_qubits):
        if abs(pdm1[i, j]) > 0.005:
            angle = 0.1 * pdm1[i, j]
            qc.cx(i, j)
            qc.rz(angle, j)
            qc.cx(i, j)

print(qc)

print("Creating fermionic Hamiltonian...")

h1 = fcidump.h1e
h2 = fcidump.g2e
core_energy = fcidump.const_e

fermion_op = FermionicOp({})

fermion_op += FermionicOp({"": core_energy})

print("Adding one-body terms...")
for i in range(n_qubits):
    for j in range(n_qubits):
        if abs(h1[i, j]) > 1e-10:
            term = f"+_{i} -_{j}"
            fermion_op += FermionicOp({term: h1[i, j]})

print("Adding two-body terms...")
importance_list = []
for i in range(n_qubits):
    for j in range(n_qubits):
        for k in range(n_qubits):
            for l in range(n_qubits):
                if abs(h2[i, j, k, l]) > 1e-6:
                    importance_list.append((abs(h2[i, j, k, l]), i, j, k, l))

importance_list.sort(reverse=True)
top_terms = importance_list[:1000]  # Increased to capture more terms
print(f"Including top {len(top_terms)} two-body terms out of {len(importance_list)} total")

for val, i, j, k, l in top_terms:
    term = f"+_{i} +_{j} -_{l} -_{k}"
    fermion_op += FermionicOp({term: 0.5 * h2[i, j, k, l]})

print("Mapping to qubit operators...")
jw_mapper = JordanWignerMapper()
qubit_op = jw_mapper.map(fermion_op)
print(f"Hamiltonian mapped to {len(qubit_op)} Pauli terms")

print("Simulating quantum circuit...")
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
statevector = Statevector.from_instruction(compiled_circuit)

energy_quantum = statevector.expectation_value(qubit_op)
print(f"Quantum circuit energy = {energy_quantum.real}")
print(f"Classical DMRG energy = {energy_classical}")
print(f"Energy difference = {abs(energy_quantum.real - energy_classical)}")
print(f"Relative error = {abs(energy_quantum.real - energy_classical)/abs(energy_classical)*100:.4f}%")

print("Calculating quantum 1PDM...")
quantum_pdm1 = np.zeros((n_qubits, n_qubits))
for i in range(n_qubits):
    for j in range(n_qubits):
        if i == j:
            number_op = FermionicOp({f"+_{i} -_{i}": 1.0})
            qubit_number_op = jw_mapper.map(number_op)
            quantum_pdm1[i, j] = statevector.expectation_value(qubit_number_op).real
        else:
            transition_op = FermionicOp({f"+_{i} -_{j}": 1.0})
            qubit_transition_op = jw_mapper.map(transition_op)
            quantum_pdm1[i, j] = statevector.expectation_value(qubit_transition_op).real

print("Quantum 1PDM:")
print(quantum_pdm1)
print("Classical 1PDM:")
print(pdm1 / 2)  # Divide by 2 to compare with quantum (no spin doubling)

n_electrons_quantum = np.sum(np.diag(quantum_pdm1))
print(f"Total electrons in quantum state: {n_electrons_quantum}")
print(f"Total electrons in classical state: {np.sum(np.diag(pdm1/2))}")

print("Analyzing energy components...")
HF_energy = -74.96  # Approximate HF energy for H2O/STO-3G

print(f"HF reference energy = {HF_energy}")
print(f"DMRG correlation energy = {energy_classical - HF_energy}")
print(f"Quantum correlation energy = {energy_quantum.real - HF_energy}")
print(f"Correlation energy recovery = {(energy_quantum.real - HF_energy)/(energy_classical - HF_energy)*100:.2f}%")

trace_distance = 0.5 * np.sum(np.abs(quantum_pdm1 - pdm1/2))
print(f"Trace distance between quantum and classical 1PDMs: {trace_distance}")
