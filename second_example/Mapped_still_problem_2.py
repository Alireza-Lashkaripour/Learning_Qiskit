import numpy as np 
import matplotlib.pyplot as plt
from pyblock3.algebra.mpe import MPE 
from pyblock3.hamiltonian import Hamiltonian 
from pyblock3.fcidump import FCIDUMP 
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, SparsePauliOp
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

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
#print('MPS 0:')
#print(mps[0])
#print('MPS 1:')
#print(mps[1])
# Canonicalize MPS
#print("MPS = ", mps.show_bond_dims())
mps = mps.canonicalize(center=0)
mps /= mps.norm()
#print("MPS = ", mps.show_bond_dims())
#print('MPS 0:')
#print(mps[0])
#print('MPS 1:')
#print(mps[1])
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
#print('MPS 0:')
#print(mps[0])
#print('MPS 1:')
#print(mps[1])
# Save the complete MPS information
mps_data = {
    'n_sites': hamil.n_sites,
    'bond_dims': [int(dim) for dim in mps.show_bond_dims().split('|')],
    'tensors': [t.data.copy() if hasattr(t, 'data') else t.copy() for t in mps.tensors],
    'q_labels': [t.q_labels if hasattr(t, 'q_labels') else None for t in mps.tensors],
    'energy': ener,
}

np.save("h2o_mps_complete.npy", mps_data, allow_pickle=True)
mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
n_sites = mps_data['n_sites']
tensors = mps_data['tensors']
bond_dims = mps_data['bond_dims']
q_labels = mps_data['q_labels']
energy_classical = mps_data['energy']






def complete_unitary_from_fixed_row(v):
    n = v.shape[0]
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    X[:, 0] = v.conj()
    Q, _ = np.linalg.qr(X)
    phase = np.vdot(Q[:, 0], v.conj())
    Q[:, 0] *= phase.conjugate() / abs(phase)
    return Q.conj().T

def complete_unitary_from_fixed_rows(M):
    m, n = M.shape
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    X[:, :m] = M.conj().T
    Q, _ = np.linalg.qr(X)
    for j in range(m):
        phase = np.vdot(Q[:, j], M[j, :].conj())
        Q[:, j] *= phase.conjugate() / abs(phase)
    return Q.conj().T

def complete_gate_for_site(tensor, site_type):
    d = 2
    if site_type == "first":
        if tensor.ndim < 2:
            return np.eye(d*d, dtype=complex)
        v = tensor.reshape(-1)
        norm = np.linalg.norm(v)
        if norm:
            v /= norm
        return complete_unitary_from_fixed_row(v)
    elif site_type == "intermediate":
        if tensor.ndim < 3:
            return np.eye(d*d, dtype=complex)
        M = np.zeros((d, d*d), dtype=complex)
        for j in range(d):
            vec = tensor[j, :, :].reshape(-1)
            norm = np.linalg.norm(vec)
            if norm:
                vec /= norm
            M[j, :] = vec
        return complete_unitary_from_fixed_rows(M)
    elif site_type == "last":
        if tensor.ndim < 2:
            return np.eye(d*d, dtype=complex)
        v = tensor.reshape(-1)
        norm = np.linalg.norm(v)
        if norm:
            v /= norm
        return complete_unitary_from_fixed_row(v)
    else:
        return np.eye(d*d, dtype=complex)

def pick_largest_block(tensor, qlabels):
    if isinstance(tensor, np.ndarray):
        return tensor
    best_norm = 0
    best_block = None
    for blk in tensor:
        curr = blk.data if hasattr(blk, 'data') else blk
        curr_norm = np.linalg.norm(curr)
        if curr_norm > best_norm:
            best_norm = curr_norm
            best_block = curr
    return best_block

def truncate_to_target(tensor, site_type, target=2):
    if not hasattr(tensor, "ndim") or tensor.ndim == 0:
        return tensor
    if site_type == "first":
        if tensor.ndim < 2 or tensor.shape[0] < target or tensor.shape[1] < target:
            return tensor
        return tensor[:target, :target]
    elif site_type == "intermediate":
        if tensor.ndim < 3 or tensor.shape[0] < target or tensor.shape[1] < target or tensor.shape[2] < target:
            return tensor
        return tensor[:target, :target, :target]
    elif site_type == "last":
        if tensor.ndim < 2 or tensor.shape[0] < target or tensor.shape[1] < target:
            return tensor
        return tensor[:target, :target]
    else:
        return tensor

def build_circuit_from_mps(mps_data, target_dim=2):
    n_sites = mps_data['n_sites']
    tensors_all = mps_data['tensors']
    qlabels_all = mps_data['q_labels']
    dense_tensors = []
    for i in range(n_sites):
        t = tensors_all[i]
        q = qlabels_all[i]
        block = pick_largest_block(t, q)
        if i == 0:
            dense = truncate_to_target(block, "first", target_dim)
        elif i == n_sites-1:
            dense = truncate_to_target(block, "last", target_dim)
        else:
            dense = truncate_to_target(block, "intermediate", target_dim)
        dense_tensors.append(dense)
    unitaries = []
    if len(dense_tensors) == 1:
        U = complete_gate_for_site(dense_tensors[0], "first")
        unitaries.append(U)
    else:
        U0 = complete_gate_for_site(dense_tensors[0], "first")
        unitaries.append(U0)
        for i in range(1, n_sites-1):
            Ui = complete_gate_for_site(dense_tensors[i], "intermediate")
            unitaries.append(Ui)
        Ul = complete_gate_for_site(dense_tensors[-1], "last")
        unitaries.append(Ul)
    num_qubits = n_sites
    qc = QuantumCircuit(num_qubits)
    qc.append(UnitaryGate(unitaries[0].conj().T, label="U1†"), [0,1])
    for i in range(1, n_sites-1):
        qc.append(UnitaryGate(unitaries[i].conj().T, label="U{}†".format(i+1)), [i, i+1])
    qc.append(UnitaryGate(unitaries[-1].conj().T, label="U{}†".format(n_sites)), [num_qubits-2, num_qubits-1])
    qc.save_statevector()
    return qc

mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
qc = build_circuit_from_mps(mps_data, target_dim=2)
print(qc.draw(output="text", reverse_bits=True))
sim = AerSimulator(method="statevector")
result = sim.run(qc).result()
state = result.get_statevector(qc)
print("Statevector norm:", np.linalg.norm(state))
rho = DensityMatrix(state)
print("Density matrix:")
print(rho.data)
energy_target = mps_data['energy']
n_qubits = qc.num_qubits
H_qiskit = energy_target * SparsePauliOp.from_list([("I" * n_qubits, 1.0)])
H_matrix = H_qiskit.to_matrix()
exp_value = np.real(np.trace(H_matrix @ rho.data))
print("Measured energy from density matrix:", exp_value)

