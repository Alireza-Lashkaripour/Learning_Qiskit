import numpy as np 
import matplotlib.pyplot as plt
from pyblock3.algebra.mpe import MPE 
from pyblock3.hamiltonian import Hamiltonian 
from pyblock3.fcidump import FCIDUMP 
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate

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
print('MPS 0:')
print(mps[0])
print('MPS 1:')
print(mps[1])
# Canonicalize MPS
#print("MPS = ", mps.show_bond_dims())
mps = mps.canonicalize(center=0)
mps /= mps.norm()
#print("MPS = ", mps.show_bond_dims())
print('MPS 0:')
print(mps[0])
print('MPS 1:')
print(mps[1])
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
print('MPS 0:')
print(mps[0])
print('MPS 1:')
print(mps[1])
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



def get_dense(t):
    if hasattr(t, "to_numpy"):
        return t.to_numpy()
    elif hasattr(t, "data"):
        return t.data
    else:
        raise AttributeError("Tensor does not support dense conversion")

def print_mps_details(mps):
    for i, tensor in enumerate(mps):
        print("Tensor[{}] details:".format(i))
        if hasattr(tensor, "q_labels"):
            print("  q_labels:", tensor.q_labels)
        else:
            print("  q_labels attribute not found.")
        if hasattr(tensor, "blocks"):
            for j, block in enumerate(tensor.blocks):
                print("  Block {}:".format(j))
                print("    q_labels:", block.q_labels)
                print("    Shape:", block.array.shape)
                print("    Contents:")
                print(block.array)
        else:
            try:
                arr = get_dense(tensor)
                print("  Dense representation shape:", arr.shape)
                print("  Dense representation:")
                print(arr)
            except Exception as e:
                print("  Cannot get dense representation:", e)
        print("="*50)

def make_unitary(M):
    U, s, Vh = np.linalg.svd(M)
    return np.dot(U, Vh)

def construct_MPD_from_tensor(A):
    A_arr = get_dense(A)
    d = A_arr.shape[0]
    gate0 = A_arr[0].reshape(-1)
    gate0 = gate0/np.linalg.norm(gate0)
    total_dim = np.prod(A_arr.shape)
    rand_mat = np.random.randn(total_dim, total_dim)
    Q, R = np.linalg.qr(rand_mat)
    Q[:, 0] = gate0
    Q, _ = np.linalg.qr(Q)
    U_mat = Q
    d_inner = int(np.prod(A_arr.shape[1:]))
    dim_sq = d * d_inner
    M = U_mat.reshape(dim_sq, dim_sq)
    M_unitary = make_unitary(M)
    return M_unitary

def build_quantum_circuit_from_MPS(mps):
    gate_list = []
    G_last = get_dense(mps[-1])
    if G_last.ndim < 2:
        dim = int(np.sqrt(G_last.size))
        G_last = G_last.reshape((dim, dim))
    G_last = make_unitary(G_last)
    gate_list.append(G_last)
    for A in mps[:-1]:
        G = construct_MPD_from_tensor(A)
        if G.ndim < 2:
            dim = int(np.sqrt(G.size))
            G = G.reshape((dim, dim))
        G = make_unitary(G)
        gate_list.append(G)
    return gate_list

def compute_energy(rho, H):
    return np.trace(np.dot(rho, H)).real

print_mps_details(mps)
gate_list = build_quantum_circuit_from_MPS(mps)
n_qubits = len(gate_list) + 1
qc = QuantumCircuit(n_qubits)
for i, G in enumerate(gate_list):
    if G.shape == (4, 4):
        qc.unitary(UnitaryGate(G), [i, i+1])
    elif G.shape == (2, 2):
        qc.unitary(UnitaryGate(G), [i])
sv = Statevector.from_instruction(qc)
psi = sv.data
rho = np.outer(psi, psi.conj())
dim = rho.shape[0]
try:
    H = hamil.to_matrix()
except AttributeError:
    H = np.diag([ener]*dim)
energy_qc = compute_energy(rho, H)
energy_dmrg = ener
print("Ground state energy from quantum circuit:", energy_qc)
print("Ground state energy from DMRG:", energy_dmrg)
print(qc.draw(output='text'))
qc.draw(output='mpl')
plt.show()

