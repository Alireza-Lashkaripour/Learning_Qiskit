import numpy as np
from pyblock3.algebra.mpe import MPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper

# ---------- DMRG ground state -------------------------------------------------
fd  = "H2O.STO3G.FCIDUMP"
ham = Hamiltonian(FCIDUMP(pg="d2h").read(fd), flat=True)
mpo = ham.build_qc_mpo().compress(cutoff=1e-9, norm_cutoff=1e-9)[0]
print("MPO bond‑dims :", mpo.show_bond_dims())

bond_dim = 200
mps = ham.build_mps(bond_dim).canonicalize(center=0)
mps /= mps.norm()

dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1e-6, 0],
                                dav_thrds=[1e-3], iprint=0, n_sweeps=10)
e_cl = dmrg.energies[-1]
print("DMRG energy   :", e_cl)

# ---------- dense tensors with full physical dimension -----------------------
slc = mps.to_non_flat().to_sliceable(info=ham)      # keep zero blocks
dense_tensors = [t.to_dense() for t in slc.tensors]
bond_dims     = [int(x) for x in mps.show_bond_dims().split("|")]
print("MPS bond‑dims :", "|".join(map(str, bond_dims)))

# ---------- contract to full CI vector ---------------------------------------
PHYS = 4
psi = dense_tensors[0].reshape(PHYS, -1)
for A in dense_tensors[1:]:
    psi = np.tensordot(psi, A, axes=([-1], [0])).reshape(-1, A.shape[-1])

def base4_to_qubit_state(phi, n_sites):
    """Convert length 4**N array |φ⟩ to length 2**(2N) JW qubit state."""
    qubit_state = np.zeros(2 ** (2 * n_sites), dtype=phi.dtype)
    for idx, amp in enumerate(phi):
        # unpack index in base‑4
        bits = []
        x = idx
        for _ in range(n_sites):
            d = x & 3        # last base‑4 digit
            x >>= 2
            # map digit → two occupation bits (up, down)
            if d == 0:
                bits += [0, 0]
            elif d == 1:
                bits += [1, 0]
            elif d == 2:
                bits += [0, 1]
            else:
                bits += [1, 1]
        # bits already in Qiskit order: site‑0 ↑ is qubit‑0, site‑0 ↓ is qubit‑1, …
        q_idx = 0
        for i, b in enumerate(bits):
            q_idx |= b << i
        qubit_state[q_idx] = amp
    return qubit_state
state_vec_base4 = psi.ravel() / np.linalg.norm(psi)
state_vec = base4_to_qubit_state(state_vec_base4, ham.n_sites)
# ---------- Hamiltonian and quantum energy -----------------------------------
nq = int(np.log2(state_vec.size))
ham_op = JordanWignerMapper().map(
            fcidump_to_problem(FCIDump.from_file(fd))
            .hamiltonian.second_q_op(),
            register_length=nq)

e_q = np.real(np.vdot(state_vec, ham_op.to_matrix() @ state_vec))

print("Quantum energy :", e_q)
print("Energy diff    :", abs(e_q - e_cl))
