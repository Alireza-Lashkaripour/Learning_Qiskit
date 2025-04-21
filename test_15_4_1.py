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
# ──────────────────── 1.  DMRG ground state ───────────────────────────────────
fd  = "H2O.STO3G.FCIDUMP"
ham = Hamiltonian(FCIDUMP(pg="d2h").read(fd), flat=True)
mpo = ham.build_qc_mpo().compress(cutoff=1e-9, norm_cutoff=1e-9)[0]
bond_dim = 200
mps = ham.build_mps(bond_dim).canonicalize(center=0)
mps /= mps.norm()
dmrg = MPE(mps, mpo, mps).dmrg(
        bdims=[bond_dim], noises=[1e-6, 0],
        dav_thrds=[1e-3], iprint=0, n_sweeps=10)
e_dmrg = dmrg.energies[-1]
print("DMRG energy  :", e_dmrg)
# ──────────────────── 2.  dense site tensors  ─────────────────────────────────
slc = mps.to_non_flat().to_sliceable(info=ham)      # keep zero blocks
tensors = [t.to_dense() for t in slc.tensors]       # (bondL, phys, bondR)
# ──────────────────── 3.  contract to 4ᴺ vector ───────────────────────────────
PHYS = 4
psi = tensors[0].reshape(PHYS, -1)
for A in tensors[1:]:
    psi = np.tensordot(psi, A, axes=([-1], [0]))    # contract bond
    psi = psi.reshape(-1, A.shape[-1])
state_base4 = psi.ravel() / np.linalg.norm(psi)     # length 4**N
# ──────────────────── 4.  map to qubit JW basis ───────────────────────────────
def base4_to_jw(phi, n_sites):
    """
    Convert |φ⟩ in occupation basis (0 v, 1 β, 2 α, 3 αβ) to Jordan‑Wigner qubits.
    Fixed to properly match Qiskit's JW mapping.
    """
    n_qubits = 2 * n_sites
    jw = np.zeros(1 << n_qubits, dtype=phi.dtype)
    
    # Create a mapping dictionary for Qiskit's expected ordering
    # In Qiskit's JW mapping: 
    # - First half of qubits (0 to n_sites-1) represent spin-up (α)
    # - Second half (n_sites to 2*n_sites-1) represent spin-down (β)
    for idx, amp in enumerate(phi):
        qidx = 0
        for s in range(n_sites):
            d = (idx // (4 ** s)) % 4  # Get base-4 digit of site s
            
            # Map digit to JW qubit encoding
            # d=0: vacuum (00), d=1: β (01), d=2: α (10), d=3: αβ (11)
            if d & 0b10:  # Alpha occupied (bit 1)
                qidx |= (1 << s)  # First n_sites qubits for α
            if d & 0b01:  # Beta occupied (bit 0)
                qidx |= (1 << (s + n_sites))  # Second n_sites qubits for β
                
        jw[qidx] = amp
    return jw

state_vec = base4_to_jw(state_base4, ham.n_sites)
state_vec /= np.linalg.norm(state_vec)  # defensive renorm
# ──────────────────── 5.  expectation value  ──────────────────────────────────
nq = int(np.log2(state_vec.size))
# Get the FCIDump and convert to problem
fcidump = FCIDump.from_file(fd)
problem = fcidump_to_problem(fcidump)
# Extract the Hamiltonian operator and map it using Jordan-Wigner transformation
second_q_op = problem.hamiltonian.second_q_op()
ham_op = JordanWignerMapper().map(second_q_op, register_length=nq)
# Calculate the energy expectation value
e_qubit = np.real(np.vdot(state_vec, ham_op.to_matrix() @ state_vec))
# ──────────────────── 6.  report  ─────────────────────────────────────────────
print("Qubit energy :", e_qubit)
print("Difference   :", abs(e_qubit - e_dmrg))
