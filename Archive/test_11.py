import numpy as np
from scipy.linalg import null_space
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit

def complete_unitary_from_fixed_row(v):
    n = v.shape[0]
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    X[:, 0] = v.conj()
    Q, _ = np.linalg.qr(X)
    phase = np.vdot(Q[:, 0], v.conj())
    Q[:, 0] *= phase.conjugate() / abs(phase)
    U = Q.conj().T
    return U

def complete_unitary_from_fixed_rows(M):
    m, n = M.shape
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    X[:, :m] = M.conj().T
    Q, _ = np.linalg.qr(X)
    for j in range(m):
        phase = np.vdot(Q[:, j], M[j, :].conj())
        Q[:, j] *= phase.conjugate() / abs(phase)
    U = Q.conj().T
    return U

def complete_gate_for_site(tensor, typ):
    d = 2
    if typ == "first":
        v = tensor.reshape(-1)
        norm = np.linalg.norm(v)
        if norm != 0:
            v = v / norm
        U = complete_unitary_from_fixed_row(v)
        return U
    if typ == "intermediate":
        M = np.zeros((d, d * d), dtype=complex)
        for j in range(d):
            vec = tensor[j, :, :].reshape(-1)
            norm = np.linalg.norm(vec)
            if norm != 0:
                vec = vec / norm
            M[j, :] = vec
        U = complete_unitary_from_fixed_rows(M)
        return U
    if typ == "last":
        v = tensor[0, :].reshape(-1)
        norm = np.linalg.norm(v)
        if norm != 0:
            v = v / norm
        U = complete_unitary_from_fixed_row(v)
        return U

def generate_random_mps(num_sites, bond_dim, d=2):
    mps = []
    A1 = np.random.rand(d, bond_dim) + 1j * np.random.rand(d, bond_dim)
    A1 = A1 / np.linalg.norm(A1)
    mps.append(A1)
    for _ in range(num_sites - 2):
        A = np.random.rand(d, bond_dim, bond_dim) + 1j * np.random.rand(d, bond_dim, bond_dim)
        A = A / np.linalg.norm(A)
        mps.append(A)
    A_last = np.random.rand(d, bond_dim) + 1j * np.random.rand(d, bond_dim)
    A_last = A_last / np.linalg.norm(A_last)
    mps.append(A_last)
    return mps

def mps_to_circuit(mps_tensors, d=2):
    num_sites = len(mps_tensors)
    chi = mps_tensors[0].shape[1]
    if chi != d:
        new_tensors = []
        new_tensors.append(mps_tensors[0][:, :d])
        for A in mps_tensors[1:-1]:
            new_tensors.append(A[:, :d, :d])
        new_tensors.append(mps_tensors[-1][:, :d])
        mps_tensors = new_tensors
    gates = []
    U_first = complete_gate_for_site(mps_tensors[0], "first")
    gates.append(U_first)
    for n in range(1, num_sites - 1):
        U_int = complete_gate_for_site(mps_tensors[n], "intermediate")
        gates.append(U_int)
    U_last = complete_gate_for_site(mps_tensors[-1], "last")
    gates.append(U_last)
    num_q = num_sites
    qc = QuantumCircuit(num_q)
    qc.append(UnitaryGate(gates[0].conj().T, label="U1†"), [0, 1])
    for n in range(1, num_sites - 1):
        qc.append(UnitaryGate(gates[n].conj().T, label=f"U{n+1}†"), [n, n+1])
    qc.append(UnitaryGate(gates[-1].conj().T, label=f"U{num_sites}†"), [num_q - 1])
    return qc

if __name__ == "__main__":
    num_sites = 12
    bond_dim = 4
    physical_dim = 2
    mps = generate_random_mps(num_sites, bond_dim, physical_dim)
    qc = mps_to_circuit(mps, physical_dim)
    print(qc.draw(output="text", reverse_bits=True))

