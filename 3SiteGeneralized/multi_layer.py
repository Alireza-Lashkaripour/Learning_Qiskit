import numpy as np
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit

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

def complete_gate_for_site(tensor, typ):
    d = 2
    if typ == "first":
        v = tensor.reshape(-1)
        nrm = np.linalg.norm(v)
        if nrm != 0:
            v = v / nrm
        return complete_unitary_from_fixed_row(v)
    if typ == "intermediate":
        M = np.zeros((d, d*d), dtype=complex)
        for j in range(d):
            vec = tensor[j, :, :].reshape(-1)
            nrm = np.linalg.norm(vec)
            if nrm != 0:
                vec = vec / nrm
            M[j, :] = vec
        return complete_unitary_from_fixed_rows(M)
    if typ == "last":
        v = tensor.reshape(-1)
        nrm = np.linalg.norm(v)
        if nrm != 0:
            v = v / nrm
        return complete_unitary_from_fixed_row(v)

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

def truncate_mps(mps, d=2):
    num_sites = len(mps)
    new_mps = []
    new_mps.append(mps[0][:, :d])
    for A in mps[1:-1]:
        new_mps.append(A[:, :d, :d])
    new_mps.append(mps[-1][:, :d])
    return new_mps

def deep_mps_to_circuit(mps, d=2, layers=2):
    num_sites = len(mps)
    num_q = num_sites + 1
    deep_qc = QuantumCircuit(num_q)
    current_mps = mps
    for layer in range(layers):
        truncated = truncate_mps(current_mps, d)
        gates = []
        U_first = complete_gate_for_site(truncated[0], "first")
        gates.append(U_first)
        for n in range(1, num_sites - 1):
            U_int = complete_gate_for_site(truncated[n], "intermediate")
            gates.append(U_int)
        U_last = complete_gate_for_site(truncated[-1], "last")
        gates.append(U_last)
        layer_qc = QuantumCircuit(num_q)
        layer_qc.append(UnitaryGate(gates[0].conj().T, label=f"U1†_L{layer+1}"), [0, 1])
        for n in range(1, num_sites - 1):
            layer_qc.append(UnitaryGate(gates[n].conj().T, label=f"U{n+1}†_L{layer+1}"), [n, n+1])
        layer_qc.append(UnitaryGate(gates[-1].conj().T, label=f"U{num_sites}†_L{layer+1}"), [num_q - 1, num_q - 2])
        deep_qc = deep_qc.compose(layer_qc)
        current_mps = truncated
    return deep_qc

if __name__ == "__main__":
    num_sites = 6
    bond_dim = 8
    physical_dim = 2
    layers = 3
    mps = generate_random_mps(num_sites, bond_dim, physical_dim)
    qc = deep_mps_to_circuit(mps, physical_dim, layers)
    print(qc.draw(output="text", reverse_bits=True))

