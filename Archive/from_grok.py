import numpy as np
from scipy.linalg import svd, qr
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, SparsePauliOp
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit import transpile

mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
n_sites = mps_data['n_sites']
tensors = mps_data['tensors']
energy_classical = mps_data['energy']
bond_dims = mps_data['bond_dims']

def truncate_mps(tensors, bond_dims, chi=2):
    new_tensors = []
    for n in range(len(tensors)):
        tensor = tensors[n]
        if n == 0:
            total_size = tensor.size
            D1 = bond_dims[1]
            d = total_size // D1
            if tensor.ndim == 1:
                tensor = tensor.reshape(d, D1)
            U, S, Vh = svd(tensor, full_matrices=False)
            k = min(chi, len(S))
            new_tensor = U[:, :k]
        elif n == len(tensors) - 1:
            total_size = tensor.size
            DN1 = bond_dims[n]
            d = total_size // DN1
            if tensor.ndim == 1:
                tensor = tensor.reshape(DN1, d)
            tensor = tensor.T
            U, S, Vh = svd(tensor, full_matrices=False)
            k = min(chi, len(S))
            new_tensor = Vh[:k, :]
        else:
            Dn = bond_dims[n]
            Dn1 = bond_dims[n+1]
            total_size = tensor.size
            d = max(1, total_size // (Dn * Dn1))
            if tensor.ndim == 1:
                padded_size = Dn * d * Dn1
                if total_size < padded_size:
                    tensor = np.pad(tensor, (0, padded_size - total_size), 'constant')
                elif total_size > padded_size:
                    tensor = tensor[:padded_size]
                tensor = tensor.reshape(Dn, d, Dn1)
            tensor = tensor.reshape(Dn * d, Dn1)
            U, S, Vh = svd(tensor, full_matrices=False)
            k = min(chi, len(S))
            new_tensor = U[:, :k].reshape(Dn, d, k)
        new_tensors.append(new_tensor)
    return new_tensors

def null_space(A, rcond=1e-15):
    u, s, vh = svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[0]
    rcond = np.max(s) * rcond
    tol = s > rcond
    num = np.sum(tol, dtype=int)
    Q = vh[num:, :].T.conj()
    return Q

def construct_mpd(tensors, n_qubits):
    qc = QuantumCircuit(n_qubits)
    for n in range(min(len(tensors), n_qubits)):
        tensor = tensors[n]
        A = tensor if n == 0 else tensor.reshape(-1, tensor.shape[-1])
        d = A.shape[0] if A.ndim > 1 else 1
        k = A.shape[1] if A.ndim > 1 else 1
        U = np.eye(4, dtype=complex)
        if A.ndim > 1:
            min_dim = min(4, min(d, k))
            U[:min_dim, :min_dim] = A[:min_dim, :min_dim]
        else:
            U[0, 0] = A[0] if A.size > 0 else 0
        Q, R = qr(U)
        gate = UnitaryGate(Q)
        if n == n_qubits - 1:
            qc.append(gate, [n_qubits - 1 - n])
        else:
            qc.append(gate, [n_qubits - 1 - n, n_qubits - 2 - n])
    return qc

def build_deep_circuit(tensors_original, bond_dims, D=4, n_qubits=6):
    qc_total = QuantumCircuit(n_qubits)
    current_tensors = tensors_original.copy()
    for k in range(D):
        truncated_tensors = truncate_mps(current_tensors, bond_dims, chi=2)
        mpd_qc = construct_mpd(truncated_tensors, n_qubits)
        qc_total.compose(mpd_qc, inplace=True)
        current_tensors = truncated_tensors
    print("Quantum Circuit:", qc_total)
    return qc_total

D = 4
n_qubits = D + 2
qc = build_deep_circuit(tensors, bond_dims, D, n_qubits)

simulator = AerSimulator(method='statevector')
qc_save = qc.copy()
qc_save.save_statevector()
compiled_circuit = transpile(qc_save, simulator)
result = simulator.run(compiled_circuit).result()
statevector = result.get_statevector()

rho = DensityMatrix(statevector)

pauli_terms = []
coefficients = []
for i in range(n_qubits):
    op_str = ['I'] * n_qubits
    op_str[i] = 'Z'
    pauli_terms.append(''.join(op_str))
    coefficients.append(0.1)
for i in range(n_qubits - 1):
    op_str = ['I'] * n_qubits
    op_str[i] = 'Z'
    op_str[i + 1] = 'Z'
    pauli_terms.append(''.join(op_str))
    coefficients.append(0.05)
hamiltonian = SparsePauliOp(pauli_terms, coefficients)

energy_quantum = rho.expectation_value(hamiltonian).real
print(f"Quantum circuit energy: {energy_quantum}")
print(f"Classical MPS energy: {energy_classical}")
print(f"Energy difference: {abs(energy_quantum - energy_classical)}")
