import numpy as np
np.random.seed(42)
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, SparsePauliOp
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper

mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()

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
        d = 2
        v = tensor.flatten()[:d]
        norm = np.linalg.norm(v)
        if norm > 1e-14:
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
        s = tensor.shape
        mat = tensor.reshape(s[0], -1)
        U, sigma, Vh = np.linalg.svd(mat, full_matrices=False)
        truncated = U[:, :target] @ np.diag(sigma[:target]) @ Vh[:target, :]
        return truncated.reshape((target, target))
    elif site_type == "intermediate":
        if tensor.ndim < 3:
            vec = tensor.flatten()
            needed = target*target*target
            if vec.size < needed:
                vec = np.pad(vec, (0, needed-vec.size))
            return vec[:needed].reshape((target, target, target))
        else:
            # A more advanced option would be an HOSVD; here we simply slice.
            return tensor[:target, :target, :target]
    elif site_type == "last":
        if tensor.ndim < 2 or tensor.shape[0] < target or tensor.shape[1] < target:
            return tensor
        s = tensor.shape
        mat = tensor.reshape(s[0], -1)
        U, sigma, Vh = np.linalg.svd(mat, full_matrices=False)
        truncated = U[:, :target] @ np.diag(sigma[:target]) @ Vh[:target, :]
        return truncated.reshape((target, target))
    else:
        return tensor

def build_deep_circuit_iterative(mps_data, target_dim=2, layers=3):
    # Make a deep copy of mps_data so we can update its tensors iteratively.
    mps_updated = {key: mps_data[key] for key in mps_data}
    n_sites = mps_updated['n_sites']
    full_qc = QuantumCircuit(n_sites)
    for L in range(layers):
        dense_tensors = []
        for i in range(n_sites):
            t = mps_updated['tensors'][i]
            q = mps_updated['q_labels'][i]
            block = pick_largest_block(t, q)
            if i == 0:
                dense = truncate_to_target(block, "first", target_dim)
            elif i == n_sites - 1:
                dense = truncate_to_target(block, "last", target_dim)
            else:
                dense = truncate_to_target(block, "intermediate", target_dim)
            dense_tensors.append(dense)
        U_list = []
        if len(dense_tensors) == 1:
            U = complete_gate_for_site(dense_tensors[0], "first")
            U_list.append(U)
        else:
            U_list.append(complete_gate_for_site(dense_tensors[0], "first"))
            for i in range(1, n_sites - 1):
                U_list.append(complete_gate_for_site(dense_tensors[i], "intermediate"))
            U_list.append(complete_gate_for_site(dense_tensors[-1], "last"))
        layer_qc = QuantumCircuit(n_sites)
        layer_qc.append(UnitaryGate(U_list[0], label="U1_layer"+str(L)), [0, 1])
        for i in range(1, n_sites - 1):
            layer_qc.append(UnitaryGate(U_list[i], label="U"+str(i+1)+"_layer"+str(L)), [i, i+1])
        layer_qc.append(UnitaryGate(U_list[-1], label="U"+str(n_sites)+"_layer"+str(L)), [n_sites-1])
        full_qc = full_qc.compose(layer_qc)
        # Iterative update (crude): replace current tensors with the dense ones
        mps_updated["tensors"] = dense_tensors
    full_qc.save_density_matrix(label="rho")
    return full_qc

qc = build_deep_circuit_iterative(mps_data, target_dim=2, layers=3)
print(qc.draw(output="text", reverse_bits=True))

sim = AerSimulator(method="density_matrix")
compiled = transpile(qc, sim)
job = sim.run(compiled, shots=1)
result = job.result()
data = result.data(0)
print(data)
print("Result data keys:", list(data.keys()))

fcidump = FCIDump.from_file("H2O.STO3G.FCIDUMP")
problem = fcidump_to_problem(fcidump)
mapper = JordanWignerMapper()
qubit_op = mapper.map(problem.second_q_ops()[0])

rho_mat = DensityMatrix(data["rho"])
def pad_density(rho, n_missing):
    zero_proj = np.zeros((2**n_missing, 2**n_missing))
    zero_proj[0,0] = 1.0
    return np.kron(rho, zero_proj)
rho_padded = pad_density(rho_mat.data, 7)
energy_circ = np.real(np.trace(qubit_op.to_matrix() @ rho_padded))
print("Energy from circuit:", energy_circ)

