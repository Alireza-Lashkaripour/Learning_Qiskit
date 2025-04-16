import numpy as np
np.random.seed(42)
from qiskit import QuantumCircuit, transpile
from scipy.linalg import polar
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper

mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()


def complete_unitary_from_fixed_row(v):
    v = np.atleast_1d(v).astype(np.complex128)
    n = v.shape[0]
    X = (np.random.randn(n, n) + 1j*np.random.randn(n, n)).astype(np.complex128)
    v_norm = np.linalg.norm(v)
    v_normed = v if v_norm < 1e-12 else v / v_norm
    X[0, :] = v_normed
    U, _ = polar(X)
    phase = np.vdot(v_normed, U[0, :])
    if np.abs(phase) < 1e-12:
        phase = 1.0+0j
    U[0, :] *= np.conj(phase)/np.abs(phase)
    return U

def complete_unitary_from_fixed_rows(M):
    M = np.atleast_2d(M).astype(np.complex128)
    m, n = M.shape
    X = (np.random.randn(n, n) + 1j*np.random.randn(n, n)).astype(np.complex128)
    X[:m, :] = M
    U, _ = polar(X)
    for j in range(m):
        phase = np.vdot(M[j, :], U[j, :])
        if np.abs(phase) < 1e-12:
            phase = 1.0+0j
        U[j, :] *= np.conj(phase)/np.abs(phase)
    return U


def complete_gate_for_site(tensor, site_type, target_dim):
    tensor = np.array(tensor, dtype=np.complex128)
    if site_type == "first":
        if tensor.ndim < 2:
            candidate = np.eye(2, dtype=np.complex128)
        else:
            v = tensor.reshape(-1)
            norm = np.linalg.norm(v)
            v = v if norm < 1e-12 else v / norm
            candidate = complete_unitary_from_fixed_row(v)
            if candidate.shape[0] != 2:
                candidate = candidate[:2, :2]
        U, _ = polar(candidate)
        return U
    elif site_type == "intermediate":
        if tensor.ndim < 3:
            candidate = np.eye(target_dim * target_dim, dtype=np.complex128)
        else:
            if tensor.shape != (target_dim, target_dim, target_dim):
                tensor = tensor[:target_dim, :target_dim, :target_dim]
            M = np.zeros((target_dim, target_dim * target_dim), dtype=np.complex128)
            for j in range(target_dim):
                vec = tensor[j, :, :].reshape(-1)
                norm = np.linalg.norm(vec)
                vec = vec if norm < 1e-12 else vec / norm
                M[j, :] = vec
            candidate = complete_unitary_from_fixed_rows(M)
            if candidate.shape[0] != target_dim * target_dim:
                candidate = candidate[:target_dim * target_dim, :target_dim * target_dim]
        U, _ = polar(candidate)
        return U
    elif site_type == "last":
        if tensor.ndim < 2:
            candidate = np.eye(2, dtype=np.complex128)
        else:
            v = tensor.flatten()
            if v.size < 2:
                v = np.pad(v, (0, 2 - v.size), mode='constant')
            else:
                v = v[:2]
            norm = np.linalg.norm(v)
            v = v if norm < 1e-12 else v / norm
            candidate = complete_unitary_from_fixed_row(v)
            if candidate.shape[0] != 2:
                candidate = candidate[:2, :2]
        U, _ = polar(candidate)
        return U
    else:
        return np.eye(target_dim * target_dim, dtype=np.complex128)

def pick_largest_block(tensor, qlabels):
    if isinstance(tensor, np.ndarray):
        return tensor
    best_block = max(
        (np.array(blk.data if hasattr(blk, 'data') else blk, dtype=np.complex128) for blk in tensor),
        key=lambda arr: np.linalg.norm(arr)
    )
    return best_block




def truncate_to_target(tensor, site_type, target=4):
    tensor = np.array(tensor, dtype=np.complex128)
    if tensor.ndim == 0:
        return tensor
    if site_type in ["first", "last"]:
        if tensor.ndim < 2:
            tensor = tensor.reshape((1, -1))
        s = tensor.shape
        pad_rows = max(0, target - s[0])
        pad_cols = max(0, target - s[1])
        if pad_rows > 0 or pad_cols > 0:
            tensor = np.pad(tensor, ((0, pad_rows), (0, pad_cols)), mode='constant')
        s = tensor.shape
        mat = tensor.reshape(s[0], -1)
        U, sigma, Vh = np.linalg.svd(mat, full_matrices=False)
        U_trunc = U[:, :target]
        S_trunc = np.diag(sigma[:target])
        Vh_trunc = Vh[:target, :]
        truncated = U_trunc @ S_trunc @ Vh_trunc
        return truncated[:target, :target]
    elif site_type == "intermediate":
        vec = tensor.flatten()
        needed = target * target * target
        if vec.size < needed:
            vec = np.pad(vec, (0, needed - vec.size), mode='constant')
        else:
            vec = vec[:needed]
        return vec.reshape((target, target, target))
    else:
        return tensor



def build_layer(mps_data, target_dim=2, layer_idx=0):
    n_sites = mps_data['n_sites']
    total_qubits = 1 + 2 * (n_sites - 2) + 1
    dense_tensors = []
    for i in range(n_sites):
        t = mps_data['tensors'][i]
        q = mps_data['q_labels'][i]
        block = pick_largest_block(t, q)
        if i == 0:
            dense = truncate_to_target(block, "first", target=2)
        elif i == n_sites - 1:
            dense = truncate_to_target(block, "last", target=2)
        else:
            dense = truncate_to_target(block, "intermediate", target=target_dim)
        dense_tensors.append(dense)
    U_list = []
    if len(dense_tensors) == 1:
        U = complete_gate_for_site(dense_tensors[0], "first", target_dim=2)
        U_list.append(U)
    else:
        U_list.append(complete_gate_for_site(dense_tensors[0], "first", target_dim=2))
        for i in range(1, n_sites - 1):
            U_list.append(complete_gate_for_site(dense_tensors[i], "intermediate", target_dim=target_dim))
        U_list.append(complete_gate_for_site(dense_tensors[-1], "last", target_dim=2))
    layer_qc = QuantumCircuit(total_qubits)
    layer_qc.append(UnitaryGate(U_list[0].conj().T, label="U1_layer{}".format(layer_idx)), [0])
    for i in range(1, n_sites - 1):
        start_index = 1 + 2 * (i - 1)
        layer_qc.append(UnitaryGate(U_list[i].conj().T, label="U{}_layer{}".format(i+1, layer_idx)), [start_index, start_index+1])
    layer_qc.append(UnitaryGate(U_list[-1].conj().T, label="U{}_layer{}".format(n_sites, layer_idx)), [total_qubits - 1])
    return layer_qc, dense_tensors

def simulate_layers(mps_data, target_dim=2, layers=3):
    mps_updated = {key: mps_data[key] for key in mps_data}
    n_sites = mps_updated['n_sites']
    total_qubits = 1 + 2 * (n_sites - 2) + 1
    full_qc = QuantumCircuit(total_qubits)
    for L in range(layers):
        layer_qc, dense_tensors = build_layer(mps_updated, target_dim=target_dim, layer_idx=L)
        full_qc = full_qc.compose(layer_qc)
        mps_updated["tensors"] = dense_tensors
    return full_qc

final_circuit = simulate_layers(mps_data, target_dim=2, layers=3)
print(final_circuit.draw(output="text", reverse_bits=True))
fcidump = FCIDump.from_file("H2O.STO3G.FCIDUMP")
problem = fcidump_to_problem(fcidump)
mapper = JordanWignerMapper()
qubit_op = mapper.map(problem.second_q_ops()[0])
final_circuit.save_expectation_value(qubit_op, list(range(final_circuit.num_qubits)), label="exp_val")
sim = AerSimulator(method="matrix_product_state", max_parallel_threads=1, max_parallel_experiments=1)
coupling_map = CouplingMap.from_full(final_circuit.num_qubits)
compiled = transpile(final_circuit, sim, coupling_map=coupling_map, optimization_level=0, layout_method="trivial")
job = sim.run(compiled, shots=1)
result = job.result()
energy_circ = result.data(0)["exp_val"]
print("Energy from circuit:", energy_circ)



