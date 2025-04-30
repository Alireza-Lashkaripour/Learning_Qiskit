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
    X = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128)
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-12:
        v_normed = v / v_norm
    else:
        v_normed = v
    X[:, 0] = v_normed.conj()
    U, _ = polar(X)
    phase = np.vdot(U[:, 0], v_normed.conj())
    if np.abs(phase) < 1e-12:
        phase = 1.0 + 0j
    U[:, 0] *= np.conj(phase) / np.abs(phase)
    return U

def complete_unitary_from_fixed_rows(M):
    M = np.atleast_2d(M).astype(np.complex128)
    m, n = M.shape
    X = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128)
    X[:, :m] = M.conj().T
    U, _ = polar(X)
    for j in range(m):
        phase = np.vdot(U[:, j], M[j, :].conj())
        if np.abs(phase) < 1e-12:
            phase = 1.0 + 0j
        U[:, j] *= np.conj(phase) / np.abs(phase)
    return U


def complete_gate_for_site(tensor, site_type, target_dim):
    tensor = np.array(tensor, dtype=np.complex128)
    if site_type == "first":
        # Always return a 2x2 unitary for the first site.
        if tensor.ndim < 2:
            candidate = np.eye(2, dtype=np.complex128)
        else:
            v = tensor.reshape(-1)
            norm = np.linalg.norm(v)
            if norm > 1e-12:
                v = v / norm
            candidate = complete_unitary_from_fixed_row(v)
            if candidate.shape[0] != 2:
                candidate = candidate[:2, :2]
        U, _ = polar(candidate)
        return U
    elif site_type == "intermediate":
        if tensor.ndim < 3:
            candidate = np.eye(target_dim * target_dim, dtype=np.complex128)
        else:
            M = np.zeros((target_dim, target_dim * target_dim), dtype=np.complex128)
            for j in range(target_dim):
                vec = tensor[j, :, :].reshape(-1)
                norm = np.linalg.norm(vec)
                if norm > 1e-12:
                    vec = vec / norm
                M[j, :] = vec
            candidate = complete_unitary_from_fixed_rows(M)
            if candidate.shape[0] != target_dim * target_dim:
                candidate = candidate[:target_dim * target_dim, :target_dim * target_dim]
        U, _ = polar(candidate)
        return U
    elif site_type == "last":
        # Always return a 2x2 unitary for the last site.
        if tensor.ndim < 2:
            candidate = np.eye(2, dtype=np.complex128)
        else:
            v = tensor.flatten()
            if v.size < 2:
                v = np.pad(v, (0, 2 - v.size), mode='constant')
            else:
                v = v[:2]
            norm = np.linalg.norm(v)
            if norm > 1e-12:
                v = v / norm
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
    best_norm = 0
    best_block = None
    for blk in tensor:
        curr = blk.data if hasattr(blk, 'data') else blk
        curr = np.array(curr, dtype=np.complex128)
        curr_norm = np.linalg.norm(curr)
        if curr_norm > best_norm:
            best_norm = curr_norm
            best_block = curr
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
        if tensor.ndim < 3:
            vec = tensor.flatten()
            needed = target * target * target
            if vec.size < needed:
                vec = np.pad(vec, (0, needed - vec.size), mode='constant')
            return vec[:needed].reshape((target, target, target))
        else:
            s = tensor.shape
            pad_width = [(0, max(0, target - s[i])) for i in range(3)]
            tensor = np.pad(tensor, pad_width, mode='constant')
            # Mode-1 unfolding
            unfold1 = tensor.reshape(tensor.shape[0], -1)
            U1, _, _ = np.linalg.svd(unfold1, full_matrices=False)
            U1_trunc = U1[:, :target]
            # Mode-2 unfolding
            unfold2 = np.transpose(tensor, (1, 0, 2)).reshape(tensor.shape[1], -1)
            U2, _, _ = np.linalg.svd(unfold2, full_matrices=False)
            U2_trunc = U2[:, :target]
            # Mode-3 unfolding
            unfold3 = np.transpose(tensor, (2, 0, 1)).reshape(tensor.shape[2], -1)
            U3, _, _ = np.linalg.svd(unfold3, full_matrices=False)
            U3_trunc = U3[:, :target]
            # Reconstruct core by mode-n products
            core = np.tensordot(np.conj(U1_trunc).T, tensor, axes=[1,0])
            core = np.tensordot(np.conj(U2_trunc).T, core, axes=[1,0])
            core = np.tensordot(np.conj(U3_trunc).T, core, axes=[1,0])
            return core
    else:
        return tensor

def truncate_mps_to_physical(mps_data, d=2):
    new_tensors = []
    for t in mps_data['tensors']:
        t_arr = np.array(t, dtype=np.complex128)
        if t_arr.ndim == 2:
            new_tensors.append(t_arr[:d, :d])
        elif t_arr.ndim == 3:
            new_tensors.append(t_arr[:d, :d, :d])
        else:
            new_tensors.append(t_arr)
    mps_data['tensors'] = new_tensors
    return mps_data

mps_data = truncate_mps_to_physical(mps_data, d=2)


def build_layer(mps_data, target_dim=4, layer_idx=0):
    n_sites = mps_data['n_sites']
    print(n_sites)
    int_q = int(np.log2(target_dim**2))  # Number of qubits per intermediate site
    total_qubits = 1 + int_q * (n_sites - 2) + 2
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
            U_list.append(complete_gate_for_site(dense_tensors[i], "intermediate", target_dim))
        U_list.append(complete_gate_for_site(dense_tensors[-1], "last", target_dim=2))
    layer_qc = QuantumCircuit(total_qubits)
    layer_qc.append(UnitaryGate(U_list[0].conj().T, label="U1_layer"+str(layer_idx)), [0])
    for i in range(1, n_sites - 1):
        start_index = 1 + int_q * (i - 1)
        gate_qubits = list(range(start_index, start_index + int_q))
        layer_qc.append(UnitaryGate(U_list[i].conj().T, label="U"+str(i+1)+"_layer"+str(layer_idx)), gate_qubits)
    layer_qc.append(UnitaryGate(U_list[-1].conj().T, label="U"+str(n_sites)+"_layer"+str(layer_idx)),
                     [total_qubits - 1])
    return layer_qc, dense_tensors

def simulate_layers(mps_data, target_dim=4, layers=3):
    mps_updated = {key: mps_data[key] for key in mps_data}
    n_sites = mps_updated['n_sites']
    int_q = int(np.log2(target_dim**2))
    total_qubits = 1 + int_q * (n_sites - 2) + 2
    full_qc = QuantumCircuit(total_qubits)
    for L in range(layers):
        layer_qc, dense_tensors = build_layer(mps_updated, target_dim, layer_idx=L)
        full_qc = full_qc.compose(layer_qc)
        mps_updated["tensors"] = dense_tensors
    return full_qc


def iterative_circuit_optimization(mps_data, target_dim=2, tol=1e-3, max_layers=10):
    mps_updated = {key: mps_data[key] for key in mps_data}
    n_sites = mps_updated['n_sites']
    int_q = int(np.log2(target_dim * target_dim))  
    total_qubits = qubit_op.num_qubits 
    qc_total = QuantumCircuit(total_qubits)
    prev_energy = None
    layers_used = 0
    for L in range(max_layers):
        layer_qc, dense_tensors = build_layer(mps_updated, target_dim=target_dim, layer_idx=L)
        qc_total = qc_total.compose(layer_qc)
        mps_updated["tensors"] = dense_tensors
        energy = measure_energy(qc_total, qubit_op, total_qubits)
        print("After layer", L, "energy =", energy)
        if prev_energy is not None and np.abs(energy - prev_energy) < tol:
            print("Convergence reached at layer", L)
            layers_used = L + 1
            break
        prev_energy = energy
        layers_used = L + 1
    return qc_total, energy, layers_used

fcidump = FCIDump.from_file("H2O.STO3G.FCIDUMP")
problem = fcidump_to_problem(fcidump)
mapper = JordanWignerMapper()
qubit_op = mapper.map(problem.second_q_ops()[0])

def measure_energy(qc, qubit_op, total_qubits):
    qc_local = qc.copy()
    qc_local.save_expectation_value(qubit_op, list(range(qubit_op.num_qubits)), label="exp_val")
    sim = AerSimulator(method="matrix_product_state", max_parallel_threads=1, max_parallel_experiments=1)
    coupling_map = CouplingMap.from_full(total_qubits)
    compiled = transpile(qc_local, sim, coupling_map=coupling_map, optimization_level=0, layout_method="trivial")
    job = sim.run(compiled, shots=1)
    result = job.result()
    print(qc_local)
    return result.data(0)["exp_val"]


final_circuit, final_energy, layers_used = iterative_circuit_optimization(mps_data, target_dim=2, tol=1e-3, max_layers=10)
print("Final energy from circuit:", final_energy)
print("Total layers used:", layers_used)

