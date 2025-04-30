import numpy as np
np.random.seed(42)
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector
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
    if abs(phase) < 1e-12:
        phase = 1.0
    Q[:, 0] *= phase.conjugate() / abs(phase)
    U = Q.conj().T
    u, s, vh = np.linalg.svd(U)
    U_unitary = u @ vh
    return U_unitary


def complete_unitary_from_fixed_rows(M):
    m, n = M.shape
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    X[:, :m] = M.conj().T
    Q, _ = np.linalg.qr(X)
    for j in range(m):
        phase = np.vdot(Q[:, j], M[j, :].conj())
        if abs(phase) < 1e-12:
            phase = 1.0
        Q[:, j] *= phase.conjugate() / abs(phase)
    U = Q.conj().T
    u, s, vh = np.linalg.svd(U)
    U_unitary = u @ vh
    return U_unitary


def complete_gate_for_site(tensor, site_type, target_dim):
    if site_type == "first":
        if tensor.ndim < 2:
            return np.eye(target_dim * target_dim, dtype=complex)
        v = tensor.reshape(-1)
        norm = np.linalg.norm(v)
        if norm:
            v /= norm
        return complete_unitary_from_fixed_row(v)
    elif site_type == "intermediate":
        if tensor.ndim < 3:
            return np.eye(target_dim * target_dim, dtype=complex)
        # Create M with shape (target_dim, target_dim^2)
        M = np.zeros((target_dim, target_dim * target_dim), dtype=complex)
        for j in range(target_dim):
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
        return np.eye(target_dim * target_dim, dtype=complex)

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

def build_deep_circuit_iterative(mps_data, target_dim=2, layers=1):
    mps_updated = {key: mps_data[key] for key in mps_data}
    n_sites = mps_updated['n_sites']
    # Total qubits: first site uses 2, each intermediate uses 4, last uses 1.
    total_qubits = 2 + 4 * (n_sites - 2) + 1
    full_qc = QuantumCircuit(total_qubits)
    for L in range(layers):
        dense_tensors = []
        for i in range(n_sites):
            t = mps_updated['tensors'][i]
            q = mps_updated['q_labels'][i]
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
        # Build layer circuit with proper qubit assignments:
        layer_qc = QuantumCircuit(total_qubits)
        # First gate on qubits [0,1]:
        first_gate_qubits = [0, 1]
        layer_qc.append(UnitaryGate(U_list[0].conj().T, label="U1_layer"+str(L)), first_gate_qubits)
        # Intermediate gates: assign 4 qubits per gate.
        for i in range(1, n_sites - 1):
            start_index = 2 + 4 * (i - 1)
            gate_qubits = list(range(start_index, start_index + 4))
            layer_qc.append(UnitaryGate(U_list[i].conj().T, label="U" + str(i+1) + "_layer" + str(L)), gate_qubits)
        # Last gate on the final qubit:
        last_gate_qubits = [total_qubits - 1]
        layer_qc.append(UnitaryGate(U_list[-1].conj().T, label="U" + str(n_sites) + "_layer" + str(L)), last_gate_qubits)
        full_qc = full_qc.compose(layer_qc)
        # Optional: iterative update (if needed)
        mps_updated["tensors"] = dense_tensors
    full_qc.save_density_matrix(label="rho")
    return full_qc



qc = build_deep_circuit_iterative(mps_data, target_dim=2, layers=1)
print(qc.draw(output="text", reverse_bits=True))

sim = AerSimulator(method="statevector")
coupling_map = CouplingMap.from_full(qc.num_qubits)
compiled = transpile(qc, sim, coupling_map=coupling_map,
                     optimization_level=0, layout_method="trivial")
job = sim.run(compiled, shots=1)
result = sim.run(compiled, shots=1).result()



state = result.get_statevector(qc)
sv = Statevector(state)

fcidump = FCIDump.from_file("H2O.STO3G.FCIDUMP")
problem = fcidump_to_problem(fcidump)
mapper = JordanWignerMapper()
#qubit_op = mapper.map(problem.second_q_ops()[0])
qubit_op = mapper.map(problem.hamiltonian)


H_mat = qubit_op.to_matrix()
num_hamiltonian_qubits = int(np.log2(H_mat.shape[0]))
print("Hamiltonian acts on", num_hamiltonian_qubits, "qubits.")
print("Circuit has", qc.num_qubits, "qubits.")

if qc.num_qubits > num_hamiltonian_qubits:
    extra_qubits = list(range(num_hamiltonian_qubits, qc.num_qubits))
    print("Tracing out qubits:", extra_qubits)
    rho_reduced = sv.partial_trace(extra_qubits)
else:
    from qiskit.quantum_info import DensityMatrix
    rho_reduced = DensityMatrix(sv)

energy_circ = np.real(np.trace(H_mat @ rho_reduced.data))
print("Energy from circuit:", energy_circ)
