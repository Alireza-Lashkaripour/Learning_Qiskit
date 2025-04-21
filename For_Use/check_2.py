import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, SparsePauliOp
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter

def complete_unitary_from_fixed_row(v):
    n = v.shape[0]
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    X[:, 0] = v.conjugate()
    Q, _ = np.linalg.qr(X)
    phase = np.vdot(Q[:, 0], v.conjugate())
    Q[:, 0] *= phase.conjugate() / abs(phase)
    return Q.conjugate().T

def complete_unitary_from_fixed_row_2x2(v):
    u = np.array([-v[1], v[0]], dtype=complex)
    u = u / np.linalg.norm(u)
    U = np.array([v.conjugate(), u.conjugate()])
    return U

def complete_gate_for_site(tensor, site_type):
    d = 2
    if site_type in ["first", "intermediate"]:
        if tensor.ndim < 2:
            return np.eye(d * d, dtype=complex)
        v = tensor.reshape(-1)
        norm = np.linalg.norm(v)
        if norm:
            v /= norm
        return complete_unitary_from_fixed_row(v)
    elif site_type == "last":
        if tensor.ndim < 2 or tensor.shape[0] < d or tensor.shape[1] < d:
            return np.eye(d, dtype=complex)
        mat = tensor[:d, :d]
        row = mat[0, :]
        norm = np.linalg.norm(row)
        if norm:
            row = row / np.linalg.norm(row)
        return complete_unitary_from_fixed_row_2x2(row)
    else:
        return np.eye(d * d, dtype=complex)

def truncate_to_target(tensor, site_type, target=2):
    if not hasattr(tensor, "ndim") or tensor.ndim == 0:
        return tensor
    if site_type == "first":
        if tensor.ndim < 2 or tensor.shape[0] < target or tensor.shape[1] < target:
            return tensor
        return tensor[:target, :target]
    elif site_type == "intermediate":
        if tensor.ndim < 3 or tensor.shape[0] < target or tensor.shape[1] < target or tensor.shape[2] < target:
            return tensor
        return tensor[:target, :target, :target]
    elif site_type == "last":
        if tensor.ndim < 2 or tensor.shape[0] < target or tensor.shape[1] < target:
            return tensor
        return tensor[:target, :target]
    else:
        return tensor

def pick_largest_block(tensor, qlabels):
    if isinstance(tensor, np.ndarray):
        return tensor
    best_norm = 0
    best_block = None
    for blk in tensor:
        curr = blk.data if hasattr(blk, "data") else blk
        curr_norm = np.linalg.norm(curr)
        if curr_norm > best_norm:
            best_norm = curr_norm
            best_block = curr
    return best_block

def compute_unitaries_from_mps(mps_data, target_dim=2):
    n_sites = mps_data["n_sites"]
    tensors_all = mps_data["tensors"]
    qlabels_all = mps_data["q_labels"]
    dense_tensors = []
    for i in range(n_sites):
        t = tensors_all[i]
        q = qlabels_all[i]
        block = pick_largest_block(t, q)
        if i == 0:
            dense = truncate_to_target(block, "first", target_dim)
        elif i == n_sites - 1:
            dense = truncate_to_target(block, "last", target_dim)
        else:
            dense = truncate_to_target(block, "intermediate", target_dim)
        dense_tensors.append(dense)
    unitaries = []
    if n_sites == 1:
        U = complete_gate_for_site(dense_tensors[0], "last")
        unitaries.append(U)
    else:
        U0 = complete_gate_for_site(dense_tensors[0], "first")
        unitaries.append(U0)
        for i in range(1, n_sites - 1):
            Ui = complete_gate_for_site(dense_tensors[i], "intermediate")
            unitaries.append(Ui)
        U_last = complete_gate_for_site(dense_tensors[-1], "last")
        unitaries.append(U_last)
    return unitaries

def build_deep_circuit_from_mps(mps_data, target_dim=2, num_layers=1):
    n_sites = mps_data["n_sites"]
    qc = QuantumCircuit(n_sites)
    for layer in range(num_layers):
        unitaries = compute_unitaries_from_mps(mps_data, target_dim)
        if n_sites == 1:
            qc.append(UnitaryGate(unitaries[0].conj().T, label=f"Layer{layer+1}-U1†"), [0])
        else:
            qc.append(UnitaryGate(unitaries[0].conj().T, label=f"Layer{layer+1}-U1†"), [0, 1])
            for i in range(1, n_sites - 1):
                qc.append(UnitaryGate(unitaries[i].conj().T, label=f"Layer{layer+1}-U{i+1}†"), [i, i+1])
            qc.append(UnitaryGate(unitaries[-1].conj().T, label=f"Layer{layer+1}-U{n_sites}†"), [n_sites - 1])
    qc.save_density_matrix(label="rho")
    return qc

mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
qc = build_deep_circuit_from_mps(mps_data, target_dim=2, num_layers=2)
print(qc.draw(output="text"))
simulator = AerSimulator(method="density_matrix")
compiled_circuit = transpile(qc, simulator)
job = simulator.run(compiled_circuit, shots=1)
result = job.result()
rho = result.data(0)["rho"]
density = DensityMatrix(rho)
fd = FCIDump.from_file("H2O.STO3G.FCIDUMP")
problem = fcidump_to_problem(fd)
mapper = JordanWignerMapper()
qubit_converter = QubitConverter(mapper=mapper, two_qubit_reduction=False)
qubit_op = qubit_converter.convert(problem.hamiltonian)
H_mat = qubit_op.to_matrix()
print("Hamiltonian acts on", qubit_op.num_qubits, "qubits.")
print("Circuit has", qc.num_qubits, "qubits.")
if qubit_op.num_qubits != qc.num_qubits:
    raise ValueError("Dimension mismatch between Hamiltonian and circuit")
exp_value = np.real(np.trace(H_mat @ density.data))
print("Measured energy:", exp_value)

