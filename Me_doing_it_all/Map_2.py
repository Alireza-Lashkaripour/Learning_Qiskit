import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, SparsePauliOp
from qiskit_aer import AerSimulator


data1 = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
n_sites = data1['n_sites']
dense_all = data1['dense_tensors']

with open("mps_data_output_1.txt", "w") as f:
    f.write(f"n_sites:   {n_sites}\n")
    f.write(f"bond_dims: {data1['bond_dims']}\n")
    f.write(f"energy:    {data1['energy']:.12f}\n\n")
    for i, (tensor, qlbl, dense) in enumerate(zip(data1['tensors'], data1['q_labels'], dense_all)):
        f.write(f"Tensor {i}  (block-sparse flat shape {tensor.shape}):\n")
        f.write(f"{tensor}\n\n")
        f.write(f"q_labels {i} (rows={len(qlbl)}):\n{qlbl}\n\n")
        f.write(f"Dense Tensor {i} (shape {dense.shape}):\n")
        f.write(np.array2string(dense, precision=8, threshold=1_000_000))
        f.write("\n" + "-"*80 + "\n\n")

def block_to_dense(vec, label_rows, phys_dim=4):
    out = np.zeros((phys_dim, label_rows.shape[0], label_rows.shape[1]//3), dtype=vec.dtype)
    out.flat[:] = vec
    return out

dense = []
for t, l in zip(data1['tensors'], data1['q_labels']):
    dense.append(block_to_dense(t, l, 4))

psi = dense[0]
for k in range(1, n_sites):
    psi = np.tensordot(psi, dense[k], axes=([-1], [0]))
psi = np.squeeze(psi).reshape(-1)

n_qubits = 2 * n_sites
psi_qubit = np.zeros(2**n_qubits, dtype=complex)
for idx, amp in enumerate(psi):
    temp = idx
    bits = []
    for _ in range(n_sites):
        level = temp & 3
        bits.append(level & 1)
        bits.append((level >> 1) & 1)
        temp >>= 2
    bit_index = sum(b << i for i, b in enumerate(bits))
    psi_qubit[bit_index] = amp

psi_qubit /= np.linalg.norm(psi_qubit)

qc = QuantumCircuit(n_qubits)
qc.initialize(psi_qubit)

sim = AerSimulator(method="statevector")
sv = Statevector.from_instruction(qc)
print(np.vdot(sv.data, sv.data))

dm = DensityMatrix(sv)
print(dm.expectation_value(pauli_op).real)

