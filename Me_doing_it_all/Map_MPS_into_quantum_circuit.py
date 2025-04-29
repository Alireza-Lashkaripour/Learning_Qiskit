import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper

data1 = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
n_sites = data1['n_sites']
dense_all = data1['dense_tensors']

psi = dense_all[0]
for k in range(1, n_sites):
    psi = np.tensordot(psi, dense_all[k], axes=([-1], [0]))
psi = np.squeeze(psi).reshape(-1)

psi_qubit = np.zeros(2 ** (2 * n_sites), dtype=complex)
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

qc = QuantumCircuit(2 * n_sites)
qc.initialize(psi_qubit)

sim = AerSimulator(method="statevector")
result = sim.run(qc).result()
sv = Statevector(result.get_statevector(qc))

fcidump = FCIDump.from_file("H2O.STO3G.FCIDUMP")
problem = fcidump_to_problem(fcidump)
ferm_op = problem.hamiltonian.second_q_op()
mapper = JordanWignerMapper()
qubit_op = mapper.map(ferm_op)

pauli_list = [(label, float(coeff)) for label, coeff in zip(qubit_op.primitive.paulis.to_labels(), qubit_op.primitive.coeffs)]
pauli_op = SparsePauliOp(pauli_list)

state = sv.data
energy = np.real(np.vdot(state, pauli_op.to_matrix().dot(state)))
print(energy)


