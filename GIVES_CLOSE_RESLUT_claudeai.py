import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper

def reconstruct_state_vector(tensors, n_qubits):
    physical_dim = 2
    
    state = np.ones(1, dtype=np.complex128)
    for i in range(n_qubits):
        coefs = np.zeros((physical_dim, state.size), dtype=np.complex128)
        if i < len(tensors):
            tensor = tensors[i]
            for j in range(min(physical_dim, len(tensor))):
                if j < len(tensor):
                    coefs[j] = tensor[j] * state
        else:
            coefs[0] = state
        state = coefs.flatten()
    
    return state / np.linalg.norm(state)

mps = np.load("h2o_mps_complete.npy", allow_pickle=True).item()

print("MPS tensor shapes:")
for i, t in enumerate(mps["tensors"]):
    tensor = t.data if hasattr(t, "data") else t
    print(f"Tensor {i}: {np.asarray(tensor).shape}")

nq = 14
state = reconstruct_state_vector(mps["tensors"], nq)
print(f"State vector size: {state.size}, qubits: {nq}")

qc = QuantumCircuit(nq)
qc.initialize(state, range(nq))
qc = transpile(qc, basis_gates=["u", "cx"])

sv = Statevector(state)

fcidump = FCIDump.from_file("H2O.STO3G.FCIDUMP")
prob = fcidump_to_problem(fcidump)
op = JordanWignerMapper().map(prob.hamiltonian.second_q_op(), register_length=nq)

e_q = np.real(np.vdot(sv.data, op.to_matrix() @ sv.data))

print("Quantum energy:", e_q)
print("MPS energy:", mps["energy"])
