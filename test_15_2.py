import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper

def mps_to_state(tensors):
    n_sites = len(tensors)
    d = 2
    state = np.zeros(d**n_sites, dtype=np.complex128)
    
    indices = [0] * n_sites
    max_idx = [0] * n_sites
    for i in range(n_sites):
        max_idx[i] = d-1
    
    def update_indices(idx):
        idx[n_sites-1] += 1
        for i in range(n_sites-1, 0, -1):
            if idx[i] > max_idx[i]:
                idx[i] = 0
                idx[i-1] += 1
    
    idx = 0
    while indices[0] <= max_idx[0]:
        coeff = 1.0
        for i, tensor in enumerate(tensors):
            coeff *= tensor[indices[i]]
        
        state[idx] = coeff
        idx += 1
        update_indices(indices)
    
    return state / np.linalg.norm(state)

mps = np.load("h2o_mps_complete.npy", allow_pickle=True).item()

print("MPS tensor shapes:")
for i, t in enumerate(mps["tensors"]):
    tensor = t.data if hasattr(t, "data") else t
    print(f"Tensor {i}: {np.asarray(tensor).shape}")

nq = 14

def reconstruct_state_vector(tensors):
    physical_dim = 2
    n_qubits = nq
    
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

state = reconstruct_state_vector(mps["tensors"])

print(f"State vector size: {state.size}, qubits: {nq}")

qc = QuantumCircuit(nq)
qc.initialize(state, range(nq))
qc = transpile(qc, basis_gates=["u", "cx"])

fcidump = FCIDump.from_file("H2O.STO3G.FCIDUMP")
prob = fcidump_to_problem(fcidump)
op = JordanWignerMapper().map(prob.hamiltonian.second_q_op(), register_length=nq)

sv = AerSimulator(method="statevector").run(qc).result().get_statevector()
e_q = np.real(np.vdot(sv, op.to_matrix() @ sv))

print("Quantum energy:", e_q)
print("MPS energy:", mps["energy"])
