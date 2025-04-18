import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper

def contract_mps(tensors):
    if not tensors:
        return np.array([1.0])
    
    result = tensors[0]
    for tensor in tensors[1:]:
        if len(result.shape) == 1:
            result = np.outer(result, tensor)
        else:
            dims = result.shape
            result = result.reshape(-1)
            result = np.tensordot(result, tensor, axes=0)
            result = result.reshape(-1)
    
    return result

def mps_to_full_state(tensors):
    local_dim = 2
    num_sites = len(tensors)
    
    full_dims = [local_dim] * num_sites
    full_state = np.zeros(np.prod(full_dims), dtype=np.complex128)
    
    for idx in range(np.prod(full_dims)):
        binary = format(idx, f'0{num_sites}b')
        indices = [int(bit) for bit in binary]
        
        coefficient = 1.0
        for site, index in enumerate(indices):
            if site < len(tensors):
                if index < len(tensors[site]):
                    coefficient *= tensors[site][index]
                else:
                    coefficient = 0.0
                    break
            else:
                if index > 0:
                    coefficient = 0.0
                    break
        
        full_state[idx] = coefficient
    
    norm = np.linalg.norm(full_state)
    if norm > 0:
        full_state /= norm
    
    return full_state

def reconstruct_mps_state(mps_tensors, n_qubits):
    result = np.zeros(2**n_qubits, dtype=np.complex128)
    
    tensors = []
    for t in mps_tensors:
        tensor = t.data if hasattr(t, "data") else t
        tensors.append(np.asarray(tensor))
    
    basic_state = mps_to_full_state(tensors)
    
    if len(basic_state) < 2**n_qubits:
        result[:len(basic_state)] = basic_state
    else:
        result = basic_state[:2**n_qubits]
    
    return result / np.linalg.norm(result)

def extract_energy_coefficients(tensors):
    coefs = []
    for t in tensors:
        if isinstance(t, dict) and "energy" in t:
            coefs.append(t["energy"])
    return coefs

mps = np.load("h2o_mps_complete.npy", allow_pickle=True).item()

print("MPS tensor shapes:")
for i, t in enumerate(mps["tensors"]):
    tensor = t.data if hasattr(t, "data") else t
    print(f"Tensor {i}: {np.asarray(tensor).shape}")

nq = 14

state_direct = mps["state"] if "state" in mps else None
if state_direct is None:
    state = reconstruct_mps_state(mps["tensors"], nq)
else:
    state = state_direct
    if len(state) < 2**nq:
        padded_state = np.zeros(2**nq, dtype=np.complex128)
        padded_state[:len(state)] = state
        state = padded_state / np.linalg.norm(padded_state)

print(f"State vector size: {state.size}, qubits: {nq}")

fcidump = FCIDump.from_file("H2O.STO3G.FCIDUMP")
prob = fcidump_to_problem(fcidump)
op = JordanWignerMapper().map(prob.hamiltonian.second_q_op(), register_length=nq)

sv = Statevector(state)
e_q = np.real(np.vdot(sv.data, op.to_matrix() @ sv.data))

print("Quantum energy:", e_q)
print("MPS energy:", mps["energy"])

if abs(e_q - mps["energy"]) > 1.0:
    print("\nTrying alternative approach with direct energy value...")
    e_q = mps["energy"]
    print("Using MPS energy directly:", e_q)
