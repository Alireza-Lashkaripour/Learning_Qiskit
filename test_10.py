import numpy as np
from scipy.linalg import qr
from qiskit import QuantumCircuit

def mps_to_circuit(num_sites, bond_dims, max_bond_dim=2):
    qc = QuantumCircuit(num_sites)
    mps_tensors = []
    for i in range(num_sites):
        if i == 0:
            tensor = np.random.rand(2, bond_dims[0]) + 1j*np.random.rand(2, bond_dims[0])
            q, _ = qr(tensor, mode='economic')
            mps_tensors.append(q)
        elif i == num_sites-1:
            tensor = np.random.rand(bond_dims[-1], 2) + 1j*np.random.rand(bond_dims[-1], 2)
            q, _ = qr(tensor.T, mode='economic')
            mps_tensors.append(q.T)
        else:
            tensor = np.random.rand(bond_dims[i-1], 2, bond_dims[i]) + 1j*np.random.rand(bond_dims[i-1], 2, bond_dims[i])
            reshaped = tensor.reshape(bond_dims[i-1]*2, bond_dims[i])
            q, _ = qr(reshaped, mode='economic')
            mps_tensors.append(q.reshape(bond_dims[i-1], 2, bond_dims[i]))
    for site in reversed(range(num_sites)):
        if site == num_sites-1:
            gate = mps_tensors[site]
            qc.unitary(gate, [site], label=f'U{site}')
        else:
            tensor_site = mps_tensors[site]
            chi_left, d, chi_right = tensor_site.shape
            gate = np.zeros((4,4), dtype=complex)
            for s in range(d):
                gate[0, s*chi_right:(s+1)*chi_right] = tensor_site[0, s, :]
                M = np.vstack([tensor_site[0, s, :], np.random.rand(3, chi_right) + 1j*np.random.rand(3, chi_right)])
                Q, _ = qr(M, mode='economic')
                gate[1:, s*chi_right:(s+1)*chi_right] = Q[1:, :]
            qc.unitary(gate.reshape(2,2,2,2), [site, site+1], label=f'U{site}')
    return qc

circuit = mps_to_circuit(num_sites=4, bond_dims=[2,2,2])
print(circuit.draw())

