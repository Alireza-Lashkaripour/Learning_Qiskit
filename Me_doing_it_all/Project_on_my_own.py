import numpy as np
np.random.seed(42)
from qiskit import QuantumCircuit, transpile
from scipy.linalg import polar
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector
#from qiskit_nature.second_q.formats.fcidump import FCIDump
#from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
#from qiskit_nature.second_q.mappers import JordanWignerMapper


data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()

with open("mps_data_output.txt", "w") as f:
    f.write(f"n_sites: {data['n_sites']}\n")
    f.write(f"bond_dims: {data['bond_dims']}\n")
    f.write(f"energy: {data['energy']}\n\n")
    for i, (tensor, qlbl) in enumerate(zip(data['tensors'], data['q_labels'])):
        f.write(f"Tensor {i} (shape {tensor.shape}):\n{tensor}\n")
        f.write(f"q_labels {i}: {qlbl}\n\n")
