import numpy as np
from pyblock3.algebra.flat import FlatSparseTensor
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper

def to_dense_tensor(T):
    if isinstance(T, np.ndarray):
        return T
    flat = FlatSparseTensor(T)
    return flat.array

def contract_mps(tensors):
    # Print shapes for debugging
    print("Tensor shapes:")
    for i, t in enumerate(tensors):
        t_dense = to_dense_tensor(t)
        print(f"Tensor {i}: {t_dense.shape}")

    ψ = to_dense_tensor(tensors[0])
    print(f"Initial ψ shape: {ψ.shape}")
    
    # Process remaining tensors
    for i, A in enumerate(tensors[1:], 1):
        A = to_dense_tensor(A)
        print(f"Contracting with tensor {i}, shape {A.shape}")
        print(f"Current ψ shape: {ψ.shape}, ndim: {ψ.ndim}")
        
        # Adapt contraction based on tensor shapes
        if ψ.ndim == 1:  # ψ is a vector
            if A.ndim == 2:  # A is a matrix
                print(f"Vector-Matrix contraction: {ψ.shape} with {A.shape}")
                # Reshape ψ to match A's first dimension if needed
                if len(ψ) != A.shape[0]:
                    print(f"Reshaping vector from {len(ψ)} to {A.shape[0]}")
                    # This might be problematic - consider your specific case
                    ψ = ψ.reshape(A.shape[0])
                ψ = ψ @ A
            elif A.ndim == 3:  # A is a 3D tensor
                print(f"Vector-Tensor contraction: {ψ.shape} with {A.shape}")
                # Reshape ψ if needed
                ψ = np.tensordot(ψ.reshape(-1, 1), A, axes=([0], [0]))
                ψ = ψ.squeeze()
        elif ψ.ndim == 2:  # ψ is a matrix
            if A.ndim == 2:  # A is a matrix
                print(f"Matrix-Matrix contraction: {ψ.shape} with {A.shape}")
                ψ = ψ @ A
            elif A.ndim == 3:  # A is a 3D tensor
                print(f"Matrix-Tensor contraction: {ψ.shape} with {A.shape}")
                ψ = np.tensordot(ψ, A, axes=([1], [0]))
                # Reshape depending on result dimensions
                if ψ.ndim > 2:
                    new_shape = (ψ.shape[0] * ψ.shape[1], ψ.shape[2]) if ψ.ndim == 3 else ψ.shape
                    ψ = ψ.reshape(new_shape)
        else:  # ψ has more dimensions
            print(f"Complex contraction with ψ ndim={ψ.ndim}, A ndim={A.ndim}")
            # Handle based on specific MPS structure
            ψ = np.tensordot(ψ, A, axes=([ψ.ndim-1], [0]))
            if ψ.ndim > 2:
                # Flatten all but the last dimension
                ψ = ψ.reshape(-1, ψ.shape[-1])
    
    print(f"Final ψ shape before flatten: {ψ.shape}")
    return ψ.flatten()

mps_data = np.load("h2o_mps_complete.npy", allow_pickle=True).item()
psi = contract_mps(mps_data['tensors'])
psi /= np.linalg.norm(psi)

n_qubits = int(np.log2(psi.size))
qc = QuantumCircuit(n_qubits)
qc.initialize(psi, list(range(n_qubits)))

fcidump = FCIDump.from_file("H2O.STO3G.FCIDUMP")
problem = fcidump_to_problem(fcidump)
qubit_op = JordanWignerMapper().map(problem.second_q_ops()[0])

qc.save_density_matrix(label="rho")
sim = AerSimulator(method="density_matrix")
compiled = transpile(qc, sim, optimization_level=0)
result = sim.run(compiled).result()
rho = result.data(0)["rho"]
H = qubit_op.to_matrix()
energy = np.real(np.trace(rho @ H))
print(energy)

