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

# ==== NEW DENSE-EXPANDER ====================================================
def inflate_tensor(full_vals: np.ndarray, qlbl: np.ndarray):
    """
    Expand a pyblock3 block-sparse tensor into a dense
    (chi_L × d × chi_R_max) array.

    A separate right-bond counter is kept for every (qL,qP) pair, so
    multiple successive values with the same (qL,qP) but different qR
    land at r = 0,1,2,… along that strip.

    Parameters
    ----------
    full_vals : (N,) ndarray
        Flattened non-zero values as stored in the .npy file.
    qlbl      : (N,3) ndarray of int64
        Symmetry keys per value: [qL, qP, qR].

    Returns
    -------
    A : ndarray, shape (chi_L, d, chi_R_max)
        Dense tensor with zeros padded where a strip is shorter than the
        global maximum length.
    """
    qlbl = np.asarray(qlbl, dtype=np.int64)
    full_vals = np.asarray(full_vals)

    # 1) index maps for left and physical legs (first-appearance order)
    qL_keys = list(dict.fromkeys(qlbl[:, 0]))
    qP_keys = list(dict.fromkeys(qlbl[:, 1]))
    Lmap = {q: i for i, q in enumerate(qL_keys)}
    Pmap = {q: i for i, q in enumerate(qP_keys)}

    # 2) running counters for right indices *per (ℓ,s) pair*
    strip_counters = {}          # {(ℓ,s): next_r}
    coords = []                  # will store (ℓ,s,r,val)

    for val, (qL, qP, _qR) in zip(full_vals, qlbl):
        ℓ = Lmap[qL]
        s = Pmap[qP]
        r = strip_counters.setdefault((ℓ, s), 0)
        strip_counters[(ℓ, s)] = r + 1
        coords.append((ℓ, s, r, val))

    chi_L = len(qL_keys)
    d_phys = len(qP_keys)
    chi_R_max = max(r + 1 for (_, _, r, _) in coords)

    A = np.zeros((chi_L, d_phys, chi_R_max), dtype=full_vals.dtype)
    for ℓ, s, r, val in coords:
        A[ℓ, s, r] = val
    return A


# ========== inflate every tensor and report the dense shapes ===============
dense_tensors = []

for i, (flat, qlbl) in enumerate(zip(data["tensors"], data["q_labels"])):
    dense = inflate_tensor(flat, qlbl)
    dense_tensors.append(dense)
    print(f"Site {i}: dense shape {dense.shape}")

# ---------------------------------------------------------------------------
# dense_tensors[k] is now the full χ_{k-1} × d_k × χ_k(max) tensor
# ready for contractions, expectation values, circuit mapping, etc.

