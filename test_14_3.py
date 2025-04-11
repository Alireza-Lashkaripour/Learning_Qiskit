import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any
from scipy.linalg import sqrtm, expm
import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, Statevector, SparsePauliOp, Operator
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit.opflow import Z2Symmetries
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.primitives import Estimator
import tensornetwork as tn
import opt_einsum as oe
from functools import lru_cache

class MPSCircuitBuilder:
    def __init__(self, mps_data: Dict[str, Any], precision: float = 1e-12):
        self.mps_data = mps_data
        self.n_sites = mps_data['n_sites']
        self.precision = precision
        self.bond_dims = self._extract_bond_dimensions()
        
    def _extract_bond_dimensions(self) -> List[int]:
        dims = []
        for i, tensor in enumerate(self.mps_data['tensors']):
            block = self._pick_largest_block(tensor, self.mps_data['q_labels'][i])
            if i == 0:
                dims.append(block.shape[1] if len(block.shape) > 1 else 1)
            elif i == self.n_sites - 1:
                dims.append(block.shape[0] if len(block.shape) > 0 else 1)
            else:
                dims.append(block.shape[0] if len(block.shape) > 0 else 1)
                dims.append(block.shape[2] if len(block.shape) > 2 else 1)
        return list(dict.fromkeys(dims))  # Remove duplicates while preserving order
        
    def _pick_largest_block(self, tensor, qlabels):
        if isinstance(tensor, np.ndarray):
            return tensor
        
        if not isinstance(tensor, (list, tuple)):
            return tensor
            
        best_norm = 0
        best_block = None
        
        for blk in tensor:
            curr = blk.data if hasattr(blk, 'data') else blk
            if not isinstance(curr, np.ndarray):
                continue
                
            curr_norm = np.linalg.norm(curr)
            if curr_norm > best_norm:
                best_norm = curr_norm
                best_block = curr
                
        return best_block if best_block is not None else np.array([0])

    def _complete_unitary_from_fixed_row(self, v: np.ndarray) -> np.ndarray:
        n = v.shape[0]
        X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        X[:, 0] = v.conj()
        
        Q, _ = np.linalg.qr(X)
        phase = np.vdot(Q[:, 0], v.conj())
        
        if abs(phase) > self.precision:
            Q[:, 0] *= phase.conjugate() / abs(phase)
            
        return Q.conj().T

    def _complete_unitary_from_fixed_rows(self, M: np.ndarray) -> np.ndarray:
        m, n = M.shape
        X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        X[:, :m] = M.conj().T
        
        Q, _ = np.linalg.qr(X)
        
        for j in range(m):
            phase = np.vdot(Q[:, j], M[j, :].conj())
            if abs(phase) > self.precision:
                Q[:, j] *= phase.conjugate() / abs(phase)
                
        return Q.conj().T

    def _svd_truncate(self, matrix: np.ndarray, max_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
        
        # Calculate truncation index based on singular value magnitude
        cum_energy = np.cumsum(S**2) / np.sum(S**2)
        trunc_idx = np.searchsorted(cum_energy, 0.9999) + 1
        trunc_idx = min(trunc_idx, max_dim, len(S))
        
        return U[:, :trunc_idx], np.diag(S[:trunc_idx]), Vh[:trunc_idx, :]

    def _tensor_to_site_operator(self, tensor: np.ndarray, site_type: str, target_dim: int) -> np.ndarray:
        d = 2  # Local Hilbert space dimension
        
        if site_type == "first":
            if tensor.ndim < 2:
                tensor = tensor.reshape(1, -1)
            elif tensor.ndim > 2:
                # Reshape higher-dimensional tensor to a matrix
                s = tensor.shape
                tensor = tensor.reshape(s[0], -1)
                
            if tensor.shape[1] > target_dim:
                U, S, Vh = self._svd_truncate(tensor, target_dim)
                tensor = U @ S @ Vh
                
            v = tensor.reshape(-1)
            norm = np.linalg.norm(v)
            if norm > self.precision:
                v /= norm
                
            return self._complete_unitary_from_fixed_row(v)
            
        elif site_type == "intermediate":
            if tensor.ndim < 3:
                # Handle insufficient dimensions
                return np.eye(d*d, dtype=complex)
                
            # Create a proper reshaping for 3-tensor
            chi1, d_loc, chi2 = tensor.shape
            target_chi = min(target_dim, chi1, chi2)
            
            # Truncate bond dimensions if needed
            if chi1 > target_chi or chi2 > target_chi:
                # SVD-based truncation in both directions
                tensor_mat = tensor.reshape(chi1, d_loc * chi2)
                U1, S1, Vh1 = self._svd_truncate(tensor_mat, target_chi)
                
                tensor_trunc = (U1 @ S1 @ Vh1).reshape(U1.shape[0], d_loc, -1)
                
                # Second truncation in the other direction
                tensor_mat2 = tensor_trunc.transpose(0, 2, 1).reshape(tensor_trunc.shape[0] * tensor_trunc.shape[2], d_loc)
                U2, S2, Vh2 = self._svd_truncate(tensor_mat2, target_chi)
                
                tensor = (U2 @ S2 @ Vh2).reshape(-1, d_loc, target_chi).transpose(0, 2, 1)
            
            # Map to unitary operators
            M = np.zeros((d, d*d), dtype=complex)
            for j in range(d):
                vec = tensor[:target_chi, j, :target_chi].reshape(-1)
                norm = np.linalg.norm(vec)
                if norm > self.precision:
                    vec /= norm
                M[j, :] = vec
                
            return self._complete_unitary_from_fixed_rows(M)
            
        elif site_type == "last":
            if tensor.ndim < 2:
                tensor = tensor.reshape(-1, 1)
            elif tensor.ndim > 2:
                # Reshape higher-dimensional tensor to a matrix
                s = tensor.shape
                tensor = tensor.reshape(-1, s[-1])
                
            if tensor.shape[0] > target_dim:
                U, S, Vh = self._svd_truncate(tensor, target_dim)
                tensor = U @ S @ Vh
                
            v = tensor.flatten()[:d]
            norm = np.linalg.norm(v)
            if norm > self.precision:
                v /= norm
                
            return self._complete_unitary_from_fixed_row(v)
            
        else:
            return np.eye(d*d, dtype=complex)

    def build_circuit(self, target_dim: int = 2, layers: int = 3, use_tensornetwork: bool = True) -> QuantumCircuit:
        np.random.seed(42)  # For reproducibility
        
        if use_tensornetwork:
            return self._build_circuit_tensornetwork(target_dim, layers)
        else:
            return self._build_circuit_iterative(target_dim, layers)

    def _build_circuit_iterative(self, target_dim: int = 2, layers: int = 3) -> QuantumCircuit:
        mps_updated = {key: self.mps_data[key] for key in self.mps_data}
        full_qc = QuantumCircuit(self.n_sites)
        
        for L in range(layers):
            dense_tensors = []
            
            for i in range(self.n_sites):
                t = mps_updated['tensors'][i]
                q = mps_updated['q_labels'][i] if 'q_labels' in mps_updated else None
                block = self._pick_largest_block(t, q)
                
                if i == 0:
                    site_type = "first"
                elif i == self.n_sites - 1:
                    site_type = "last"
                else:
                    site_type = "intermediate"
                    
                dense_tensors.append(block)
                
            U_list = []
            if len(dense_tensors) == 1:
                U = self._tensor_to_site_operator(dense_tensors[0], "first", target_dim)
                U_list.append(U)
            else:
                U_list.append(self._tensor_to_site_operator(dense_tensors[0], "first", target_dim))
                
                for i in range(1, self.n_sites - 1):
                    U_list.append(self._tensor_to_site_operator(dense_tensors[i], "intermediate", target_dim))
                    
                U_list.append(self._tensor_to_site_operator(dense_tensors[-1], "last", target_dim))
                
            # Apply quantum gates from the unitaries
            layer_qc = QuantumCircuit(self.n_sites)
            
            # Apply first site unitary
            layer_qc.append(UnitaryGate(U_list[0], label=f"U1_L{L}"), [0, 1] if self.n_sites > 1 else [0])
            
            # Apply intermediate site unitaries
            for i in range(1, self.n_sites - 1):
                layer_qc.append(UnitaryGate(U_list[i], label=f"U{i+1}_L{L}"), [i, i+1])
                
            # Apply last site unitary if there are multiple sites
            if self.n_sites > 1:
                layer_qc.append(UnitaryGate(U_list[-1], label=f"U{self.n_sites}_L{L}"), 
                               [self.n_sites-2, self.n_sites-1] if self.n_sites > 2 else [self.n_sites-1])
                
            full_qc = full_qc.compose(layer_qc)
            
            # Update MPS tensors for next layer
            mps_updated["tensors"] = dense_tensors
            
        # Save density matrix for later use
        full_qc.save_density_matrix(label="rho")
        return full_qc

    def _build_circuit_tensornetwork(self, target_dim: int = 2, layers: int = 3) -> QuantumCircuit:
        # Convert MPS to TensorNetwork representation
        tn_nodes = []
        for i in range(self.n_sites):
            t = self.mps_data['tensors'][i]
            q = self.mps_data['q_labels'][i] if 'q_labels' in self.mps_data else None
            tensor = self._pick_largest_block(t, q)
            
            if i == 0:
                # First tensor has one virtual bond and one physical index
                tensor = tensor.reshape(-1, target_dim) if tensor.ndim != 2 else tensor
                tensor = tensor[:, :target_dim]
            elif i == self.n_sites - 1:
                # Last tensor has one virtual bond and one physical index
                tensor = tensor.reshape(target_dim, -1) if tensor.ndim != 2 else tensor
                tensor = tensor[:target_dim, :]
            else:
                # Middle tensors have two virtual bonds and one physical index
                if tensor.ndim != 3:
                    # Create proper intermediate tensor
                    tensor = np.zeros((target_dim, 2, target_dim), dtype=complex)
                else:
                    tensor = tensor[:target_dim, :, :target_dim]
                    
            tn_nodes.append(tn.Node(tensor))
            
        # Connect virtual bonds
        for i in range(self.n_sites - 1):
            tn.connect(tn_nodes[i][1], tn_nodes[i+1][0])
            
        # Build the circuit using the tensor network representation
        full_qc = QuantumCircuit(self.n_sites)
        
        for L in range(layers):
            layer_qc = QuantumCircuit(self.n_sites)
            
            # Apply unitaries based on tensor network structure
            for i in range(self.n_sites):
                if i == 0:
                    site_type = "first"
                elif i == self.n_sites - 1:
                    site_type = "last"
                else:
                    site_type = "intermediate"
                    
                # Extract tensor from node
                tensor = tn_nodes[i].tensor
                
                # Convert tensor to unitary gate
                U = self._tensor_to_site_operator(tensor, site_type, target_dim)
                
                # Apply the gate
                if i < self.n_sites - 1:
                    layer_qc.append(UnitaryGate(U, label=f"U{i+1}_L{L}"), [i, i+1])
                else:
                    layer_qc.append(UnitaryGate(U, label=f"U{i+1}_L{L}"), [i])
                    
            full_qc = full_qc.compose(layer_qc)
            
            # Evolve the tensor network for the next layer
            if L < layers - 1:
                # Apply some evolution to the tensor network for next layer
                for i in range(self.n_sites):
                    # Simple evolution: Add small random perturbation
                    delta = np.random.randn(*tn_nodes[i].tensor.shape) * 0.01
                    new_tensor = tn_nodes[i].tensor + delta
                    # Normalize
                    new_tensor /= np.linalg.norm(new_tensor)
                    tn_nodes[i].tensor = new_tensor
                    
        # Save density matrix for later use
        full_qc.save_density_matrix(label="rho")
        return full_qc

class QuantumChemistryMPS:
    def __init__(self, mps_data_file: str, fcidump_file: str, mapper_type: str = "jw"):
        self.mps_data = np.load(mps_data_file, allow_pickle=True).item()
        self.fcidump = FCIDump.from_file(fcidump_file)
        self.problem = fcidump_to_problem(self.fcidump)
        self.mapper_type = mapper_type
        self.mapper = self._get_mapper()
        self.qubit_op = self.mapper.map(self.problem.second_q_ops()[0])
        
        # Extract symmetries
        self.z2_symmetries = self._extract_z2_symmetries()
        
        # Calculate FCI energy as reference
        self.fci_energy = self._compute_fci_energy()
        
    def _get_mapper(self):
        if self.mapper_type.lower() == "jw":
            return JordanWignerMapper()
        elif self.mapper_type.lower() == "parity":
            return ParityMapper()
        elif self.mapper_type.lower() == "bk":
            return BravyiKitaevMapper()
        else:
            raise ValueError(f"Unknown mapper type: {self.mapper_type}")
    
    def _extract_z2_symmetries(self):
        try:
            if hasattr(self.qubit_op, "paulis"):
                # For older Qiskit versions
                return Z2Symmetries.find_z2_symmetries(self.qubit_op)
            else:
                # For newer Qiskit versions with SparsePauliOp
                # This is a simplified approach - actual implementation would be more complex
                return None
        except:
            return None
    
    def _compute_fci_energy(self):
        try:
            # Use NumPy eigensolver to get exact ground state energy
            solver = NumPyMinimumEigensolver()
            result = solver.compute_minimum_eigenvalue(self.qubit_op)
            return result.eigenvalue.real
        except:
            # If the operator is too large, return None
            return None
    
    @lru_cache(maxsize=4)
    def pad_density_matrix(self, density_matrix: np.ndarray, target_qubits: int) -> np.ndarray:
        """Pad density matrix with |0⟩⟨0| states to match target qubit count"""
        current_qubits = int(np.log2(density_matrix.shape[0]))
        n_missing = target_qubits - current_qubits
        
        if n_missing <= 0:
            return density_matrix
            
        zero_proj = np.zeros((2**n_missing, 2**n_missing))
        zero_proj[0, 0] = 1.0
        
        return np.kron(density_matrix, zero_proj)
    
    def build_mps_circuit(self, target_dim: int = 2, layers: int = 5, 
                         use_tensornetwork: bool = True) -> QuantumCircuit:
        """Build quantum circuit from MPS data"""
        circuit_builder = MPSCircuitBuilder(self.mps_data)
        return circuit_builder.build_circuit(target_dim, layers, use_tensornetwork)
    
    def simulate_and_get_energy(self, circuit: QuantumCircuit, 
                               backend: Optional[Any] = None) -> Dict[str, Any]:
        """Simulate the circuit and calculate energy expectation value"""
        # Use default simulator if none provided
        if backend is None:
            backend = AerSimulator(method="density_matrix")
            
        # Transpile and run
        compiled = transpile(circuit, backend)
        job = backend.run(compiled, shots=1)
        result = job.result()
        data = result.data(0)
        
        # Extract density matrix
        if "rho" in data:
            rho_data = data["rho"]
            if isinstance(rho_data, np.ndarray):
                rho_mat = rho_data
            else:
                rho_mat = DensityMatrix(rho_data).data
                
            # Calculate energy
            n_op_qubits = self.qubit_op.num_qubits
            n_circ_qubits = int(np.log2(rho_mat.shape[0]))
            
            # Pad density matrix if needed
            if n_op_qubits > n_circ_qubits:
                rho_padded = self.pad_density_matrix(rho_mat, n_op_qubits)
            else:
                rho_padded = rho_mat
                
            # Calculate energy expectation value
            if hasattr(self.qubit_op, "to_matrix"):
                ham_matrix = self.qubit_op.to_matrix()
                energy = np.real(np.trace(ham_matrix @ rho_padded))
            else:
                # For newer Qiskit versions with SparsePauliOp
                estimator = Estimator()
                energy = estimator.run([rho_padded], [self.qubit_op]).result().values[0].real
                
            return {
                "energy": energy,
                "density_matrix": rho_mat,
                "fci_energy": self.fci_energy,
                "energy_error": abs(energy - self.fci_energy) if self.fci_energy is not None else None
            }
        else:
            raise ValueError("Density matrix 'rho' not found in simulation results")

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load MPS data and initialize quantum chemistry solver
    mps_file = "h2o_mps_complete.npy"
    fcidump_file = "H2O.STO3G.FCIDUMP"
    
    # Try different qubit mappings
    mappers = ["jw", "parity", "bk"]
    best_energy = float('inf')
    best_result = None
    best_mapper = None
    best_circuit = None
    
    for mapper in mappers:
        try:
            qc_solver = QuantumChemistryMPS(mps_file, fcidump_file, mapper)
            
            # Build and simulate circuit with tensornetwork approach
            circuit = qc_solver.build_mps_circuit(target_dim=4, layers=5, use_tensornetwork=True)
            result = qc_solver.simulate_and_get_energy(circuit)
            
            print(f"Mapper: {mapper}, Energy: {result['energy']}")
            
            if abs(result['energy']) < abs(best_energy):
                best_energy = result['energy']
                best_result = result
                best_mapper = mapper
                best_circuit = circuit
        except Exception as e:
            print(f"Error with mapper {mapper}: {str(e)}")
    
    # Print best results
    print("\nBest Results:")
    print(f"Mapper: {best_mapper}")
    print(f"Energy: {best_energy}")
    print(f"FCI Energy: {best_result['fci_energy']}")
    if best_result['energy_error'] is not None:
        print(f"Energy Error: {best_result['energy_error']}")
    
    # Draw the circuit
    print("\nCircuit:")
    print(best_circuit.draw(output="text", reverse_bits=True))

if __name__ == "__main__":
    main()
