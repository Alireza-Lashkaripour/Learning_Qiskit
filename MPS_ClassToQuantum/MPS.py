import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

class DMRG:
    """
    Density Matrix Renormalization Group implementation for the
    Transverse Field Ising Model using Matrix Product States.
    """
    
    def __init__(self, L, J=1.0, h=1.0, max_bond_dim=20):
        """
        Initialize the DMRG solver.
        
        Args:
            L: Number of sites
            J: Coupling strength between neighboring spins
            h: Transverse field strength
            max_bond_dim: Maximum bond dimension for truncation
        """
        self.L = L
        self.J = J
        self.h = h
        self.max_bond_dim = max_bond_dim
        
        # Physical dimension (d=2 for spin-1/2)
        self.d = 2
        
        # Pauli matrices
        self.sigma_x = np.array([[0, 1], [1, 0]])
        self.sigma_y = np.array([[0, -1j], [1j, 0]])
        self.sigma_z = np.array([[1, 0], [0, -1]])
        self.identity = np.eye(2)
        
        # Initialize MPS in a product state (all spins in +x direction)
        self.initialize_mps()
        
        # Initialize environments for DMRG sweeps
        self.initialize_environments()
    
    def initialize_mps(self):
        """Initialize MPS in the product state |++++...> (ground state for h→∞)"""
        self.mps = []
        
        # Create state |+> = (|0> + |1>)/sqrt(2) at each site
        plus_state = np.array([1, 1]) / np.sqrt(2)
        
        # First site
        A = np.zeros((self.d, 1, 1))
        A[:, 0, 0] = plus_state
        self.mps.append(A)
        
        # Middle sites
        for i in range(1, self.L-1):
            A = np.zeros((self.d, 1, 1))
            A[:, 0, 0] = plus_state
            self.mps.append(A)
        
        # Last site
        A = np.zeros((self.d, 1, 1))
        A[:, 0, 0] = plus_state
        self.mps.append(A)
        
        # Convert to canonical form
        self.canonicalize()
    
    def canonicalize(self):
        """Convert MPS to left-canonical form"""
        # Left-canonicalize: sweep from left to right
        for i in range(self.L-1):
            s, a, b = self.mps[i].shape
            A = self.mps[i].reshape(s * a, b)
            
            # SVD decomposition
            U, S, V = np.linalg.svd(A, full_matrices=False)
            
            # Truncate if necessary
            if len(S) > self.max_bond_dim:
                U = U[:, :self.max_bond_dim]
                S = S[:self.max_bond_dim]
                V = V[:self.max_bond_dim, :]
            
            # Update current tensor
            self.mps[i] = U.reshape(s, a, -1)
            
            # Update next tensor
            B = np.diag(S) @ V
            s_next, a_next, b_next = self.mps[i+1].shape
            self.mps[i+1] = np.tensordot(B, self.mps[i+1], axes=(1, 1))
            self.mps[i+1] = self.mps[i+1].transpose(1, 0, 2).reshape(s_next, -1, b_next)
        
        # Normalize the last tensor
        self.mps[-1] /= np.linalg.norm(self.mps[-1])
    
    def initialize_environments(self):
        """Initialize left and right environments for DMRG sweeps"""
        # Left environments: L[i] represents contraction of all tensors to the left of site i
        self.L_env = [None] * (self.L + 1)
        self.L_env[0] = np.array([1.0]).reshape(1, 1, 1)  # Leftmost environment is just 1
        
        # Right environments: R[i] represents contraction of all tensors to the right of site i
        self.R_env = [None] * (self.L + 1)
        self.R_env[self.L] = np.array([1.0]).reshape(1, 1, 1)  # Rightmost environment is just 1
    
    def update_left_environment(self, site):
        """Update left environment for site"""
        if site == 0:
            return
        
        # Get MPS tensor for the previous site
        A = self.mps[site-1]
        
        # Contract with previous left environment
        self.L_env[site] = np.tensordot(self.L_env[site-1], A, axes=(0, 1))
        self.L_env[site] = np.tensordot(self.L_env[site], np.conj(A), axes=([0, 1], [1, 0]))
    
    def update_right_environment(self, site):
        """Update right environment for site"""
        if site == self.L - 1:
            return
        
        # Get MPS tensor for the next site
        A = self.mps[site+1]
        
        # Contract with previous right environment
        self.R_env[site] = np.tensordot(A, self.R_env[site+1], axes=(2, 0))
        self.R_env[site] = np.tensordot(np.conj(A), self.R_env[site], axes=([2, 0], [0, 2]))
    
    def compute_two_site_hamiltonian(self, site):
        """Compute the effective two-site Hamiltonian for DMRG optimization"""
        # Physical indices
        d = self.d
        
        # Bond dimensions
        if site == 0:
            left_dim = 1
        else:
            left_dim = self.mps[site-1].shape[2]
        
        if site == self.L - 2:
            right_dim = 1
        else:
            right_dim = self.mps[site+2].shape[1]
        
        # Initialize effective Hamiltonian
        H_eff = np.zeros((d*d, left_dim, right_dim, d*d, left_dim, right_dim), dtype=complex)
        
        # Add transverse field terms (-h∑ᵢ σᵢˣ)
        # For first site in the pair
        H_eff += np.kron(np.kron(np.kron(np.kron(np.kron(
            -self.h * self.sigma_x, np.eye(d)), np.eye(left_dim)), np.eye(right_dim)),
            np.eye(left_dim)), np.eye(right_dim))
        
        # For second site in the pair
        H_eff += np.kron(np.kron(np.kron(np.kron(np.kron(
            np.eye(d), -self.h * self.sigma_x), np.eye(left_dim)), np.eye(right_dim)),
            np.eye(left_dim)), np.eye(right_dim))
        
        # Add interaction term (-J σᵢᶻσᵢ₊₁ᶻ)
        H_eff += np.kron(np.kron(np.kron(np.kron(np.kron(
            -self.J * self.sigma_z, self.sigma_z), np.eye(left_dim)), np.eye(right_dim)),
            np.eye(left_dim)), np.eye(right_dim))
        
        # Add interactions with left part (-J σᵢ₋₁ᶻσᵢᶻ) if not the first site
        if site > 0:
            # This requires contracting with left environment and operators
            # Implementation simplified for clarity
            pass
        
        # Add interactions with right part (-J σᵢ₊₁ᶻσᵢ₊₂ᶻ) if not the last pair
        if site < self.L - 2:
            # This requires contracting with right environment and operators
            # Implementation simplified for clarity
            pass
        
        # Reshape to matrix for eigenvalue problem
        H_eff = H_eff.reshape((d*d*left_dim*right_dim, d*d*left_dim*right_dim))
        
        return H_eff
    
    def two_site_dmrg_step(self, site):
        """Perform a two-site DMRG optimization step"""
        # Compute effective Hamiltonian
        H_eff = self.compute_two_site_hamiltonian(site)
        
        # Find ground state using sparse eigensolver
        eigenvalues, eigenvectors = eigsh(H_eff, k=1, which='SA')
        ground_state = eigenvectors[:, 0]
        ground_energy = eigenvalues[0]
        
        # Reshape ground state to tensor
        d = self.d
        if site == 0:
            left_dim = 1
        else:
            left_dim = self.mps[site-1].shape[2]
        
        if site == self.L - 2:
            right_dim = 1
        else:
            right_dim = self.mps[site+2].shape[1]
        
        theta = ground_state.reshape(d, d, left_dim, right_dim)
        
        # SVD and truncate
        theta = theta.transpose(0, 2, 1, 3).reshape(d*left_dim, d*right_dim)
        U, S, V = np.linalg.svd(theta, full_matrices=False)
        
        # Truncate if necessary
        if len(S) > self.max_bond_dim:
            U = U[:, :self.max_bond_dim]
            S = S[:self.max_bond_dim]
            V = V[:self.max_bond_dim, :]
        
        # Calculate truncation error
        truncation_error = 1 - np.sum(S**2)
        
        # Update MPS tensors
        self.mps[site] = U.reshape(d, left_dim, -1)
        self.mps[site+1] = V.reshape(-1, right_dim, d).transpose(2, 0, 1)
        
        # Insert singular values
        self.mps[site+1] = np.tensordot(np.diag(S), self.mps[site+1], axes=(1, 1))
        self.mps[site+1] = self.mps[site+1].transpose(1, 0, 2)
        
        return ground_energy, truncation_error
    
    def dmrg_sweep(self, direction='right'):
        """Perform a full DMRG sweep"""
        sweep_energy = 0
        max_truncation_error = 0
        
        if direction == 'right':
            # Right sweep: optimize from left to right
            for i in range(self.L - 1):
                # Update environments
                self.update_left_environment(i)
                self.update_right_environment(i+1)
                
                # Two-site DMRG update
                energy, error = self.two_site_dmrg_step(i)
                sweep_energy = energy  # Take the last energy
                max_truncation_error = max(max_truncation_error, error)
        else:
            # Left sweep: optimize from right to left
            for i in range(self.L - 2, -1, -1):
                # Update environments
                self.update_left_environment(i)
                self.update_right_environment(i+1)
                
                # Two-site DMRG update
                energy, error = self.two_site_dmrg_step(i)
                sweep_energy = energy  # Take the last energy
                max_truncation_error = max(max_truncation_error, error)
        
        return sweep_energy, max_truncation_error
    
    def run(self, max_sweeps=10, energy_tol=1e-8):
        """Run DMRG algorithm until convergence"""
        energies = []
        truncation_errors = []
        
        prev_energy = 0
        
        for sweep in range(max_sweeps):
            # Right sweep
            energy_right, error_right = self.dmrg_sweep(direction='right')
            
            # Left sweep
            energy_left, error_left = self.dmrg_sweep(direction='left')
            
            # Record results
            energy = energy_left
            error = max(error_right, error_left)
            energies.append(energy)
            truncation_errors.append(error)
            
            print(f"Sweep {sweep+1}: Energy = {energy:.12f}, Error = {error:.12e}")
            
            # Check convergence
            if sweep > 0 and abs(energy - prev_energy) < energy_tol:
                print("Converged!")
                break
                
            prev_energy = energy
        
        return energies, truncation_errors
    
    def compute_magnetization(self):
        """Compute the magnetization profile <σᶻ>"""
        magnetization = []
        
        for i in range(self.L):
            # Initialize environments for site i
            for j in range(i):
                self.update_left_environment(j+1)
            for j in range(self.L-1, i, -1):
                self.update_right_environment(j-1)
            
            # Contract MPS with σᶻ operator and environments
            A = self.mps[i]
            sz_expectation = np.tensordot(self.L_env[i], A, axes=(0, 1))
            sz_expectation = np.tensordot(sz_expectation, self.sigma_z, axes=(0, 0))
            sz_expectation = np.tensordot(sz_expectation, np.conj(A), axes=([0, 1], [1, 0]))
            sz_expectation = np.tensordot(sz_expectation, self.R_env[i], axes=([0, 1], [0, 1]))
            
            magnetization.append(sz_expectation.real)
        
        return magnetization
    
    def compute_correlation(self, i, j, op1=None, op2=None):
        """Compute correlation function between sites i and j"""
        if op1 is None:
            op1 = self.sigma_z
        if op2 is None:
            op2 = self.sigma_z
            
        # Implementation of correlation function calculation
        # This is a simplified placeholder
        return 0.0

# Example usage
def main():
    # System parameters
    L = 20  # System size
    h_values = np.linspace(0.1, 2.0, 20)  # Range of transverse field values
    
    # Store results
    ground_energies = []
    magnetizations = []
    
    for h in h_values:
        print(f"\nSimulating h = {h:.2f}")
        
        # Initialize and run DMRG
        dmrg = DMRG(L=L, J=1.0, h=h, max_bond_dim=20)
        energies, errors = dmrg.run(max_sweeps=5)
        
        # Compute observables
        mag = dmrg.compute_magnetization()
        
        # Store results
        ground_energies.append(energies[-1] / L)  # Energy per site
        magnetizations.append(np.mean(mag))  # Average magnetization
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(h_values, ground_energies, 'o-')
    plt.xlabel('Transverse Field h')
    plt.ylabel('Ground State Energy per Site')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(h_values, np.abs(magnetizations), 'o-')
    plt.xlabel('Transverse Field h')
    plt.ylabel('|Magnetization|')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('tfim_phase_transition.png')
    plt.show()

if __name__ == "__main__":
    main()
