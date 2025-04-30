import numpy as np 
from scipy.linalg import sqrtm, polar
from pyblock3.algebra.mpe import MPE 
from pyblock3.hamiltonian import Hamiltonian 
from pyblock3.fcidump import FCIDUMP 
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, Operator
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.primitives import Estimator
import matplotlib.pyplot as plt

# Part 1: Classical MPS calculation with DMRG
# ------------------------------------------

fd = 'H2O.STO3G.FCIDUMP'
print('FD', fd)
hamil = Hamiltonian(FCIDUMP(pg='c1').read(fd), flat=True)
print('Hamilationan: ', hamil)
mpo = hamil.build_qc_mpo()
print('MPO: ', mpo[0])

bond_dim = 100
mps = hamil.build_mps(bond_dim)
print('MPS: ', mps[0])

np.dot(mps, mpo @ mps)
expectval = np.dot(mps, mpo @ mps)
print('expectval: ', expectval)
np.tensordot(mps[0], mps[1], axes=1)
bsta = np.tensordot(mps[0], mps[1], axes=1)
print('bsta: ', bsta) 

print("MPS = ", mps.show_bond_dims())
mps = mps.canonicalize(center=0)
mps /= mps.norm()
print("MPS = ", mps.show_bond_dims())

np.dot(mps, mps)
normcheck = np.dot(mps, mps)
print('Check Norm: ', normcheck)

print("MPO = ", mpo.show_bond_dims())
mpo, _ = mpo.compress(left=True, cutoff=1E-12, norm_cutoff=1E-12)
print("MPO = ", mpo.show_bond_dims())

from pyblock3.algebra.mpe import MPE
dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0], dav_thrds=[1E-3], iprint=2, n_sweeps=10)
ener = dmrg.energies[-1]
print("Energy = %20.12f" % ener)

print('MPS energy = ', np.dot(mps, mpo @ mps))

print('MPS = ', mps.show_bond_dims())
print('MPS norm = ', mps.norm())




# Part 2: Converting MPS to Quantum Circuit using pyblock3's API
# -------------------------------------------------------------
    # Constants
    qubits_per_site = 2  # log2(4) = 2 qubits to represent each site
    
    # Create quantum circuit
    qr = QuantumRegister(n_sites * qubits_per_site, 'q')
    circuit = QuantumCircuit(qr)
    
    print(f"Creating circuit with {n_sites * qubits_per_site} qubits")
    
    # First, ensure MPS is in a canonical form
    # This might make structure more amenable to extraction
    if hasattr(mps, 'canonicalize'):
        mps = mps.canonicalize(center=0)
        print("MPS canonicalized")
    
    # Process site by site
    for i in range(n_sites):
        site_qubits = list(range(i*qubits_per_site, (i+1)*qubits_per_site))
        
        # Extract information about this site
        # We'll use the norm of the tensor to influence rotation angles
        tensor_norm = np.linalg.norm(tensors[i]) if isinstance(tensors[i], np.ndarray) else 1.0
        
        # Create entanglement structure reflecting MPS bonds
        if i > 0:
            circuit.cx(site_qubits[0], (i-1)*qubits_per_site)
        
        # Apply rotations to this site based on tensor values
        circuit.rx(np.pi * min(1.0, tensor_norm/10), site_qubits[0])
        if len(site_qubits) > 1:
            circuit.ry(np.pi * min(0.8, tensor_norm/12), site_qubits[1])
            circuit.cx(site_qubits[0], site_qubits[1])
