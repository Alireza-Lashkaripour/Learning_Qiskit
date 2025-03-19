import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_circuit_layout, plot_bloch_multivector
from qiskit.quantum_info import Statevector

def create_5site_mps_circuit():
    # Create a 5-qubit quantum circuit
    qc = QuantumCircuit(5, 5)
    
    # MPS-like state preparation
    # 1. Initial Hadamard gates for superposition
    for qubit in range(5):
        qc.h(qubit)
    
    # 2. Entangle adjacent qubits (nearest-neighbor interactions)
    qc.cx(0, 1)  # Entangle first two qubits
    qc.cx(1, 2)  # Entangle next two
    qc.cx(2, 3)  # Continue entanglement
    qc.cx(3, 4)  # Final entanglement
    
    # 3. Add some local rotations to create more complex state
    qc.rx(np.pi/4, 0)
    qc.ry(np.pi/3, 2)
    qc.rz(np.pi/2, 4)
    
    # 4. Measurement (optional)
    qc.measure_all()
    
    return qc

def visualize_mps_circuit():
    # Create the MPS circuit
    mps_circuit = create_5site_mps_circuit()
    
    # Plotting options
    plt.figure(figsize=(15, 10))
    
    # 1. Circuit Layout Visualization
    plt.subplot(2, 2, 1)
    plot_circuit_layout(mps_circuit)
    plt.title('Circuit Layout')
    
    # 2. Circuit Diagram
    plt.subplot(2, 2, 2)
    mps_circuit.draw(output='mpl')
    plt.title('MPS Quantum Circuit')
    
    # 3. State Vector Visualization
    backend = Aer.get_backend('statevector_simulator')
    job = execute(mps_circuit, backend)
    statevector = job.result().get_statevector()
    
    plt.subplot(2, 2, 3)
    plot_bloch_multivector(statevector)
    plt.title('Bloch Sphere Representation')
    
    # 4. Probability Distribution
    plt.subplot(2, 2, 4)
    probabilities = np.abs(statevector.data)**2
    plt.bar(range(len(probabilities)), probabilities)
    plt.title('State Probability Distribution')
    plt.xlabel('State Index')
    plt.ylabel('Probability')
    
    plt.tight_layout()
    plt.show()

# Run the visualization
visualize_mps_circuit()

# Optional: Detailed State Analysis
def analyze_mps_state():
    mps_circuit = create_5site_mps_circuit()
    backend = Aer.get_backend('statevector_simulator')
    job = execute(mps_circuit, backend)
    statevector = job.result().get_statevector()
    
    print("Statevector Shape:", statevector.shape)
    print("Total Probability:", np.sum(np.abs(statevector.data)**2))
    
    # Entanglement entropy calculation (simplified)
    def von_neumann_entropy(probabilities):
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    # Example of local reduced density matrices
    reduced_entropies = []
    for i in range(4):  # Check entanglement between adjacent sites
        # This is a simplified demonstration
        local_probs = np.abs(statevector.data[i:i+2])**2
        reduced_entropies.append(von_neumann_entropy(local_probs))
    
    print("Local Entanglement Entropies:", reduced_entropies)

# Run state analysis
analyze_mps_state()
