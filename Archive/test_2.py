import numpy as np
import re
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def create_water_mps_circuit(fcidump_filename='H2O.STO3G.FCIDUMP'):
    # Read the FCIDUMP file and extract basic parameters
    try:
        with open(fcidump_filename, 'r') as f:
            fcidump_data = f.read()
        
        norb_match = re.search(r'NORB=\s*(\d+)', fcidump_data)
        norb = int(norb_match.group(1)) if norb_match else 7
        
        nelec_match = re.search(r'NELEC=(\d+)', fcidump_data)
        nelec = int(nelec_match.group(1)) if nelec_match else 10
        
        # Also extract the nuclear repulsion energy (last line in FCIDUMP)
        lines = fcidump_data.strip().split('\n')
        last_line = lines[-1].strip().split()
        nuclear_repulsion = float(last_line[0])
        
        print(f"Water molecule: {norb} orbitals, {nelec} electrons")
    except Exception as e:
        print(f"Error reading FCIDUMP file: {e}")
        print("Using default values")
        norb, nelec = 7, 10
        nuclear_repulsion = 9.8194765097824046  # Default for water STO-3G
    
    # Create quantum circuit - one qubit per orbital
    circuit = QuantumCircuit(norb)
    
    # Key point: Initialize electrons in lower energy orbitals
    for i in range(min(nelec//2, norb)):
        circuit.x(i)
    
    # Key point: Create entanglement mimicking molecular bonds
    for i in range(norb-1):
        circuit.cx(i, i+1)
    
    for i in range(norb):
        circuit.h(i)
    
    for i in range(0, norb-1, 2):
        if i+1 < norb:
            circuit.cx(i, i+1)
    
    # Set up the MPS simulator
    simulator = AerSimulator(method='matrix_product_state')
    
    # Save MPS and statevector snapshots
    snapshot_circuit = circuit.copy()
    snapshot_circuit.save_statevector(label='water_sv')
    snapshot_circuit.save_matrix_product_state(label='water_mps')
    snapshot_circuit.measure_all()
    
    tcirc = transpile(snapshot_circuit, simulator)
    result = simulator.run(tcirc).result()
    
    counts = result.get_counts(0)
    data = result.data(0)
    
    print("\nQuantum circuit for water molecule:")
    print(circuit)
    
    print("\nMeasurement outcomes:")
    print(counts)
    
    print("\nMPS representation created")

    # Calculate approximate ground state energy
    energy = calculate_ground_state_energy(fcidump_data, data, counts, nuclear_repulsion)
    
    return circuit, result, data, energy

def calculate_ground_state_energy(fcidump_data, data, counts, nuclear_repulsion):
    """
    Calculate an approximate ground state energy from the simulation results.
    """
    # Extract one-electron integrals from FCIDUMP
    one_electron_terms = []
    two_electron_terms = []
    
    lines = fcidump_data.strip().split('\n')
    for line in lines:
        if '&' in line:  # Skip header lines
            continue
        
        parts = line.strip().split()
        if len(parts) >= 5:
            value = float(parts[0])
            i, j, k, l = [int(x) for x in parts[1:5]]
            
            if k == 0 and l == 0:  # One-electron integral
                one_electron_terms.append((i, j, value))
            elif k > 0 and l > 0:  # Two-electron integral
                two_electron_terms.append((i, j, k, l, value))
    
    # Get most probable state from measurements
    max_count_state = max(counts.items(), key=lambda x: x[1])[0]
    
    # Convert binary state to occupations (simplified approach)
    occupations = [int(bit) for bit in max_count_state[::-1]]
    
    # Calculate electronic energy (very simplified)
    electronic_energy = 0.0
    
    # One-electron terms (simplified)
    for i, j, value in one_electron_terms:
        if i == j and i <= len(occupations) and occupations[i-1] == 1:
            electronic_energy += value
    
    # Add nuclear repulsion energy
    total_energy = electronic_energy + nuclear_repulsion
    
    print(f"\nApproximate Ground State Energy: {total_energy:.6f} Hartree")
    print(f"  Electronic Energy: {electronic_energy:.6f} Hartree")
    print(f"  Nuclear Repulsion: {nuclear_repulsion:.6f} Hartree")
    
    return total_energy

if __name__ == "__main__":
    circuit, result, data, energy = create_water_mps_circuit()
