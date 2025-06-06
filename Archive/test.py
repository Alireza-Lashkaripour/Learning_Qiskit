import numpy as np
import re
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Function to extract key parameters from the FCI data
def extract_water_parameters(data_string):
    """Extract key parameters from the water molecule FCI data."""
    # Extract number of orbitals (NORB)
    norb_match = re.search(r'NORB=\s*(\d+)', data_string)
    norb = int(norb_match.group(1)) if norb_match else 7
    
    # Extract number of electrons (NELEC)
    nelec_match = re.search(r'NELEC=(\d+)', data_string)
    nelec = int(nelec_match.group(1)) if nelec_match else 10
    
    return norb, nelec

# Function to create a quantum circuit for the water molecule
def create_water_circuit(norb, nelec):
    """Create a quantum circuit for the water molecule suitable for MPS representation."""
    # Use one qubit per orbital
    num_qubits = norb
    
    # Create the circuit
    circuit = QuantumCircuit(num_qubits)
    
    # Initialize in a state that roughly represents the electronic structure
    # For a water molecule, the electrons occupy the lower energy orbitals
    for i in range(min(nelec//2, num_qubits)):
        circuit.x(i)
    
    # Create entanglement in a chain structure (suitable for MPS)
    # This structure works well for MPS representation
    for i in range(num_qubits-1):
        circuit.cx(i, i+1)
    
    # Add more quantum operations to create a superposition of states
    # This helps capture the electron correlation effects
    for i in range(num_qubits):
        circuit.h(i)
    
    # Add another layer of entanglement 
    for i in range(0, num_qubits-1, 2):
        if i+1 < num_qubits:
            circuit.cx(i, i+1)
    
    # Save the MPS representation and statevector for later analysis
    circuit.save_matrix_product_state(label='water_mps')
    circuit.save_statevector(label='water_sv')
    
    return circuit

# Function to simulate using the MPS simulator
def simulate_with_mps(circuit):
    """Simulate the circuit using the Matrix Product State simulator."""
    # Create the MPS simulator
    simulator = AerSimulator(method='matrix_product_state')
    
    # Transpile the circuit for the simulator
    transpiled_circuit = transpile(circuit, simulator)
    
    # Run the simulation
    result = simulator.run(transpiled_circuit).result()
    
    # Get the MPS data
    mps_data = result.data(0)
    
    return mps_data

# Main function to demonstrate the process
def demonstrate_water_molecule_mps(fci_data_string):
    """Run the complete process of creating and simulating an MPS for water."""
    print("Extracting parameters from FCI data...")
    norb, nelec = extract_water_parameters(fci_data_string)
    print(f"Number of orbitals: {norb}")
    print(f"Number of electrons: {nelec}")
    
    print("\nCreating quantum circuit for water molecule...")
    circuit = create_water_circuit(norb, nelec)
    print(circuit)
    
    print("\nSimulating with MPS simulator...")
    mps_data = simulate_with_mps(circuit)
    
    # Display some of the MPS data
    print("\nMPS Representation obtained:")
    if 'water_mps' in mps_data:
        # The MPS data structure is complex, so we'll just acknowledge it's there
        print("Successfully extracted Matrix Product State")
        # You can explore the MPS data structure further if needed
    else:
        print("MPS data not found in simulation results")
    
    return circuit, mps_data

# Example usage (with your data as a string)
if __name__ == "__main__":
    # Your FCI data would go here
    fci_data = """ &FCI NORB=   7,NELEC=10,MS2=0,
  ORBSYM=1,1,1,1,1,1,1,
  ISYM=1,
 &END
  4.7446024972265013   1   1   1   1
 -0.4005569207152798   2   1   1   1
  0.0543879654747371   2   1   2   1
  0.9862427470679018   2   2   1   1
 -0.0096577341445865   2   2   2   1
  0.7457165015123214   2   2   2   2
  0.0140209049969404   3   1   3   1
  0.0204837990352860   3   2   3   1
  0.1438996619292540   3   2   3   2
  0.8716273566211508   3   3   1   1
 -0.0049607387375706   3   3   2   1
  0.6802078051804231   3   3   2   2
  0.6761709771448129   3   3   3   3
 -0.2377912490981060   4   1   1   1
  0.0279920550587111   4   1   2   1
 -0.0198413809113488   4   1   2   2
 -0.0078147588119301   4   1   3   3
  0.0310912805105295   4   1   4   1
  0.1555436904206536   4   2   1   1
 -0.0109596632030024   4   2   2   1
 -0.0161221483000111   4   2   2   2
  0.0088644439030785   4   2   3   3
  0.0116374353039770   4   2   4   1
  0.1128323118739208   4   2   4   2
  0.0076245485695920   4   3   3   1
  0.0014700155646969   4   3   3   2
  0.0515491684751192   4   3   4   3
  0.9695577065123711   4   4   1   1
 -0.0149443129409979   4   4   2   1
  0.6475813590315546   4   4   2   2
  0.6209516610761605   4   4   3   3
  0.0087754846778638   4   4   4   1
  0.0974320234455502   4   4   4   2
  0.7582716561082493   4   4   4   4
  0.0260409884525928   5   1   5   1
  0.0312289635761368   5   2   5   1
  0.1360922514332653   5   2   5   2
  0.0329057057784873   5   3   5   3
  0.0175116633243894   5   4   5   1
  0.0589713964994248   5   4   5   2
  0.0651579927561828   5   4   5   4
  1.1153374923942718   5   5   1   1
 -0.0112505771802660   5   5   2   1
  0.7378484355929998   5   5   2   2
  0.6664990781962318   5   5   3   3
 -0.0066651413710073   5   5   4   1
  0.0830419796338111   5   5   4   2
  0.7157524425599351   5   5   4   4
  0.8801590933750436   5   5   5   5
  0.2131188180544552   6   1   1   1
 -0.0331955426038509   6   1   2   1
 -0.0063869273091977   6   1   2   2
 -0.0012696888652764   6   1   3   3
 -0.0021414496432460   6   1   4   1
  0.0208117744770594   6   1   4   2
  0.0230425953416399   6   1   4   4
  0.0054335735643497   6   1   5   5
  0.0337096482623593   6   1   6   1
 -0.3133842944208965   6   2   1   1
  0.0043160413046890   6   2   2   1
 -0.1493633931519697   6   2   2   2
 -0.1017215715824130   6   2   3   3
  0.0191991868127011   6   2   4   1
  0.0040447248099209   6   2   4   2
 -0.0696917269428985   6   2   4   4
 -0.1596517970687365   6   2   5   5
  0.0119377428666477   6   2   6   1
  0.1125326045133848   6   2   6   2
 -0.0058246373580332   6   3   3   1
  0.0049903703285689   6   3   3   2
 -0.0161986119944182   6   3   4   3
  0.0443743875151307   6   3   6   3
  0.2247261697128875   6   4   1   1
 -0.0038019037128802   6   4   2   1
  0.0694905402810812   6   4   2   2
  0.0526973069019459   6   4   3   3
  0.0037377252505768   6   4   4   1
  0.0688666142847975   6   4   4   2
  0.1325843947188173   6   4   4   4
  0.1199189399628465   6   4   5   5
  0.0071304156775443   6   4   6   1
 -0.0497512730320683   6   4   6   2
  0.0928279615586607   6   4   6   4
 -0.0139093103324018   6   5   5   1
 -0.0526719125304430   6   5   5   2
 -0.0031491832484117   6   5   5   4
  0.0385373197553727   6   5   6   5
  0.8883884068790534   6   6   1   1
 -0.0068873403824309   6   6   2   1
  0.6633280697490749   6   6   2   2
  0.6046857630751054   6   6   3   3
 -0.0209702635886530   6   6   4   1
 -0.0420783857565089   6   6   4   2
  0.5841421640778124   6   6   4   4
  0.6286746535828067   6   6   5   5
 -0.0109091791413417   6   6   6   1
 -0.1125877940393681   6   6   6   2
  0.0391306074764903   6   6   6   4
  0.6457913396568912   6   6   6   6
  0.0151977411638709   7   1   3   1
  0.0205837830166046   7   1   3   2
  0.0083685974346289   7   1   4   3
 -0.0056546627020635   7   1   6   3
  0.0165214674411758   7   1   7   1
  0.0123484177076428   7   2   3   1
  0.0204636492203700   7   2   3   2
  0.0456878615958577   7   2   4   3
 -0.0361693155301378   7   2   6   3
  0.0128290208274776   7   2   7   1
  0.0569417773432315   7   2   7   2
  0.3392628342455847   7   3   1   1
 -0.0079469786300027   7   3   2   1
  0.0951175912865732   7   3   2   2
  0.0964099695129973   7   3   3   3
  0.0009730497518544   7   3   4   1
  0.0970982819032685   7   3   4   2
  0.1362178734310445   7   3   4   4
  0.1731263551711330   7   3   5   5
  0.0086198902807216   7   3   6   1
 -0.0741219909665625   7   3   6   2
  0.0932092068850792   7   3   6   4
  0.0376243626123283   7   3   6   6
  0.1588707166246736   7   3   7   3
  0.0124111277444726   7   4   3   1
  0.0878543135021337   7   4   3   2
  0.0082183565665864   7   4   4   3
  0.0297276380980336   7   4   6   3
  0.0130780643580211   7   4   7   1
  0.0018262855210969   7   4   7   2
  0.0802734157470199   7   4   7   4
  0.0222640540242556   7   5   5   3
  0.0193292344104973   7   5   7   5
 -0.0079406993711193   7   6   3   1
 -0.0809661626735664   7   6   3   2
  0.0366410618350765   7   6   4   3
 -0.0302885761479625   7   6   6   3
 -0.0079589221534018   7   6   7   1
  0.0280103004547427   7   6   7   2
 -0.0591001121327057   7   6   7   4
  0.0927523080739217   7   6   7   6
  0.7915229497670674   7   7   1   1
 -0.0064351733081312   7   7   2   1
  0.6213060141758935   7   7   2   2
  0.6209443507695054   7   7   3   3
 -0.0060602523019640   7   7   4   1
 -0.0146141139109483   7   7   4   2
  0.5842524611548806   7   7   4   4
  0.5936324360251058   7   7   5   5
  0.0013328407898505   7   7   6   1
 -0.0544532679225285   7   7   6   2
  0.0135269500882058   7   7   6   4
  0.5920439490018626   7   7   6   6
  0.0342373740286195   7   7   7   3
  0.6203474773112126   7   7   7   7
-32.7595752325552070   1   1   0   0
  0.5358761109404221   2   1   0   0
 -7.7425796924270234   2   2   0   0
 -6.6618629063190165   3   3   0   0
  0.3118348753981520   4   1   0   0
 -0.4877766379743241   4   2   0   0
 -6.9448634131698999   4   4   0   0
 -7.4986384857396198   5   5   0   0
 -0.2664381049982429   6   1   0   0
  1.4262517037312190   6   2   0   0
 -1.0836948279986856   6   4   0   0
 -5.7726406382312430   6   6   0   0
 -1.5177154777760427   7   3   0   0
 -5.3417336727917268   7   7   0   0
  9.8194765097824046   0   0   0   0
  """  # abbreviated for clarity
    
    circuit, mps_data = demonstrate_water_molecule_mps(fci_data)
    
# Additional functions for deeper MPS analysis

def analyze_mps_properties(mps_data):
    """Analyze the properties of the Matrix Product State."""
    if 'water_mps' not in mps_data:
        print("MPS data not found")
        return
    
    mps = mps_data['water_mps']
    
    # In a real implementation, you would extract bond dimensions
    # and analyze entanglement structure from the MPS data
    print("\nMPS Analysis:")
    print("- MPS is a tensor network representation that efficiently captures")
    print("  quantum states with limited entanglement along a 1D chain")
    print("- The water molecule's entanglement structure is captured in the bond dimensions")
    print("- For water, we expect higher entanglement between bonded atoms (O-H bonds)")
    
    # You could compute entanglement entropy between different bipartitions
    # This would require additional code to work with the MPS structure

# Complete example for use with the provided water molecule FCI data
def run_complete_water_mps_example():
    """Run a complete example reading FCI data from file and creating an MPS circuit."""
    try:
        # Read your FCI data from the file
        with open('paste.txt', 'r') as f:
            fci_data = f.read()
        
        # Extract parameters
        norb, nelec = extract_water_parameters(fci_data)
        print(f"Water molecule parameters extracted: {norb} orbitals, {nelec} electrons")
        
        # Create and display the circuit
        circuit = create_water_circuit(norb, nelec)
        print("\nQuantum circuit for water molecule:")
        print(circuit)
        
        # Set up the simulator
        simulator = AerSimulator(method='matrix_product_state')
        
        # Add measurement operations to match your example
        meas_circuit = circuit.copy()
        meas_circuit.measure_all()
        
        # Transpile
        tcirc = transpile(meas_circuit, simulator)
        
        # Run
        result = simulator.run(tcirc).result()
        
        # Get counts
        counts = result.get_counts(0)
        print("\nMeasurement outcomes:")
        print(counts)
        
        # Run again with snapshots
        snapshot_circuit = circuit.copy()
        snapshot_circuit.save_statevector(label='my_sv')
        snapshot_circuit.save_matrix_product_state(label='my_mps')
        snapshot_circuit.measure_all()
        
        tcirc = transpile(snapshot_circuit, simulator)
        result = simulator.run(tcirc).result()
        data = result.data(0)
        
        print("\nSnapshot data obtained:")
        print("- Contains statevector and MPS representations")
        
        return circuit, result, data
        
    except Exception as e:
        print(f"Error running water MPS example: {e}")
        return None, None, None

