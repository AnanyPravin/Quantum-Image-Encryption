import matplotlib
import qiskit
from qiskit import assemble, transpile
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, circuit_drawer
import matplotlib.pyplot as plt
import numpy as np


def nxor(circuit, ctrl, target):
    circuit.cx(ctrl, target)
    circuit.x(target)


def apply_T1(qr, qc, ancilla):
    # Input: 0000 -> Output: 0110
    qc.x([qr[3], qr[2], qr[1], qr[0]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2])
    qc.x([qr[3], qr[2], qr[1], qr[0]]) 
    qc.barrier()

    # Input: 0001 -> Output: 1111
    qc.x([qr[3], qr[2], qr[1]]) 
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1])
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2])
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3])
    qc.x([qr[3], qr[2], qr[1]]) 
    qc.barrier()

    # Input: 0010 -> Output: 1011
    qc.x([qr[3], qr[2], qr[0]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0]) 
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1])
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3])
    qc.x([qr[3], qr[2], qr[0]]) 
    qc.barrier()

    # Input: 0011 -> Output: 0111
    qc.x([qr[3], qr[2]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1])
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2])
    qc.x([qr[3], qr[2]])  
    qc.barrier()

    # Input: 0100 -> Output: 1110
    qc.x([qr[3], qr[1], qr[0]]) 
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2])
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1])
    qc.x([qr[3], qr[1], qr[0]]) 
    qc.barrier()

    # Input: 0101 -> Output: 1000
    qc.x([qr[3], qr[1]]) 
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3])  
    qc.x([qr[3], qr[1]]) 
    qc.barrier()

    # Input: 0110 -> Output: 0000
    
    # Input: 0111 -> Output: 0011
    qc.x([qr[3]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0]) 
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1])
    qc.x([qr[3]])  
    qc.barrier()

    # Input: 1000 -> Output: 0101
    qc.x([qr[2], qr[1], qr[0]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2])
    qc.x([qr[2], qr[1], qr[0]]) 
    qc.barrier()

    # Input: 1001 -> Output: 1101
    qc.x([qr[2], qr[1]]) 
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2])
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3])
    qc.x([qr[2], qr[1]])  
    qc.barrier()
    
    # Input: 1010 -> Output: 1100
    qc.x([qr[2], qr[0]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3]) 
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2])
    qc.x([qr[2], qr[0]]) 
    qc.barrier()
    
    # Input: 1011 -> Output: 0010
    qc.x([qr[2]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1]) 
    qc.x([qr[2]]) 
    qc.barrier()

    # Input: 1100 -> Output: 1010
    qc.x([qr[1], qr[0]]) 
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1]) 
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3])
    qc.x([qr[1], qr[0]]) 
    qc.barrier()

    # Input: 1101 -> Output: 1001
    qc.x([qr[1]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0]) 
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3])
    qc.x([qr[1]])  
    qc.barrier()

    # Input: 1110 -> Output: 0100
    qc.x([qr[0]]) 
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2]) 
    qc.x([qr[0]]) 
    qc.barrier()

    # Input: 1111 -> Output: 0001
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0]) 
    
    for i in range(4):
        qc.swap(qr[i], ancilla[i])
    return qc

def apply_T2(qr, qc, ancilla):
    qc.x([qr[3], qr[2], qr[1], qr[0]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1])
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2])
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3])
    qc.x([qr[3], qr[2], qr[1], qr[0]])  
    qc.barrier()

    # Input: 0001 -> Output: 0110
    qc.x([qr[3], qr[2], qr[1]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2])
    qc.x([qr[3], qr[2], qr[1]])  
    qc.barrier()

    # Input: 0010 -> Output: 1110
    qc.x([qr[3], qr[2], qr[0]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2])
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3])
    qc.x([qr[3], qr[2], qr[0]])  
    qc.barrier()

    # Input: 0011 -> Output: 1100
    qc.x([qr[3], qr[2]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1])
    qc.x([qr[3], qr[2]])  
    qc.barrier()

    # Input: 0100 -> Output: 1101
    qc.x([qr[3], qr[1], qr[0]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2])
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3])
    qc.x([qr[3], qr[1], qr[0]])  
    qc.barrier()

    # Input: 0101 -> Output: 1010
    qc.x([qr[3], qr[1]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3])
    qc.x([qr[3], qr[1]])  
    qc.barrier()

    # Input: 0110 -> Output: 0001
    qc.x([qr[3], qr[0]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0])  
    qc.x([qr[3], qr[0]])  
    qc.barrier()

    # Input: 0111 -> Output: 1011
    qc.x([qr[3]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1])
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3])
    qc.x([qr[3]])  
    qc.barrier()

    # Input: 1000 -> Output: 1001
    qc.x([qr[2], qr[1], qr[0]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3])
    qc.x([qr[2], qr[1], qr[0]])  
    qc.barrier()

    # Input: 1001 -> Output: 1000
    qc.x([qr[2], qr[1]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[3])  
    qc.x([qr[2], qr[1]])  
    qc.barrier()

    # Input: 1010 -> Output: 0101
    qc.x([qr[2], qr[0]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2])
    qc.x([qr[2], qr[0]])  
    qc.barrier()

    qc.x([qr[2]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[0])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1])
    qc.x([qr[2]])  
    qc.barrier()

    qc.x([qr[1], qr[0]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1])
    qc.x([qr[1], qr[0]])  
    qc.barrier()

    qc.x([qr[1]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[2])  
    qc.x([qr[1]])  
    qc.barrier()

    qc.x([qr[0]])  
    qc.mcx([qr[3], qr[2], qr[1], qr[0]], ancilla[1])  
    qc.x([qr[0]])  
    qc.barrier()

    for i in range(4):
        qc.swap(qr[i], ancilla[i])

# F function used in Feistel rounds
def apply_F_function(qc, intensity_register, ancilla):
    # Split 8-bit intensity register into two 4-bit blocks
    qr1 = intensity_register[:4]
    qr2 = intensity_register[4:]

    # Apply transformations to the two blocks
    apply_T1(qr1, qc, ancilla[:4])
    apply_T2(qr2, qc, ancilla[4:8])

    # Swap bits between the two blocks (within the intensity register)
    qr3 = [qr1[0], qr1[1], qr2[2], qr1[3]]
    qr4 = [qr2[0], qr2[1], qr1[2], qr2[3]]
    
    apply_T1(qr3, qc, ancilla[:4])
    apply_T2(qr4, qc, ancilla[4:8])

# Feistel round function
def feistel_round(qc, intensity_register, key, ancilla):
    # Apply F function to the intensity register
    apply_F_function(qc, intensity_register, ancilla)
    
    # Apply NXOR with the key to the intensity register
    for i in range(len(intensity_register)):
        nxor(qc, key[i], intensity_register[i])

# Generate subkey for each round
def generate_subkey(circuit, key, round_index):
    for i in range(len(key)):
        if round_index & (1 << i):
            circuit.x(key[i])
    return key

# Feistel structure with multiple rounds
def feistel_structure(qc, intensity_register, key, ancilla, num_rounds=2):
    # Apply multiple Feistel rounds
    for round_index in range(num_rounds):
        subkey = generate_subkey(qc, key, round_index)
        feistel_round(qc, intensity_register, subkey, ancilla)

# Encryption function that applies the Feistel structure
def encrypt(circuit, f, key, ancilla, n_rounds=10):
    # Apply Feistel encryption to the intensity register
    feistel_structure(circuit, f, key, ancilla, num_rounds=n_rounds)

# Function to display the grayscale image
def display_image(image, title="Image"):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Input image (4x4 grayscale image)
input_image = [[0, 15, 31, 47],
               [63, 79, 95, 111],
               [127, 143, 159, 175],
               [191, 207, 223, 239]]
# Convert image to binary NEQR format
input_image_binary = [[bin(j)[2:].zfill(8) for j in i] for i in input_image]
n = int(np.log2(len(input_image_binary)))  # Number of qubits per coordinate

coded_image = [(input_image_binary[y][x], bin(y)[2:].zfill(n), bin(x)[2:].zfill(n)) for y in range(len(input_image_binary)) for x in range(len(input_image_binary))]

# Initialize quantum registers
f = QuantumRegister(8, 'f')  # Intensity register
yr = QuantumRegister(n, 'y')  # y-coordinate register
xr = QuantumRegister(n, 'x')  # x-coordinate register
c = ClassicalRegister(2*n + 8, 'c')  # Classical register
ancilla = QuantumRegister(8, 'anc')  # Ancilla qubits for Feistel network
key = QuantumRegister(8, 'key')  # Key for Feistel encryption

# Create the circuit
circuit = QuantumCircuit(f, yr, xr, ancilla, key, c)

# Prepare the NEQR state (superposition over coordinates)
for xyc in range(n):
    circuit.h(yr[xyc])
    circuit.h(xr[xyc])

# Encode pixel intensities using multi-controlled X gates
for pixel in coded_image:
    # Apply X gates to set coordinates
    for xc in range(n):
        if pixel[2][xc] == '0':
            circuit.x(xr[xc])
    for yc in range(n):
        if pixel[1][yc] == '0':
            circuit.x(yr[yc])
    
    # mcx for intensity
    for fc in range(8):
        if pixel[0][fc] == '1':
            circuit.mcx([xr[x] for x in range(n)] + [yr[y] for y in range(n)], f[fc])

    # Reset coordinates with X gates
    for xc in range(n):
        if pixel[2][xc] == '0':
            circuit.x(xr[xc])
    for yc in range(n):
        if pixel[1][yc] == '0':
            circuit.x(yr[yc])
    
    circuit.barrier()

encrypt(circuit, f, key, ancilla, n_rounds=2)

for l in range(2*n + 8):
    circuit.measure(l, c[2*n + 7 - l])

state_sim = AerSimulator()
job = state_sim.run(circuit, shots=100)
result = job.result()
counts = result.get_counts()

# Reconstruct the image from the measured counts
output_image = np.zeros((2**n, 2**n), dtype=np.uint8)

for pixel in counts:
    intensity = int(pixel[:8], 2)
    y_coord = int(pixel[8:8+n], 2)
    x_coord = int(pixel[8+n:], 2)
    output_image[y_coord, x_coord] = intensity

display_image(output_image, "Encrypted Image")
