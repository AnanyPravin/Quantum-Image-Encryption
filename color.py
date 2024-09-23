import numpy as np
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("/Users/ananypravin/Desktop/feistelcode/lena_color_512.tif")
image_array = np.array(image)

def nxor(circuit, ctrl, target):
    circuit.cx(ctrl, target)
    circuit.x(target)

def transformation1(input):
    output = []
    # Convert input integers to boolean (assuming input as 0 or 1)
    A = input[0]
    B = input[1]
    C = input[2]
    D = input[3]
    
    # Calculate outputs using XOR (^) and NOT (~) operations
    W = A ^ B  # XOR of A and B
    X = (not B & 0b1) ^ C  # XOR of NOT B and C
    Y = A ^ (not D & 0b1)  # XOR of A and NOT D
    Z = C ^ D  # XOR of C and D
    
    # Convert boolean results back to integers
    W = int(W)
    X = int(X)
    Y = int(Y)
    Z = int(Z)
    output.append(W)
    output.append(X)
    output.append(Y)
    output.append(Z)
    
    return output

def transformation2(input):
    output = []
    # Convert input integers to boolean (assuming input as 0 or 1)
    A = input[0]
    B = input[1]
    C = input[2]
    D = input[3]
    
    W = A ^ (not B)  # XOR of A and NOT B
    X = B ^ C        # XOR of B and C
    Y = (A ^ C) ^ D  # XOR of (A XOR C) with D
    Z = (not A) ^ (not D)  # XOR of NOT A and NOT D
    
    # Convert boolean results back to integers
    W = int(W)
    X = int(X)
    Y = int(Y)
    Z = int(Z)
    output.append(W)
    output.append(X)
    output.append(Y)
    output.append(Z)
    return output

def F_function(input):
    output = []
    l1_T1 = transformation1(input[:4])
    l1_T2 = transformation2(input[4:8])
    l2_T1 = transformation1(l1_T2)
    l2_T2 = transformation2(l1_T1)
    l3_T1 = transformation1(l2_T2)
    l3_T2 = transformation2(l2_T1)
    for i in l3_T1:
        output.append(i)
    for j in l3_T2:
        output.append(j)
    return output

import random
import numpy as np

# Use transformation1 and transformation2 as defined above

def generate_subkey():
    return random.randint(0, 255)  # Generate a random 8-bit subkey

# Feistel round for classical encryption
def feistel_round(left, right, subkey):
    if len(right) < 8:
        right += [0] * (8 - len(right))  # Pad with zeros to make it 8 bits

    # Apply the F function to the right half
    f_output = F_function(right)
    
    # XOR the output of F function with the left half
    new_right = [l ^ f for l, f in zip(left, f_output)]
    
    # Swap left and right
    return right[:4], new_right[:4]  # Only return 4 bits for left and right halvest

# Feistel encryption function for image bits (8-bit block)
def feistel_encrypt(image_bits, rounds=4):
    # Split the image bits into left and right halves (first 4 and last 4 bits)
    left = image_bits[:4]
    right = image_bits[4:]

    # Apply the Feistel rounds
    for _ in range(rounds):
        subkey = generate_subkey()  # Generate a new subkey for each round
        left, right = feistel_round(left, right, subkey)
    
    # Concatenate the final left and right halves
    return left + right

def threedlf(size, p1, p2, p3, h1=0.1, h2=0.2, h3=0.3):
    flat_size = size**2
    H1 = np.zeros(flat_size)
    H2 = np.zeros(flat_size)
    H3 = np.zeros(flat_size)
    H1[0], H2[0], H3[0] = h1, h2, h3
    
    for i in range(1, flat_size):
        H1[i] = abs(np.sin(np.pi * (4 * p1 * H1[i-1] * (1 - H1[i-1]) + 1 / ((H2[i-1] ** 2) + 0.1) - p2 * H2[i-1] + p3 * np.sin(np.pi * H3[i-1]))))
        H2[i] = abs(np.sin(np.pi * (4 * p1 * H2[i-1] * (1 - H2[i-1]) + 1 / ((H3[i-1] ** 2) + 0.1) - p2 * H3[i-1] + p3 * np.sin(np.pi * H1[i-1]))))
        H3[i] = abs(np.sin(np.pi * (4 * p1 * H3[i-1] * (1 - H3[i-1]) + 1 / ((H1[i-1] ** 2) + 0.1) - p2 * H1[i-1] + p3 * np.sin(np.pi * H2[i-1]))))
    
    ind_h = [np.argsort(H1), np.argsort(H2), np.argsort(H3)]
    
    return ind_h

# Example function to encrypt an image using the Feistel structure
# def encrypt_image(image, rounds=4):
#     # Convert image to binary format
#     height, width = image.shape
#     encrypted_image = np.zeros_like(image)

#     for y in range(height):
#         for x in range(width):
#             # Convert each pixel to an 8-bit binary number (assuming grayscale image)
#             pixel = format(image[y, x], '08b')
#             pixel_bits = [int(bit) for bit in pixel]  # Convert to list of bits

#             # Apply Feistel encryption to the pixel bits
#             encrypted_bits = feistel_encrypt(pixel_bits, rounds)

#             # Convert the encrypted bits back to an integer
#             encrypted_pixel = int(''.join(map(str, encrypted_bits)), 2)
#             encrypted_image[y, x] = encrypted_pixel

#     return encrypted_image

# # Example to apply the encryption on a 4x4 grayscale image
# input_image = np.array([[0, 15, 31, 47],
#                         [63, 79, 95, 111],
#                         [127, 143, 159, 175],
#                         [191, 207, 223, 239]], dtype=np.uint8)

# # Encrypt the image using the Feistel encryption scheme
# encrypted_image = encrypt_image(input_image, rounds=4)

# # Display the original and encrypted images
# print("Original Image:")
# print(input_image)

# print("Encrypted Image:")
# print(encrypted_image)








# def display_image(image, title="Image"):
#     plt.imshow(image, cmap='gray', vmin=0, vmax=255)
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

# def encrypt_image(image_array, rounds=4):
#     height, width = image_array.shape
#     encrypted_image = np.zeros_like(image_array)

#     for y in range(height):
#         for x in range(width):
#             pixel = format(image_array[y, x], '08b')
#             pixel_bits = [int(bit) for bit in pixel]  # Convert to list of bits
#             encrypted_bits = feistel_encrypt(pixel_bits, rounds)
#             encrypted_pixel = int(''.join(map(str, encrypted_bits)), 2)
#             encrypted_image[y, x] = encrypted_pixel

#     return encrypted_image

# # ---- Main Usage ----

# if __name__ == "__main__":
#     # Example: 4x4 grayscale NumPy array as input image
#     input_image = np.array([[0, 15, 31, 47],
#                             [63, 79, 95, 111],
#                             [127, 143, 159, 175],
#                             [191, 207, 223, 239]], dtype=np.uint8)
    
#     print("Original Image:")
#     display_image(input_image, "Original Image")

#     # Encrypt the image using the Feistel encryption
#     encrypted_image = encrypt_image(input_image, rounds=4)

#     print("Encrypted Image:")
#     display_image(encrypted_image, "Encrypted Image")

def display_image(image, title="Image"):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
# def swap_pixels(image, coord1, coord2):
#     image[coord1], image[coord2] = image[coord2].copy(), image[coord1].copy()

# def encrypt_rgb_image(image_array, rounds=4, swap_probability=0.9):
#     height, width, _ = image_array.shape
#     encrypted_image = np.zeros_like(image_array)

#     # First pass: Encrypt each pixel's RGB channels
#     for y in range(height):
#         for x in range(width):
#             # Extract RGB channels
#             red = format(image_array[y, x, 0], '08b')
#             green = format(image_array[y, x, 1], '08b')
#             blue = format(image_array[y, x, 2], '08b')
            
#             # Convert to list of bits for each channel
#             red_bits = [int(bit) for bit in red]
#             green_bits = [int(bit) for bit in green]
#             blue_bits = [int(bit) for bit in blue]
            
#             # Encrypt each channel separately
#             encrypted_red_bits = feistel_encrypt(red_bits, rounds)
#             encrypted_green_bits = feistel_encrypt(green_bits, rounds)
#             encrypted_blue_bits = feistel_encrypt(blue_bits, rounds)
            
#             # Convert encrypted bits back to integers
#             encrypted_red = int(''.join(map(str, encrypted_red_bits)), 2)
#             encrypted_green = int(''.join(map(str, encrypted_green_bits)), 2)
#             encrypted_blue = int(''.join(map(str, encrypted_blue_bits)), 2)
            
#             encrypted_image[y, x, 0] = encrypted_red
#             encrypted_image[y, x, 1] = encrypted_green
#             encrypted_image[y, x, 2] = encrypted_blue

#     # Second pass: Apply pixel swapping
#     for y in range(height):
#         for x in range(width):
#             # Randomly decide whether to swap the current pixel
#             if random.random() < swap_probability:
#                 # Pick a random pixel to swap with
#                 swap_y = random.randint(0, height - 1)
#                 swap_x = random.randint(0, width - 1)
#                 # Swap the current pixel (y, x) with a random pixel (swap_y, swap_x)
#                 swap_pixels(encrypted_image, (y, x), (swap_y, swap_x))

#     return encrypted_image

def apply_chaotic_mapping(image_array, chaotic_indices, iteration):
    # Flatten the image to apply chaotic mapping based on iteration (H1, H2, or H3)
    height, width, _ = image_array.shape
    flat_image = image_array.reshape(height * width, 3)
    mapped_image = flat_image[chaotic_indices[iteration]]  # Apply chaotic reordering
    return mapped_image.reshape(height, width, 3)  # Reshape back to the original format

# def encrypt_rgb_image(image_array, rounds=4, p1=2.299, p2=1.332, p3=6.887):
#     height, width, _ = image_array.shape
#     encrypted_image = np.zeros_like(image_array)

#     # First pass: Encrypt each pixel's RGB channels
#     for y in range(height):
#         for x in range(width):
#             # Extract RGB channels
#             red = format(image_array[y, x, 0], '08b')
#             green = format(image_array[y, x, 1], '08b')
#             blue = format(image_array[y, x, 2], '08b')
            
#             # Convert to list of bits for each channel
#             red_bits = [int(bit) for bit in red]
#             green_bits = [int(bit) for bit in green]
#             blue_bits = [int(bit) for bit in blue]
            
#             # Encrypt each channel separately
#             encrypted_red_bits = feistel_encrypt(red_bits, rounds)
#             encrypted_green_bits = feistel_encrypt(green_bits, rounds)
#             encrypted_blue_bits = feistel_encrypt(blue_bits, rounds)
            
#             # Convert encrypted bits back to integers
#             encrypted_red = int(''.join(map(str, encrypted_red_bits)), 2)
#             encrypted_green = int(''.join(map(str, encrypted_green_bits)), 2)
#             encrypted_blue = int(''.join(map(str, encrypted_blue_bits)), 2)
            
#             # Store the encrypted RGB pixel
#             encrypted_image[y, x, 0] = encrypted_red
#             encrypted_image[y, x, 1] = encrypted_green
#             encrypted_image[y, x, 2] = encrypted_blue

#     # Second pass: Apply chaotic pixel swapping based on 3D-LFS chaotic map
#     flat_image = encrypted_image.reshape(height * width, 3)  # Flatten the image for easy swapping

#     # Generate chaotic indices using the chaotic map
#     ind_h = threedlf(height, p1, p2, p3)

#     # Apply chaotic swapping based on the indices
#     for iter in range(3):  # Perform swapping based on each chaotic index list (H1, H2, H3)
#         flat_image = flat_image[ind_h[iter]]  # Reorder the pixels based on chaotic index
    
#     encrypted_image = flat_image.reshape(height, width, 3)  # Reshape back to original image shape
    
#     return encrypted_image


def encrypt_rgb_image(image_array, rounds=4, p1=2.299, p2=1.332, p3=6.887):
    height, width, _ = image_array.shape
    encrypted_image = np.zeros_like(image_array)

    # Generate chaotic indices using the chaotic map
    chaotic_indices = threedlf(height, p1, p2, p3)

    # Interleave Feistel and Chaotic rounds
    for round_num in range(rounds):
        # Step 1: Apply Feistel encryption on each pixel's RGB channels
        for y in range(height):
            for x in range(width):
                # Extract RGB channels
                red = format(image_array[y, x, 0], '08b')
                green = format(image_array[y, x, 1], '08b')
                blue = format(image_array[y, x, 2], '08b')
                
                # Convert to list of bits for each channel
                red_bits = [int(bit) for bit in red]
                green_bits = [int(bit) for bit in green]
                blue_bits = [int(bit) for bit in blue]
                
                # Encrypt each channel separately using Feistel
                encrypted_red_bits = feistel_encrypt(red_bits, 1)  # Apply 1 round of Feistel
                encrypted_green_bits = feistel_encrypt(green_bits, 1)
                encrypted_blue_bits = feistel_encrypt(blue_bits, 1)
                
                # Convert encrypted bits back to integers
                encrypted_red = int(''.join(map(str, encrypted_red_bits)), 2)
                encrypted_green = int(''.join(map(str, encrypted_green_bits)), 2)
                encrypted_blue = int(''.join(map(str, encrypted_blue_bits)), 2)
                
                # Store the encrypted RGB pixel
                encrypted_image[y, x, 0] = encrypted_red
                encrypted_image[y, x, 1] = encrypted_green
                encrypted_image[y, x, 2] = encrypted_blue

        # Step 2: Apply chaotic mapping after every alternate round of Feistel
        if round_num % 2 == 1:  # On odd rounds, apply chaotic mapping
            iteration = (round_num // 2) % 3  # Use H1, H2, H3 in a cyclic manner
            encrypted_image = apply_chaotic_mapping(encrypted_image, chaotic_indices, iteration)

    return encrypted_image


# ---- Main Usage ----

if __name__ == "__main__":
    # input_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
    #                         [[255, 0, 255], [0, 255, 255], [255, 255, 255], [128, 128, 128]],
    #                         [[64, 64, 64], [32, 32, 32], [16, 16, 16], [8, 8, 8]],
    #                         [[0, 0, 0], [255, 127, 127], [127, 255, 127], [127, 127, 255]]], dtype=np.uint8)
    input_image = image_array 
    print("Original RGB Image:")
    display_image(input_image, "Original RGB Image")

    # Encrypt the RGB image using the Feistel encryption
    encrypted_image = encrypt_rgb_image(input_image, rounds=4)

    print("Encrypted RGB Image:")
    display_image(encrypted_image, "Encrypted RGB Image")

