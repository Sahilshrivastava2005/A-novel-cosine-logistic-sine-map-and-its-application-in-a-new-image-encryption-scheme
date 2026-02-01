import numpy as np
import matplotlib.pyplot as plt
from ordpy import permutation_entropy
from sampen import sampen2
import cv2
import math

def CLSM(x0, y0, u, N):
    x = np.zeros(N)
    y = np.zeros(N)

    x[0] = x0
    y[0] = y0

    for i in range(N-1):
        x[i+1] = np.sin(np.pi * u * (y[i] + 3) * x[i] * (1 - x[i]))
        y[i+1] = np.sin(np.pi * u * (x[i+1] + 3) * y[i] * (1 - y[i]))

    return x, y


def plot_trajectory():
    x, y = CLSM(0.312, 0.723, 0.53, 5000)

    plt.scatter(x, y, s=1)
    plt.title("2D-CLSM Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def bifurcation_diagram():
    u_values = np.linspace(0, 1, 300)
    x0, y0 = 0.312, 0.723

    for u in u_values:
        x, y = CLSM(x0, y0, u, 300)
        plt.scatter([u]*100, x[-100:], s=0.2)

    plt.title("Bifurcation Diagram of 2D-CLSM")
    plt.xlabel("u")
    plt.ylabel("x values")
    plt.show()

def lyapunov_exponent(sequence):
    N = len(sequence)
    le = 0

    for i in range(N-1):
        diff = abs(sequence[i+1] - sequence[i])
        if diff != 0:
            le += math.log(diff)

    return le / N

def compute_permutation_entropy(seq):
    pe = permutation_entropy(seq, dx=3, taux=1)
    return pe


def compute_sample_entropy(seq):
    se = sampen2(seq)
    return se

def basic_randomness_test(seq):
    binary = ['1' if i > 0 else '0' for i in seq]
    ones = binary.count('1')
    zeros = binary.count('0')

    print("Ones:", ones)
    print("Zeros:", zeros)
    print("Balance Ratio:", ones / (zeros + 1))
    
def encrypt_image(image_path):
    img = cv2.imread(image_path, 0)
    M, N = img.shape

    x, y = CLSM(0.312, 0.723, 0.53, M*N)

    key = (np.array(x) * 255).astype(np.uint8)

    flat = img.flatten()

    encrypted = flat ^ key[:M*N]
    encrypted = encrypted.reshape(M, N)

    cv2.imwrite("encrypted.png", encrypted)
    print("Encrypted image saved as encrypted.png")

def decrypt_image(enc_image):
    img = cv2.imread(enc_image, 0)
    M, N = img.shape

    x, y = CLSM(0.312, 0.723, 0.53, M*N)

    key = (np.array(x) * 255).astype(np.uint8)

    flat = img.flatten()

    decrypted = flat ^ key[:M*N]
    decrypted = decrypted.reshape(M, N)

    cv2.imwrite("decrypted.png", decrypted)
    print("Decrypted image saved as decrypted.png")

if __name__ == "__main__":

    # Generate sequence
    x, y = CLSM(0.312, 0.723, 0.53, 5000)

    print("Lyapunov Exponent:", lyapunov_exponent(x))

    print("Permutation Entropy:", compute_permutation_entropy(x))

    print("Sample Entropy:", compute_sample_entropy(x))

    basic_randomness_test(x)

    plot_trajectory()

    bifurcation_diagram()

    # Image Encryption Demo
    encrypt_image("penguin.png")
    decrypt_image("encrypted.png")

orig = cv2.imread("penguin.png", 0)
enc  = cv2.imread("encrypted.png", 0)
dec  = cv2.imread("decrypted.png", 0)

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(orig, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(enc, cmap='gray')
plt.title("Encrypted")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(dec, cmap='gray')
plt.title("Decrypted")
plt.axis('off')

plt.show()
