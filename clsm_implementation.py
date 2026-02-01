import numpy as np
import math
import matplotlib.pyplot as plt

# Constant π with high precision (20 decimal places as paper states)
PI = 3.14159265358979323846

def cls_map(u, x0, y0, n):
    """
    Implements the 2D-CLSM chaotic map

    Parameters:
    u  : control parameter
    x0 : initial value of x
    y0 : initial value of y
    n  : number of iterations

    Returns:
    x, y sequences
    """

    x = np.zeros(n)
    y = np.zeros(n)

    x[0] = x0
    y[0] = y0

    for i in range(n - 1):

        x[i+1] = math.cos(
            4 * u * x[i] * (1 - x[i]) *
            math.sin(PI * (y[i] * (1 - y[i]))) *
            (PI ** 10) + PI ** 2
        )

        y[i+1] = math.cos(
            4 * u * y[i] * (1 - y[i]) *
            math.sin(PI * (x[i] + y[i])) *
            (PI ** 10) + PI ** 2
        )

    return x, y


# Example usage
u = 0.9
x0 = 0.123456
y0 = 0.654321
iterations = 500

x_seq, y_seq = cls_map(u, x0, y0, iterations)

print("First 10 x values:", x_seq[:10])
print("First 10 y values:", y_seq[:10])



plt.figure(figsize=(10, 4))

plt.plot(x_seq, label='x sequence')
plt.plot(y_seq, label='y sequence')

plt.title("2D-CLSM Chaotic Sequences")
plt.legend()
plt.show()


def normalize(seq):
    return (seq - np.min(seq)) / (np.max(seq) - np.min(seq))

rand_x = normalize(x_seq)
rand_y = normalize(y_seq)
rand_x
rand_y

x1, y1 = cls_map(u, 0.123456, 0.654321, 200)
x2, y2 = cls_map(u, 0.123457, 0.654321, 200)

plt.plot(x1 - x2)
plt.title("Difference with tiny change in initial value")
plt.show()




PI = 3.14159265358979323846

# -------- Logistic Map --------
def logistic_map(u, x0, n):

    x = np.zeros(n)
    x[0] = x0

    for i in range(n-1):
        x[i+1] = 4 * u * x[i] * (1 - x[i])

    return x

x = logistic_map(0.9, 0.3, 200)

plt.plot(x)
plt.title("Logistic Map Output")
plt.show()


def sine_map(x0, n):

    x = np.zeros(n)
    x[0] = x0

    for i in range(n-1):
        x[i+1] = math.sin(PI * x[i])

    return x

x = sine_map(0.3, 200)

plt.plot(x)
plt.title("Sine Map Output")
plt.show()
