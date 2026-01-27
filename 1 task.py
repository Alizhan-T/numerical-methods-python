# ============================================================
# Assignment 4
# Curve Fitting using Least Squares + Error Analysis
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Error metrics
# -------------------------
def sse_and_r2(y, y_hat):
    y = np.array(y, dtype=float)
    y_hat = np.array(y_hat, dtype=float)

    sse = np.sum((y - y_hat) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - sse / sst

    return sse, r2


# -------------------------
# Linear model: y = a + b x
# -------------------------
def linear_model(x, y):
    A = np.column_stack((np.ones(len(x)), x))
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    y_hat = a + b * x
    return y_hat, f"y = {a:.4f} + {b:.4f}x"


# -------------------------
# Parabolic model: y = a + b x + c x^2
# -------------------------
def parabolic_model(x, y):
    A = np.column_stack((np.ones(len(x)), x, x**2))
    a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]

    y_hat = a + b * x + c * x**2
    return y_hat, f"y = {a:.4f} + {b:.4f}x + {c:.4f}x²"


# -------------------------
# Power model: y = a x^b
# ln(y) = ln(a) + b ln(x)
# -------------------------
def power_model(x, y):
    ln_x = np.log(x)
    ln_y = np.log(y)

    A = np.column_stack((np.ones(len(x)), ln_x))
    ln_a, b = np.linalg.lstsq(A, ln_y, rcond=None)[0]

    a = np.exp(ln_a)
    y_hat = a * x**b

    return y_hat, f"y = {a:.4f} x^{b:.4f}"


# -------------------------
# Exponential model: y = a e^(b x)
# ln(y) = ln(a) + b x
# -------------------------
def exponential_model(x, y):
    ln_y = np.log(y)

    A = np.column_stack((np.ones(len(x)), x))
    ln_a, b = np.linalg.lstsq(A, ln_y, rcond=None)[0]

    a = np.exp(ln_a)
    y_hat = a * np.exp(b * x)

    return y_hat, f"y = {a:.4f} e^({b:.4f}x)"


# -------------------------
# Logarithmic model: y = a + b ln(x)
# -------------------------
def logarithmic_model(x, y):
    ln_x = np.log(x)

    A = np.column_stack((np.ones(len(x)), ln_x))
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    y_hat = a + b * ln_x
    return y_hat, f"y = {a:.4f} + {b:.4f} ln(x)"


# ============================================================
# DATA SET A
# ============================================================

x = np.array([2.5, 3.5, 5, 6, 7.5, 10, 12.5, 15, 17.5, 20])
y = np.array([5, 3.4, 2, 1.6, 1.2, 0.8, 0.6, 0.4, 0.3, 0.3])

models = [
    linear_model,
    parabolic_model,
    power_model,
    exponential_model,
    logarithmic_model
]

plt.scatter(x, y, color="black", label="Data points")

for model in models:
    y_hat, eq = model(x, y)
    sse, r2 = sse_and_r2(y, y_hat)

    print(eq)
    print(f"SSE = {sse:.6f}, R^2 = {r2:.6f}\n")

    plt.plot(x, y_hat, label=eq)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Curve Fitting – Data Set A")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# DATA SET B
# ============================================================

x = np.array([10, 15, 22.5, 33.7, 50.6, 75.9])
y = np.array([0.3, 0.675, 1.519, 3.417, 7.639, 11.3])

plt.scatter(x, y, color="black", label="Data points")

for model in models:
    y_hat, eq = model(x, y)
    sse, r2 = sse_and_r2(y, y_hat)

    print(eq)
    print(f"SSE = {sse:.6f}, R^2 = {r2:.6f}\n")

    plt.plot(x, y_hat, label=eq)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Curve Fitting – Data Set B")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# DATA SET C
# ============================================================

x = np.array([0.05, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4])
y = np.array([550, 750, 1000, 1400, 2000, 2700, 3750])

plt.scatter(x, y, color="black", label="Data points")

for model in models:
    y_hat, eq = model(x, y)
    sse, r2 = sse_and_r2(y, y_hat)

    print(eq)
    print(f"SSE = {sse:.6f}, R^2 = {r2:.6f}\n")

    plt.plot(x, y_hat, label=eq)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Curve Fitting – Data Set C")
plt.legend()
plt.grid(True)
plt.show()