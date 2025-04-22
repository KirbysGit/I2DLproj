# Imports.
import numpy as np
import sympy as sp

# Define ReLU Func.
def relu(x):
    return np.maximum(0, x)

# Define ReLU Func for Symbolic Computations.
def relu_sym(x):
    return sp.Piecewise((0, x <= 0), (x, x > 0))

# Define Softmax Func.
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Just to prevent overflow issues as I was getting.
    return exp_z / np.sum(exp_z)

# -- GIVEN PARAMETERS --

# Inputs
x1, x2 = 1, 2  # Input values

# Weights & Biases.
w1, w2, w3, w4 = 0.2, 0.1, 0.3, 0.2
w5, w6, w7, w8 = 0.1, 0.2, 0.4, 0.3
b1, b2 = 0.1, 0.2

# Ground Truth.
y_true = np.array([0, 1])  # For Class 2

# Ground Truth for Squared Loss.
delta1, delta2 = 0, 1

# Forward Pass Computations (Softmax & Cross Entropy).
h1 = relu(w1 * x1 + w2 * x2 + b1)
h2 = relu(w3 * x1 + w4 * x2 + b1)

z1 = w5 * h1 + w6 * h2 + b2 # o1 
z2 = w7 * h1 + w8 * h2 + b2 # o2 

# Exponentials For Softmax.
exp_z1 = np.exp(z1)
exp_z2 = np.exp(z2)
sum_exp_z = exp_z1 + exp_z2

# Softmax Output (ŷ₁ and ŷ₂)
o = softmax(np.array([z1, z2]))
o1, o2 = o[0], o[1]

# Compute Cross-Entropy Loss.
L_cross_entropy = -np.sum(y_true * np.log(o))

# Compute Gradients for Cross-Entropy Loss.
dL_dz1 = 2 * (o1 - delta1)
dL_dz2 = 2 * (o2 - delta2)
dL_dz = o - y_true
dL_dw7 = dL_dz[1] * h1  # Gradient wrt w7
dL_dh2 = dL_dz[0] * w6 + dL_dz[1] * w8
relu_derivative = 1 if h2 > 0 else 0
dL_dw3 = dL_dh2 * relu_derivative * x1  # Gradient wrt w3

# Forward Pass Computations (Squared Loss).
o1_sq = relu(w5 * h1 + w6 * h2 + b2)
o2_sq = relu(w7 * h1 + w8 * h2 + b2)

# Compute Squared Loss.
L_squared = (o1_sq - delta1) ** 2 + (o2_sq - delta2) ** 2

# Symbolic Computation for Squared Loss.
w3_sym, w7_sym = sp.symbols('w3 w7')
h1_sym = relu_sym(w1 * x1 + w2 * x2 + b1)
h2_sym = relu_sym(w3_sym * x1 + w4 * x2 + b1)
o1_sym = relu_sym(w5 * h1_sym + w6 * h2_sym + b2)
o2_sym = relu_sym(w7_sym * h1_sym + w8 * h2_sym + b2)

# Define Loss Function.
L_sym = (o1_sym - delta1)**2 + (o2_sym - delta2)**2

# Compute Gradients.
dL_dw3_sym = sp.diff(L_sym, w3_sym)
dL_dw7_sym = sp.diff(L_sym, w7_sym)

# Substitute the Realv Values.
dL_dw3_val = dL_dw3_sym.subs({w3_sym: w3, w7_sym: w7})
dL_dw7_val = dL_dw7_sym.subs({w3_sym: w3, w7_sym: w7})

# Print the Solved Values (Just providing our intermittentent values to check for errors).
print("\n-- Solved Values --")
print(f"h1 = {h1:.6f}, h2 = {h2:.6f}")
print(f"o1 = {o1:.6f}, o2 = {o2:.6f}")
print(f"L : {L_squared:.6f}")
print(f"∂L/∂o1 = 2 * ({o1:.6f} - {delta1}) = {dL_dz1:.6f}")
print(f"∂L/∂o2 = 2 * ({o2:.6f} - {delta2}) = {dL_dz2:.6f}")
print(f"e^(z1) = e^{z1:.6f} ≈ {exp_z1:.6f}")
print(f"e^(z2) = e^{z2:.6f} ≈ {exp_z2:.6f}")
print(f"Σ e^z = {sum_exp_z:.6f}")
print(f"(y1_hat) = {exp_z1:.6f} / {sum_exp_z:.6f} ≈ {o1:.6f}")
print(f"(y2_hat) = {exp_z2:.6f} / {sum_exp_z:.6f} ≈ {o2:.6f}")

# Print Final Answers.
print("\n-- Final Answers --")
print(f"Question # 1 w/ Output1: {z1:.2f}")
print(f"Question # 1 w/ Output2: {z2:.2f}")
print(f"-")
print(f"Question # 2 w/ Weight3: {dL_dw3_val:.3f}")
print(f"Question # 2 w/ Weight7: {dL_dw7_val:.3f}")
print(f"-")
print(f"Question # 3 w/ Weight3: {dL_dw3:.6f}")
print(f"Question # 3 w/ Weight7: {dL_dw7:.6f}")
