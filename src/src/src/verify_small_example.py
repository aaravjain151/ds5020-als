
"""
T18: Independent verification of the small ALS example.

Checks the exact Fraction-based calculations in small_example.py
against independently solved normal equations using NumPy.
"""

import numpy as np
from fractions import Fraction

# Original rating matrix (None = missing rating)
R = [
    [Fraction(5), None, Fraction(1)],
    [Fraction(4), Fraction(2), None],
    [None, Fraction(3), Fraction(5)]
]

lam = 1

# Initial movie factors
V = [Fraction(1), Fraction(1), Fraction(1)]

# Exact calculations from the first ALS update
U_exact = [
    Fraction(2),
    Fraction(2),
    Fraction(8, 3)
]

V_exact = [
    Fraction(2),
    Fraction(108, 109),
    Fraction(138, 109)
]

# Independent user update using NumPy normal equations
U_numpy = []

for i in range(3):
    observed = [
        j for j in range(3) if R[i][j] is not None
    ]

    A = np.array([[float(V[j])] for j in observed])
    b = np.array([float(R[i][j]) for j in observed])

    result = np.linalg.solve(
        A.T @ A + lam * np.eye(1),
        A.T @ b
    )

    U_numpy.append(result[0])

# Independent movie update using NumPy normal equations
V_numpy = []

for j in range(3):
    observed = [
        i for i in range(3) if R[i][j] is not None
    ]

    A = np.array([[U_numpy[i]] for i in observed])
    b = np.array([float(R[i][j]) for i in observed])

    result = np.linalg.solve(
        A.T @ A + lam * np.eye(1),
        A.T @ b
    )

    V_numpy.append(result[0])

# Compare exact and numerical results
print("User factor verification:")

for i in range(3):
    print(
        f"User {i + 1}: exact={U_exact[i]}, "
        f"numpy={U_numpy[i]:.10f}"
    )
    assert np.isclose(U_numpy[i], float(U_exact[i]))

print("\nMovie factor verification:")

for j in range(3):
    print(
        f"Movie {j + 1}: exact={V_exact[j]}, "
        f"numpy={V_numpy[j]:.10f}"
    )
    assert np.isclose(V_numpy[j], float(V_exact[j]))

print("\nSUCCESS: All ALS updates independently verified.")
