
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
    [Fraction(4), Fraction(3), Fraction(5)]
]

lam = 1

# Initial movie factors
V = [Fraction(1), Fraction(1), Fraction(1)]

# Exact calculations from the first ALS update
U_exact = [
    Fraction(2),
    Fraction(2),
    Fraction(3)
]

V_exact = [
    Fraction(5, 3),
    Fraction(13, 14),
    Fraction(17, 14)
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

# Second way (T18, Aarav): the same step with the team's ALS in
# src/methods.py, which solves each system by our own Gaussian
# elimination instead of numpy.linalg.solve
from methods import my_update_users, my_update_movies

R_float = np.array([[0.0 if r is None else float(r) for r in row] for row in R])
known = np.array([[r is not None for r in row] for row in R])
U_methods = my_update_users(R_float, known, np.ones((3, 1)), float(lam))
V_methods = my_update_movies(R_float, known, U_methods, float(lam))

print("\nSecond way with src/methods.py:")
print("largest |U methods - U exact| =",
      f"{np.abs(U_methods.ravel() - [float(u) for u in U_exact]).max():.2e}")
print("largest |V methods - V exact| =",
      f"{np.abs(V_methods.ravel() - [float(v) for v in V_exact]).max():.2e}")
assert np.allclose(U_methods.ravel(), [float(u) for u in U_exact])
assert np.allclose(V_methods.ravel(), [float(v) for v in V_exact])

print("\nSUCCESS: All ALS updates independently verified.")
