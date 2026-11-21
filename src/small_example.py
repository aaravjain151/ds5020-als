
"""
T17: Small worked example of Alternating Least Squares (ALS)

Rating matrix: 3 users x 3 movies
Missing ratings: 2
Latent factor dimension: k = 1
Regularization parameter: lambda = 1

Perform one complete ALS iteration:
1. Update all user factors.
2. Update all movie factors.

All calculations use exact fractions.
"""

from fractions import Fraction


def print_matrix(matrix, title):
    """Print a matrix with missing entries displayed as ?."""
    print(f"\n{title}")
    for row in matrix:
        print(["?" if value is None else str(value) for value in row])


def update_users(R, V, lam):
    """
    Update user factors while keeping movie factors fixed.

    u_i = sum(r_ij * v_j) / (sum(v_j^2) + lambda)
    """

    U = []

    for i in range(len(R)):
        numerator = Fraction(0)
        denominator = lam

        for j in range(len(V)):
            if R[i][j] is not None:
                numerator += R[i][j] * V[j]
                denominator += V[j] ** 2

        U.append(numerator / denominator)

        print(
            f"User {i + 1}: "
            f"numerator = {numerator}, "
            f"denominator = {denominator}, "
            f"u = {U[-1]}"
        )

    return U


def update_movies(R, U, lam):
    """
    Update movie factors while keeping updated user factors fixed.

    v_j = sum(r_ij * u_i) / (sum(u_i^2) + lambda)
    """

    V = []

    for j in range(len(R[0])):
        numerator = Fraction(0)
        denominator = lam

        for i in range(len(R)):
            if R[i][j] is not None:
                numerator += R[i][j] * U[i]
                denominator += U[i] ** 2

        V.append(numerator / denominator)

        print(
            f"Movie {j + 1}: "
            f"numerator = {numerator}, "
            f"denominator = {denominator}, "
            f"v = {V[-1]}"
        )

    return V


def main():

    # Rating matrix: None represents a missing rating.
    R = [
        [Fraction(5), None, Fraction(1)],
        [Fraction(4), Fraction(2), None],
        [None, Fraction(3), Fraction(5)]
    ]

    # Initial movie latent factors (k = 1).
    V = [
        Fraction(1),
        Fraction(1),
        Fraction(1)
    ]

    # Regularization parameter.
    lam = Fraction(1)

    print("=" * 50)
    print("ALS SMALL WORKED EXAMPLE")
    print("=" * 50)

    print_matrix(R, "Original rating matrix:")

    print("\nInitial movie factors:")
    print(V)

    print("\nRegularization lambda:", lam)

    # Step 1: Update user factors.
    print("\n" + "=" * 50)
    print("STEP 1: UPDATE USER FACTORS")
    print("=" * 50)

    U = update_users(R, V, lam)

    print("\nUpdated user factors:")
    print(U)

    # Step 2: Update movie factors using new user factors.
    print("\n" + "=" * 50)
    print("STEP 2: UPDATE MOVIE FACTORS")
    print("=" * 50)

    V_new = update_movies(R, U, lam)

    print("\nUpdated movie factors:")
    print(V_new)

    # Final result.
    print("\n" + "=" * 50)
    print("FINAL RESULTS AFTER ONE COMPLETE ALS ITERATION")
    print("=" * 50)

    print("User factors (U):", U)
    print("Movie factors (V):", V_new)

    print("\nAll calculations completed using exact fractions.")


if __name__ == "__main__":
    main()
