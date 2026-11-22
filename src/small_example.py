
"""
T17: Small worked example of Alternating Least Squares (ALS)

Rating matrix: 3 users x 3 movies
Missing ratings: 2 (user 1 movie 2, user 2 movie 3)
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


def objective(R, U, V, lam):
    """
    J(U, V) = sum over known (i, j) of (r_ij - u_i v_j)^2
              + lambda * (sum of u_i^2 + sum of v_j^2)
    """

    J = Fraction(0)

    for i in range(len(R)):
        for j in range(len(R[0])):
            if R[i][j] is not None:
                J += (R[i][j] - U[i] * V[j]) ** 2

    J += lam * (sum(u ** 2 for u in U) + sum(v ** 2 for v in V))

    return J


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
        [Fraction(4), Fraction(3), Fraction(5)]
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

    J_after_users = objective(R, U, V, lam)
    print("J after the user update =", J_after_users, f"= {float(J_after_users):.4f}")

    # Step 2: Update movie factors using new user factors.
    print("\n" + "=" * 50)
    print("STEP 2: UPDATE MOVIE FACTORS")
    print("=" * 50)

    V_new = update_movies(R, U, lam)

    print("\nUpdated movie factors:")
    print(V_new)

    J_after_movies = objective(R, U, V_new, lam)
    print("J after the movie update =", J_after_movies, f"= {float(J_after_movies):.4f}")
    print("J went down:", J_after_movies <= J_after_users)

    # Final result.
    print("\n" + "=" * 50)
    print("FINAL RESULTS AFTER ONE COMPLETE ALS ITERATION")
    print("=" * 50)

    print("User factors (U):", U)
    print("Movie factors (V):", V_new)

    # Predicted ratings: r_hat_ij = u_i * v_j for every cell.
    print("\nPredicted rating matrix U V^T (missing cells marked with *):")
    for i in range(len(R)):
        row = []
        for j in range(len(R[0])):
            mark = "*" if R[i][j] is None else ""
            row.append(f"{U[i] * V_new[j]}{mark}")
        print(row)

    print("\nGuesses for the missing cells:")
    for i in range(len(R)):
        for j in range(len(R[0])):
            if R[i][j] is None:
                guess = U[i] * V_new[j]
                print(f"User {i + 1}, movie {j + 1}: {guess} = {float(guess):.4f}")

    print("\nAll calculations completed using exact fractions.")


if __name__ == "__main__":
    main()
