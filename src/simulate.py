#simulation with a known answer
#we build a true rating matrix of rank 3 then hide 70 percent of it
#then we check that alternating least squares finds the hidden values
#the step by step run with plots is in notebooks/iteration2.ipynb

import numpy as np


def simulate_ratings(nu=100, ni=80, k=3, frac=0.3, seed=0):
    #fixed seed so every team member gets the same numbers
    rng = np.random.default_rng(seed)

    #two thin random matrices of size nu by k and k by ni
    A = rng.normal(size=(nu, k))
    B = rng.normal(size=(k, ni))

    #their product is the true matrix and its rank is k
    R_true = A @ B

    #add small noise so the data is not perfect like real ratings
    noise = 0.1 * rng.normal(size=(nu, ni))
    R = R_true + noise

    #keep each cell with chance frac and this set of cells is omega
    mask = rng.random((nu, ni)) < frac
    return R, R_true, mask


def objective(R, mask, U, V, lam):
    #this is J of U and V from question 1
    #error on every cell then zero on the cells we do not know
    error = (R - U @ V.T) * mask

    #squared error on known cells
    data_part = np.sum(error ** 2)

    #penalty on the size of the numbers in U and V
    penalty_part = lam * (np.sum(U ** 2) + np.sum(V ** 2))
    return data_part + penalty_part


def als(R, mask, k=3, lam=1.0, iters=50, seed=0):
    #start from random U and V
    rng = np.random.default_rng(seed)
    nu, ni = R.shape
    U = rng.normal(size=(nu, k))
    V = rng.normal(size=(ni, k))

    #keep J after every step to check it never goes up
    J_hist = [objective(R, mask, U, V, lam)]

    for step in range(iters):
        #fix V and find a new row of U for each user
        for u in range(nu):
            #rows of V for the movies this user rated
            V_u = V[mask[u]]
            #ratings of this user on those movies
            r_u = R[u, mask[u]]
            #solve the small k by k system from question 2
            left = V_u.T @ V_u + lam * np.eye(k)
            right = V_u.T @ r_u
            U[u] = np.linalg.solve(left, right)

        #fix U and find a new row of V for each movie
        for i in range(ni):
            #rows of U for the users who rated this movie
            U_i = U[mask[:, i]]
            #ratings of this movie from those users
            r_i = R[mask[:, i], i]
            left = U_i.T @ U_i + lam * np.eye(k)
            right = U_i.T @ r_i
            V[i] = np.linalg.solve(left, right)

        J_hist.append(objective(R, mask, U, V, lam))

    return U, V, np.array(J_hist)


def rmse(P, T, M):
    #root mean squared error only on the cells where M is true
    difference = (P - T)[M]
    return np.sqrt(np.mean(difference ** 2))


def svd_baseline(R, mask, k=3):
    #fill unknown cells with the mean of the known ones
    known_mean = R[mask].mean()
    R_filled = np.where(mask, R, known_mean)

    #full svd of the filled matrix
    U_s, s, Vt_s = np.linalg.svd(R_filled, full_matrices=False)

    #keep only the top k singular values and rebuild the matrix
    R_k = U_s[:, :k] @ np.diag(s[:k]) @ Vt_s[:k, :]
    return R_k


if __name__ == "__main__":
    #quick check from the terminal
    R, R_true, mask = simulate_ratings()
    U, V, J_hist = als(R, mask)
    print("objective never increases", bool(np.all(np.diff(J_hist) <= 1e-9)))
    print("ALS rmse on hidden cells", round(rmse(U @ V.T, R_true, ~mask), 4))
    print("filled SVD rmse on hidden cells", round(rmse(svd_baseline(R, mask), R_true, ~mask), 4))
