#checks of our als step by step for iteration 3 (aarav, tasks T20 and T23)
#part 1 solves every small system of one als step with our gaussian elimination
#       and with numpy.linalg.solve and compares them (question 2)
#part 2 records J after every half step, after the users and after the movies,
#       on the simulation and on the real data, and checks it never goes up (question 5)
#part 3 repeats the size test of check_simulation.py with a smaller penalty
#       to see where the error on the simulation comes from (check of T24)
#run it from the project folder with   python src/check_als_steps.py

#Path helps to build folder paths that work on every computer
from pathlib import Path

#matplotlib makes the plots
import matplotlib
#Agg saves plots to files and does not open a window
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#our own functions from src/methods.py
from methods import my_als, my_gauss_solve, my_init, my_objective, my_rmse, my_update_movies, my_update_users
#iteration 2 functions
from simulate import simulate_ratings
from load_data import build_rating_matrix, load_ratings

#figures folder of the project
FIGURES = Path(__file__).resolve().parent.parent / "figures"

#settings, the same as in check_simulation.py and run_real.py
ITERS = 50
SEED = 0
#numbers smaller than this are rounding error and not a real increase
TOL = 1e-9


#solves the system of every user (or every movie) once with our elimination
#and once with numpy, and returns the largest difference and the largest residual
#F is the fixed factor matrix (V for users, U for movies)
#R and mask are already turned so that each row is one user (or one movie)
def compare_all_systems(R, mask, F, lam):
    k = F.shape[1]
    #largest |mine - numpy| over all systems, largest |A x - b| over all systems
    worst_diff = 0.0
    worst_res = 0.0
    #largest condition number, it says how hard a system is to solve
    worst_cond = 0.0
    for row in range(R.shape[0]):
        #rows of F that belong to the known cells of this row
        F_r = F[mask[row]]
        r = R[row, mask[row]]
        A = F_r.T @ F_r + lam * np.eye(k)
        b = F_r.T @ r
        mine = my_gauss_solve(A, b)
        ref = np.linalg.solve(A, b)
        worst_diff = max(worst_diff, np.abs(mine - ref).max())
        worst_res = max(worst_res, np.abs(A @ mine - b).max())
        worst_cond = max(worst_cond, np.linalg.cond(A))
    return worst_diff, worst_res, worst_cond, R.shape[0]


#runs als like my_als but records J after the users and after the movies
#J_half[0] is the start, J_half[1] is after the first user update,
#J_half[2] after the first movie update, and so on
def als_half_steps(R, mask, k, lam):
    nu, ni = R.shape
    U, V = my_init(nu, ni, k, SEED)
    J_half = [my_objective(R, mask, U, V, lam)]
    #we keep U and V from the middle of the run for part 1
    middle = None
    for step in range(ITERS):
        if step == ITERS // 2:
            middle = (U.copy(), V.copy())
        U = my_update_users(R, mask, V, lam)
        J_half.append(my_objective(R, mask, U, V, lam))
        V = my_update_movies(R, mask, U, lam)
        J_half.append(my_objective(R, mask, U, V, lam))
    return np.array(J_half), middle


#prints the checks of one data set
def check_one(name, R, mask, k, lam):
    print(f"{name}   {R.shape[0]} by {R.shape[1]}   known cells {int(mask.sum())}   k = {k}   lambda = {lam}")
    J_half, (U, V) = als_half_steps(R, mask, k, lam)

    #part 1   every system of the step in the middle of the run
    #users use V and the rows of R, movies use U and the columns of R (R.T)
    du, ru, cu, nu = compare_all_systems(R, mask, V, lam)
    dv, rv, cv, nv = compare_all_systems(R.T, mask.T, U, lam)
    print(f"  question 2   {nu + nv} systems ({nu} users, {nv} movies) solved two ways")
    print(f"    largest |mine - numpy.linalg.solve| = {max(du, dv):.2e}")
    print(f"    largest |A x - b| for our x         = {max(ru, rv):.2e}")
    print(f"    largest condition number of A       = {max(cu, cv):.1f}")

    #part 2   J after every half step
    #np.diff gives the change of J from one half step to the next
    changes = np.diff(J_half)
    after_users = changes[0::2]
    after_movies = changes[1::2]
    print(f"  question 5   J from {J_half[0]:.2f} to {J_half[-1]:.2f} in {ITERS} steps ({2 * ITERS} half steps)")
    print(f"    largest change after a user update  = {after_users.max():.2e}")
    print(f"    largest change after a movie update = {after_movies.max():.2e}")
    print(f"    J never increases                   = {bool(np.all(changes <= TOL))}")
    #how much of the total drop happens in the first step
    first = (J_half[0] - J_half[2]) / (J_half[0] - J_half[-1])
    print(f"    share of the total drop in step 1   = {first:.3f}")
    #relative change in the last step, to show that J has settled
    print(f"    relative change in the last step    = {(J_half[-3] - J_half[-1]) / J_half[-1]:.2e}")
    return J_half


def plot(J_sim, J_real):
    #two pictures side by side, one for each data set
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, J_half, title in [(axes[0], J_sim, "simulation 100 by 80, k = 3, lambda = 1"),
                              (axes[1], J_real, "MovieLens u1.base, k = 2, lambda = 1")]:
        #half step t sits at t / 2 on the axis, so whole numbers are full steps
        steps = np.arange(len(J_half)) / 2
        ax.semilogy(steps, J_half, "-", color="gray", linewidth=1)
        #dots after the user updates and squares after the movie updates
        ax.semilogy(steps[1::2], J_half[1::2], "o", markersize=3, label="after user update")
        ax.semilogy(steps[2::2], J_half[2::2], "s", markersize=3, label="after movie update")
        ax.set_xlabel("ALS step")
        ax.set_ylabel("objective J")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "objective_half_steps.png", dpi=150)
    plt.close()


#error on the hidden cells for three sizes, for a large and a small penalty
#if noise were the only error, it would be about
#sigma sqrt( k (n_u + n_i) / N ), the number of unknowns over the number of known cells
def penalty_and_size():
    print("part 3  where the error on the simulation comes from")
    users = [100, 200, 400]
    movies = [80, 160, 320]
    for lam in [1.0, 0.01]:
        errors = []
        for n in range(len(users)):
            R, R_true, mask = simulate_ratings(nu=users[n], ni=movies[n], k=3, frac=0.3, seed=0)
            U, V, J_hist = my_als(R, mask, k=3, lam=lam, iters=ITERS, seed=SEED)
            errors.append(my_rmse(U @ V.T, R_true, ~mask))
        #slope on the log log plot from 100 users to 400 users
        slope = (np.log(errors[2]) - np.log(errors[0])) / (np.log(400) - np.log(100))
        print(f"  lambda = {lam:4.2f}   rmse on hidden cells " + "  ".join(f"{e:.4f}" for e in errors)
              + f"   slope {slope:.2f}")
    noise_only = []
    for n in range(len(users)):
        R, R_true, mask = simulate_ratings(nu=users[n], ni=movies[n], k=3, frac=0.3, seed=0)
        noise_only.append(0.1 * np.sqrt(3 * (users[n] + movies[n]) / mask.sum()))
    print("  noise only estimate          " + "  ".join(f"{e:.4f}" for e in noise_only) + "   slope -0.50")


#this part runs only when we start the file from the terminal
if __name__ == "__main__":
    #the iteration 2 simulation
    R, R_true, mask = simulate_ratings(nu=100, ni=80, k=3, frac=0.3, seed=0)
    J_sim = check_one("simulation", R, mask, k=3, lam=1.0)
    print()
    #the real data, train part only, best setting from run_real.py
    R_train, mask_train = build_rating_matrix(load_ratings("u1.base"))
    J_real = check_one("MovieLens u1.base", R_train, mask_train, k=2, lam=1.0)
    plot(J_sim, J_real)
    print()
    penalty_and_size()
