#simulation test for iteration 3
#the data is the iteration 2 simulation so we know the true matrix
#part 1 compares our als with the truth and with numpy
#part 2 repeats it for bigger matrices and plots the error
#part 3 is the stress test with fewer known cells

#Path helps to build folder paths that work on every computer
from pathlib import Path

#matplotlib makes the plots
import matplotlib
#Agg saves plots to files and does not open a window
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#our own functions from src/methods.py
from methods import my_als, my_gauss_solve, my_rmse, report
#iteration 2 functions from src/simulate.py
#als there uses numpy.linalg.solve so we use it as the library answer
from simulate import als, simulate_ratings, svd_baseline

#figures folder of the project
#__file__ is this script, parent is src/ and parent of that is the project folder
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def part_1_truth_and_library():
    print("part 1  simulation 100 by 80 with 30 percent known")
    #R is what we see, R_true is the answer, mask says which cells we see
    R, R_true, mask = simulate_ratings(nu=100, ni=80, k=3, frac=0.3, seed=0)
    U, V, J_hist = my_als(R, mask, k=3, lam=1.0, iters=50, seed=0)

    #question 5  the objective never goes up
    #np.diff gives the change of J from one step to the next
    #.max() is the biggest change, it should be 0 or below
    print("largest change of J", f"{np.diff(J_hist).max():.2e}")
    #1e-9 means 0.000000001, we allow this tiny amount for rounding
    #np.all is true only if every change passes
    print("J never increases", bool(np.all(np.diff(J_hist) <= 1e-9)))

    #compare with the truth on the cells that were hidden
    #~mask flips true and false so it picks the hidden cells
    #round(..., 4) keeps 4 digits after the point
    print("ALS rmse on hidden cells", round(my_rmse(U @ V.T, R_true, ~mask), 4))

    #question 3  truncated svd after filling with the mean
    R_svd = svd_baseline(R, mask, k=3)
    print("filled SVD rmse on hidden cells", round(my_rmse(R_svd, R_true, ~mask), 4))

    #compare with a library  the iteration 2 als uses numpy.linalg.solve
    #same data and same seed, so the only difference is how each system is solved
    U_lib, V_lib, J_lib = als(R, mask, k=3, lam=1.0, iters=50, seed=0)
    report("U V^T mine vs numpy als", U @ V.T, U_lib @ V_lib.T)
    report("J history mine vs numpy als", J_hist, J_lib)

    #question 2  one update for user 0 with our elimination and with numpy
    #user 0 is the first user, python counts from 0
    V_u = V[mask[0]]
    r_u = R[0, mask[0]]
    #lambda is 1.0 and k is 3 here
    left = V_u.T @ V_u + 1.0 * np.eye(3)
    right = V_u.T @ r_u
    report("one user update vs numpy.solve", my_gauss_solve(left, right), np.linalg.solve(left, right))

    #second way  put the answer back into the system
    #if x is right then left times x minus right is 0
    print("max |left x - right|", f"{np.abs(left @ my_gauss_solve(left, right) - right).max():.2e}")


def part_2_bigger_matrices():
    #empty print gives an empty line between parts
    print()
    print("part 2  error against the truth for bigger matrices")
    #same shape as the iteration 2 simulation   80 movies for every 100 users
    users = [50, 100, 200, 400]
    movies = [40, 80, 160, 320]
    #we add one error for each size to this list
    errors = []
    #n goes 0, 1, 2, 3 and picks the n-th size from both lists
    for n in range(len(users)):
        R, R_true, mask = simulate_ratings(nu=users[n], ni=movies[n], k=3, frac=0.3, seed=0)
        U, V, J_hist = my_als(R, mask, k=3, lam=1.0, iters=50, seed=0)
        error = my_rmse(U @ V.T, R_true, ~mask)
        errors.append(error)
        #mask.sum() counts the true cells, which is the number of known cells
        #:4d pads a whole number to 4 places and :.4f keeps 4 digits after the point
        print(f"{users[n]:4d} by {movies[n]:4d}   known cells {int(mask.sum()):6d}   rmse on hidden cells {error:.4f}")

    #slope of the line on the log log plot from 100 users to 400 users
    #slope = (log e_400 - log e_100) / (log 400 - log 100)
    #errors[1] is the error for 100 users and errors[3] for 400 users
    slope = (np.log(errors[3]) - np.log(errors[1])) / (np.log(400) - np.log(100))
    print("slope from 100 users up", round(slope, 2))

    #new empty picture 5 by 3.5 inches
    plt.figure(figsize=(5, 3.5))
    #loglog draws both axes on a log scale, "o-" means dots joined by a line
    plt.loglog(users, errors, "o-", label="ALS")
    #axhline draws a flat dashed line at the noise level
    plt.axhline(0.1, color="gray", linestyle="--", label="noise 0.1")
    plt.xlabel("number of users")
    plt.ylabel("RMSE on hidden cells")
    #legend shows which line is which
    plt.legend()
    #tight_layout stops labels from being cut off
    plt.tight_layout()
    #dpi=150 is the picture sharpness
    plt.savefig(FIGURES / "simulation_error_vs_size.png", dpi=150)
    #close frees the picture before the next one
    plt.close()


def part_3_stress_test():
    print()
    print("part 3  stress test with fewer known cells")
    #share of cells we keep, from 30 percent down to 5 percent
    fractions = [0.30, 0.20, 0.15, 0.10, 0.07, 0.05]
    errors = []
    for frac in fractions:
        R, R_true, mask = simulate_ratings(nu=100, ni=80, k=3, frac=frac, seed=0)
        U, V, J_hist = my_als(R, mask, k=3, lam=1.0, iters=50, seed=0)
        error = my_rmse(U @ V.T, R_true, ~mask)
        errors.append(error)
        #mask.sum(axis=1) counts known ratings in each row, so one count per user
        #known ratings of the user with the fewest ratings
        fewest = int(mask.sum(axis=1).min())
        never_up = bool(np.all(np.diff(J_hist) <= 1e-9))
        #two f strings next to each other are joined into one line
        print(f"known {frac:.2f}   per user mean {mask.sum(axis=1).mean():5.1f} min {fewest:2d}"
              f"   rmse on hidden cells {error:.4f}   J never increases {never_up}")

    plt.figure(figsize=(5, 3.5))
    plt.plot(fractions, errors, "o-", label="ALS")
    #pass line from the iteration 3 instructions
    plt.axhline(0.3, color="gray", linestyle="--", label="pass line 0.3")
    plt.xlabel("share of known cells")
    plt.ylabel("RMSE on hidden cells")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "stress_test_known_share.png", dpi=150)
    plt.close()


#this part runs only when we start the file from the terminal
#python src/check_simulation.py
if __name__ == "__main__":
    part_1_truth_and_library()
    part_2_bigger_matrices()
    part_3_stress_test()
