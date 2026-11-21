#run als on movielens 100k and check it
#train on u1.base and test on u1.test so the result can be compared with other teams
#part 1 loads the split and prints every cleaning number
#part 2 gives the simple baselines
#part 3 gives the test rmse for every k and lambda
#part 4 compares our als with the numpy version on the real data

#Path helps to build folder paths that work on every computer
from pathlib import Path

#matplotlib makes the plots
import matplotlib
#Agg saves plots to files and does not open a window
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#iteration 2 functions that read the files and build the rating matrix
from load_data import build_rating_matrix, load_ratings
#our own functions from src/methods.py
from methods import my_als, my_movie_mean, my_rmse, report
#als there uses numpy.linalg.solve so we use it as the library answer
from simulate import als, svd_baseline

#figures folder of the project
#__file__ is this script, parent is src/ and parent of that is the project folder
FIGURES = Path(__file__).resolve().parent.parent / "figures"

#values asked in the iteration 3 instructions
#k is the number of factors and lambda is the penalty weight
K_VALUES = [2, 5, 10, 20]
LAM_VALUES = [0.1, 1.0, 10.0]
#number of full als steps, the same as in iteration 2
ITERS = 50
#fixed seed so every team member gets the same numbers
SEED = 0
#the setting with the lowest test rmse in part 3
BEST_K = 2
BEST_LAM = 1.0


def part_1_load():
    print("part 1  the split")
    #train is what the method learns from, test is hidden from it
    train = load_ratings("u1.base")
    test = load_ratings("u1.test")
    #both matrices are 943 by 1682 so the cells line up
    R_train, mask_train = build_rating_matrix(train)
    R_test, mask_test = build_rating_matrix(test)

    #len gives the number of rows, which is the number of ratings
    print("train ratings", len(train), "  test ratings", len(test))
    #we do not remove any rows, this line is here so the report shows it
    print("rows removed", 0)
    #& means "and", so this counts cells that are in train and in test at once
    #it must be 0 or the test would not be hidden
    print("cells in both train and test", int((mask_train & mask_test).sum()))

    #users and movies that the method never sees in training
    #sum(axis=1) counts known cells in each row (user)
    #== 0 marks the users with none and the last sum counts them
    users_no_train = int((mask_train.sum(axis=1) == 0).sum())
    #sum(axis=0) counts known cells in each column (movie)
    #this gives a true or false for each movie
    movies_no_train = mask_train.sum(axis=0) == 0
    print("users with no train rating", users_no_train)
    print("movies with no train rating", int(movies_no_train.sum()))
    #test ratings that fall on those movies
    print("test ratings on those movies", int(mask_test[:, movies_no_train].sum()))
    print("fewest train ratings for one user", int(mask_train.sum(axis=1).min()))
    return R_train, mask_train, R_test, mask_test


def part_2_baselines(R_train, mask_train, R_test, mask_test):
    print()
    print("part 2  baselines on the test ratings")
    #first baseline   guess the average of all train ratings for every cell
    mean_all = R_train[mask_train].mean()
    #np.full makes a matrix of this size where every cell is mean_all
    P_all = np.full(R_train.shape, mean_all)
    #:.3f keeps 3 digits after the point
    print(f"one number for everything ({mean_all:.3f})   rmse {my_rmse(P_all, R_test, mask_test):.4f}")

    #second baseline   guess the average rating of each movie
    P_movie = my_movie_mean(R_train, mask_train)
    movie_rmse = my_rmse(P_movie, R_test, mask_test)
    print(f"mean rating of each movie            rmse {movie_rmse:.4f}")

    #question 3  fill with the mean then keep k singular values
    svd_rmse = []
    for k in K_VALUES:
        P_svd = svd_baseline(R_train, mask_train, k=k)
        svd_rmse.append(my_rmse(P_svd, R_test, mask_test))
        #svd_rmse[-1] is the last number we added to the list
        print(f"filled SVD k = {k:2d}                    rmse {svd_rmse[-1]:.4f}")
    return movie_rmse, svd_rmse


def part_3_grid(R_train, mask_train, R_test, mask_test):
    print()
    print("part 3  als test rmse for every k and lambda")
    #one row for each k and one column for each lambda
    test_rmse = np.zeros((len(K_VALUES), len(LAM_VALUES)))
    #a picks the k and b picks the lambda, so we try all 12 pairs
    for a in range(len(K_VALUES)):
        for b in range(len(LAM_VALUES)):
            k = K_VALUES[a]
            lam = LAM_VALUES[b]
            U, V, J_hist = my_als(R_train, mask_train, k=k, lam=lam, iters=ITERS, seed=SEED)
            #P is the guess for every cell
            P = U @ V.T
            test_rmse[a, b] = my_rmse(P, R_test, mask_test)
            #error on the ratings it learned from, to compare with the test error
            train_rmse = my_rmse(P, R_train, mask_train)

            #question 5  the objective never goes up
            #np.diff gives the change of J from one step to the next
            never_up = bool(np.all(np.diff(J_hist) <= 1e-9))
            print(f"k = {k:2d}  lambda = {lam:4.1f}   train rmse {train_rmse:.4f}"
                  f"   test rmse {test_rmse[a, b]:.4f}   J never increases {never_up}")
    return test_rmse


def part_4_library(R_train, mask_train, R_test, mask_test):
    print()
    print(f"part 4  our als against the numpy als on the real data (k = {BEST_K}, lambda = {BEST_LAM})")
    #same data, same start, so any difference comes from how the small systems are solved
    U, V, J_hist = my_als(R_train, mask_train, k=BEST_K, lam=BEST_LAM, iters=ITERS, seed=SEED)
    U_lib, V_lib, J_lib = als(R_train, mask_train, k=BEST_K, lam=BEST_LAM, iters=ITERS, seed=SEED)
    report("U V^T mine vs numpy als", U @ V.T, U_lib @ V_lib.T)
    report("J history mine vs numpy als", J_hist, J_lib)
    #:.10f prints 10 digits after the point so we can see they match
    print("test rmse mine", f"{my_rmse(U @ V.T, R_test, mask_test):.10f}")
    print("test rmse numpy", f"{my_rmse(U_lib @ V_lib.T, R_test, mask_test):.10f}")
    return J_hist


def make_plots(test_rmse, movie_rmse, J_hist):
    #test rmse against k with one line for each lambda
    plt.figure(figsize=(5.5, 3.5))
    for b in range(len(LAM_VALUES)):
        #test_rmse[:, b] is column b, so all k values for one lambda
        plt.plot(K_VALUES, test_rmse[:, b], "o-", label=f"ALS lambda = {LAM_VALUES[b]}")
    #flat dashed line for the movie mean baseline
    plt.axhline(movie_rmse, color="gray", linestyle="--", label="movie mean")
    #log scale so 2, 5, 10, 20 are spread evenly
    plt.xscale("log")
    #show exactly 2, 5, 10, 20 under the axis
    plt.xticks(K_VALUES, K_VALUES)
    plt.xlabel("k")
    plt.ylabel("test RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "real_test_rmse.png", dpi=150)
    plt.close()

    #question 5  objective after each step for the best run
    plt.figure(figsize=(5, 3.5))
    #semilogy puts only the up axis on a log scale
    plt.semilogy(range(len(J_hist)), J_hist, "o-")
    plt.xlabel("ALS step")
    plt.ylabel("objective J")
    plt.tight_layout()
    plt.savefig(FIGURES / "real_objective.png", dpi=150)
    plt.close()


#this part runs only when we start the file from the terminal
#python src/run_real.py
if __name__ == "__main__":
    R_train, mask_train, R_test, mask_test = part_1_load()
    movie_rmse, svd_rmse = part_2_baselines(R_train, mask_train, R_test, mask_test)
    test_rmse = part_3_grid(R_train, mask_train, R_test, mask_test)
    J_hist = part_4_library(R_train, mask_train, R_test, mask_test)
    make_plots(test_rmse, movie_rmse, J_hist)
