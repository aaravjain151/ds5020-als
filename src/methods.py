#alternating least squares from scratch
#one function for one step of the algorithm from the iteration 2 report
#only numpy arrays and matrix products are used here
#numpy.linalg is used only in report and only to check an answer
#the same functions run the small example and the simulation and the real data

#numpy is the python library for arrays (tables of numbers) and matrix math
import numpy as np


#solves A x = b by gaussian elimination
#forward step   row_i = row_i - (a_ij / a_jj) row_j   for every row i below row j
#back step      x_j = (b_j - sum over l > j of a_jl x_l) / a_jj
#our A is positive definite (iteration 2 question 3) so the pivot a_jj is never zero
def my_gauss_solve(A, b):
    #k is the number of unknowns, the same as the length of b
    k = len(b)

    #put A and b side by side so every row operation changes both
    #T has k rows and k + 1 columns and the last column is b
    #astype(float) makes sure the numbers can have decimals
    T = np.column_stack([A, b]).astype(float)

    #forward step makes zeros below the diagonal
    #j is the column we clean and i is a row under row j
    for j in range(k):
        for i in range(j + 1, k):
            #how many times row j fits into row i at column j
            factor = T[i, j] / T[j, j]
            #subtract that many copies of row j so T[i, j] becomes 0
            T[i] = T[i] - factor * T[j]

    #back step finds x from the last row up
    #x starts as k zeros and we fill it from the end
    x = np.zeros(k)
    #range(k - 1, -1, -1) counts down k - 1, k - 2, ..., 0
    for j in range(k - 1, -1, -1):
        #part of row j that uses the x values we already found
        #@ is matrix product, here a dot product of two vectors
        known_part = T[j, j + 1:k] @ x[j + 1:]
        #T[j, k] is the number from b in this row
        x[j] = (T[j, k] - known_part) / T[j, j]
    return x


#J(U,V) = || M o (R - U V^T) ||_F^2 + lam ( ||U||_F^2 + ||V||_F^2 )
def my_objective(R, mask, U, V, lam):
    #V.T is V turned on its side (transpose) and U @ V.T is every guess at once
    #times mask keeps the error on known cells and makes it 0 on empty cells
    error = (R - U @ V.T) * mask
    #** 2 squares every number and np.sum adds them all
    return np.sum(error ** 2) + lam * (np.sum(U ** 2) + np.sum(V ** 2))


#step 1   U and V start as random normal numbers
#U is n_u by k and V is n_i by k
def my_init(nu, ni, k, seed):
    #random number maker with a fixed seed so every run gives the same numbers
    rng = np.random.default_rng(seed)
    #normal means numbers around 0, most of them between -2 and 2
    U = rng.normal(size=(nu, k))
    V = rng.normal(size=(ni, k))
    return U, V


#step 2   for every user u with V fixed
#(V_u^T V_u + lam I) u_u = V_u^T r_u
def my_update_users(R, mask, V, lam):
    #shape gives (rows, columns) so shape[0] is the number of users
    nu = R.shape[0]
    #shape[1] of V is the number of columns, which is k
    k = V.shape[1]
    #empty U that we fill one row at a time
    U = np.zeros((nu, k))
    for u in range(nu):
        #rows of V for the movies this user rated
        #mask[u] is a list of true and false, one for each movie
        V_u = V[mask[u]]
        #ratings of this user on those movies
        r_u = R[u, mask[u]]
        #np.eye(k) is the identity matrix I with 1 on the diagonal
        left = V_u.T @ V_u + lam * np.eye(k)
        right = V_u.T @ r_u
        #the answer is the new row u of U
        U[u] = my_gauss_solve(left, right)
    return U


#step 3   for every movie i with U fixed
#(U_i^T U_i + lam I) v_i = U_i^T r_i
def my_update_movies(R, mask, U, lam):
    #shape[1] of R is the number of movies
    ni = R.shape[1]
    k = U.shape[1]
    V = np.zeros((ni, k))
    for i in range(ni):
        #rows of U for the users who rated this movie
        #mask[:, i] is column i of the mask, : means all rows
        U_i = U[mask[:, i]]
        #ratings of this movie from those users
        r_i = R[mask[:, i], i]
        left = U_i.T @ U_i + lam * np.eye(k)
        right = U_i.T @ r_i
        #the answer is the new row i of V
        V[i] = my_gauss_solve(left, right)
    return V


#the whole algorithm   step 1 once then step 2 and step 3 iters times
#J_hist[t] is J after t full steps
def my_als(R, mask, k=3, lam=1.0, iters=50, seed=0):
    #number of users and number of movies
    nu, ni = R.shape
    U, V = my_init(nu, ni, k, seed)
    #list of J values, the first one is before any step
    J_hist = [my_objective(R, mask, U, V, lam)]
    #step is not used inside, the loop only repeats iters times
    for step in range(iters):
        U = my_update_users(R, mask, V, lam)
        V = my_update_movies(R, mask, U, lam)
        #append adds the new J at the end of the list
        J_hist.append(my_objective(R, mask, U, V, lam))
    #np.array turns the list into an array so we can do math on it later
    return U, V, np.array(J_hist)


#rmse = sqrt( (1 / |S|) sum over (u,i) in S of (p_ui - t_ui)^2 )
#S is the set of cells where M is true
#P is the guess, T is the true value, M says which cells to use
def my_rmse(P, T, M):
    #[M] keeps only the cells where M is true
    difference = (P - T)[M]
    #np.mean is the average and np.sqrt is the square root
    return np.sqrt(np.mean(difference ** 2))


#baseline   the guess for movie i is the mean of its known ratings
#m_i = (1 / |omega_i|) sum over u in omega_i of r_ui
#a movie with no known rating gets the mean of all known ratings
def my_movie_mean(R, mask):
    nu, ni = R.shape
    #average of all known ratings, used when a movie has none
    mean_all = R[mask].mean()
    P = np.zeros((nu, ni))
    for i in range(ni):
        #known ratings of movie i
        r_i = R[mask[:, i], i]
        if len(r_i) > 0:
            #every user gets the same guess for this movie
            P[:, i] = r_i.mean()
        else:
            P[:, i] = mean_all
    return P


#prints max |mine - ref| and the relative error ||mine - ref|| / ||ref||
#this helper is given in the iteration 3 instructions
#mine is our answer and ref is the answer we compare with
def report(name, mine, ref):
    #turn both into arrays of decimal numbers
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    #largest difference in any one cell
    abs_err = np.abs(mine - ref).max()
    #np.linalg.norm is the length of all the numbers together
    #so rel_err is the size of the difference compared with the size of ref
    rel_err = np.linalg.norm(mine - ref) / np.linalg.norm(ref)
    #:32s pads the name to 32 letters and :.2e writes the number like 1.23e-15
    print(f"{name:32s} max|diff| = {abs_err:.2e}   rel = {rel_err:.2e}")
    return abs_err
