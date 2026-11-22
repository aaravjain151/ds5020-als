# Movie Recommender with Matrix Factorization

DS5020 final project, Fall 2026. Team members are Aarav Jain, Madhu Vadhani Madasamy and Margarita Kholostova.

## What this project does

We have a table of movie ratings where one row is one user and one column is one movie. Most cells are empty because each person rates only a few movies (only 6.3% of the cells have a rating). We want to guess the empty cells, so that we can recommend movies a person has not seen yet but would probably like. To do this we write the big table as a product of two thin tables U and V (matrix factorization) and find U and V with alternating least squares (ALS). Then we compare ALS with a simpler method, the truncated SVD. So in the end we can say how well each method guesses ratings that were hidden from it.

## Folders

| Folder | What is inside |
|---|---|
| `src/` | all Python code |
| `notebooks/` | `iteration2.ipynb` runs the code step by step and shows every table and plot |
| `data/` | the dataset (not in git, see below) |
| `figures/` | plots made by the code |
| `docs/` | report PDFs (`report_iteration2.pdf`, `report_iteration3.pdf`), each member's LaTeX part, and the Excel plan |

## How to get the data

The raw data is not stored in git because the rules ask us to keep large raw files out. Download it once from the project folder with the commands below.

```
mkdir -p data/raw && cd data/raw
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

After this the files are in `data/raw/ml-100k/`, so the code can find them.

## How to run

We use only numpy, pandas, matplotlib and scipy. The easiest way to see everything is to open `notebooks/iteration2.ipynb` and run all cells. The scripts below are a quick check from the terminal (run them from the project folder).

| Command | What it does |
|---|---|
| `python src/load_data.py` | loads MovieLens 100K and builds the rating matrix R, prints its size |
| `python src/simulate.py` | builds a rating matrix where we know the true answer, hides 70% of it, runs ALS and prints the checks |
| `python src/small_example.py` | Iteration 3: one full ALS step on a 3 by 3 matrix with exact fractions |
| `python src/verify_small_example.py` | Iteration 3: checks the small example a second way with numpy |
| `python src/check_simulation.py` | Iteration 3: our ALS against the truth and against numpy, bigger matrices, stress test |
| `python src/run_real.py` | Iteration 3: train on `u1.base`, test on `u1.test`, baselines, every k and lambda, numpy check (about 2 minutes) |
| `python src/check_als_steps.py` | Iteration 3: every small system of one ALS step against `numpy.linalg.solve`, J after every half step on the simulation and on MovieLens, and where the simulation error comes from (about 10 seconds) |

All scripts use seed 0, so every run gives the same numbers. The last digits of the tiny differences (numbers like `1e-15`) can change from one computer to another, because they are rounding error.

### What you should see (Iteration 3)

| Script | Main numbers it prints | Figure it saves |
|---|---|---|
| `small_example.py` | U = (2, 2, 3), V = (5/3, 13/14, 17/14), J goes from 39 to 212/7, guesses 13/7 and 17/7 for the 2 missing cells | none |
| `verify_small_example.py` | numpy and `methods.py` give the same U and V as the fractions | none |
| `check_simulation.py` | RMSE 0.147 on the hidden cells (filled SVD 1.394), slope -1.08 over sizes, stress test passes down to 20% known | `simulation_error_vs_size.png`, `stress_test_known_share.png` |
| `run_real.py` | best test RMSE 0.940 at k = 2, lambda = 1 (movie mean 1.033), same as numpy to 10 digits | `real_test_rmse.png`, `real_objective.png` |
| `check_als_steps.py` | all 2,805 systems match numpy to about 1e-15, J never increases in 200 half steps, slope -0.63 with lambda = 0.01 | `objective_half_steps.png` |

## Who did what

| Part | Owner |
|---|---|
| Data loading and description, simulation check, repository | Margarita |
| Notation, assumptions, derivations Q2 to Q5 (math) | Madhu |
| Algorithm as numbered steps with sizes and operation counts, Overleaf report, check of the data, simulation, notation and Q2 to Q5 parts | Aarav |

The full Iteration 2 report is `docs/report_iteration2.pdf`. Each section of the report names its owner and its checker, and Margarita leads Iteration 2.

### Iteration 3

| Part | Owner | Checker |
|---|---|---|
| Small worked example (Q1), `small_example.py`, `verify_small_example.py` | Madhu | Margarita, Aarav |
| ALS from scratch (`methods.py`), simulation test, real data, SVD baseline, test RMSE grid, verification table | Margarita | Aarav, Madhu |
| Q2 and Q5 write-up, `check_als_steps.py` (every system against numpy, J after every half step), README, Iteration 3 report PDF | Aarav | Madhu, Margarita |

The full Iteration 3 report is `docs/report_iteration3.pdf`.
