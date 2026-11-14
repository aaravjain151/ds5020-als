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
| `docs/` | Overleaf PDF and the Excel plan |

## How to get the data

The raw data is not stored in git because the rules ask us to keep large raw files out. Download it once from the project folder with the commands below.

```
cd data && mkdir -p raw && cd raw
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

## Who did what

| Part | Owner |
|---|---|
| Data loading and description, simulation check, repository | Margarita |
