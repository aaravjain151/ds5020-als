#load movielens 100k and turn it into the rating matrix R
#the raw files must be in data/raw/ml-100k see README for the download
#exploring and plots are in notebooks/iteration2.ipynb

from pathlib import Path

import numpy as np
import pandas as pd

#folder with the raw files
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "ml-100k"


def load_ratings(file_name="u.data"):
    #each line is one rating with user id then movie id then stars then time
    #the file has no header and columns are split by tabs
    column_names = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = pd.read_csv(RAW / file_name, sep="\t", names=column_names)
    return ratings


def load_movies():
    #each line is one movie with id then title then dates then link then genre flags
    #the file uses the bar sign as a separator and old latin encoding
    column_names = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"]

    #genre names are kept in a separate small file
    genre_table = pd.read_csv(RAW / "u.genre", sep="|", names=["genre", "id"])
    genre_names = genre_table["genre"].tolist()

    movies = pd.read_csv(RAW / "u.item", sep="|", names=column_names + genre_names, encoding="latin-1")
    return movies


def build_rating_matrix(ratings):
    #one row of R is one user and one column of R is one movie
    n_users = ratings["user_id"].max()
    n_movies = ratings["movie_id"].max()

    #start with an empty matrix full of zeros
    R = np.zeros((n_users, n_movies))

    #mask says which cells have a real rating and this set of cells is omega
    mask = np.zeros((n_users, n_movies), dtype=bool)

    #ids in the file start at 1 but python counts from 0 so we subtract 1
    rows = ratings["user_id"].to_numpy() - 1
    cols = ratings["movie_id"].to_numpy() - 1

    #put every rating in its cell and mark the cell as known
    R[rows, cols] = ratings["rating"].to_numpy()
    mask[rows, cols] = True
    return R, mask


if __name__ == "__main__":
    #quick check from the terminal
    ratings = load_ratings()
    R, mask = build_rating_matrix(ratings)
    print("ratings", len(ratings))
    print("R shape", R.shape)
    print("known cells", int(mask.sum()))
