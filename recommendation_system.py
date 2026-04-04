import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


def load_rating_matrix(filepath: str = "movie_ratings.data", max_ratings: int = None):
    """
    Reads movie_ratings.data and returns a sparse ratings matrix.
    """
    data = pd.read_csv(filepath, sep=r"\s+", engine="python", header=None)
    data.columns = ["UserID", "MovieID", "Rating", "Date"]
    data = data.drop(columns=["Date"])

    if max_ratings is not None:
        data = data.iloc[:max_ratings]

    unique_users = data["UserID"].unique()
    unique_movies = data["MovieID"].unique()
    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    movie_to_idx = {mid: i for i, mid in enumerate(unique_movies)}

    rows = data["UserID"].map(user_to_idx).values
    cols = data["MovieID"].map(movie_to_idx).values
    ratings = data["Rating"].values.astype(np.float32)

    n_users = len(unique_users)
    n_movies = len(unique_movies)

    return rows, cols, ratings, n_users, n_movies, user_to_idx, movie_to_idx

