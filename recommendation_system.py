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


def matrix_factorization(A_sparse, k=10, alpha=0.005, lam=0.01, iterations=1000):
    """
    Matrix Factorization via Gradient Descent.
    """
    A = A_sparse.toarray().astype(np.float32)
    n_users, n_movies = A.shape

    U = np.random.rand(n_users, k).astype(np.float32)
    F = np.random.rand(k, n_movies).astype(np.float32)

    mask = (A > 0).astype(np.float32)

    for t in range(1, iterations + 1):
        E = (A - U @ F) * mask

        dU = -2 * (E @ F.T) + 2 * lam * U
        dF = -2 * (U.T @ E) + 2 * lam * F

        U = U - alpha * dU
        F = F - alpha * dF

        loss = np.sum(E ** 2) + lam * (np.sum(U ** 2) + np.sum(F ** 2))
        print(f"Iteration {t:>4} / {iterations} — Loss: {loss:.4f}")

    return U, F


def split_train_test(rows, cols, ratings, test_ratio=0.2):
    """
    Splits ratings into train and test BEFORE training.
    Returns two sparse matrices: A_train and A_test.
    """
    n_users = rows.max() + 1
    n_movies = cols.max() + 1

    idx = np.arange(len(ratings))
    train_idx, test_idx = train_test_split(idx, test_size=test_ratio, random_state=42)

    A_train = sp.csr_matrix(
        (ratings[train_idx], (rows[train_idx], cols[train_idx])),
        shape=(n_users, n_movies), dtype=np.float32
    )
    A_test = sp.csr_matrix(
        (ratings[test_idx], (rows[test_idx], cols[test_idx])),
        shape=(n_users, n_movies), dtype=np.float32
    )

    return A_train, A_test


def evaluate_rmse(U, F, A_test_sparse):
    """
    Calculate RMSE.
    """
    A_test = A_test_sparse.toarray()
    users, movies = A_test.nonzero()
    actual = A_test[users, movies]
    predicted = np.array([U[u] @ F[:, m] for u, m in zip(users, movies)])

    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    return rmse


if __name__ == "__main__":
    rows, cols, ratings, n_users, n_movies, user_to_idx, movie_to_idx = load_rating_matrix(
        "movie_ratings.data"
    )

    A_train, A_test = split_train_test(rows, cols, ratings, test_ratio=0.2)

    U, F = matrix_factorization(
        A_train,
        k=5,
        alpha=0.0005,
        lam=0.001,
        iterations=2000
    )

    rmse = evaluate_rmse(U, F, A_test)
    print(f"\nRMSE: {rmse:.4f}")