import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# ==========================================
# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ==========================================
def load_and_prepare_data(filepath="data/movie_ratings.data"):
    print("Завантаження та підготовка даних...")
    data = pd.read_csv(filepath, sep=r"\s+", engine="python", header=None)
    data.columns = ["user", "item", "rating", "timestamp"]
    data = data.drop(columns=["timestamp"])

    unique_users = data["user"].unique()
    unique_movies = data["item"].unique()
    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    movie_to_idx = {mid: i for i, mid in enumerate(unique_movies)}

    data['u_idx'] = data['user'].map(user_to_idx)
    data['i_idx'] = data['item'].map(movie_to_idx)

    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    n_users = len(user_to_idx)
    n_movies = len(movie_to_idx)

    A_train = sp.csr_matrix(
        (train_df["rating"].values, (train_df["u_idx"].values, train_df["i_idx"].values)),
        shape=(n_users, n_movies), dtype=np.float32
    )
    A_test = sp.csr_matrix(
        (test_df["rating"].values, (test_df["u_idx"].values, test_df["i_idx"].values)),
        shape=(n_users, n_movies), dtype=np.float32
    )

    return train_df, test_df, A_train, A_test, n_users, n_movies


# ==========================================
# 2. АЛГОРИТМ 1: Custom MF (з твого algorithm.py)
# ==========================================
def train_mf(A_train, A_test, iterations, k=10, alpha=0.0005, lam=0.01):
    print(f"\n--- 1. Навчання MF ({iterations} ітерацій) ---")
    A = A_train.toarray().astype(np.float32)
    n_users, n_movies = A.shape
    U = np.random.rand(n_users, k).astype(np.float32)
    F = np.random.rand(k, n_movies).astype(np.float32)
    mask = (A > 0).astype(np.float32)
    
    test_users, test_movies = A_test.nonzero()
    actual_test = A_test.data

    loss_hist, rmse_hist = [], []

    for t in range(iterations):
        E = (A - U @ F) * mask
        U -= alpha * (-2 * (E @ F.T) + 2 * lam * U)
        F -= alpha * (-2 * (U.T @ E) + 2 * lam * F)

        loss = np.sum(E ** 2) + lam * (np.sum(U ** 2) + np.sum(F ** 2))
        loss_hist.append(loss)
        
        preds = np.sum(U[test_users, :] * F[:, test_movies].T, axis=1)
        rmse_hist.append(np.sqrt(np.mean((preds - actual_test) ** 2)))

        if (t + 1) % 100 == 0:
            print(f"MF Ітерація {t+1:>4} | Loss: {loss:.2f} | RMSE: {rmse_hist[-1]:.4f}")

    return loss_hist, rmse_hist


# ==========================================
# 3. АЛГОРИТМ 2: FunkSVD (Stochastic Gradient Descent)
# ==========================================
def train_svd(train_df, test_df, n_users, n_items, iterations, n_factors=10, lr=0.005, reg=0.02):
    print(f"\n--- 2. Навчання SVD ({iterations} ітерацій) ---")
    global_mean = train_df["rating"].mean()
    b_u, b_i = np.zeros(n_users), np.zeros(n_items)
    P = np.random.normal(0, 0.1, (n_users, n_factors))
    Q = np.random.normal(0, 0.1, (n_items, n_factors))
    
    u_tr, i_tr, r_tr = train_df["u_idx"].values, train_df["i_idx"].values, train_df["rating"].values
    u_te, i_te, r_te = test_df["u_idx"].values, test_df["i_idx"].values, test_df["rating"].values
    
    loss_hist, rmse_hist = [], []
    
    for epoch in range(iterations):
        for u, i, r in zip(u_tr, i_tr, r_tr):
            err = r - (global_mean + b_u[u] + b_i[i] + np.dot(P[u], Q[i]))
            b_u[u] += lr * (err - reg * b_u[u])
            b_i[i] += lr * (err - reg * b_i[i])
            P_u = P[u].copy()
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * P_u - reg * Q[i])
        
        tr_preds = global_mean + b_u[u_tr] + b_i[i_tr] + np.sum(P[u_tr] * Q[i_tr], axis=1)
        loss = np.sum((r_tr - tr_preds)**2) + reg * (np.sum(b_u**2) + np.sum(b_i**2) + np.sum(P**2) + np.sum(Q**2))
        loss_hist.append(loss)
        
        te_preds = global_mean + b_u[u_te] + b_i[i_te] + np.sum(P[u_te] * Q[i_te], axis=1)
        rmse_hist.append(np.sqrt(mean_squared_error(r_te, te_preds)))
        
        if (epoch + 1) % 100 == 0:
            print(f"SVD Ітерація {epoch+1:>4} | Loss: {loss:.2f} | RMSE: {rmse_hist[-1]:.4f}")
            
    return loss_hist, rmse_hist


# ==========================================
# 4. АЛГОРИТМ 3: ALS (Alternating Least Squares)
# ==========================================
def train_als(A_train, A_test, iterations, k=10, lam=0.1):
    print(f"\n--- 3. Навчання ALS ({iterations} ітерацій) ---")
    A_csr, A_csc = A_train.tocsr(), A_train.tocsc()
    n_users, n_movies = A_train.shape
    U = np.random.rand(n_users, k).astype(np.float32)
    F = np.random.rand(k, n_movies).astype(np.float32)
    I = np.eye(k, dtype=np.float32)

    tr_users, tr_movies = A_train.nonzero()
    te_users, te_movies = A_test.nonzero()

    loss_hist, rmse_hist = [], []

    for t in range(iterations):
        for u in range(n_users):
            idx = A_csr.indices[A_csr.indptr[u]:A_csr.indptr[u+1]]
            if len(idx) > 0:
                F_u = F[:, idx]
                U[u, :] = np.linalg.solve(F_u @ F_u.T + lam * I, F_u @ A_csr.data[A_csr.indptr[u]:A_csr.indptr[u+1]])

        for i in range(n_movies):
            idx = A_csc.indices[A_csc.indptr[i]:A_csc.indptr[i+1]]
            if len(idx) > 0:
                U_i = U[idx, :]
                F[:, i] = np.linalg.solve(U_i.T @ U_i + lam * I, U_i.T @ A_csc.data[A_csc.indptr[i]:A_csc.indptr[i+1]])

        tr_preds = np.sum(U[tr_users, :] * F[:, tr_movies].T, axis=1)
        loss = np.sum((A_train.data - tr_preds) ** 2) + lam * (np.sum(U ** 2) + np.sum(F ** 2))
        loss_hist.append(loss)
        
        te_preds = np.sum(U[te_users, :] * F[:, te_movies].T, axis=1)
        rmse_hist.append(np.sqrt(np.mean((te_preds - A_test.data) ** 2)))

        if (t + 1) % 100 == 0:
            print(f"ALS Ітерація {t+1:>4} | Loss: {loss:.2f} | RMSE: {rmse_hist[-1]:.4f}")

    return loss_hist, rmse_hist


# ==========================================
# 5. ПОБУДОВА ГРАФІКІВ
# ==========================================
def plot_all_algorithms(iters_count, losses, rmses):
    print("\nПобудова графіків...")
    x = np.arange(1, iters_count + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Графік Loss
    ax1.plot(x, losses['mf'], label="Gradient Descent (Our algorithm)", color='#2ca02c', linewidth=2)
    ax1.plot(x, losses['svd'], label='FunkSVD (SGD)', color='#1f77b4', linewidth=2)
    ax1.plot(x, losses['als'], label='Alternating Least Squares (ALS)', color='#ff7f0e', linewidth=2)
    
    ax1.set_title('Losses on the train data', fontsize=14)
    ax1.set_xlabel('Iterations number', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_yscale('log') # Логарифмічна шкала для коректного порівняння різних порядків
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Графік RMSE
    ax2.plot(x, rmses['mf'], label="Gradient Descent (Our algorithm)", color='#2ca02c', linewidth=2)
    ax2.plot(x, rmses['svd'], label='FunkSVD (SGD)', color='#1f77b4', linewidth=2)
    ax2.plot(x, rmses['als'], label='Alternating Least Squares (ALS)', color='#ff7f0e', linewidth=2)
    
    ax2.set_title('Change of RMSE on test data', fontsize=14)
    ax2.set_xlabel('Iterations number', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(f'The comparison of 3 algorithms (over {iters_count} iterations)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Параметри
    N_ITERATIONS = 2000
    K_FACTORS = 5  # Однакова кількість латентних факторів для всіх моделей

    # 1. Підготовка
    train_df, test_df, A_train, A_test, n_users, n_movies = load_and_prepare_data("data/movie_ratings.data")

    # 2. Навчання
    mf_loss, mf_rmse = train_mf(A_train, A_test, N_ITERATIONS, k=K_FACTORS)
    svd_loss, svd_rmse = train_svd(train_df, test_df, n_users, n_movies, N_ITERATIONS, n_factors=K_FACTORS)
    als_loss, als_rmse = train_als(A_train, A_test, N_ITERATIONS, k=K_FACTORS)

    # 3. Формуємо словники з результатами та будуємо графік
    losses = {'mf': mf_loss, 'svd': svd_loss, 'als': als_loss}
    rmses = {'mf': mf_rmse, 'svd': svd_rmse, 'als': als_rmse}

    plot_all_algorithms(N_ITERATIONS, losses, rmses)