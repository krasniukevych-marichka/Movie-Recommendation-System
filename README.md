# Movie Recommendation System

A collaborative filtering system for movie rating prediction based on Matrix Factorization with gradient descent optimization.

---

## Links to the video presentation
Krasniukevych Mariia: https://youtu.be/ex1D9ArDJxU
Kukurudza Viktoria: https://youtu.be/Oyqh00EUd_0
Svyrydenko Iryna: https://youtu.be/M0lbopi9-Gw?si=f3nNJyAgQ-YR9Tvx 

---

## Overview

The system learns low-rank approximations of a user–item rating matrix using gradient descent. Given a sparse matrix of observed ratings **A ∈ ℝ^(n×m)**, the model seeks factor matrices **U ∈ ℝ^(n×k)** and **F ∈ ℝ^(k×m)** such that:

$$A \approx U \cdot F$$

Training minimizes the regularized squared reconstruction error over observed entries:

$$\mathcal{L} = \sum_{(i,j) \in \Omega} \left( A_{ij} - (UF)_{ij} \right)^2 + \lambda \left( \|U\|_F^2 + \|F\|_F^2 \right)$$

where **Ω** denotes the set of observed ratings and **λ** is the L2 regularization coefficient.

---


## Dependencies

| Package      | Version  |
|--------------|----------|
| Python       | ≥ 3.8    |
| numpy        | latest   |
| pandas       | latest   |
| scipy        | latest   |
| scikit-learn | latest   |


---

## Data Format

The model expects a whitespace-delimited file `movie_ratings.data` with the following schema:

```
UserID   MovieID   Rating   Timestamp
```

The `Timestamp` column is parsed but discarded prior to matrix construction. `UserID` and `MovieID` are re-indexed to contiguous integer indices internally.

---

## Methodology

### 1. Data Loading (`load_rating_matrix`)

Reads the ratings file and constructs a coordinate-format sparse matrix. Supports optional truncation via `max_ratings` for rapid prototyping.

### 2. Train/Test Split (`split_train_test`)

Ratings are partitioned into training and test subsets prior to any model fitting, eliminating data leakage. Default split ratio: 80% train / 20% test, with a fixed random seed for reproducibility.

### 3. Matrix Factorization (`matrix_factorization`)

Gradient descent updates at each iteration `t`:

```
E  = (A - U·F) ⊙ Mask
∂U = -2·E·Fᵀ + 2λ·U
∂F = -2·Uᵀ·E + 2λ·F

U ← U - α·∂U
F ← F - α·∂F
```

where **Mask** is a binary matrix indicating observed entries and **α** is the learning rate.

### 4. Evaluation (`evaluate_rmse`)

Model quality is assessed using Root Mean Square Error on held-out ratings:

$$\text{RMSE} = \sqrt{ \frac{1}{|\Omega_{\text{test}}|} \sum_{(i,j) \in \Omega_{\text{test}}} \left( A_{ij} - (UF)_{ij} \right)^2 }$$

---

## Usage

```bash
python recommendation_system.py
```

---

## Hyperparameters

| Parameter    | Default  | Description                           |
|--------------|----------|---------------------------------------|
| `k`          | `5`      | Rank of latent factor matrices        |
| `alpha`      | `0.0005` | Gradient descent learning rate        |
| `lam`        | `0.001`  | L2 regularization coefficient         |
| `iterations` | `2000`   | Number of gradient descent iterations |
| `test_ratio` | `0.2`    | Proportion of ratings held out        |

---

## Output

At each iteration, the training loss is reported to stdout:

```
Iteration    1 / 2000 — Loss: 14823.1234
...
Iteration 2000 / 2000 — Loss: 312.4401

RMSE: 1.0342
```

---

