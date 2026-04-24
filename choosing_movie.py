import pandas as pd
import numpy as np
import json
from algorithm import load_rating_matrix

rows, cols, ratings, n_users, n_movies, user_to_idx, movie_to_idx = load_rating_matrix("data/movie_ratings.data")
titles_df = pd.read_csv("data/movie_titles.csv")

def get_bayesian_top(rows, cols, ratings, movie_to_idx, n=20):
    idx_to_movie = {v: k for k, v in movie_to_idx.items()}
    
    counts = np.zeros(len(movie_to_idx))
    sums = np.zeros(len(movie_to_idx))
    np.add.at(counts, cols, 1)
    np.add.at(sums, cols, ratings)
    
    global_mean = ratings.mean()
    prior = 50
    
    bayesian_avg = (sums + prior * global_mean) / (counts + prior)
    
    top_indices = np.argsort(bayesian_avg)[::-1][:n]
    return [int(idx_to_movie[i]) for i in top_indices]

top_ids = get_bayesian_top(rows, cols, ratings, movie_to_idx, n=20)
selected = titles_df[titles_df['item_id'].isin(top_ids)].copy()

result_json = []
for _, row in selected.iterrows():
    result_json.append({
        "id": str(row['item_id']),
        "title": row['title']
    })

with open("static/selected_movies.json", "w", encoding="utf-8") as f:
    json.dump(result_json, f, ensure_ascii=False, indent=2)

print("Файл selected_movies.json успішно створено.")
