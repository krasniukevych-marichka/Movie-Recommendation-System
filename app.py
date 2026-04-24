from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

U = np.load("models/matrix_U.npy")
F = np.load("models/matrix_F.npy")
with open("models/movie_to_idx.pkl", "rb") as f:
    movie_to_idx = pickle.load(f)

idx_to_movie = {v: k for k, v in movie_to_idx.items()}

titles_df = pd.read_csv("data/movie_titles.csv")
movie_titles_map = dict(zip(titles_df['item_id'].astype(str), titles_df['title']))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_data = request.json  
    
    k = 5
    v_user = np.random.rand(k).astype(np.float32)
    
    rated_movies = [m for m in user_data if m['rating'] is not None]
    
    if not rated_movies:
        return jsonify([]) 

    alpha = 0.005
    for _ in range(100): 
        for m in rated_movies:
            m_idx = movie_to_idx.get(str(m['id']))
            if m_idx is None: continue
            
            rating = float(m['rating'])
            prediction = v_user @ F[:, m_idx]
            error = rating - prediction
            
            v_user += alpha * (2 * error * F[:, m_idx])

    all_predictions = v_user @ F
    
    results = []
    idx_to_movie = {v: k for k, v in movie_to_idx.items()}
    
    for i, score in enumerate(all_predictions):
        m_id = idx_to_movie[i]  
        title = movie_titles_map.get(str(m_id), f"Unknown Movie (ID: {m_id})")
        
        results.append({
            "id": int(m_id),           
            "title": str(title),      
            "predicted_rating": round(float(score), 2)  
        })

    results.sort(key=lambda x: x['predicted_rating'], reverse=True)
    return jsonify(results[:5])

if __name__ == '__main__':
    app.run(debug=True)