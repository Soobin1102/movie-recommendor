import pandas as pd
from scipy import sparse
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("ratings.csv")

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
id_to_title = dict(zip(movies['movieId'], movies['title']))
title_to_id = dict(zip(movies['title'], movies['movieId']))
user_item_matrix = df.pivot_table(index='userId',columns='movieId',values='rating').fillna(0)

mov_sim=cosine_similarity(user_item_matrix.T)
mov_sim_df = pd.DataFrame (
    mov_sim, 
    index=user_item_matrix.columns, 
    columns=user_item_matrix.columns)

def recommend(movie_name, n=10):
    if movie_name not in title_to_id:
        return f"Movie '{movie_name}' not found!"
    movie_id = title_to_id[movie_name]
    sim_scores = mov_sim_df[movie_id]
    sim_mov=sim_scores.sort_values(ascending=False)
    recom_ids=sim_mov.iloc[1:n+1].index.tolist()
    recom_titles=[id_to_title[mid] for mid in recom_ids]
    return recom_titles

# --- Streamlit UI ---
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender System")
st.write("Find similar movies using Collaborative Filtering (cosine similarity).")

# --- Search input ---
search_text = st.text_input("Type a movie name:")

# --- Filter top 10 matches ---
if search_text:
    matches = [m for m in movies['title'] if search_text.lower() in m.lower()]
    matches = matches[:10]  # Limit to top 10 matches
    if matches:
        selected_movie = st.selectbox("Select from top matches:", matches)
    else:
        st.write("No matching movies found.")
        selected_movie = None
else:
    selected_movie = None

# --- Recommend button ---
if selected_movie and st.button("Recommend"):
    recs = recommend(selected_movie)
    if recs:
        st.subheader("Recommended Movies:")
        for m in recs:
            st.write(f"- {m}")
    else:
        st.write("No recommendations found.")
#$env:PATH += ";C:\Users\subha\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts"
