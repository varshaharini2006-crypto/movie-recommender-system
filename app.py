import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

@st.cache_data
def load_data():
    with open('item_similarity_df.pkl', 'rb') as f:
        item_similarity_df = pickle.load(f)
    with open('train_matrix.pkl', 'rb') as f:
        train_matrix = pickle.load(f)
    movies = pd.read_csv('movies_clean.csv')
    return item_similarity_df, train_matrix, movies

item_similarity_df, train_matrix, movies = load_data()

def recommend_for_user(user_id, n=10, min_similarity_sum=3.0):
    if user_id not in train_matrix.index:
        return None
    
    user_ratings = train_matrix.loc[user_id].dropna()
    seen_movies = set(user_ratings.index)
    
    weighted_scores = pd.Series(dtype=float)
    similarity_sums = pd.Series(dtype=float)
    
    for movie_id, rating in user_ratings.items():
        sims = item_similarity_df[movie_id]
        positive_sims = sims[sims > 0]
        weighted_scores = weighted_scores.add(positive_sims * rating, fill_value=0)
        similarity_sums = similarity_sums.add(positive_sims, fill_value=0)
    
    similarity_sums = similarity_sums[similarity_sums >= min_similarity_sum]
    weighted_scores = weighted_scores[similarity_sums.index]
    predicted = weighted_scores / similarity_sums
    predicted = predicted.drop(labels=seen_movies, errors='ignore')
    
    top_scores = predicted.sort_values(ascending=False).head(n)
    result = movies[movies['movieId'].isin(top_scores.index)][['movieId', 'title', 'genres']].copy()
    result['predicted_rating'] = result['movieId'].map(top_scores).round(2)
    return result.sort_values('predicted_rating', ascending=False)

st.title("🎬 Movie Recommender")
st.write("Item-based collaborative filtering, trained on the MovieLens dataset.")

user_ids = sorted(train_matrix.index.tolist())
selected_user = st.selectbox("Select a user ID to generate recommendations for:", user_ids)

n_recs = st.slider("Number of recommendations:", 5, 20, 10)

if st.button("Get Recommendations"):
    recs = recommend_for_user(selected_user, n=n_recs)
    if recs is not None and len(recs) > 0:
        st.subheader(f"Top {n_recs} recommendations for User {selected_user}")
        st.dataframe(recs[['title', 'genres', 'predicted_rating']], use_container_width=True)
    else:
        st.warning("Not enough data to generate recommendations for this user.")