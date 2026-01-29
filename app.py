import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
st.set_page_config(
    page_title="Sistem Rekomendasi Film",
    page_icon="🎬",
    layout="wide"
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    ratings = pd.read_csv("ml-latest-small/ratings.csv")
    movies = pd.read_csv("ml-latest-small/movies.csv")
    return ratings, movies

ratings, movies = load_data()

# ================= MODEL =================
def build_model(ratings):
    user_item_matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)

    similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    return user_item_matrix, similarity_df

def recommend_movies(user_id, ratings, movies, top_n=5):
    user_item_matrix, similarity_df = build_model(ratings)

    if user_id not in similarity_df.index:
        return pd.DataFrame(columns=["title", "genres"])

    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6]
    weighted_ratings = user_item_matrix.loc[similar_users.index].T.dot(similar_users)
    recommendations = weighted_ratings.sort_values(ascending=False).head(top_n)

    result = movies[movies['movieId'].isin(recommendations.index)]
    return result[['title', 'genres']]

# ================= UI =================
st.title("🎬 Sistem Rekomendasi Film")
st.markdown(
    """
    **Prototipe sistem rekomendasi film berbasis web**  
    Metode yang digunakan adalah **User-Based Collaborative Filtering**  
    Dataset: **MovieLens**
    """
)

st.divider()

# ================= INPUT AREA =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Pilih Pengguna")
    user_id = st.selectbox(
        "User ID",
        sorted(ratings['userId'].unique())
    )

with col2:
    st.subheader("⭐ Simulasi Transaksi Rating")
    movie_title = st.selectbox(
        "Pilih Film",
        movies['title'].values
    )
    rating = st.slider(
        "Beri Rating",
        0.5, 5.0, 0.5
    )

if st.button("💾 Simpan Rating"):
    movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]
    new_row = {
        "userId": user_id,
        "movieId": movie_id,
        "rating": rating,
        "timestamp": 999999999
    }
    ratings = pd.concat([ratings, pd.DataFrame([new_row])], ignore_index=True)
    st.success("Rating berhasil disimpan sebagai data transaksi")

st.divider()

# ================= RECOMMENDATION =================
st.subheader("📌 Rekomendasi Film untuk Pengguna")

if st.button("🎯 Tampilkan Rekomendasi"):
    result = recommend_movies(user_id, ratings, movies)

    if result.empty:
        st.warning("Belum cukup data untuk memberikan rekomendasi")
    else:
        for _, row in result.iterrows():
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; background-color:#f4f6f8; margin-bottom:10px">
                    <h4>🎥 {row['title']}</h4>
                    <p><strong>Genre:</strong> {row['genres']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

st.divider()

# ================= FOOTER =================
st.caption(
    "Dikembangkan sebagai prototipe Sistem Rekomendasi menggunakan Collaborative Filtering | UAS"
)
