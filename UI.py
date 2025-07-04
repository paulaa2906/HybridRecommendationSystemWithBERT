import streamlit as st
import pandas as pd
import base64
import sys
import os

from run_model import run_svd_recommender
from recommendation import get_recommendations_for_user

# --- Konstanta untuk Path File ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "svd_model_0.8503.joblib")
USER_DATA_PATH = os.path.join(BASE_DIR, "dataset", "ratings.csv")
MOVIES_DATA_PATH = os.path.join(BASE_DIR, "dataset", "movies.csv")
SENTIMENT_DATA_PATH = os.path.join(BASE_DIR, "dataset", "movie_sentiments_bert(fadli).csv")
BACKGROUND_IMAGE_PATH = os.path.join(BASE_DIR, "FilmCollage.png")

# --- Fungsi Utilitas untuk Tampilan ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background_image(image_file):
    if not os.path.exists(image_file):
        st.warning(f"File gambar latar belakang tidak ditemukan di: {image_file}")
        return
        
    bin_str = get_base64_of_bin_file(image_file)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        /* Label di luar kotak tetap putih */
        label {{
            color: white !important;
        }}
        /* Teks yang diketik pengguna di dalam kotak menjadi hitam */
        input[type="text"] {{
            color: black !important;
        }}
        /* Teks placeholder menjadi abu-abu gelap */
        input[type="text"]::placeholder {{
            color: #888888 !important; 
        }}
        .stButton > button {{
            background-color: #e63946; color: white; border-radius: 20px;
            border: 1px solid #e63946; padding: 0.5em 1.2em; font-weight: bold;
            transition: all 0.2s ease-in-out;
        }}
        .stButton > button:hover {{ background-color: white; color: #e63946; border: 1px solid #e63946; }}
        div[data-testid="stExpander"] {{
            background-color: rgba(40, 40, 40, 0.8); border-radius: 10px;
            border: 1px solid #4a4a4a; margin-bottom: 1rem;
        }}
        div[data-testid="stExpander"] summary {{ color: white; font-weight: bold; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Fungsi Pemuatan Data & Model ---
@st.cache_resource
def load_all_data():
    with st.spinner("Memuat model dan data awal... Ini mungkin butuh beberapa saat."):
        svd_model = run_svd_recommender()
        if svd_model is None:
            st.error("Gagal memuat model rekomendasi. Aplikasi tidak dapat berjalan.")
            st.stop()
        
        try:
            movies_df = pd.read_csv(MOVIES_DATA_PATH)
            df_movies_sentiment = pd.read_csv(SENTIMENT_DATA_PATH)
            user_df = pd.read_csv(USER_DATA_PATH)
        except FileNotFoundError as e:
            st.error(f"Error: File data tidak ditemukan! Pastikan path file benar. Detail: {e}")
            st.stop()

    return svd_model, movies_df, df_movies_sentiment, user_df

# --- Fungsi untuk Tampilan Halaman ---

def show_login_page(valid_user_ids):
    st.markdown("""
        <div style='text-align: center; color: white; text-shadow: 2px 2px 8px #000000;'>
            <h1 style='margin-bottom: 0.5rem;'>🎬 Movie Recommendation System</h1>
            <p style='margin-top: 0; font-size: 1.2rem;'>Temukan film yang akan kamu sukai berikutnya!</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    user_input = st.text_input("Masukkan User ID Anda", placeholder="Contoh: 123")
    if st.button("Login"):
        if user_input.strip() and user_input.strip() in valid_user_ids:
            st.session_state.logged_in = True
            st.session_state.user_id = user_input.strip()
            st.success("Login berhasil! Mengalihkan ke halaman utama...")
            st.rerun()
        else:
            st.error("❌ User ID tidak valid atau tidak ditemukan. Silakan coba lagi.")

def show_main_app(svd_model, movies_df, df_movies_sentiment, user_df):
    st.markdown(
        f"<h2 style='color: white; text-shadow: 1px 1px 3px #000;'>Selamat Datang, User {st.session_state.user_id}!</h2>",
        unsafe_allow_html=True
    )
    
    with st.expander("Lihat 10 Film dengan Rating Tertinggi yang Pernah Anda Berikan"):
        user_id_int = int(st.session_state.user_id)
        user_rated_df = user_df[user_df['userId'] == user_id_int]
        if not user_rated_df.empty:
            top_10_rated = user_rated_df.sort_values(by='rating', ascending=False).head(10)
            top_10_with_titles = pd.merge(top_10_rated, movies_df, on='movieId', how='left')
            for index, row in top_10_with_titles.iterrows():
                st.markdown(
                    f"""
                    <div style='padding: 8px 10px; border-radius: 5px; margin-bottom: 5px; border-bottom: 1px solid #4a4a4a;'>
                        <span style='color: #fafafa; font-weight: bold;'>{row['title']}</span>
                        <span style='float: right; color: #fafafa;'><strong>Rating Anda: {row['rating']} ⭐</strong></span>
                        <p style='font-size: 13px; color: #cccccc; margin-top: 5px; margin-bottom: 0;'>
                            {row['genres'].replace('|', ', ')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Anda belum memberikan rating untuk film manapun.")

    st.markdown("<h3 style='color: white; text-shadow: 1px 1px 3px #000;'>✨ Berikut Rekomendasi Film Untukmu:</h3>", unsafe_allow_html=True)

    with st.spinner("Sedang mengambil rekomendasi terbaik..."):
        recommendations = get_recommendations_for_user(
            user_id=int(st.session_state.user_id), 
            svd_model=svd_model, 
            df_movies=movies_df, 
            df_movies_sentiment=df_movies_sentiment, 
            #df_ratings=user_df,
            top_n=10
        )

    if not recommendations.empty:
        for index, movie in recommendations.iterrows():
            st.markdown(
                f"""
                <div style='background-color: rgba(0,0,0,0.7); padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #e63946;'>
                    <h4 style='color: white; margin-bottom: 5px;'>{movie['Title']}</h4>
                    <p style='margin-top: -5px; margin-bottom: 10px; font-size: 14px; color: #cccccc;'>
                        <strong>Genre:</strong> {movie['Genres'].replace('|', ', ')}
                    </p>
                    <span style='color: #f1faee; font-size: 14px;'>
                        Prediksi Rating: <strong>{movie['Predicted Rating']:.2f} ⭐</strong> | 
                        Sentimen Positif: <strong>{movie['Positive Sentiment (%)']}% 👍</strong>
                    </span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Tidak ada rekomendasi yang bisa ditampilkan untuk Anda saat ini.")

# --- Alur Utama Aplikasi ---
def main():
    st.set_page_config(page_title="Rekomendasi Film", layout="centered")
    set_background_image(BACKGROUND_IMAGE_PATH)

    svd_model, movies_df, df_movies_sentiment, user_df = load_all_data()
    valid_user_ids = set(user_df['userId'].astype(str).tolist())

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        show_login_page(valid_user_ids)
    else:
        show_main_app(svd_model, movies_df, df_movies_sentiment, user_df)

if __name__ == "__main__":
    main()
