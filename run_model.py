#from dotenv import load_dotenv
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split


# from src.extract.extract_data import mount_drive
from src.model.model import SVDMatrixFactorization
from src.utils.helper import ratings, movies, movies_sentiment

#create df function to help change genre data frame into dict
def to_dict_genres(genres_data):
    id_col = next((col for col in genres_data if 'movieId' in col.lower()), genres_data.columns[0])
    genre_col = next((col for col in genres_data if 'genre' in col.lower()), genres_data.columns[1])
    
    genres_dict = {}
    for _, row in genres_data.iterrows():
        movie_id = row[id_col]
        genres = row[genre_col]
        if isinstance(genres, str):
            if '|' in genres:
                genre_list = genres.split('|')
            elif ',' in genres:
                genre_list = genres.split(',')
            else:
                genre_list = [genres]
            
            # Clean genres (strip whitespace, etc.)
            genre_list = [g.strip() for g in genre_list if g]
        elif isinstance(genres, (list, tuple)):
            genre_list = genres
        else:
            genre_list = []
        
        genres_dict[movie_id] = genre_list
    return genres_dict
        
# def get_active_user_input(df_ratings):
#     """Get active user input from terminal with validation"""
#     while True:
#         try:
#             print("\nDaftar 25 User Paling Aktif:")
#             active_users = df_ratings['userId'].value_counts().head(25)
#             for user_id, n_ratings in active_users.items():
#                 print(f"User ID: {user_id} - Jumlah Rating: {n_ratings}")
            
#             user_id = int(input("\nMasukkan User ID untuk melihat rekomendasi: "))
            
#             if user_id not in df_ratings['userId'].unique():
#                 print(f"Error: User ID {user_id} tidak ditemukan dalam dataset")
#                 continue
                
#             return user_id
            
#         except ValueError:
#             print("Error: Masukkan ID berupa angka")
#         except Exception as e:
#             print(f"Error: {str(e)}")

def run_svd_recommender():
    #total_start_time = time.time()
    df_ratings = ratings()
    df_movies = movies()
    df_movies_sentiment = movies_sentiment()
    train_data, test_data = train_test_split(df_ratings, test_size=0.2, random_state=42)

    # Parameter model
    n_factors = 120
    learning_rate = 1.5e-2
    regularization = 0.1
    n_epochs = 100
    sentiment_weight=0.1
    similarity_weight = 0.2# Bisa disesuaikan

    # Inisialisasi dan latih model
    svd = SVDMatrixFactorization(
        ratings=df_ratings,
        sentiment_scores=df_movies_sentiment,
        n_factors=n_factors,
        learning_rate=learning_rate,
        regularization=regularization,
        sentiment_weight= sentiment_weight,
        similarity_weight= similarity_weight,
        n_epochs=n_epochs
    )

    # svd.load_model("HybridRecommendationSystemWithBERT/saved_models/svd_model_0.8503.joblib")
    svd.load_model("saved_models/svd_model_0.8503.joblib")
    # #transform the movie df
    # if similarity_weight != 0.0 :
    #     genres_dict = to_dict_genres(df_movies)
    #     svd.load_movie_genres(genres_dict)
    #     compute_time = time.time()
    #     svd.precompute_similarity_scores()
    #     # svd.load_or_precompute_similarity()
    #     total_compute_time = time.time() - compute_time
    #     total_compute_minutes = total_compute_time / 60
    #     total_compute_hours = total_compute_minutes / 60
    #     print(f"\nTotal Compute time:{total_compute_hours:.2f} hours - {total_compute_minutes:.2f} minutes - {total_compute_time:.2f} seconds")
    
    # train_rmse_log, test_rmse_log, train_mae_log, test_mae_log = svd.train(train_data, test_data)
    # total_training_time = time.time() - total_start_time
    # total_minutes = total_training_time / 60
    # total_hours = total_minutes / 60

    # print(f"\nTotal training time:{total_hours:.2f} hours - {total_minutes:.2f} minutes - {total_training_time:.2f} seconds")
    #     # Plot learning curve
    # svd.plot_learning_curve()
    # # Evaluasi model dengan RMSE
    # print("\nEvaluasi Model:")
    # print(f"Final Train RMSE: {train_rmse_log[-1]:.4f}")
    # print(f"Final Train MAE: {train_mae_log[-1]:.4f}")
    # if test_rmse_log:
    #     print(f"Final Test RMSE: {test_rmse_log[-1]:.4f}")
    #     print(f"Final Test MAE: {test_mae_log[-1]:.4f}")
    


    # print("\n Merekomendasikan Film untuk Pengguna:")
    
    # # 1. Get active user (most active)
    # active_user = get_active_user_input(df_ratings)

    # # 2. Show top 5 movies the user already rated and genres they liked
    # print(f"\n🎬 Film yang sudah diberi rating oleh User {active_user} (Top 5):")
    # user_rated = df_ratings[df_ratings['userId'] == active_user]
    # top_rated = user_rated.sort_values(by='rating', ascending=False).head(5)
    
    # #get user liked ratings
    # # user_rated_movies = 
        

    # for _, row in top_rated.iterrows():
    #     movie_title = df_movies[df_movies['movieId'] == row['movieId']]['title'].values[0]
    #     movie_genre = df_movies[df_movies['movieId'] == row['movieId']]['genres'].values[0]
    #     print(f"✅ {movie_title} (ID: {row['movieId']}) - {movie_genre} — Rating: {row['rating']:.1f}")

    # high_rated = df_ratings[(df_ratings['userId'] == active_user) & (df_ratings['rating'] > 4.5)]

    # # Step 2: Merge with movie genres
    # rated_with_genres = pd.merge(high_rated, df_movies, on='movieId')

    # # Step 3: Expand genres (explode)
    # rows = []
    # for _, row in rated_with_genres.iterrows():
    #     genres = row['genres'].split('|')
    #     for genre in genres:
    #         if genre != '(no genres listed)':
    #             rows.append({'genre': genre.strip(), 'rating': row['rating']})

    # # Step 4: Create DataFrame from exploded genre list
    # genre_df = pd.DataFrame(rows)

    # # Step 5: Group by genre and calculate count and average rating
    # genre_stats = genre_df.groupby('genre').agg(
    #     count=('rating', 'count'),
    #     avg_rating=('rating', 'mean')
    # ).reset_index()

    # # Step 6: Sort by average rating
    # genre_stats = genre_stats.sort_values(by='avg_rating', ascending=False)

    # # Print the final genre stats
    # print(genre_stats)
    # # 3. Get recommendations
    # recommended_movies = svd.recommend_movies(active_user, top_n=10)

    # # 4. Show recommendations with titles and sentiment if available
    # print(f"\n✨ Rekomendasi film untuk User {active_user}:")

    # for movie_id, pred_rating in recommended_movies:
    #     # Get title
    #     title = df_movies[df_movies['movieId'] == movie_id]['title'].values[0]
    #     genre = df_movies[df_movies['movieId'] == movie_id]['genres'].values[0]
    #     # Get sentiment (if available)
    #     sentiment_row = df_movies_sentiment[df_movies_sentiment['movieId'] == movie_id]
    #     if not sentiment_row.empty:
    #         sentiment = sentiment_row['positive_percent'].values[0]
    #         sentiment_str = f"👍 Sentimen Positif: {sentiment:.1f}%"
    #     else:
    #         sentiment_str = "🚫 Sentimen Tidak Tersedia"

    #     print(f"🔹 {title} (ID: {movie_id}) - {genre} — Prediksi Rating: {pred_rating:.2f} | {sentiment_str}")

    return svd