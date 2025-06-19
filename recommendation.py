import pandas as pd

def get_recommendations_for_user(user_id, svd_model, df_movies, df_movies_sentiment, top_n=10):
    # Ambil rekomendasi movieId dan prediksi rating
    recommended_movies = svd_model.recommend_movies(user_id, top_n=top_n)

    # Buat dataframe hasil rekomendasi
    recommendations = []
    for movie_id, pred_rating in recommended_movies:
        # Ambil baris data film dari df_movies
        movie_info = df_movies[df_movies['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info['title'].values[0]
            genres = movie_info['genres'].values[0]
        else:
            title = f"Unknown Title (ID: {movie_id})"
            genres = "N/A"

        # Ambil sentiment
        sentiment_row = df_movies_sentiment[df_movies_sentiment['movieId'] == movie_id]
        if not sentiment_row.empty:
            positive_percent = sentiment_row['positive_percent'].values[0]
        else:
            positive_percent = None

        recommendations.append({
            'Title': title,
            'Genres': genres,
            'Predicted Rating': round(pred_rating, 2),
            'Positive Sentiment (%)': round(positive_percent, 1) if positive_percent is not None else 'N/A'
        })

    return pd.DataFrame(recommendations)