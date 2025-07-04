import pandas as pd
import numpy as np

#load_dotenv(".env")

# ratings_path = os.getenv("ratings_path")
# movies_path = os.getenv("movies_path")
# movies_sentiment_path = os.getenv("movie_sentiments")

ratings_path = "HybridRecommendationSystemWithBERT/dataset/ratings.csv"
movies_path = "HybridRecommendationSystemWithBERT/dataset/movies.csv" 
movies_sentiment_path = "HybridRecommendationSystemWithBERT/dataset/movie_sentiments_bert(fadli).csv"


def ratings():
    df_ratings = pd.read_csv(ratings_path)
    return df_ratings

def movies():
    df_movies = pd.read_csv(movies_path)
    return df_movies

def full_sentiment():
    df_movies_sentiment = pd.read_csv(movies_sentiment_path)
    return df_movies_sentiment

def movies_sentiment():
    df_movies_sentiment = pd.read_csv(movies_sentiment_path)
    return df_movies_sentiment[['movieId','positive_percent', 'num_reviews', 'negative_percent']]

def fit(self, train_data, test_data=None):
        self.train(train_data, test_data)

def calculate_novelty(recommendations, item_popularity):
    """
    Menghitung novelty berdasarkan popularitas film.
    Novelty dihitung sebagai rata-rata self-information dari film-film yang direkomendasikan.
    """
    if not recommendations:
        return 0

    total_popularity = sum(item_popularity.values())
    novelty_sum = 0

    for item_id, _ in recommendations:
        probability = item_popularity.get(item_id, 0) / total_popularity
        if probability > 0:
            self_information = -np.log2(probability)
            novelty_sum += self_information

    average_novelty = novelty_sum / len(recommendations)
    return average_novelty


