import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import functools
import os
import pickle
import gzip
import sys 
from scipy.sparse import csr_matrix
from itertools import combinations
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import mean_squared_error

from src.utils.helper import calculate_novelty
from src.utils.helper import movies ,movies_sentiment, ratings


df_movies = movies()
df_ratings = ratings()
df_sentiment = movies_sentiment()

#global 
global_genre_sets = {}

#grlobal function
def _calculate_batch(movie_pairs):
    """Helper for parallel processing"""
    movie_id1, movie_id2 = movie_pairs
    genres1 = global_genre_sets(movie_id1, set())
    genres2 = global_genre_sets(movie_id2, set())
        
    if not genres1 or not genres2:
        similarity = 0.0
    elif movie_id1 == movie_id2:
        similarity = 1.0
    else:
        intersection = len(genres1 & genres2)
        union = len(genres1) + len(genres2) - intersection
        similarity = intersection / union if union > 0 else 0.0  
        return(movie_id1,movie_id2,similarity)
    
class SVDMatrixFactorization:
    def __init__(self, ratings, sentiment_scores, n_factors, learning_rate, regularization, sentiment_weight, similarity_weight, n_epochs):
        self.ratings = ratings
        self.n_users = ratings['userId'].nunique()
        self.n_items = ratings['movieId'].nunique()
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs

        # Mapping untuk user_id dan movie_id
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(ratings['userId'].unique())}
        self.item_mapping = {movie_id: idx for idx, movie_id in enumerate(ratings['movieId'].unique())}
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: movie_id for movie_id, idx in self.item_mapping.items()}
        #user rated movies
        self.user_rated_movies = ratings.groupby('userId')['movieId'].apply(set).to_dict()

        # Inisialisasi matriks latent factor
        np.random.seed(42)
        self.user_factors = np.random.normal(scale=0.1, size=(self.n_users, self.n_factors))
        self.item_factors = np.random.normal(scale=0.1, size=(self.n_items, self.n_factors))
        #movie genres
        self.movie_genres = {}
        self.genre_sets = {} 
        # Global bias, user bias, item bias
        self.global_mean = np.mean(ratings['rating'])
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)

        # Log untuk menyimpan error
        self.train_rmse_log = []
        self.train_mae_log = []
        self.test_rmse_log = []
        self.test_mae_log = []

        #added parameters for weighted sentiment analysis
        self.sentiment_features = None
        self.sentiment_weight = sentiment_weight
        self.sentiment_scores = sentiment_scores.set_index('movieId')
        #added parameters for similarity weight
        self.similarity_weight = similarity_weight
        self.genre_similarity_matrix = defaultdict(dict)
        self.genre_to_index = {} 
        
    #precompute similarity score
    def precompute_similarity_scores(self, batch_size=1000):
        """
        Precompute all pairwise similarities with optimized memory usage
        """
        print("Precomputing all similarities...")
        start_time = time.time()
        
        # Make sure we have the genre matrix
        if not hasattr(self, 'genre_matrix'):
            self.create_genre_matrix()
            
        movie_ids = list(self.movie_to_index.keys())
        n_movies = len(movie_ids)
        
        # Calculate total pairs for progress tracking
        total_pairs = (n_movies * (n_movies - 1)) // 2
        processed_pairs = 0
        last_report = 0
        
        print(f"Total movies: {n_movies}, Total pairs to compute: {total_pairs}")
        
        # Process in batches to manage memory
        for i in range(0, n_movies, batch_size):
            batch_start_time = time.time()
            batch_end = min(i + batch_size, n_movies)
            batch_movie_ids = movie_ids[i:batch_end]
            batch_indices = [self.movie_to_index[m] for m in batch_movie_ids]
            
            # Get genre vectors for this batch
            batch_genres = self.genre_matrix[batch_indices]
            
            # For each movie in batch, compute similarities with ALL movies
            # (we'll filter to keep only the upper triangle)
            for batch_idx, movie_id1 in enumerate(batch_movie_ids):
                global_idx = i + batch_idx
                
                # Get this movie's genres
                movie1_genres = self.genre_sets[movie_id1]
                if not movie1_genres:
                    continue
                    
                # Only compute for movies with higher indices (upper triangle)
                for j in range(global_idx + 1, n_movies):
                    movie_id2 = movie_ids[j]
                    movie2_genres = self.genre_sets[movie_id2]
                    
                    if not movie2_genres:
                        continue
                    
                    # Compute Jaccard similarity directly
                    intersection = len(movie1_genres & movie2_genres)
                    union = len(movie1_genres) + len(movie2_genres) - intersection
                    
                    if union > 0:
                        similarity = intersection / union
                        
                        # Only store non-zero similarities to save memory
                        if similarity > 0.2:
                            if movie_id1 not in self.genre_similarity_matrix:
                                self.genre_similarity_matrix[movie_id1] = {}
                            if movie_id2 not in self.genre_similarity_matrix:
                                self.genre_similarity_matrix[movie_id2] = {}
                                
                            self.genre_similarity_matrix[movie_id1][movie_id2] = similarity
                            self.genre_similarity_matrix[movie_id2][movie_id1] = similarity
                    
                    processed_pairs += 1
                
                # Report progress at reasonable intervals
                if processed_pairs - last_report >= total_pairs // 100:
                    pct_complete = processed_pairs / total_pairs * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / pct_complete) * (100 - pct_complete) if pct_complete > 0 else float('inf')
                    
                    print(f"Progress: {processed_pairs:,}/{total_pairs:,} pairs ({pct_complete:.1f}%) - "
                        f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
                    last_report = processed_pairs
            
            batch_time = time.time() - batch_start_time
            print(f"Processed batch {i//batch_size + 1}/{(n_movies-1)//batch_size + 1} "
                f"in {batch_time:.2f}s")
        
        # Report final statistics
        total_time = time.time() - start_time
        similarity_count = sum(len(v) for v in self.genre_similarity_matrix.values())
        memory_usage = sys.getsizeof(self.genre_similarity_matrix) / (1024 * 1024)  # MB
        
        print(f"Precomputation complete!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Total similarities stored: {similarity_count:,}")
        print(f"Estimated memory usage: {memory_usage:.1f} MB")
        
        return self
    # def precompute_similarity_scores(self):
    #     all_movie_ids = list(self.movie_genres.keys())
    #     n = len(all_movie_ids)

    #     # Initialize the matrix
    #     for movie_id in all_movie_ids:
    #         self.genre_similarity_matrix[movie_id] = {}

    #     # Only compute upper triangular matrix
    #     for i in range(n):
    #         movie_id1 = all_movie_ids[i]
    #         for j in range(i+1, n):  # Start from i+1 to avoid redundancy
    #             movie_id2 = all_movie_ids[j]
    #             similarity = self.calculate_genre_similarity(movie_id1, movie_id2)

    #             # Store the value in both directions (symmetry)
    #             self.genre_similarity_matrix[movie_id1][movie_id2] = similarity
    #             self.genre_similarity_matrix[movie_id2][movie_id1] = similarity
    
    # def precompute_similarity_scores(self, n_jobs=4):
    #     """Precompute with parallel processing"""
    #     print("Computing...")
    #     global global_genre_sets
    #     all_movie_ids = list(self.movie_genres.keys())
    #     n = len(all_movie_ids)
    #     global_genre_sets = self.genre_sets
        
    #     # Generate all pairs to compute (upper triangle)
    #     pairs = list(combinations(all_movie_ids, 2))
    #     # Split into batches
    #     # batch_size = max(1, len(pairs_to_compute) // (n_jobs * 10))
    #     # batches = [pairs_to_compute[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
        
    #     # Process in parallel
    #     with ProcessPoolExecutor(max_workers=n_jobs) as executor:
    #         results = executor.map(_calculate_batch, pairs)
            
    #     # Combine results
    #     for movie_id1, movie_id2, similarity in results:
    #         if similarity > 0.01: 
    #             self.genre_similarity_matrix[movie_id1][movie_id2] = similarity
    #             self.genre_similarity_matrix[movie_id2][movie_id1] = similarity
    #     #store similarity
    #     path="genre_similarity_matrix.pkl"
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     with open("genre_similarity_matrix.pkl", "wb") as f:
    #         pickle.dump(dict(self.genre_similarity_matrix), f)
    #     print("Computed!")
        

    def fit(self, train_data, test_data=None):
        self.train(train_data, test_data)

    def get_user_idx(self, user_id):
        return self.user_mapping.get(user_id)

    def get_item_idx(self, movie_id):
        return self.item_mapping.get(movie_id)

    def get_original_user_id(self, user_idx):
        return self.reverse_user_mapping.get(user_idx)

    def get_original_movie_id(self, item_idx):
        return self.reverse_item_mapping.get(item_idx)
    
    def get_similar_movies(self, movie_id, top_n=10):
        """
        Menemukan film yang mirip berdasarkan latent factors
        """
        item_idx = self.get_item_idx(movie_id)

        if item_idx is None:
            print(f"Film dengan ID {movie_id} tidak ditemukan.")
            return []

        # Hitung cosine similarity antara film yang dicari dan semua film lainnya
        target_factors = self.item_factors[item_idx]
        similarities = []

        for idx, factors in enumerate(self.item_factors):
            if idx != item_idx:
                similarity = np.dot(target_factors, factors) / (np.linalg.norm(target_factors) * np.linalg.norm(factors))
                similarities.append((self.get_original_movie_id(idx), similarity))

        # Urutkan berdasarkan similaritas dan kembalikan top_n
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    
    def recommend_movies(self, user_id, top_n=10, exclude_rated=True):
        """
        Rekomendasikan film untuk user tertentu
        """
        user_idx = self.get_user_idx(user_id)

        if user_idx is None:
            print(f"User dengan ID {user_id} tidak ditemukan.")
            return []

        # Dapatkan film yang sudah dinilai oleh user jika exclude_rated=True
        rated_movies = set()
        if exclude_rated:
            rated_movies = set(self.ratings[self.ratings['userId'] == user_id]['movieId'])

        predictions = []
        # Prediksi rating untuk semua film
        for movie_idx in range(self.n_items):
            movie_id = self.get_original_movie_id(movie_idx)

            # Lewati film yang sudah dinilai jika exclude_rated=True
            if exclude_rated and movie_id in rated_movies:
                continue

            pred_rating = self.predict(user_id, movie_id, self.sentiment_weight, self.similarity_weight)
            predictions.append((movie_id, pred_rating))

        return sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]


    def load_movie_genres(self, genres_data):
        print("Loading genres...")
        # Load genres data
        start_time = time.time()
        self.movie_genres = genres_data
        
        # Precompute all genre sets once
        for movie_id, genres in self.movie_genres.items():
            self.genre_sets[movie_id] = set(genres)
        all_genres = set()
        for genres in self.genre_sets.values():
            all_genres.update(genres)
            
        self.genre_to_index = {genre: idx for idx, genre in enumerate(sorted(all_genres))}
        
        print(f"Loaded genres for {len(self.movie_genres)} movies with {len(self.genre_to_index)} unique genres")
        print(f"Loading time: {time.time() - start_time:.2f} seconds")
        # print("Genre sets:\n")
        # print(self.genre_sets)
        return self
        print("Loaded!")
    
    def create_genre_matrix(self):
        """Create a sparse matrix representation of movies x genres"""
        print("Creating genre matrix...")
        start_time = time.time()
        
        movie_ids = list(self.movie_genres.keys())
        self.movie_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
        
        # Create sparse matrix: rows=movies, cols=genres
        rows, cols, data = [], [], []
        
        for movie_id, genres in self.movie_genres.items():
            movie_idx = self.movie_to_index[movie_id]
            for genre in genres:
                if genre in self.genre_to_index:  # Handle case where genre might not be in mapping
                    genre_idx = self.genre_to_index[genre]
                    rows.append(movie_idx)
                    cols.append(genre_idx)
                    data.append(1)
        
        self.genre_matrix = csr_matrix(
            (data, (rows, cols)), 
            shape=(len(movie_ids), len(self.genre_to_index))
        )
        
        print(f"Created genre matrix with shape {self.genre_matrix.shape}")
        print(f"Matrix creation time: {time.time() - start_time:.2f} seconds")
        
        return self
    
    def calculate_genre_similarity(self, movie_id1, movie_id2):
        """Optimized genre similarity calculation"""
        # Check if already in matrix
        if movie_id2 in self.genre_similarity_matrix.get(movie_id1, {}):
            return self.genre_similarity_matrix[movie_id1][movie_id2]
            
        if movie_id1 == movie_id2:
            return 1.0
            
        # Use precomputed sets
        genres1 = self.genre_sets.get(movie_id1, set())
        genres2 = self.genre_sets.get(movie_id2, set())
        
        if not genres1 or not genres2:
            return 0.0
            
        # Faster calculation of intersection size and union size
        intersection_size = len(genres1 & genres2)  # Bitwise & is faster for sets
        union_size = len(genres1) + len(genres2) - intersection_size
        
        similarity = intersection_size / union_size if union_size > 0 else 0.0
        
        # Cache result
        self.genre_similarity_matrix[movie_id1][movie_id2] = similarity
        self.genre_similarity_matrix[movie_id2][movie_id1] = similarity
        
        return similarity
      
    def get_similar_movies_enhanced(self, movie_id, top_n=10, alpha=0.7):
        """
        Menemukan film yang mirip berdasarkan kombinasi latent factors dan kesamaan genre.

        Parameters:
        - movie_id: ID film yang menjadi referensi
        - top_n: Jumlah film yang akan dikembalikan
        - alpha: Bobot untuk similarity latent factor (1-alpha untuk similarity genre)

        Returns:
        - List of tuples: (movie_id, combined_similarity)
        """
        item_idx = self.get_item_idx(movie_id)

        if item_idx is None:
            print(f"Film dengan ID {movie_id} tidak ditemukan.")
            return []

        if not hasattr(self, 'movie_genres'):
            print("Data genre belum dimuat. Menggunakan similarity berbasis latent factor saja.")
            return self.get_similar_movies(movie_id, top_n)

        target_factors = self.item_factors[item_idx]
        similarities = []

        for idx, factors in enumerate(self.item_factors):
            if idx != item_idx:
                other_movie_id = self.get_original_movie_id(idx)
                latent_similarity = np.dot(target_factors, factors) / (np.linalg.norm(target_factors) * np.linalg.norm(factors))
                genre_similarity = self.calculate_genre_similarity(movie_id, other_movie_id)
                combined_similarity = alpha * latent_similarity + (1 - alpha) * genre_similarity
                similarities.append((other_movie_id, combined_similarity, latent_similarity, genre_similarity))

        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

        print(f"\nBreakdown similarity untuk film ID {movie_id}:")
        print(f"{'Movie ID':<10} {'Combined':<10} {'Latent':<10} {'Genre':<10}")
        print("-" * 40)
        for movie_id, combined, latent, genre in sorted_similarities[:5]:
            print(f"{movie_id:<10} {combined:.4f} {latent:.4f} {genre:.4f}")

        return [(movie_id, similarity) for movie_id, similarity, _, _ in sorted_similarities]
    
    def load_movie_titles(self, movies_path):
        """
        Memuat judul film untuk meningkatkan interpretasi hasil.

        Parameters:
        - movies_path: Path ke file CSV yang berisi data film
        """
        df_movies = pd.read_csv(movies_path)
        self.movie_titles = {}
        for _, row in df_movies.iterrows():
            movie_id = row['movieId']
            title = row['title']
            self.movie_titles[movie_id] = title

        print(f"Berhasil memuat judul untuk {len(self.movie_titles)} film")
        return self.movie_titles
    
    def predict(self, user_id, movie_id, sentiment_weight, similarity_weight):
        """
        Memprediksi rating untuk pasangan user-movie
        """
        user_idx = self.get_user_idx(user_id)
        item_idx = self.get_item_idx(movie_id)

        max_rating = 5.0  # Assuming 5-star rating scale

        # If unknown user or item
        if user_idx is None or item_idx is None:
            return self.global_mean

        # Retrieve sentiment safely
        sentiment_score = None
        try:
            if movie_id in self.sentiment_scores.index:
                sentiment_score = self.sentiment_scores.at[movie_id, 'positive_percent']
        except Exception as e:
            print(f"Sentiment lookup failed for movie_id={movie_id}: {e}")
            sentiment_score = None

        #get rated movies
        rated_movies = self.user_rated_movies.get(user_id, set())
        similarities = []
        for rated_movie_id in rated_movies:
            if rated_movie_id in self.genre_similarity_matrix and movie_id in self.genre_similarity_matrix[rated_movie_id]:
                similarity = self.genre_similarity_matrix[rated_movie_id][movie_id]
                similarities.append(similarity)

        if similarities:
            similarity_score = np.mean(similarities)
        else:
            similarity_score = 0.0


        # CF prediction
        cf_pred = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx] + \
                  np.dot(self.user_factors[user_idx], self.item_factors[item_idx])

        cf_weight = 1.0 - self.sentiment_weight - self.similarity_weight
        # Blend with sentiment if available
        if sentiment_score is not None:
            sentiment_score /= 100  # Normalize to 0–1
            pred = cf_weight * cf_pred + self.sentiment_weight * (sentiment_score * max_rating) + self.similarity_weight * similarity_score
        else:
            pred = (1 - self.similarity_weight) * cf_pred + self.similarity_weight * similarity_score
        # Clamp to valid rating range
        return max(1, min(5, pred))

    def save_model(self, filepath):
        
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_bias': self.user_bias,
            'item_bias': self.item_bias,
            'user_to_index': self.user_mapping,             # ← use your existing name
            'item_to_index': self.item_mapping,
            'index_to_user': self.reverse_user_mapping,
            'index_to_item': self.reverse_item_mapping,
            'genre_similarity_matrix': getattr(self, 'genre_similarity_matrix', None),
            'movie_genres': getattr(self, 'movie_genres', None),
            'user_genre_counts': getattr(self, 'user_genre_counts', None)
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True) 
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """
        Memuat model yang sudah dilatih dari sebuah file.
        Fungsi ini bisa menangani file pickle biasa (.pkl) atau yang terkompresi (.pkl.gz).
        """
        print(f"Mencoba memuat model dari: {filepath}")
    
        try:
            # Cek apakah file terkompresi atau tidak berdasarkan ekstensinya
            if filepath.endswith(".gz"):
                # Jika terkompresi, gunakan gzip.open
                with gzip.open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
            else:
                # Jika tidak, gunakan open biasa
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)

            self.user_factors = model_data['user_factors']
            self.item_factors = model_data['item_factors']
            self.user_bias = model_data['user_bias']
            self.item_bias = model_data['item_bias']
            self.user_mapping = model_data['user_to_index']
            self.item_mapping = model_data['item_to_index']
            self.reverse_user_mapping = model_data['index_to_user']
            self.reverse_item_mapping = model_data['index_to_item']
            self.genre_similarity_matrix = model_data.get('genre_similarity_matrix', None)
            self.movie_genres = model_data.get('movie_genres', None)
            self.user_genre_counts = model_data.get('user_genre_counts', None)

            print(f"Model loaded from {filepath}")

        except FileNotFoundError:
            print(f"ERROR: File model tidak ditemukan di '{filepath}'. Pastikan path sudah benar.")
        except Exception as e:
            print(f"ERROR: Terjadi kesalahan saat memuat model: {e}")

    def train(self, train_data, test_data=None, batch_size=1024):
        """
        Melatih model SVD dengan SGD menggunakan mini-batch processing
        """
        print("Mulai pelatihan model...")
        best_rmse = float('inf')
        best_epoch = -1
        no_improve_count = 0
        patience = 15  # or whatever you want
        delta = 1e-4  # minimum change to count as improvement
        best_model_state = None
        
        for epoch in range(self.n_epochs):
            #add time
            start_time = time.time()
            # Shuffle data once per epoch
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            
            # Process in mini-batches
            n_batches = len(train_data) // batch_size + (1 if len(train_data) % batch_size != 0 else 0)
            epoch_errors = []
            epoch_abs_errors = []

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(train_data))
                batch_data = train_data.iloc[start_idx:end_idx]

                # Convert to numpy arrays for faster processing
                user_ids = batch_data['userId'].values
                movie_ids = batch_data['movieId'].values
                ratings = batch_data['rating'].values

                # Get indices in bulk
                user_indices = np.array([self.get_user_idx(uid) for uid in user_ids])
                item_indices = np.array([self.get_item_idx(mid) for mid in movie_ids])

                # Filter valid indices
                mask = (user_indices != None) & (item_indices != None)
                user_indices = user_indices[mask]
                item_indices = item_indices[mask]
                batch_ratings = ratings[mask]

                if len(batch_ratings) == 0:
                    continue

                # Compute predictions for entire batch
                predictions = np.zeros(len(batch_ratings))
                for i in range(len(batch_ratings)):
                    predictions[i] = self.predict(
                        user_ids[i], 
                        movie_ids[i], 
                        self.sentiment_weight,
                        self.similarity_weight
                    )

                # Compute errors
                errors = batch_ratings - predictions
                epoch_errors.extend(errors ** 2)
                epoch_abs_errors.extend(np.abs(errors))

                # Update parameters
                for i in range(len(batch_ratings)):
                    u_idx = user_indices[i]
                    i_idx = item_indices[i]
                    error = errors[i]

                    # Update biases
                    self.user_bias[u_idx] += self.learning_rate * (error - self.regularization * self.user_bias[u_idx])
                    self.item_bias[i_idx] += self.learning_rate * (error - self.regularization * self.item_bias[i_idx])

                    # Update latent factors
                    user_factors_prev = self.user_factors[u_idx].copy()
                    item_factors_prev = self.item_factors[i_idx].copy()

                    self.user_factors[u_idx] += self.learning_rate * (error * item_factors_prev - self.regularization * user_factors_prev)
                    self.item_factors[i_idx] += self.learning_rate * (error * user_factors_prev - self.regularization * item_factors_prev)

            # Calculate RMSE for epoch
            train_rmse = np.sqrt(np.mean(epoch_errors))
            self.train_rmse_log.append(train_rmse)
            
            train_mae = np.mean(epoch_abs_errors)
            self.train_mae_log.append(train_mae)
            
            if test_data is not None:
                test_rmse, test_mae = self._evaluate_test_data(test_data)
                self.test_rmse_log.append(test_rmse)
                self.test_mae_log.append(test_mae)
            # Evaluate test data less frequently
            if (epoch + 1) % 5 == 0:
                # test_rmse = self._evaluate_test_data(test_data)
                # self.test_rmse_log.append(test_rmse)
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{self.n_epochs} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}\nTrain MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}, \nTime: {elapsed_time:.2f} sec")
                start_time = time.time()
            elif (epoch + 1) % 5 == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{self.n_epochs} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}\nTrain MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}, \nTime: {elapsed_time:.2f} sec")
                start_time = time.time()
            if test_rmse + delta < best_rmse:
                best_rmse = test_rmse
                best_mae = test_mae
                best_train_rmse = train_rmse
                best_train_mae = train_mae
                best_epoch = epoch
                no_improve_count = 0
                best_model_state = {
                    'user_factors': self.user_factors.copy(),
                    'item_factors': self.item_factors.copy(),
                    'user_bias': self.user_bias.copy(),
                    'item_bias': self.item_bias.copy()
                }
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
        # print(no_improve_count)
        if best_model_state is not None:
            self.user_factors = best_model_state['user_factors']
            self.item_factors = best_model_state['item_factors']
            self.user_bias = best_model_state['user_bias']
            self.item_bias = best_model_state['item_bias']
            self.best_epoch = best_epoch+1
            print(f"Model restored to best state from epoch {best_epoch + 1} with:\n")
            print(f"Train RMSE = {best_train_rmse:.4f}\n")
            print(f"Train MAE = {best_train_mae:.4f}\n")
            print(f"Test RMSE = {best_rmse:.4f}\n")
            print(f"Test MAE = {best_mae:.4f}\n")
            filename = f"saved_models/svd_model_{best_rmse:.4f}.pkl"
            self.save_model(filepath=filename)

        print("Train Finished!")
        return self.train_rmse_log, self.test_rmse_log, self.train_mae_log, self.test_mae_log
    
    def plot_learning_curve(self):
        """
        Plot learning curve untuk melihat performa model
        """
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(1, self.n_epochs + 1), self.train_rmse_log, label='Training RMSE')
        # if self.test_rmse_log:
        #     plt.plot(range(1, self.n_epochs + 1), self.test_rmse_log, label='Test RMSE')
        # plt.xlabel('Epoch')
        # plt.ylabel('RMSE')
        # plt.title('Learning Curve')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

            # Plot RMSE
        axs[0].plot(range(1, len(self.train_rmse_log) + 1), self.train_rmse_log, label='Training RMSE')
        if self.test_rmse_log:
            axs[0].plot(range(1, len(self.test_rmse_log) + 1), self.test_rmse_log, label='Test RMSE')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('RMSE')
        axs[0].set_title('Learning Curve - RMSE')
        axs[0].legend()
        axs[0].grid(True)

        # Plot MAE
        axs[1].plot(range(1, len(self.train_mae_log) + 1), self.train_mae_log, label='Training MAE', color='orange')
        if hasattr(self, 'test_mae_log') and self.test_mae_log:
            axs[1].plot(range(1, len(self.test_mae_log) + 1), self.test_mae_log, label='Test MAE', color='red')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('MAE')
        axs[1].set_title('Learning Curve - MAE')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()
    def _evaluate_test_data(self, test_data):
        """Helper method to evaluate test data"""
        test_predictions = []
        test_true_ratings = []
        
        # Process test data in batches
        batch_size = 1024
        for start_idx in range(0, len(test_data), batch_size):
            batch = test_data.iloc[start_idx:start_idx + batch_size]
            
            for _, row in batch.iterrows():
                user_id, movie_id, rating = row['userId'], row['movieId'], row['rating']
                if self.get_user_idx(user_id) is not None and self.get_item_idx(movie_id) is not None:
                    pred = self.predict(user_id, movie_id, self.sentiment_weight, self.similarity_weight)
                    test_predictions.append(pred)
                    test_true_ratings.append(rating)

        return np.sqrt(mean_squared_error(test_true_ratings, test_predictions)), np.mean(np.abs(np.array(test_true_ratings) - np.array(test_predictions)))
    
    def recommend_movies(self, user_id, top_n=10, exclude_rated=True):
        """
        Rekomendasikan film untuk user tertentu
        """
        user_idx = self.get_user_idx(user_id)

        if user_idx is None:
            print(f"User dengan ID {user_id} tidak ditemukan.")
            return []

        # Dapatkan film yang sudah dinilai oleh user jika exclude_rated=True
        rated_movies = set()
        if exclude_rated:
            rated_movies = set(self.ratings[self.ratings['userId'] == user_id]['movieId'])

        predictions = []
        # Prediksi rating untuk semua film
        for movie_idx in range(self.n_items):
            movie_id = self.get_original_movie_id(movie_idx)

            # Lewati film yang sudah dinilai jika exclude_rated=True
            if exclude_rated and movie_id in rated_movies:
                continue

            pred_rating = self.predict(user_id, movie_id, self.sentiment_weight, self.similarity_weight)
            predictions.append((movie_id, pred_rating))

        return sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
