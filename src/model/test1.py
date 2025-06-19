def train(self, train_data, test_data=None):
        """
        Melatih model SVD dengan vectorized SGD untuk performa yang lebih baik
        """
        with np.errstate(over='ignore', invalid='ignore'):
            for epoch in range(self.n_epochs):
                # Shuffle data
                train_data = train_data.sample(frac=1).reset_index(drop=True)
                
                # Convert user and item IDs to indices
                user_indices = np.array([self.get_user_idx(uid) for uid in train_data['userId']])
                item_indices = np.array([self.get_item_idx(mid) for mid in train_data['movieId']])
                ratings = train_data['rating'].values
                
                # Filter out entries with unknown users or items
                valid_entries = (user_indices != None) & (item_indices != None)
                user_indices = np.array(user_indices[valid_entries], dtype=int)
                item_indices = np.array(item_indices[valid_entries], dtype=int)
                ratings = ratings[valid_entries]
                
                if len(ratings) == 0:
                    continue
                    
                # Vectorized prediction
                user_bias_vec = self.user_bias[user_indices]
                item_bias_vec = self.item_bias[item_indices]
                user_factors_vec = self.user_factors[user_indices]
                item_factors_vec = self.item_factors[item_indices]
                
                # Calculate predictions
                dot_products = np.sum(user_factors_vec * item_factors_vec, axis=1)
                predictions = self.global_mean + user_bias_vec + item_bias_vec + dot_products
                
                # Add sentiment weight if needed
                if hasattr(self, 'sentiment_weight') and self.sentiment_weight is not None:
                    predictions += self.sentiment_weight
                    
                # Calculate errors
                errors = ratings - predictions
                squared_errors = errors ** 2
                train_rmse = np.sqrt(np.mean(squared_errors))
                self.train_rmse_log.append(train_rmse)
                
                # Update biases
                user_bias_updates = self.learning_rate * (errors - self.regularization * user_bias_vec)
                item_bias_updates = self.learning_rate * (errors - self.regularization * item_bias_vec)
                
                for i in range(len(user_indices)):
                    self.user_bias[user_indices[i]] += user_bias_updates[i]
                    self.item_bias[item_indices[i]] += item_bias_updates[i]
                
                # Update latent factors - this part needs to be done as a loop since
                # we need to properly handle the dependencies between updates
                for i in range(len(user_indices)):
                    u_idx = user_indices[i]
                    i_idx = item_indices[i]
                    error = errors[i]
                    
                    # Store previous factors for update calculation
                    user_factors_prev = self.user_factors[u_idx].copy()
                    item_factors_prev = self.item_factors[i_idx].copy()
                    
                    # Update factors
                    self.user_factors[u_idx] += self.learning_rate * (error * item_factors_prev - self.regularization * user_factors_prev)
                    self.item_factors[i_idx] += self.learning_rate * (error * user_factors_prev - self.regularization * item_factors_prev)
                
                # Evaluate on test data if provided
                if test_data is not None:
                    test_predictions = []
                    test_true_ratings = []
                    
                    for _, row in test_data.iterrows():
                        user_id, movie_id, rating = row['userId'], row['movieId'], row['rating']
                        user_idx = self.get_user_idx(user_id)
                        item_idx = self.get_item_idx(movie_id)
                        
                        if user_idx is not None and item_idx is not None:
                            pred = self.predict(user_id, movie_id, 
                                                self.sentiment_weight if hasattr(self, 'sentiment_weight') else None,
                                                self.similarity_weight if hasattr(self, 'similarity_weight') else None)
                            test_predictions.append(pred)
                            test_true_ratings.append(rating)
                    
                    if test_predictions:
                        test_rmse = np.sqrt(mean_squared_error(test_true_ratings, test_predictions))
                        self.test_rmse_log.append(test_rmse)
                        
                        # Optional: Early stopping check
                        if hasattr(self, '_check_early_stopping'):
                            if self._check_early_stopping(test_rmse):
                                break

                if (epoch + 1) % 10 == 0:
                    if test_rmse is not None:
                        print(f"Epoch {epoch+1}/{self.n_epochs} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
                    else:
                        print(f"Epoch {epoch+1}/{self.n_epochs} - Train RMSE: {train_rmse:.4f}")
                    
        return self.train_rmse_log, self.test_rmse_log