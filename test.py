from src.utils.helper import movies, full_sentiment, ratings
from src.profiling.data_profile import create_movie_dataset_profiling

df1 = ratings()
df2 = movies()
df3 = full_sentiment()
# df3 = df3[['movieId','positive_percent', 'num_reviews', 'negative_percent']]

# print(df1.head())
# print(df2.head())
# print(df3.head())

create_movie_dataset_profiling(df2, df1, df3)

